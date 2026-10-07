"""Page quadrilateral -> perspective warp -> deskew -> upright orientation.

Images are uint8 grayscale or OpenCV BGR arrays. All transforms use pixel (x, y)
coordinates and map the original image to the returned image.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import shutil
import subprocess
from typing import Callable

import cv2
import numpy as np

OrientationDetector = Callable[[np.ndarray], int | None]


@dataclass(frozen=True)
class RectificationConfig:
    detection_max_side: int = 1600
    min_page_area: float = 0.35
    min_page_contrast: float = 15.0
    max_skew_degrees: float = 15.0
    min_line_consensus: float = 0.6
    detect_page: bool = True
    deskew: bool = True

    def __post_init__(self) -> None:
        if self.detection_max_side < 64:
            raise ValueError("detection_max_side must be at least 64")
        if not 0 < self.min_page_area <= 1:
            raise ValueError("min_page_area must be in (0, 1]")
        if not 0 <= self.min_page_contrast <= 255:
            raise ValueError("min_page_contrast must be in [0, 255]")
        if not 0 < self.max_skew_degrees < 45:
            raise ValueError("max_skew_degrees must be in (0, 45)")
        if not 0 < self.min_line_consensus <= 1:
            raise ValueError("min_line_consensus must be in (0, 1]")


@dataclass
class RectificationResult:
    image: np.ndarray
    transform: np.ndarray
    page_quad: np.ndarray | None
    deskew_degrees: float | None
    clockwise_rotation: int | None
    warnings: tuple[str, ...]

    def map_points(self, points: np.ndarray, *, inverse: bool = False) -> np.ndarray:
        """Map (..., 2) points; inverse=True maps predictions to the input."""
        points = np.asarray(points, dtype=np.float64)
        if points.ndim < 2 or points.shape[-1] != 2:
            raise ValueError("points must have shape (..., 2)")
        if not np.isfinite(points).all():
            raise ValueError("points must be finite")
        matrix = np.linalg.inv(self.transform) if inverse else self.transform
        homogeneous = np.column_stack(
            (points.reshape(-1, 2), np.ones(points.size // 2))
        )
        mapped = homogeneous @ matrix.T
        if np.any(np.abs(mapped[:, 2]) < 1e-12):
            raise ValueError("Some points map to infinity")
        return (mapped[:, :2] / mapped[:, 2:]).reshape(points.shape)

    def map_boxes(self, boxes: np.ndarray, *, inverse: bool = False) -> np.ndarray:
        """Map pixel xyxy boxes to quadrilaterals (N, 4, 2), preserving geometry."""
        boxes = np.asarray(boxes, dtype=np.float64)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("boxes must have shape (N, 4) in pixel xyxy format")
        if np.any(boxes[:, 2:] < boxes[:, :2]):
            raise ValueError("boxes must have ordered xyxy coordinates")
        corners = boxes[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(-1, 4, 2)
        return self.map_points(corners, inverse=inverse)


def _gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _thumbnail(image: np.ndarray, max_side: int) -> np.ndarray:
    scale = min(1.0, max_side / max(image.shape[:2]))
    if scale == 1:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _order_quad(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("page_quad must contain four finite (x, y) points")
    centered = points - points.mean(axis=0)
    points = points[np.argsort(np.arctan2(centered[:, 1], centered[:, 0]))]
    points = np.roll(points, -int(np.argmin(points.sum(axis=1))), axis=0)
    if not cv2.isContourConvex(points.reshape(-1, 1, 2)) or cv2.contourArea(points) < 4:
        raise ValueError("page_quad must be a nondegenerate convex quadrilateral")
    return points


def detect_page_quad(
    image: np.ndarray, config: RectificationConfig
) -> np.ndarray | None:
    """Find a large bright page against a darker background; otherwise abstain.

    A contrast check rejects most internal tables and boxes on full-page scans.
    This heuristic expects one flat, fully visible page, not overlapping pages.
    """
    gray = _gray(_thumbnail(image, config.detection_max_side))
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.morphologyEx(
        cv2.Canny(blur, 50, 150), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    _, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours = []
    for mask in (edges, bright):
        found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(found)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < gray.size * config.min_page_area:
            continue
        quad = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(quad) != 4 or not cv2.isContourConvex(quad):
            continue
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, quad.reshape(4, 2), 255)
        # Require enough surrounding background to assess a real page boundary.
        outside = gray[mask == 0]
        if outside.size < gray.size * 0.02:
            continue
        inside = gray[mask != 0]
        if (
            float(np.median(inside)) - float(np.median(outside))
            < config.min_page_contrast
        ):
            continue
        scale_xy = np.array(
            [image.shape[1] / gray.shape[1], image.shape[0] / gray.shape[0]]
        )
        return _order_quad(quad.reshape(4, 2) * scale_xy)
    return None


def warp_page(
    image: np.ndarray, page_quad: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rectify a quadrilateral, retaining approximately its source resolution."""
    quad = _order_quad(page_quad)
    height, width = image.shape[:2]
    if np.any(quad < 0) or np.any(quad > np.array([width - 1, height - 1])):
        raise ValueError("page_quad must lie within the input image")
    tl, tr, br, bl = quad
    width = max(
        2, int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))) + 1
    )
    height = max(
        2, int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))) + 1
    )
    target = np.float32(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    )
    matrix = cv2.getPerspectiveTransform(quad, target)
    return (
        cv2.warpPerspective(
            image, matrix, (width, height), borderValue=(255, 255, 255)
        ),
        matrix,
    )


def estimate_skew(image: np.ndarray, config: RectificationConfig) -> float | None:
    """Estimate the small OpenCV correction angle from text and ruling lines.

    Horizontal and vertical lines vote for the same residual angle, so this also
    works before a 90-degree orientation correction. None means insufficient
    evidence; zero means the detected lines are already aligned.
    """
    gray = _gray(_thumbnail(image, config.detection_max_side))
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    joined = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((1, 7), np.uint8))
    # Combine raw edges (including vertical rules) with joined text-line edges.
    edges = cv2.bitwise_or(cv2.Canny(gray, 50, 150), cv2.Canny(joined, 50, 150))
    length = max(20, int(min(gray.shape) * 0.08))
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 720,
        threshold=max(20, length // 2),
        minLineLength=length,
        maxLineGap=10,
    )
    if lines is None:
        return None
    delta = lines[:, 0, 2:].astype(float) - lines[:, 0, :2]
    angles = (np.degrees(np.arctan2(delta[:, 1], delta[:, 0])) + 45) % 90 - 45
    weights = np.linalg.norm(delta, axis=1)
    valid = np.abs(angles) <= config.max_skew_degrees
    angles, weights = angles[valid], weights[valid]
    if len(angles) < 3:
        return None
    bin_count = max(1, int(np.ceil(2 * config.max_skew_degrees / 0.5)))
    bins = np.linspace(-config.max_skew_degrees, config.max_skew_degrees, bin_count + 1)
    votes, boundaries = np.histogram(angles, bins=bins, weights=weights)
    peak = int(np.argmax(votes))
    center = (boundaries[peak] + boundaries[peak + 1]) / 2
    inliers = np.abs(angles - center) <= 1.5
    if (
        np.count_nonzero(inliers) < 3
        or weights[inliers].sum() / weights.sum() < config.min_line_consensus
    ):
        return None
    order = np.argsort(angles[inliers])
    selected, selected_weights = angles[inliers][order], weights[inliers][order]
    index = np.searchsorted(np.cumsum(selected_weights), selected_weights.sum() / 2)
    return float(selected[index])


def rotate_expanded(image: np.ndarray, degrees: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate counterclockwise with white padding, without clipping corners."""
    height, width = image.shape[:2]
    matrix = np.eye(3)
    matrix[:2] = cv2.getRotationMatrix2D(
        ((width - 1) / 2, (height - 1) / 2), degrees, 1
    )
    corners = (
        np.array(
            [
                [0, 0, 1],
                [width - 1, 0, 1],
                [width - 1, height - 1, 1],
                [0, height - 1, 1],
            ]
        )
        @ matrix.T
    )
    lower = corners[:, :2].min(axis=0)
    size = np.ceil(corners[:, :2].max(axis=0) - lower).astype(int) + 1
    matrix[:2, 2] -= lower
    return (
        cv2.warpAffine(
            image, matrix[:2], tuple(int(x) for x in size), borderValue=(255, 255, 255)
        ),
        matrix,
    )


def normalize_orientation(
    image: np.ndarray, clockwise: int
) -> tuple[np.ndarray, np.ndarray]:
    """Apply an exact quarter-turn correction and return its pixel transform."""
    height, width = image.shape[:2]
    matrices = {
        0: np.eye(3),
        90: np.array([[0, -1, height - 1], [1, 0, 0], [0, 0, 1]]),
        180: np.array([[-1, 0, width - 1], [0, -1, height - 1], [0, 0, 1]]),
        270: np.array([[0, 1, 0], [-1, 0, width - 1], [0, 0, 1]]),
    }
    if clockwise not in matrices:
        raise ValueError("Clockwise orientation correction must be 0, 90, 180, or 270")
    return np.rot90(image, k=-(clockwise // 90)).copy(), matrices[clockwise].astype(
        float
    )


def tesseract_orientation(
    image: np.ndarray, *, min_confidence: float = 15.0
) -> int | None:
    """Optional upright detector; requires Tesseract and its osd language data.

    Returns the clockwise correction, or None if unavailable/low confidence.
    A caller can replace this with a learned classifier of the same signature.
    """
    executable = shutil.which("tesseract")
    if executable is None:
        return None
    ok, encoded = cv2.imencode(".png", _gray(_thumbnail(image, 1600)))
    if not ok:
        return None
    try:
        completed = subprocess.run(
            [executable, "stdin", "stdout", "--psm", "0", "-l", "osd", "--dpi", "300"],
            input=encoded.tobytes(),
            capture_output=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = completed.stdout.decode("utf-8", errors="replace")
    rotation = re.search(r"Rotate:\s*(\d+)", output)
    confidence = re.search(r"Orientation confidence:\s*([\d.]+)", output)
    if completed.returncode or not rotation or not confidence:
        return None
    correction = int(rotation[1])
    return (
        correction
        if correction in (0, 90, 180, 270) and float(confidence[1]) >= min_confidence
        else None
    )


def rectify_document(
    image: np.ndarray,
    *,
    config: RectificationConfig | None = None,
    page_quad: np.ndarray | None = None,
    orientation_detector: OrientationDetector | None = None,
    clockwise_rotation: int | None = None,
) -> RectificationResult:
    """Run four independent stages; preserve input when geometric evidence fails.

    A supplied page_quad bypasses contour detection. Orientation detectors receive
    the warped, deskewed image and return a clockwise correction or None.
    A known clockwise_rotation overrides the detector. By default orientation
    is unresolved; pass tesseract_orientation to enable the optional OCR adapter.
    """
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim not in (2, 3)
    ):
        raise ValueError("image must be a uint8 grayscale or BGR numpy array")
    if min(image.shape[:2]) < 2 or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError("image must have height/width >= 2 and one or three channels")
    config = config or RectificationConfig()
    warnings = []
    output, matrix = image.copy(), np.eye(3)
    if page_quad is None and config.detect_page:
        page_quad = detect_page_quad(image, config)
        if page_quad is None:
            warnings.append("Page boundary uncertain; perspective warp skipped.")
    if page_quad is not None:
        page_quad = _order_quad(page_quad)
        output, matrix = warp_page(output, page_quad)
    skew = estimate_skew(output, config) if config.deskew else None
    if config.deskew and skew is None:
        warnings.append("Line evidence insufficient; deskew skipped.")
    if skew is not None and abs(skew) >= 0.1:
        output, rotation = rotate_expanded(output, skew)
        matrix = rotation @ matrix
    correction = clockwise_rotation
    if correction is None and orientation_detector is not None:
        correction = orientation_detector(output)
    if correction is None:
        warnings.append(
            "Upright orientation unresolved; quarter-turn correction skipped."
        )
    else:
        output, rotation = normalize_orientation(output, correction)
        matrix = rotation @ matrix
    return RectificationResult(
        output, matrix, page_quad, skew, correction, tuple(warnings)
    )


def main() -> int:
    """Keep the original CLI command working for both files and folders."""
    from src.preprocessing.rectify_images import main as run_cli

    return run_cli()


if __name__ == "__main__":
    raise SystemExit(main())
