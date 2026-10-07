"""Mild document augmentation with shared image/box geometry."""

from dataclasses import dataclass
from io import BytesIO
import math

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


@dataclass(frozen=True)
class AugmentationConfig:
    geometry_probability: float = 0.5
    max_rotation_degrees: float = 3.0
    perspective_fraction: float = 0.02
    color_probability: float = 0.5
    blur_probability: float = 0.15
    noise_probability: float = 0.15
    jpeg_probability: float = 0.2

    def __post_init__(self):
        for name in ("geometry_probability", "color_probability", "blur_probability",
                     "noise_probability", "jpeg_probability"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0 <= self.max_rotation_degrees <= 10:
            raise ValueError("max_rotation_degrees must be in [0, 10]")
        if not 0 <= self.perspective_fraction <= 0.1:
            raise ValueError("perspective_fraction must be in [0, 0.1]")


def transform_regions(image, annotations, matrix):
    """Warp pixels and all four box corners with the same forward homography."""
    width, height = image.size
    pixels = cv2.warpPerspective(
        np.asarray(image), matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )
    transformed = []
    for annotation in annotations:
        x, y, w, h = annotation["bbox"]
        corners = np.array([[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]], dtype=np.float64)
        corners = cv2.perspectiveTransform(corners, matrix)[0]
        lower = np.maximum(corners.min(axis=0), [0, 0])
        upper = np.minimum(corners.max(axis=0), [width, height])
        bw, bh = upper - lower
        if bw <= 0 or bh <= 0:
            continue
        transformed.append({**annotation, "bbox": [float(lower[0]), float(lower[1]), float(bw), float(bh)],
                            "area": float(bw * bh)})
    return Image.fromarray(pixels), transformed


class DocumentAugmenter:
    """Sample fresh variants on access; NumPy RNG follows Trainer/worker seeds.

    An optional NumPy Generator allows reproducible previews and tests. Geometry
    fits the whole warped page into the original canvas to retain edge regions.
    """

    def __init__(self, config=None, rng=None):
        self.config = config or AugmentationConfig()
        self.rng = rng if rng is not None else np.random

    def geometry_matrix(self, width, height):
        cfg, rng = self.config, self.rng
        source = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        angle = math.radians(rng.uniform(-cfg.max_rotation_degrees, cfg.max_rotation_degrees))
        rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        center = np.array([width / 2, height / 2])
        destination = (source - center) @ rotation.T + center
        destination += rng.uniform(-1, 1, (4, 2)) * [width * cfg.perspective_fraction, height * cfg.perspective_fraction]
        lower, upper = destination.min(axis=0), destination.max(axis=0)
        scale = min(width / (upper[0] - lower[0]), height / (upper[1] - lower[1]))
        destination = (destination - lower) * scale
        destination += (np.array([width, height]) - (upper - lower) * scale) / 2
        return cv2.getPerspectiveTransform(source, destination.astype(np.float32))

    def __call__(self, image, annotations):
        cfg, rng = self.config, self.rng
        # Own boxes even when geometry is skipped; never mutate COCO targets.
        annotations = [{**a, "bbox": list(a["bbox"])} for a in annotations]
        if rng.random() < cfg.geometry_probability:
            image, annotations = transform_regions(image, annotations, self.geometry_matrix(*image.size))
        if rng.random() < cfg.color_probability:
            image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.85, 1.15))
            image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.85, 1.15))
            image = ImageEnhance.Color(image).enhance(rng.uniform(0.9, 1.1))
        if rng.random() < cfg.blur_probability:
            image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.8)))
        if rng.random() < cfg.noise_probability:
            pixels = np.asarray(image, dtype=np.float32)
            pixels += rng.normal(0, rng.uniform(1.0, 3.0), pixels.shape)
            image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
        if rng.random() < cfg.jpeg_probability:
            with BytesIO() as buffer:
                image.save(buffer, format="JPEG", quality=int(rng.uniform(75, 96)))
                buffer.seek(0)
                with Image.open(buffer) as compressed:
                    image = compressed.convert("RGB")
        return image, annotations
