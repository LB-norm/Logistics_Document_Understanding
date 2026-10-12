"""Reusable PP-OCRv6 text detection/recognition, independent of layout and LLMs."""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from src.utils.inference_tracking import InferenceTracker, aggregate_vram


@dataclass(frozen=True)
class OCRConfig:
    device: str = "cpu"
    model_size: str = "medium"
    min_confidence: float = 0.0
    pdf_dpi: int = 240
    text_detection_model_dir: Path | None = None
    text_recognition_model_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.model_size not in {"medium", "small", "tiny"}:
            raise ValueError("model_size must be medium, small, or tiny")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        if self.pdf_dpi <= 0:
            raise ValueError("pdf_dpi must be positive")

    def pipeline_settings(self) -> dict[str, Any]:
        # Explicit model names avoid language-dependent selection or future defaults.
        settings = {
            "device": self.device,
            "text_detection_model_name": f"PP-OCRv6_{self.model_size}_det",
            "text_recognition_model_name": f"PP-OCRv6_{self.model_size}_rec",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }
        for key in ("text_detection_model_dir", "text_recognition_model_dir"):
            if getattr(self, key) is not None:
                settings[key] = str(Path(getattr(self, key)).resolve())
        return settings


def normalize_page(result: Any, page_number: int, image_size: tuple[int, int]) -> dict[str, Any]:
    """Normalize aligned recognition polygons in the supplied page's pixel frame."""
    payload = getattr(result, "json", result)
    if callable(payload):
        payload = payload()
    if isinstance(payload, str):
        payload = json.loads(payload)
    payload = payload.get("res", payload)
    texts, scores, polygons = (payload[key] for key in ("rec_texts", "rec_scores", "rec_polys"))
    if not len(texts) == len(scores) == len(polygons):
        raise ValueError("PP-OCR returned misaligned texts, scores, and recognition polygons")
    lines = []
    for index, (text, score, polygon) in enumerate(zip(texts, scores, polygons, strict=True)):
        polygon = polygon.tolist() if hasattr(polygon, "tolist") else polygon
        if len(polygon) != 4 or any(len(point) != 2 for point in polygon):
            raise ValueError("Expected a quadrilateral recognition polygon")
        polygon = [[float(x), float(y)] for x, y in polygon]
        score = float(score)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid OCR confidence")
        if not all(math.isfinite(value) for point in polygon for value in point):
            raise ValueError("Invalid OCR polygon coordinate")
        xs, ys = zip(*polygon)
        lines.append({
            "id": f"p{page_number:04d}-l{index:06d}",
            "line_index": index,
            "text": str(text),
            "confidence": score,
            "polygon": polygon,
            "bbox": [min(xs), min(ys), max(xs), max(ys)],
        })
    return {
        "page_number": page_number,
        "width": image_size[0],
        "height": image_size[1],
        "coordinate_space": "input_image_pixels",
        "lines": lines,
        "text": "\n".join(line["text"] for line in lines if line["text"].strip()),
        "text_detection_settings": payload.get("text_det_params", {}),
    }


class OCRExtractor:
    """Load OCR once; extract paths, PIL images, or uint8 BGR/grayscale arrays.

    No files are written by extraction. Arrays follow OpenCV/Paddle's BGR
    convention; PIL images are converted from RGB. Geometry always refers to
    the supplied image (or our explicitly rendered PDF page). The instance is
    intended for sequential reuse, not concurrent inference calls.
    """

    def __init__(self, config: OCRConfig | None = None, *, pipeline: Any = None) -> None:
        self.config = config or OCRConfig()
        self._pipeline = pipeline
        self._package_versions: dict[str, str] = {}
        self._injected_pipeline = pipeline is not None

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            try:
                installed = version("paddleocr")
            except PackageNotFoundError as exc:
                raise RuntimeError("Install paddleocr>=3.7 to run the PP-OCRv6 stage") from exc
            match = re.match(r"(\d+)\.(\d+)", installed)
            if match is None or tuple(map(int, match.groups())) < (3, 7):
                raise RuntimeError(
                    f"PP-OCRv6 requires paddleocr>=3.7; installed: {installed}. "
                    "Upgrade using python -m pip install 'paddleocr>=3.7,<4'."
                )
            from paddleocr import PaddleOCR

            self._pipeline = PaddleOCR(**self.config.pipeline_settings())
            self._package_versions = {"paddleocr": installed, "paddlex": version("paddlex")}
            for name in ("paddlepaddle", "paddlepaddle-gpu"):
                try:
                    self._package_versions[name] = version(name)
                except PackageNotFoundError:
                    pass
        return self._pipeline

    def _extract_page(self, pipeline: Any, image: Any, page_number: int) -> dict[str, Any]:
        with InferenceTracker(backend="paddle", device=self.config.device) as tracker:
            results = list(pipeline.predict(
                input=image, text_rec_score_thresh=self.config.min_confidence,
            ))
            if len(results) != 1:
                raise RuntimeError(f"Expected one OCR result for page {page_number}, got {len(results)}")
            height, width = image.shape[:2]
            page = normalize_page(results[0], page_number, (width, height))
        page.update(tracker.metrics)
        return page

    def extract(self, source: Any, *, source_id: str | None = None) -> dict[str, Any]:
        """Return JSON-serializable OCR; source_id identifies an in-memory source."""
        import numpy as np
        from PIL import Image

        path = Path(source).resolve() if isinstance(source, (str, Path)) else None
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
        # Initialize before timing so inference_seconds excludes model startup.
        pipeline = self._get_pipeline()
        started = time.perf_counter()
        if path is not None and path.suffix.lower() == ".pdf":
            import fitz

            pages = []
            with fitz.open(path) as document:
                for index, page in enumerate(document):
                    pixmap = page.get_pixmap(dpi=self.config.pdf_dpi, colorspace=fitz.csRGB, alpha=False)
                    rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)
                    pages.append(self._extract_page(pipeline, rgb[:, :, ::-1].copy(), index + 1))
        else:
            if path is not None:
                with Image.open(path) as image:
                    if getattr(image, "n_frames", 1) > 1:
                        raise ValueError("Multi-frame images are unsupported; supply individual pages or a PDF")
                    array = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
            elif isinstance(source, Image.Image):
                array = np.asarray(source.convert("RGB"))[:, :, ::-1].copy()
            elif isinstance(source, np.ndarray):
                if source.dtype != np.uint8 or source.ndim not in (2, 3):
                    raise ValueError("Arrays must be uint8 grayscale or three-channel BGR")
                if source.ndim == 3 and source.shape[2] != 3:
                    raise ValueError("Arrays must be uint8 grayscale or three-channel BGR")
                if not source.shape[0] or not source.shape[1]:
                    raise ValueError("Image array is empty")
                array = np.repeat(source[:, :, None], 3, axis=2) if source.ndim == 2 else source.copy()
            else:
                raise TypeError("source must be an image/PDF path, PIL image, or numpy array")
            pages = [self._extract_page(pipeline, array, 1)]
        if not pages:
            raise RuntimeError("Document contains no pages")
        text = "\n\n".join(
            (f"[Page {page['page_number']}]\n" if len(pages) > 1 else "") + page["text"]
            for page in pages
        )
        return {
            "format_version": 2,
            "parser": "pp_ocr",
            "source_id": source_id or (path.stem if path is not None else None),
            "source_path": str(path) if path is not None else None,
            "pages": pages,
            "text": text,
            "has_text": any(page["text"].strip() for page in pages),
            "inference_seconds": time.perf_counter() - started,
            "vram": aggregate_vram(page["vram"] for page in pages),
            "settings": {**self.config.pipeline_settings(), "min_confidence": self.config.min_confidence,
                         "pdf_dpi": self.config.pdf_dpi},
            "package_versions": self._package_versions.copy(),
            "backend": "injected" if self._injected_pipeline else "paddleocr",
        }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_ocr(document: dict[str, Any], output_dir: Path, stem: str) -> tuple[Path, Path]:
    """Persist an extraction separately from inference; return JSON and text paths."""
    if not stem or Path(stem).name != stem or stem in {".", ".."}:
        raise ValueError("stem must be a filename without directory components")
    json_path, text_path = output_dir / f"{stem}.ocr.json", output_dir / f"{stem}.ocr.txt"
    write_json(json_path, document)
    text_path.write_text(document["text"] + "\n", encoding="utf-8")
    return json_path, text_path
