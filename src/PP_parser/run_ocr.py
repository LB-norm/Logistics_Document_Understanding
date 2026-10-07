"""Plain PP-OCR baseline: document -> reusable OCR JSON and UTF-8 text."""
from __future__ import annotations

import argparse
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.Qwen.run_inference import write_json


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-paths", type=Path, nargs="+", required=True,
                        help="Images or PDFs; each file is one document.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output" / "pp_ocr")
    parser.add_argument("--device", default="cpu", help="Paddle device, e.g. cpu or gpu:0.")
    parser.add_argument("--lang", default="de", help="Recognition language (default: de).")
    parser.add_argument("--ocr-version", default="PP-OCRv5", choices=["PP-OCRv5", "PP-OCRv4", "PP-OCRv3"])
    parser.add_argument("--text-detection-model-dir", type=Path)
    parser.add_argument("--text-recognition-model-dir", type=Path)
    parser.add_argument("--text-detection-model-name")
    parser.add_argument("--text-recognition-model-name")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    args = parser.parse_args(argv)
    if not 0 <= args.min_confidence <= 1:
        parser.error("--min-confidence must be between 0 and 1")
    if (args.text_detection_model_dir or args.text_recognition_model_dir
            or args.text_detection_model_name or args.text_recognition_model_name):
        if not (args.text_detection_model_name and args.text_recognition_model_name):
            parser.error("Custom models require both --text-detection-model-name and "
                         "--text-recognition-model-name; Paddle ignores --lang for custom models")
    return args


def normalize_page(result: Any, page_number: int) -> dict[str, Any]:
    """Keep Paddle's recognition order and aligned, filtered recognition polygons."""
    payload = getattr(result, "json", result)
    if callable(payload):
        payload = payload()
    if isinstance(payload, str):
        payload = json.loads(payload)
    payload = payload.get("res", payload)
    texts = payload["rec_texts"]
    scores = payload["rec_scores"]
    polygons = payload["rec_polys"]
    if not len(texts) == len(scores) == len(polygons):
        raise ValueError("PP-OCR returned misaligned texts, scores, and polygons")
    lines = [
        {"text": str(text), "confidence": float(score),
         "polygon": polygon.tolist() if hasattr(polygon, "tolist") else polygon}
        for text, score, polygon in zip(texts, scores, polygons, strict=True)
    ]
    return {"page_number": page_number, "lines": lines,
            "text": "\n".join(line["text"] for line in lines if line["text"].strip())}


def extract_document(pipeline: Any, path: Path, min_confidence: float) -> dict[str, Any]:
    started = time.perf_counter()
    pages = [normalize_page(result, index + 1) for index, result in enumerate(
        pipeline.predict(input=str(path), text_rec_score_thresh=min_confidence)
    )]
    if not pages:
        raise RuntimeError("PP-OCR returned no pages")
    text = "\n\n".join(
        (f"[Page {page['page_number']}]\n" if len(pages) > 1 else "") + page["text"]
        for page in pages
    )
    return {"format_version": 1, "source_path": str(path.resolve()),
            "parser": "pp_ocr", "pages": pages, "text": text,
            "has_text": any(page["text"].strip() for page in pages),
            "inference_seconds": time.perf_counter() - started}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stems = [path.stem for path in args.input_paths]
    if len(set(stems)) != len(stems):
        raise ValueError("Input files must have unique stems to avoid output collisions")
    for path in args.input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    from paddleocr import PaddleOCR

    settings = {"device": args.device, "lang": args.lang, "ocr_version": args.ocr_version,
                "use_doc_orientation_classify": False, "use_doc_unwarping": False,
                "use_textline_orientation": False}
    for key in ("text_detection_model_dir", "text_recognition_model_dir",
                "text_detection_model_name", "text_recognition_model_name"):
        if getattr(args, key) is not None:
            settings[key] = str(getattr(args, key))
    pipeline = PaddleOCR(**settings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failed = False
    records = []
    for path in args.input_paths:
        json_path = args.output_dir / f"{path.stem}.ocr.json"
        text_path = args.output_dir / f"{path.stem}.ocr.txt"
        try:
            document = extract_document(pipeline, path, args.min_confidence)
            document["settings"] = {**settings, "min_confidence": args.min_confidence}
            document["paddleocr_version"] = version("paddleocr")
            write_json(json_path, document)
            text_path.write_text(document["text"] + "\n", encoding="utf-8")
            status = "ok" if document["has_text"] else "empty_ocr"
            records.append({"source_path": str(path.resolve()), "status": status,
                            "ocr_json_path": str(json_path), "ocr_text_path": str(text_path)})
            failed |= status != "ok"
            print(f"{status}: {text_path}")
        except Exception as exc:
            failed = True
            records.append({"source_path": str(path.resolve()), "status": "ocr_error",
                            "error": f"{type(exc).__name__}: {exc}"})
            print(f"OCR failed for {path}: {exc}", file=sys.stderr)
    write_json(args.output_dir / "manifest.json", records)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
