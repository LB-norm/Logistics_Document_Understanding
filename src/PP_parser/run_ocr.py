"""Run PP-OCRv6 on documents or prepare an OCR dataset for downstream LLM SFT."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.PP_parser.ocr import OCRConfig, OCRExtractor, save_ocr, write_json
from src.PP_parser.ocr_dataset import ensure_output_directory, prepare_ocr_dataset


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-paths", type=Path, nargs="+", help="Images or PDFs, one document per file.")
    inputs.add_argument("--dataset-root", type=Path, help="Project dataset with split/metadata.jsonl files.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output/pp_ocr_v6")
    parser.add_argument("--device", default="cpu", help="cpu or gpu:0 (requires a compatible Paddle GPU wheel).")
    parser.add_argument("--model-size", default="medium", choices=["medium", "small", "tiny"])
    parser.add_argument("--text-detection-model-dir", type=Path)
    parser.add_argument("--text-recognition-model-dir", type=Path)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--pdf-dpi", type=int, default=240)
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Existing splits to prepare; default: train val.")
    parser.add_argument("--annotation-target-key", default="content", help="Dotted annotation key, or root.")
    parser.add_argument("--schema-path", type=Path, default=REPO_ROOT / "json_schema/content.schema.json")
    parser.add_argument("--template-path", type=Path, default=REPO_ROOT / "json_schema/content.empty.json")
    parser.add_argument("--limit", type=int, help="First N metadata rows per split, for a smoke run.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without loading OCR or writing files.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing artifacts in a nonempty output directory.")
    args = parser.parse_args(argv)
    try:
        config_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.limit is not None and (args.limit < 1 or args.dataset_root is None):
        parser.error("--limit must be positive and requires --dataset-root")
    return args


def config_from_args(args: argparse.Namespace) -> OCRConfig:
    return OCRConfig(
        device=args.device, model_size=args.model_size, min_confidence=args.min_confidence,
        pdf_dpi=args.pdf_dpi, text_detection_model_dir=args.text_detection_model_dir,
        text_recognition_model_dir=args.text_recognition_model_dir,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    extractor = OCRExtractor(config_from_args(args))
    if args.dataset_root is not None:
        summary = prepare_ocr_dataset(
            extractor, args.dataset_root, args.output_dir, splits=args.splits,
            annotation_target_key=args.annotation_target_key, overwrite=args.overwrite,
            dry_run=args.dry_run, limit=args.limit,
            schema=json.loads(args.schema_path.read_text(encoding="utf-8")),
            template=json.loads(args.template_path.read_text(encoding="utf-8")),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return int(any(split.get("empty_ocr", 0) or split.get("ocr_error", 0) for split in summary["splits"].values()))
    stems = [path.stem for path in args.input_paths]
    if len(set(stems)) != len(stems):
        raise ValueError("Input files must have unique stems to avoid output collisions")
    for path in args.input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.dry_run:
        print(f"Validated {len(stems)} input documents; no OCR performed.")
        return 0
    ensure_output_directory(args.output_dir, overwrite=args.overwrite)
    records = []
    for path in args.input_paths:
        record = {"source_path": str(path.resolve())}
        try:
            document = extractor.extract(path)
            json_path, text_path = save_ocr(document, args.output_dir, path.stem)
            record.update({"status": "ok" if document["has_text"] else "empty_ocr",
                           "ocr_json_path": str(json_path), "ocr_text_path": str(text_path)})
        except Exception as exc:
            record.update({"status": "ocr_error", "error": f"{type(exc).__name__}: {exc}"})
            print(f"OCR failed for {path}: {record['error']}", file=sys.stderr)
        records.append(record)
        print(f"{record['status']}: {path}", flush=True)
    write_json(args.output_dir / "manifest.json", records)
    return int(any(record["status"] != "ok" for record in records))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
