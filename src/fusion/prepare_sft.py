"""Batch saved OCR and YOLO predictions into text-only fusion SFT JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fusion.dataset import prepare_fusion_dataset


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path,
                        default=REPO_ROOT / "data/datasets/250_CMRS_240dpi_20260707")
    parser.add_argument("--ocr-dir", type=Path, help="Default: <dataset-root>/ocr_v6_cmr")
    parser.add_argument("--layout-dir", type=Path, help="Directory with <split>.json lists; default: <dataset-root>/yolo_v11s/layout_detections")
    parser.add_argument("--output-dir", type=Path, help="Default: <dataset-root>/fusion_sft")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--modes", nargs="+", choices=["1", "2", "3", "separate", "hard", "soft"],
                        default=["separate", "hard", "soft"])
    parser.add_argument("--min-overlap", type=float, default=0.1)
    parser.add_argument("--allow-layout-rescale", action="store_true", help="Confirm differing dimensions are a scaling of the same whole raster")
    parser.add_argument("--annotation-target-key", help="Dotted key or root; default: OCR preparation setting, or content")
    parser.add_argument("--schema-path", type=Path, help="Default: <ocr-dir>/schema.json")
    parser.add_argument("--template-path", type=Path, help="Default: <ocr-dir>/template.json")
    parser.add_argument("--limit", type=int, help="First N OCR training records per split, identical across modes")
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize all examples without writing files")
    parser.add_argument("--overwrite", action="store_true", help="Replace generated files in an existing output directory")
    args = parser.parse_args(argv)
    try:
        summary = prepare_fusion_dataset(
            args.dataset_root, args.output_dir or args.dataset_root / "fusion_sft",
            ocr_dir=args.ocr_dir, layout_dir=args.layout_dir, splits=args.splits, modes=args.modes,
            min_overlap=args.min_overlap, allow_layout_rescale=args.allow_layout_rescale,
            annotation_target_key=args.annotation_target_key, schema_path=args.schema_path,
            template_path=args.template_path, limit=args.limit, dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
