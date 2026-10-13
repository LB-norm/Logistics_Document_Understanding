"""CLI for saved OCR/layout predictions: python -m src.fusion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .pipeline import FusionConfig, fuse_document, to_llm_text


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr", type=Path, required=True, help="PP-OCR document .ocr.json")
    parser.add_argument("--layout", type=Path, required=True, help="YOLO image prediction or inference JSON list")
    parser.add_argument("--mode", choices=["1", "2", "3", "separate", "hard", "soft"], required=True)
    parser.add_argument("--min-overlap", type=float, default=0.1, help="Minimum fraction of an OCR polygon covered by a region")
    parser.add_argument("--allow-layout-rescale", action="store_true", help="Confirm mismatched dimensions differ only by page resizing")
    parser.add_argument("--output", type=Path, required=True, help="Intermediate representation JSON")
    parser.add_argument("--llm-output", type=Path, help="Also write compact JSON as UTF-8 text for a downstream prompt")
    args = parser.parse_args(argv)
    try:
        config = FusionConfig(args.mode, args.min_overlap, args.allow_layout_rescale)
        inputs = [args.ocr.resolve(), args.layout.resolve()]
        outputs = [path.resolve() for path in (args.output, args.llm_output) if path is not None]
        if any(path in inputs for path in outputs) or len(outputs) != len(set(outputs)):
            raise ValueError("Output paths must differ from input paths and from each other")
        document = fuse_document(json.loads(args.ocr.read_text(encoding="utf-8")),
                                 json.loads(args.layout.read_text(encoding="utf-8")), config)
        # Serialize before writing either artifact so invalid data cannot leave partial JSON.
        formatted = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        compact = to_llm_text(document) + "\n"
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(formatted, encoding="utf-8")
    if args.llm_output is not None:
        args.llm_output.parent.mkdir(parents=True, exist_ok=True)
        args.llm_output.write_text(compact, encoding="utf-8")
    print(f"Saved {document['mode']} representation for {len(document['pages'])} page(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
