"""Unified document inference: python -m src.inference_pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from src.PP_parser import OCRConfig
from .pipeline import InferencePipeline, PipelineConfig


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, PipelineConfig]:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False, epilog=(
        "Additional flags are forwarded to the selected model's existing inference parser, "
        "for example --resolution high or --max-new-tokens 4096 for Qwen, "
        "or --task-prompt '<s_lieferschein>' for Donut. Use --model-help to list them."
    ))
    parser.add_argument("--mode", choices=["end-to-end", "parser"], required=True)
    parser.add_argument("--model-type", choices=["qwen", "donut"], default="qwen")
    parser.add_argument("--model-id", help="Base/merged model ID or local checkpoint; Qwen adapter directories are recognized")
    parser.add_argument("--adapter-path", type=Path, help="Qwen LoRA adapter; its metadata can select the base model")
    parser.add_argument("--images", type=Path, nargs="+", required=True, help="One or more page images")
    parser.add_argument("--fusion-mode", choices=["1", "2", "3", "separate", "hard", "soft"])
    parser.add_argument("--yolo-model-dir", type=Path)
    parser.add_argument("--yolo-device", help="Native YOLO device, e.g. cpu or cuda:0; default remains automatic")
    parser.add_argument("--yolo-threshold", type=float, help="Override YOLO's default detection confidence")
    parser.add_argument("--ocr-device", help="Native Paddle device, e.g. cpu or gpu:0; default remains CPU")
    parser.add_argument("--ocr-model-size", choices=["medium", "small", "tiny"])
    parser.add_argument("--ocr-min-confidence", type=float)
    parser.add_argument("--ocr-detection-model-dir", type=Path)
    parser.add_argument("--ocr-recognition-model-dir", type=Path)
    parser.add_argument("--ocr-disable-mkldnn", action="store_true",
                        help="Disable PaddleOCR's CPU oneDNN backend to avoid PIR attribute conversion failures")
    parser.add_argument("--min-overlap", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=Path("output/inference_pipeline"))
    parser.add_argument("--dry-run", action="store_true", help="Validate paths/settings without loading models or writing files")
    # Handle backend help before required wrapper arguments are enforced.
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--model-help" in raw:
        from src.Qwen import run_inference as qwen
        from src.Donut import run_donut_inference as donut

        selector = argparse.ArgumentParser(add_help=False)
        selector.add_argument("--model-type", default="qwen", choices=["qwen", "donut"])
        selected, _ = selector.parse_known_args(raw)
        (donut if selected.model_type == "donut" else qwen).parse_args(["--help"])
    args, model_args = parser.parse_known_args(raw)
    ocr_options = {key: value for key, value in (
        ("device", args.ocr_device), ("model_size", args.ocr_model_size),
        ("min_confidence", args.ocr_min_confidence),
        ("text_detection_model_dir", args.ocr_detection_model_dir),
        ("text_recognition_model_dir", args.ocr_recognition_model_dir),
        ("enable_mkldnn", False if args.ocr_disable_mkldnn else None),
    ) if value is not None}
    try:
        config = PipelineConfig(
            mode=args.mode, model_type=args.model_type, model_id=args.model_id,
            adapter_path=args.adapter_path, fusion_mode=args.fusion_mode,
            yolo_model_dir=args.yolo_model_dir, yolo_device=args.yolo_device,
            yolo_threshold=args.yolo_threshold, ocr_config=OCRConfig(**ocr_options),
            min_overlap=args.min_overlap, model_args=tuple(model_args),
        )
    except ValueError as exc:
        parser.error(str(exc))
    return args, config


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args, config = parse_args(argv)
    try:
        if len({path.stem for path in args.images}) != len(args.images):
            raise ValueError("Image stems must be unique to avoid prediction collisions")
        for path in args.images:
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.suffix.lower() == ".pdf":
                raise ValueError("Render PDFs to page images before inference")
        pipeline = InferencePipeline(config)
        if config.mode == "parser":
            weights = config.yolo_model_dir
            weights = weights / "model.pt" if weights.is_dir() else weights
            if not weights.is_file() or not (weights.parent / "layout_config.json").is_file():
                raise ValueError("YOLO requires weights and their adjacent layout_config.json")
        for path in (config.ocr_config.text_detection_model_dir, config.ocr_config.text_recognition_model_dir):
            if path is not None and not Path(path).is_dir():
                raise FileNotFoundError(path)
        manifest_path = args.output_dir / "inference_manifest.jsonl"
        destinations = [manifest_path] + [args.output_dir / f"{path.stem}.json" for path in args.images]
        if config.mode == "parser":
            destinations.extend(args.output_dir / f"{path.stem}.fusion.json" for path in args.images)
        resolved = [path.resolve() for path in destinations]
        protected = {path.resolve() for path in args.images}
        protected.add(pipeline.args.schema_path.resolve())
        protected.add((pipeline.args.template_path if config.model_type == "qwen" else pipeline.args.example_path).resolve())
        if len(resolved) != len(set(resolved)) or any(path in protected for path in resolved):
            raise ValueError("Output paths collide with inputs, output contracts, or each other")
        if args.dry_run:
            print(f"Validated {len(args.images)} image(s) for {config.mode}/{config.model_type}; no models loaded")
            return 0
        pipeline.load()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failed = 0
        # Write records as images finish, so a long run retains completed measurements.
        with manifest_path.open("w", encoding="utf-8") as manifest:
            for path in args.images:
                record = pipeline.predict(path)
                prediction = record.pop("prediction")
                record["prediction_path"] = None
                if prediction is not None:
                    prediction_path = args.output_dir / f"{path.stem}.json"
                    _write_json(prediction_path, prediction)
                    record["prediction_path"] = str(prediction_path)
                if "fusion" in record:
                    fusion_path = args.output_dir / f"{path.stem}.fusion.json"
                    _write_json(fusion_path, record.pop("fusion"))
                    record["fusion_path"] = str(fusion_path)
                manifest.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                manifest.flush()
                failed += record["status"] != "ok"
                print(f"{path}: {record['status']} ({record['inference_seconds']:.3f}s)")
                if record.get("error"):
                    print(f"  {record['error']}", file=sys.stderr, flush=True)
        print(f"Saved predictions and measurements to {args.output_dir}")
        return int(failed > 0)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
