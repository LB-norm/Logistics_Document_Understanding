"""Infer completed hybrid models, one fresh process per run, on val or test images."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.Qwen.qwen_finetune_logic import resolve_existing_path


def build_commands(args: argparse.Namespace, forwarded: list[str]) -> list[tuple[str, list[str]]]:
    runs = sorted(path for path in args.runs_dir.glob("qwen35-9b-hybrid-*")
                  if (path / "best_model").is_dir() and (path / "training_config.json").is_file())
    if args.run:
        selected = set(args.run)
        runs = [path for path in runs if path.name in selected]
        if {path.name for path in runs} != selected:
            raise ValueError("One or more requested runs have no completed best_model and training_config.json")
    if not runs:
        raise ValueError(f"No completed hybrid runs under {args.runs_dir}")
    split_dir = args.dataset_root / args.split
    metadata = split_dir / "metadata.jsonl"
    rows = [json.loads(line) for line in metadata.read_text(encoding="utf-8").splitlines() if line.strip()] if metadata.is_file() else []
    if rows:
        images = [resolve_existing_path(row["image"], args.dataset_root, split_dir) for row in rows]
    else:
        images = sorted(path for path in (split_dir / "images").rglob("*")
                        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"})
    if not images or len({path.stem for path in images}) != len(images):
        raise ValueError("Expected a nonempty split with unique image stems")
    commands = []
    for run in runs:
        config = json.loads((run / "training_config.json").read_text(encoding="utf-8"))
        if config.get("input_mode") != "hybrid" or config.get("load_in_8bit"):
            raise ValueError(f"Expected a BF16 or NF4 hybrid campaign run: {run}")
        if not (run / "best_model" / "hybrid_contract.json").is_file():
            raise ValueError(f"Missing retained hybrid prompt contract: {run}")
        contract = json.loads((run / "best_model/hybrid_contract.json").read_text(encoding="utf-8"))
        evidence_mode = contract.get("evidence_mode", "hard")
        if config.get("load_in_4bit") and (config.get("bnb_4bit_quant_type", "nf4") != "nf4" or config.get("no_bnb_double_quant", False)):
            raise ValueError("Inference launcher expects NF4 with double quantization")
        destination = args.output_dir / args.split
        if args.omit_parser_evidence:
            destination = destination / "without-parser"
        command = [sys.executable, "-u", "-m", "src.inference_pipeline",
                   "--mode", "hybrid", "--model-id", str(run / "best_model"),
                   "--fusion-mode", "hard", "--hybrid-evidence", evidence_mode,
                   "--resolution", config["resolution"], "--compute-dtype", config["compute_dtype"],
                   "--load-in-4bit" if config.get("load_in_4bit") else "--no-load-in-4bit",
                   "--max-new-tokens", str(args.max_new_tokens),
                   "--local-files-only", "--output-dir", str(destination / run.name)]
        if evidence_mode in {"layout", "hard"}:
            command += ["--yolo-model-dir", str(args.yolo_model_dir)]
        if evidence_mode in {"ocr", "hard"}:
            command += ["--ocr-device", args.ocr_device,
                        "--ocr-detection-model-dir", str(args.ocr_models_dir / "PP-OCRv6_medium_det"),
                        "--ocr-recognition-model-dir", str(args.ocr_models_dir / "PP-OCRv6_medium_rec")]
        if config.get("cache_dir"):
            command += ["--cache-dir", config["cache_dir"]]
        if args.omit_parser_evidence:
            command.append("--omit-parser-evidence")
        if args.dry_run:
            command.append("--dry-run")
        command += [*forwarded, "--images", *map(str, images)]
        commands.append((run.name, command))
    return commands


def main(argv: Sequence[str] | None = None, *, defaults: dict | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/qwen-hybrid"))
    parser.add_argument("--run", action="append", help="Completed run name; repeat to select several")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/datasets/250_CMRS_240dpi_20260707"))
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("output/qwen-hybrid"))
    parser.add_argument("--yolo-model-dir", type=Path, default=Path(os.environ.get(
        "QWEN_HYBRID_YOLO_MODEL_DIR", "runs/yolo_v11s/cmr_100epochs/best_model")))
    parser.add_argument("--ocr-models-dir", type=Path, default=Path("output/ocr_v6_model_cache/official_models"))
    parser.add_argument("--ocr-device", default="gpu:0")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--omit-parser-evidence", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate inference settings and paths without loading models")
    if defaults:
        parser.set_defaults(**defaults)
    args, forwarded = parser.parse_known_args(argv)
    # Resolve all relative paths against the repo before spawning workers.
    for key in ("runs_dir", "dataset_root", "output_dir", "yolo_model_dir", "ocr_models_dir"):
        path = getattr(args, key)
        setattr(args, key, path if path.is_absolute() else REPO_ROOT / path)
    try:
        commands = build_commands(args, forwarded)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    failed = []
    for index, (name, command) in enumerate(commands, 1):
        print(f"[{index}/{len(commands)}] {name}", flush=True)
        result = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if result.returncode:
            failed.append(name)
    print("Runs reporting errors:", failed or "none", flush=True)
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
