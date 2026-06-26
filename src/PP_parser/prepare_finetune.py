from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "datasets" / "raw_data_20260527"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "pp_ocr_vl_sft"
DEFAULT_ANNOTATION_TARGET_KEY = "content"
DEFAULT_PROMPT = (
    "Extract the CMR/Lieferschein information from this document image. "
    "Return only the target JSON object, without markdown or commentary."
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass(frozen=True)
class SplitSummary:
    split: str
    source: str
    output_jsonl: str
    output_manifest_jsonl: str
    examples: int
    max_target_chars: int
    copied_images: bool


@dataclass(frozen=True)
class PreparationSummary:
    dataset_root: str
    output_dir: str
    annotation_target_key: str
    train_split: SplitSummary
    validation_split: SplitSummary | None
    test_split: SplitSummary | None
    estimated_max_steps: int
    estimated_warmup_steps: int
    command_file: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare ERNIEKit SFT JSONL files for fine-tuning the PaddleOCR-VL "
            "VLM component on the current project annotations."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=(
            "Project dataset root. Expected layout: split/metadata.jsonl with "
            "image and annotation entries, or a flat image/json folder."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where SFT JSONL files, manifests, and command template are written.",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument(
        "--validation-split",
        default=None,
        help="Validation split name. If omitted, validation, val, then dev are tried.",
    )
    parser.add_argument(
        "--test-split",
        default="test",
        help="Optional test split name. Pass an empty value to skip test preparation.",
    )
    parser.add_argument(
        "--annotation-target-key",
        default=DEFAULT_ANNOTATION_TARGET_KEY,
        help=(
            "Key inside annotation JSON files to use as the supervised answer. "
            "Use 'root' to train on the complete annotation wrapper."
        ),
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Masked user prompt placed before each image in ERNIEKit text_info.",
    )
    parser.add_argument(
        "--absolute-image-paths",
        action="store_true",
        help=(
            "Use absolute source image paths in image_info instead of copying images "
            "into the prepared dataset directory."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing non-empty output directory.",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=3.0,
        help="Epoch count used only to estimate max_steps for the generated command template.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="GPU count used only to estimate max_steps for the generated command template.",
    )
    parser.add_argument(
        "--packing-size",
        type=int,
        default=8,
        help="ERNIEKit packing_size used only to estimate max_steps.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="ERNIEKit gradient_accumulation_steps used only to estimate max_steps.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and validate the dataset, print the summary, but do not write files.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def resolve_dataset_root(path: Path) -> Path:
    dataset_root = path.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {dataset_root}")
    return dataset_root


def ensure_output_dir(path: Path, overwrite: bool, dry_run: bool) -> None:
    if dry_run:
        return
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {path}. "
                "Pass --overwrite or choose another --output-dir."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def resolve_existing_path(path_value: str, dataset_root: Path, split_dir: Path) -> Path:
    path = Path(path_value)
    candidates = [path] if path.is_absolute() else [dataset_root / path, split_dir / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Referenced path does not exist: {path_value}. Checked: {checked}")


def extract_annotation_target(annotation: Any, target_key: str) -> Any:
    if target_key in {"", ".", "root"}:
        target = annotation
    else:
        target = annotation
        for key in target_key.split("."):
            if not isinstance(target, dict) or key not in target:
                raise KeyError(f"Annotation target key {target_key!r} not found.")
            target = target[key]

    if not isinstance(target, dict):
        raise TypeError(f"Annotation target {target_key!r} must resolve to a JSON object.")
    return target


def find_image_for_annotation(annotation_path: Path, image_files: dict[str, Path]) -> Path:
    stems = [annotation_path.stem]
    if annotation_path.stem.endswith("_0"):
        stems.append(annotation_path.stem[:-2])
    for stem in stems:
        if stem in image_files:
            return image_files[stem].resolve()
    raise FileNotFoundError(f"No matching image found for annotation: {annotation_path}")


def choose_validation_split(dataset_root: Path, requested_split: str | None) -> str | None:
    if requested_split is not None:
        return requested_split
    for split_name in ("validation", "val", "dev"):
        if (dataset_root / split_name / "metadata.jsonl").exists():
            return split_name
    return None


def load_project_split(
    dataset_root: Path,
    split_name: str,
    annotation_target_key: str,
) -> list[dict[str, Any]]:
    split_dir = dataset_root / split_name
    metadata_path = split_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.jsonl for split {split_name!r}: {metadata_path}")

    examples: list[dict[str, Any]] = []
    for record in read_jsonl(metadata_path):
        if "image" not in record or "annotation" not in record:
            raise ValueError(
                f"Unsupported metadata row in {metadata_path}. Expected image and annotation keys: {record}"
            )
        image_path = resolve_existing_path(record["image"], dataset_root, split_dir)
        annotation_path = resolve_existing_path(record["annotation"], dataset_root, split_dir)
        annotation = load_json(annotation_path)
        target = extract_annotation_target(annotation, annotation_target_key)
        examples.append(
            {
                "id": record.get("id", image_path.stem),
                "image_path": image_path,
                "annotation_path": annotation_path,
                "target": target,
                "source_split": split_name,
            }
        )
    return examples


def load_flat_examples(dataset_root: Path, annotation_target_key: str) -> list[dict[str, Any]]:
    image_files = {
        path.stem: path
        for path in sorted(dataset_root.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    annotation_paths = [
        path
        for path in sorted(dataset_root.iterdir())
        if path.is_file() and path.suffix.lower() == ".json"
    ]

    examples: list[dict[str, Any]] = []
    for annotation_path in annotation_paths:
        image_path = find_image_for_annotation(annotation_path, image_files)
        annotation = load_json(annotation_path)
        target = extract_annotation_target(annotation, annotation_target_key)
        examples.append(
            {
                "id": image_path.stem,
                "image_path": image_path,
                "annotation_path": annotation_path.resolve(),
                "target": target,
                "source_split": "flat",
            }
        )

    if not examples:
        raise ValueError(
            f"No flat image/json pairs found in {dataset_root}. "
            "Expected files such as sample.jpg and sample_0.json."
        )
    return examples


def copy_image_for_split(
    image_path: Path,
    dataset_root: Path,
    output_dir: Path,
    split_name: str,
    dry_run: bool,
) -> str:
    try:
        relative = image_path.relative_to(dataset_root)
    except ValueError:
        relative = Path(image_path.name)

    if relative.parts and relative.parts[0] == split_name:
        relative = Path(*relative.parts[1:])
    if relative.parts and relative.parts[0] == "images":
        relative = Path(*relative.parts[1:])

    output_image_path = output_dir / "images" / split_name / relative
    if not dry_run:
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, output_image_path)
    return f"./{output_image_path.relative_to(output_dir).as_posix()}"


def build_sft_record(image_url: str, prompt: str, target: dict[str, Any]) -> dict[str, Any]:
    target_text = json.dumps(target, ensure_ascii=False, separators=(",", ":"))
    return {
        "image_info": [{"image_url": image_url, "matched_text_index": 0}],
        "text_info": [
            {"text": prompt, "tag": "mask"},
            {"text": target_text, "tag": "no_mask"},
        ],
    }


def prepare_split(
    examples: list[dict[str, Any]],
    split_name: str,
    dataset_root: Path,
    output_dir: Path,
    annotation_target_key: str,
    prompt: str,
    absolute_image_paths: bool,
    dry_run: bool,
) -> SplitSummary:
    records: list[dict[str, Any]] = []
    manifest_records: list[dict[str, Any]] = []
    max_target_chars = 0

    for example in examples:
        image_path: Path = example["image_path"]
        annotation_path: Path = example["annotation_path"]
        target: dict[str, Any] = example["target"]
        image_url = (
            str(image_path)
            if absolute_image_paths
            else copy_image_for_split(image_path, dataset_root, output_dir, split_name, dry_run)
        )
        sft_record = build_sft_record(image_url=image_url, prompt=prompt, target=target)
        records.append(sft_record)

        target_text = sft_record["text_info"][1]["text"]
        max_target_chars = max(max_target_chars, len(target_text))
        manifest_records.append(
            {
                "id": example["id"],
                "source_split": example["source_split"],
                "source_image": str(image_path),
                "source_annotation": str(annotation_path),
                "sft_image_url": image_url,
                "annotation_target_key": annotation_target_key,
                "target_chars": len(target_text),
            }
        )

    output_jsonl = output_dir / f"{split_name}.jsonl"
    output_manifest_jsonl = output_dir / f"{split_name}_manifest.jsonl"
    if not dry_run:
        write_jsonl(output_jsonl, records)
        write_jsonl(output_manifest_jsonl, manifest_records)

    return SplitSummary(
        split=split_name,
        source=examples[0]["source_split"] if examples else split_name,
        output_jsonl=str(output_jsonl),
        output_manifest_jsonl=str(output_manifest_jsonl),
        examples=len(examples),
        max_target_chars=max_target_chars,
        copied_images=not absolute_image_paths,
    )


def estimate_steps(
    train_examples: int,
    epochs: float,
    num_gpus: int,
    packing_size: int,
    gradient_accumulation_steps: int,
) -> tuple[int, int]:
    if epochs <= 0:
        raise ValueError("--epochs must be greater than 0.")
    for name, value in {
        "num_gpus": num_gpus,
        "packing_size": packing_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }.items():
        if value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1.")

    samples_per_update = num_gpus * packing_size * gradient_accumulation_steps
    max_steps = max(1, math.ceil((train_examples * epochs) / samples_per_update))
    warmup_steps = max(1, round(max_steps * 0.01))
    return max_steps, warmup_steps


def write_command_template(
    output_dir: Path,
    train_jsonl: Path,
    validation_jsonl: Path | None,
    max_steps: int,
    warmup_steps: int,
    packing_size: int,
    gradient_accumulation_steps: int,
    dry_run: bool,
) -> Path:
    command_path = output_dir / "run_erniekit_train.sh"
    validation_note = (
        f"# Validation data prepared at: {validation_jsonl}\n"
        if validation_jsonl is not None
        else "# No validation JSONL was prepared.\n"
    )
    content = f"""#!/usr/bin/env bash
set -euo pipefail

# Run this inside an ERNIEKit environment from the ERNIE repository root.
# Install guide: https://github.com/PaddlePaddle/ERNIE/blob/release/v1.4/docs/paddleocr_vl_sft.md
{validation_note}
CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES:-0}}" \\
erniekit train examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml \\
  model_name_or_path="${{MODEL_NAME_OR_PATH:-PaddlePaddle/PaddleOCR-VL}}" \\
  train_dataset_path="{train_jsonl}" \\
  output_dir="${{OUTPUT_DIR:-./PaddleOCR-VL-SFT-lieferschein}}" \\
  max_steps="${{MAX_STEPS:-{max_steps}}}" \\
  warmup_steps="${{WARMUP_STEPS:-{warmup_steps}}}" \\
  packing_size="${{PACKING_SIZE:-{packing_size}}}" \\
  gradient_accumulation_steps="${{GRADIENT_ACCUMULATION_STEPS:-{gradient_accumulation_steps}}}"
"""
    if not dry_run:
        command_path.write_text(content, encoding="utf-8")
        command_path.chmod(0o755)
    return command_path


def prepare(args: argparse.Namespace) -> PreparationSummary:
    dataset_root = resolve_dataset_root(args.dataset_root)
    output_dir = args.output_dir.resolve()
    ensure_output_dir(output_dir, overwrite=args.overwrite, dry_run=args.dry_run)

    has_project_splits = (dataset_root / args.train_split / "metadata.jsonl").exists()
    validation_summary: SplitSummary | None = None
    test_summary: SplitSummary | None = None

    if has_project_splits:
        train_examples = load_project_split(dataset_root, args.train_split, args.annotation_target_key)
        validation_split = choose_validation_split(dataset_root, args.validation_split)
        validation_examples = (
            load_project_split(dataset_root, validation_split, args.annotation_target_key)
            if validation_split is not None
            else []
        )
        test_split = args.test_split if args.test_split else None
        test_examples = (
            load_project_split(dataset_root, test_split, args.annotation_target_key)
            if test_split is not None and (dataset_root / test_split / "metadata.jsonl").exists()
            else []
        )
    else:
        train_examples = load_flat_examples(dataset_root, args.annotation_target_key)
        validation_split = None
        validation_examples = []
        test_split = None
        test_examples = []

    if not train_examples:
        raise ValueError("Training split did not contain any examples.")

    train_summary = prepare_split(
        train_examples,
        "train",
        dataset_root,
        output_dir,
        args.annotation_target_key,
        args.prompt,
        args.absolute_image_paths,
        args.dry_run,
    )

    if validation_examples:
        validation_summary = prepare_split(
            validation_examples,
            "validation",
            dataset_root,
            output_dir,
            args.annotation_target_key,
            args.prompt,
            args.absolute_image_paths,
            args.dry_run,
        )

    if test_examples:
        test_summary = prepare_split(
            test_examples,
            "test",
            dataset_root,
            output_dir,
            args.annotation_target_key,
            args.prompt,
            args.absolute_image_paths,
            args.dry_run,
        )

    max_steps, warmup_steps = estimate_steps(
        train_examples=len(train_examples),
        epochs=args.epochs,
        num_gpus=args.num_gpus,
        packing_size=args.packing_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    command_file = write_command_template(
        output_dir=output_dir,
        train_jsonl=Path(train_summary.output_jsonl),
        validation_jsonl=Path(validation_summary.output_jsonl) if validation_summary else None,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        packing_size=args.packing_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dry_run=args.dry_run,
    )

    summary = PreparationSummary(
        dataset_root=str(dataset_root),
        output_dir=str(output_dir),
        annotation_target_key=args.annotation_target_key,
        train_split=train_summary,
        validation_split=validation_summary,
        test_split=test_summary,
        estimated_max_steps=max_steps,
        estimated_warmup_steps=warmup_steps,
        command_file=str(command_file),
    )

    if not args.dry_run:
        with (output_dir / "preparation_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(asdict(summary), handle, ensure_ascii=False, indent=2)

    return summary


def main() -> int:
    args = parse_args()
    summary = prepare(args)
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
