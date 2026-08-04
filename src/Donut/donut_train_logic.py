from __future__ import annotations

import argparse
import html
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.run_utils import RunContext, namespace_to_dict, normalize_trainer_metrics, write_json
from src.utils.training_history import (
    prune_checkpoints_to_best_and_last,
    summarize_checkpoints,
    summarize_training_history,
)
from src.utils.training_plots import generate_training_plots
from src.eval_suite import JsonEvaluator, make_compute_metrics

DEFAULT_SCHEMA_PATH = REPO_ROOT / "json_schema" / "content.schema.json"
DEFAULT_TARGET_SKELETON_PATH = REPO_ROOT / "json_schema" / "content.empty.json"
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "datasets" / "250_CMRS_240dpi_20260707"
DEFAULT_MODEL_ID = "naver-clova-ix/donut-base"
DEFAULT_RUNS_DIR = REPO_ROOT / "runs" / "donut"
DEFAULT_TASK_START_TOKEN = "<s_lieferschein>"
DEFAULT_ANNOTATION_TARGET_KEY = "content"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def parse_args(
    argv: Sequence[str] | None = None,
    defaults: Mapping[str, Any] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune Donut for document information extraction. The trainer accepts "
            "the local data/datasets layout with separate image and annotation paths, "
            "and the official Donut metadata.jsonl / gt_parse layout."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=(
            "Dataset root. Supported layouts: data/datasets-style split folders with "
            "metadata.jsonl rows containing image/annotation paths; official Donut split "
            "folders with file_name/ground_truth rows; or a flat folder with image/json pairs "
            "such as data/small testing."
        ),
    )
    parser.add_argument("--train-split", default="train", help="Training split directory name.")
    parser.add_argument(
        "--validation-split",
        default=None,
        help="Validation split directory name. If omitted, the trainer tries validation, val, then dev.",
    )
    parser.add_argument(
        "--annotation-target-key",
        default=DEFAULT_ANNOTATION_TARGET_KEY,
        help=(
            "Key inside project annotation JSON files to use as gt_parse. "
            "The default 'content' ignores annotation metadata. Use 'root' to train on the full JSON object."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for checkpoints, metadata, and the final fine-tuned model. "
            "If omitted, a timestamped directory is created under --runs-dir."
        ),
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Parent directory for timestamped Donut fine-tuning runs when --output-dir is omitted.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional stable run folder name. Defaults to a timestamp plus dataset name.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Base Hugging Face model id or local Donut checkpoint.",
    )
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
        help="Schema file used to derive custom field special tokens.",
    )
    parser.add_argument(
        "--target-skeleton-path",
        type=Path,
        default=DEFAULT_TARGET_SKELETON_PATH,
        help=(
            "Empty target JSON skeleton used as the structured output contract. "
            "Its field names are added as Donut special tokens and recorded in run metadata."
        ),
    )
    parser.add_argument(
        "--task-start-token",
        default=DEFAULT_TASK_START_TOKEN,
        help="Task start token used to prompt the decoder for this extraction task.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=(1280, 960),
        help="Processor resize target. Use a larger size only if the document quality requires it.",
    )
    parser.add_argument(
        "--align-long-axis",
        action="store_true",
        help="Enable Donut long-axis alignment preprocessing before resizing.",
    )
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum decoder sequence length.")
    parser.add_argument(
        "--no-resize-decoder-position-embeddings",
        action="store_true",
        help=(
            "Do not automatically extend decoder position embeddings when --max-length exceeds "
            "the base Donut decoder limit."
        ),
    )
    parser.add_argument("--num-train-epochs", type=float, default=10.0, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=3e-5, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--warmup-steps", type=int, default=200, help="Warmup steps.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=2, help="Train batch size per GPU.")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2, help="Eval batch size per GPU.")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps.",
    )
    checkpointing_group = parser.add_mutually_exclusive_group()
    checkpointing_group.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce activation memory.",
    )
    checkpointing_group.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Disable gradient checkpointing.",
    )
    parser.set_defaults(gradient_checkpointing=None)
    parser.add_argument("--dataloader-num-workers", type=int, default=4, help="PyTorch dataloader workers.")
    parser.add_argument("--eval-steps", type=int, default=250, help="Evaluation interval in optimizer steps.")
    parser.add_argument("--save-steps", type=int, default=250, help="Checkpoint save interval.")
    parser.add_argument("--logging-steps", type=int, default=25, help="Logging interval.")
    parser.add_argument(
        "--validation-preview-samples",
        type=int,
        default=0,
        help=(
            "Generate side-by-side predictions for this many fixed validation examples at each "
            "training logging step. Disabled by default because autoregressive generation adds "
            "extra training time."
        ),
    )
    parser.add_argument(
        "--validation-preview-max-length",
        type=int,
        default=None,
        help=(
            "Maximum generation length for validation previews. Defaults to --max-length and "
            "cannot exceed it."
        ),
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help="Maximum number of checkpoints to keep on disk. Donut training keeps best and last only.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 mixed precision. Recommended on modern NVIDIA GPUs.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use fp16 mixed precision if bf16 is not available.",
    )
    local_files_group = parser.add_mutually_exclusive_group()
    local_files_group.add_argument(
        "--local-files-only",
        dest="local_files_only",
        action="store_true",
        help="Load Hugging Face model files only from the local cache.",
    )
    local_files_group.add_argument(
        "--no-local-files-only",
        dest="local_files_only",
        action="store_false",
        help="Allow Hugging Face model files to be downloaded when they are not cached locally.",
    )
    parser.set_defaults(local_files_only=False)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Optional checkpoint path to resume training from.",
    )
    parser.add_argument(
        "--predict-with-generate",
        action="store_true",
        help=(
            "Generate validation outputs and report structured JSON metrics from eval_suite. "
            "This is slower than loss-only evaluation."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Optional hard cap on optimizer steps. Use 1 for a smoke test.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap for quick debugging on a subset of the training examples.",
    )
    parser.add_argument(
        "--max-validation-samples",
        type=int,
        default=None,
        help="Optional cap for quick debugging on a subset of the validation examples.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse the dataset and schema, print a summary, then exit before loading model dependencies.",
    )
    if defaults:
        known_destinations = {action.dest for action in parser._actions}
        unknown_defaults = sorted(set(defaults) - known_destinations)
        if unknown_defaults:
            raise ValueError(
                "Unknown Donut training default(s): " + ", ".join(unknown_defaults)
            )
        parser.set_defaults(**defaults)

    return parser.parse_args(argv)


def apply_checkpoint_policy(args: argparse.Namespace) -> None:
    if args.save_steps != args.eval_steps:
        print(
            "Overriding --save-steps to match --eval-steps so eval-loss checkpoint selection is exact "
            f"({args.save_steps} -> {args.eval_steps}).",
            file=sys.stderr,
        )
        args.save_steps = args.eval_steps
    if args.save_total_limit != 2:
        print(
            "Overriding --save-total-limit to 2 so only the best and last checkpoints are retained "
            f"({args.save_total_limit} -> 2).",
            file=sys.stderr,
        )
        args.save_total_limit = 2


def load_runtime_dependencies() -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    missing: list[str] = []

    try:
        import torch
    except ImportError:
        missing.append("torch")
        torch = None

    try:
        from PIL import Image
    except ImportError:
        missing.append("Pillow")
        Image = None

    try:
        from transformers import (
            DonutProcessor,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            TrainerCallback,
            VisionEncoderDecoderModel,
        )
    except ImportError:
        missing.append("transformers")
        DonutProcessor = None
        Seq2SeqTrainer = None
        Seq2SeqTrainingArguments = None
        TrainerCallback = None
        VisionEncoderDecoderModel = None

    try:
        from datasets import Dataset
    except ImportError:
        missing.append("datasets")
        Dataset = None

    if missing:
        missing_csv = ", ".join(missing)
        raise RuntimeError(
            "Missing runtime dependencies: "
            f"{missing_csv}. Install them before training, for example: "
            "`pip install torch torchvision transformers datasets pillow sentencepiece accelerate evaluate`."
        )

    return (
        torch,
        Image,
        DonutProcessor,
        VisionEncoderDecoderModel,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
        TrainerCallback,
        Dataset,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


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


def resolve_dataset_root(args: argparse.Namespace) -> Path:
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {dataset_root}")
    return dataset_root


def resolve_existing_path(path_value: str, dataset_root: Path, split_dir: Path) -> Path:
    path = Path(path_value)
    candidates = [path] if path.is_absolute() else [dataset_root / path, split_dir / path]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Referenced path does not exist: {path_value}. Checked: {checked}")


def extract_annotation_target(annotation: Any, target_key: str) -> Any:
    if target_key in {"", ".", "root"}:
        return annotation

    target = annotation
    for key in target_key.split("."):
        if not isinstance(target, dict) or key not in target:
            raise KeyError(f"Annotation target key {target_key!r} not found.")
        target = target[key]

    if not isinstance(target, dict):
        raise TypeError(f"Annotation target {target_key!r} must resolve to a JSON object.")

    return target


def build_project_example_record(
    raw_record: dict[str, Any],
    dataset_root: Path,
    split_dir: Path,
    annotation_target_key: str,
) -> dict[str, Any]:
    if "image" not in raw_record:
        raise ValueError(f"Missing 'image' in project dataset metadata row: {raw_record}")
    if "annotation" not in raw_record:
        raise ValueError(f"Missing 'annotation' in project dataset metadata row: {raw_record}")

    image_path = resolve_existing_path(raw_record["image"], dataset_root, split_dir)
    annotation_path = resolve_existing_path(raw_record["annotation"], dataset_root, split_dir)
    annotation = load_json(annotation_path)
    gt_parse = extract_annotation_target(annotation, annotation_target_key)

    return {
        "id": raw_record.get("id", image_path.stem),
        "image_path": str(image_path),
        "annotation_path": str(annotation_path),
        "gt_parse": gt_parse,
    }


def build_donut_example_record(raw_record: dict[str, Any], split_dir: Path) -> dict[str, Any]:
    if "file_name" not in raw_record:
        raise ValueError(f"Missing 'file_name' in Donut metadata row: {raw_record}")
    if "ground_truth" not in raw_record:
        raise ValueError(f"Missing 'ground_truth' in Donut metadata row: {raw_record}")

    image_path = split_dir / raw_record["file_name"]
    if not image_path.exists():
        raise FileNotFoundError(f"Referenced image does not exist: {image_path}")

    ground_truth = raw_record["ground_truth"]
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)

    if "gt_parse" not in ground_truth:
        raise ValueError(
            f"Donut fine-tuning expects 'gt_parse' in ground_truth. "
            f"Found keys {sorted(ground_truth.keys())} for image {image_path}"
        )

    gt_parse = ground_truth["gt_parse"]
    return {
        "id": raw_record.get("id", image_path.stem),
        "image_path": str(image_path),
        "annotation_path": None,
        "gt_parse": gt_parse,
    }


def build_metadata_example_record(
    raw_record: dict[str, Any],
    dataset_root: Path,
    split_dir: Path,
    annotation_target_key: str,
) -> dict[str, Any]:
    if "image" in raw_record and "annotation" in raw_record:
        return build_project_example_record(raw_record, dataset_root, split_dir, annotation_target_key)
    if "file_name" in raw_record and "ground_truth" in raw_record:
        return build_donut_example_record(raw_record, split_dir)

    raise ValueError(
        "Unsupported metadata.jsonl row. Expected project keys 'image'/'annotation' "
        f"or Donut keys 'file_name'/'ground_truth'. Found keys: {sorted(raw_record.keys())}"
    )


def load_metadata_split(
    dataset_root: Path,
    split_dir: Path,
    annotation_target_key: str,
) -> list[dict[str, Any]]:
    metadata_path = split_dir / "metadata.jsonl"
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.jsonl in split directory: {split_dir}")

    return [
        build_metadata_example_record(record, dataset_root, split_dir, annotation_target_key)
        for record in read_jsonl(metadata_path)
    ]


def find_image_for_annotation(annotation_path: Path, image_files: dict[str, Path]) -> Path:
    stems = [annotation_path.stem]
    without_numeric_suffix = re.sub(r"_\d+$", "", annotation_path.stem)
    if without_numeric_suffix not in stems:
        stems.append(without_numeric_suffix)

    for stem in stems:
        if stem in image_files:
            return image_files[stem]

    raise FileNotFoundError(f"No matching image found for annotation: {annotation_path}")


def load_flat_examples(dataset_root: Path, annotation_target_key: str) -> list[dict[str, Any]]:
    image_files = {
        path.stem: path
        for path in sorted(dataset_root.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    annotation_files = [
        path
        for path in sorted(dataset_root.iterdir())
        if path.is_file() and path.suffix.lower() == ".json"
    ]

    examples: list[dict[str, Any]] = []
    for annotation_path in annotation_files:
        annotation = load_json(annotation_path)
        gt_parse = extract_annotation_target(annotation, annotation_target_key)
        image_path = find_image_for_annotation(annotation_path, image_files)
        examples.append(
            {
                "id": image_path.stem,
                "image_path": str(image_path),
                "annotation_path": str(annotation_path),
                "gt_parse": gt_parse,
            }
        )

    if not examples:
        raise ValueError(
            f"No flat image/json pairs found in {dataset_root}. "
            "Expected files such as example.jpg and example_0.json."
        )

    return examples


def choose_validation_split(dataset_root: Path, requested_split: str | None) -> str | None:
    if requested_split is not None:
        return requested_split

    for split_name in ("validation", "val", "dev"):
        if (dataset_root / split_name / "metadata.jsonl").exists():
            return split_name

    return None


def collect_json_field_paths(obj: Any) -> set[str]:
    paths: set[str] = set()

    def visit(node: Any, prefix: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                path = f"{prefix}.{key}" if prefix else key
                paths.add(path)
                visit(value, path)
        elif isinstance(node, list):
            for item in node:
                visit(item, f"{prefix}[]")

    visit(obj, "")
    return paths


def summarize_target_shape(
    examples: list[dict[str, Any]],
    skeleton: Any,
) -> dict[str, Any]:
    skeleton_paths = collect_json_field_paths(skeleton)
    example_paths: set[str] = set()
    for example in examples:
        example_paths.update(collect_json_field_paths(example["gt_parse"]))

    return {
        "skeleton_field_count": len(skeleton_paths),
        "dataset_field_count": len(example_paths),
        "missing_from_dataset": sorted(skeleton_paths - example_paths),
        "extra_in_dataset": sorted(example_paths - skeleton_paths),
    }


def cap_examples(examples: list[dict[str, Any]], sample_limit: int | None) -> list[dict[str, Any]]:
    if sample_limit is None:
        return examples
    if sample_limit < 1:
        raise ValueError("Sample limits must be positive integers.")
    return examples[:sample_limit]


def split_flat_examples(
    examples: list[dict[str, Any]],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(examples) == 1:
        return examples, examples

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    validation_size = max(1, round(len(shuffled) * 0.1))
    return shuffled[validation_size:], shuffled[:validation_size]


def build_trainer_dataset_records(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": example["id"],
            "image_path": example["image_path"],
            "target_sequence": example["target_sequence"],
        }
        for example in examples
    ]


def load_dataset_splits(
    dataset_root: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, str, str]:
    train_split_dir = dataset_root / args.train_split

    if (train_split_dir / "metadata.jsonl").exists():
        validation_split = choose_validation_split(dataset_root, args.validation_split)
        if validation_split is None:
            raise FileNotFoundError(
                "Could not find a validation split. Expected one of validation/, val/, or dev/ "
                "with metadata.jsonl, or pass --validation-split."
            )
        validation_split_dir = dataset_root / validation_split
        train_examples = load_metadata_split(dataset_root, train_split_dir, args.annotation_target_key)
        validation_examples = load_metadata_split(dataset_root, validation_split_dir, args.annotation_target_key)
        source_layout = "project_or_donut_metadata_splits"
    elif (dataset_root / "metadata.jsonl").exists():
        examples = load_metadata_split(dataset_root, dataset_root, args.annotation_target_key)
        train_examples, validation_examples = split_flat_examples(examples, args.seed)
        validation_split = "auto"
        source_layout = "single_metadata_split"
    else:
        examples = load_flat_examples(dataset_root, args.annotation_target_key)
        train_examples, validation_examples = split_flat_examples(examples, args.seed)
        validation_split = "auto"
        source_layout = "flat_image_json_pairs"

    train_examples = cap_examples(train_examples, args.max_train_samples)
    validation_examples = cap_examples(validation_examples, args.max_validation_samples)

    if not train_examples:
        raise ValueError("Training split did not contain any examples.")
    if not validation_examples:
        raise ValueError("Validation split did not contain any examples.")

    return train_examples, validation_examples, args.train_split, validation_split, source_layout


def resolve_schema_ref(schema: dict[str, Any], node: Any) -> Any:
    while isinstance(node, dict) and "$ref" in node:
        ref = node["$ref"]
        if not ref.startswith("#/"):
            return node
        resolved: Any = schema
        for part in ref.removeprefix("#/").split("/"):
            resolved = resolved[part]
        node = resolved
    return node


def select_schema_node_for_target(schema: Any, target_key: str) -> Any:
    if not isinstance(schema, dict) or target_key in {"", ".", "root"}:
        return schema

    node: Any = schema
    for key in target_key.split("."):
        node = resolve_schema_ref(schema, node)
        properties = node.get("properties") if isinstance(node, dict) else None
        if not isinstance(properties, dict) or key not in properties:
            return schema
        node = properties[key]

    return resolve_schema_ref(schema, node)


def collect_schema_tokens(schema: Any, root_schema: Any | None = None) -> set[str]:
    tokens: set[str] = set()
    root = root_schema if root_schema is not None else schema
    seen_refs: set[str] = set()

    def resolve_ref(ref: str) -> Any:
        if not ref.startswith("#/") or not isinstance(root, dict):
            return None
        node: Any = root
        for part in ref.removeprefix("#/").split("/"):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str):
                if ref in seen_refs:
                    return
                seen_refs.add(ref)
                resolved = resolve_ref(ref)
                if resolved is not None:
                    visit(resolved)
                return

            for key, value in node.items():
                if key == "properties" and isinstance(value, dict):
                    for field_name, child_node in value.items():
                        tokens.add(f"<s_{field_name}>")
                        tokens.add(f"</s_{field_name}>")
                        visit(child_node)
                elif key == "$defs":
                    for child_node in value.values():
                        visit(child_node)
                elif key in {"items", "anyOf", "allOf", "oneOf"}:
                    visit(value)
                else:
                    visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(schema)
    return tokens


def collect_field_tokens_from_gt_parse(gt_parse: Any) -> set[str]:
    tokens: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                tokens.add(f"<s_{key}>")
                tokens.add(f"</s_{key}>")
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(gt_parse)
    return tokens


def ordered_json_keys(obj: dict[str, Any], order_template: Any) -> list[str]:
    """Order known keys by the explicit template and unknown keys deterministically."""
    template_keys = list(order_template) if isinstance(order_template, dict) else []
    template_key_set = set(template_keys)
    known_keys = [key for key in template_keys if key in obj]
    unknown_keys = sorted(key for key in obj if key not in template_key_set)
    return known_keys + unknown_keys


def json_to_donut_tokens(obj: Any, order_template: Any) -> str:
    """Serialize JSON using the recursive field order declared by a target template."""
    if isinstance(obj, dict):
        output = ""
        for key in ordered_json_keys(obj, order_template):
            child_template = order_template.get(key) if isinstance(order_template, dict) else None
            output += f"<s_{key}>"
            output += json_to_donut_tokens(obj[key], child_template)
            output += f"</s_{key}>"
        return output

    if isinstance(obj, list):
        item_template = order_template[0] if isinstance(order_template, list) and order_template else None
        return "<sep/>".join(json_to_donut_tokens(item, item_template) for item in obj)

    if obj is None:
        return "<null/>"

    if isinstance(obj, bool):
        return "true" if obj else "false"

    return str(obj)


def add_target_sequences(
    examples: list[dict[str, Any]], target_skeleton: Any
) -> list[dict[str, Any]]:
    """Attach canonical, target-skeleton-ordered Donut sequences to dataset examples."""
    return [
        {
            **example,
            "target_sequence": json_to_donut_tokens(example["gt_parse"], target_skeleton),
        }
        for example in examples
    ]


def select_validation_preview_examples(
    examples: list[dict[str, Any]], sample_count: int, seed: int
) -> list[dict[str, Any]]:
    """Choose a fixed, reproducible validation subset for the whole run."""
    if sample_count <= 0:
        return []
    if sample_count >= len(examples):
        return list(examples)

    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    return [examples[index] for index in sorted(indices[:sample_count])]


def json_differences(expected: Any, predicted: Any, path: str = "$") -> list[dict[str, Any]]:
    """Return compact, path-oriented differences between two JSON-compatible values."""
    if type(expected) is not type(predicted):
        return [
            {
                "kind": "type_mismatch",
                "path": path,
                "ground_truth": expected,
                "prediction": predicted,
            }
        ]

    if isinstance(expected, dict):
        differences: list[dict[str, Any]] = []
        for key in expected.keys() - predicted.keys():
            differences.append(
                {
                    "kind": "missing_in_prediction",
                    "path": f"{path}.{key}",
                    "ground_truth": expected[key],
                }
            )
        for key in predicted.keys() - expected.keys():
            differences.append(
                {
                    "kind": "unexpected_in_prediction",
                    "path": f"{path}.{key}",
                    "prediction": predicted[key],
                }
            )
        for key in expected.keys() & predicted.keys():
            differences.extend(json_differences(expected[key], predicted[key], f"{path}.{key}"))
        return sorted(differences, key=lambda item: (item["path"], item["kind"]))

    if isinstance(expected, list):
        differences = []
        common_length = min(len(expected), len(predicted))
        for index in range(common_length):
            differences.extend(
                json_differences(expected[index], predicted[index], f"{path}[{index}]")
            )
        for index in range(common_length, len(expected)):
            differences.append(
                {
                    "kind": "missing_in_prediction",
                    "path": f"{path}[{index}]",
                    "ground_truth": expected[index],
                }
            )
        for index in range(common_length, len(predicted)):
            differences.append(
                {
                    "kind": "unexpected_in_prediction",
                    "path": f"{path}[{index}]",
                    "prediction": predicted[index],
                }
            )
        return differences

    if expected != predicted:
        return [
            {
                "kind": "value_mismatch",
                "path": path,
                "ground_truth": expected,
                "prediction": predicted,
            }
        ]
    return []


def clean_generated_sequence(sequence: str, processor: Any) -> str:
    cleaned = sequence
    if processor.tokenizer.eos_token:
        cleaned = cleaned.replace(processor.tokenizer.eos_token, "")
    if processor.tokenizer.pad_token:
        cleaned = cleaned.replace(processor.tokenizer.pad_token, "")
    return re.sub(r"<.*?>", "", cleaned, count=1).strip()


def parse_generated_sequence(sequence: str, processor: Any) -> tuple[Any, str | None]:
    if not hasattr(processor, "token2json"):
        return {"text_sequence": sequence}, "Processor does not provide token2json()."
    try:
        return processor.token2json(sequence), None
    except Exception as exc:
        return {"text_sequence": sequence}, f"{type(exc).__name__}: {exc}"


def schema_types(schema_node: Any, root_schema: Any) -> set[str]:
    """Return the JSON types declared by a schema node after resolving local refs."""
    resolved = resolve_schema_ref(root_schema, schema_node)
    if not isinstance(resolved, dict):
        return set()

    declared = resolved.get("type")
    if isinstance(declared, str):
        return {declared}
    if isinstance(declared, list):
        return {value for value in declared if isinstance(value, str)}
    return set()


def normalize_prediction_to_schema(
    prediction: Any,
    schema_node: Any,
    root_schema: Any | None = None,
) -> Any:
    """Repair Donut parser ambiguities that can be resolved from the JSON schema.

    Donut's ``token2json`` represents ``<null/>`` as the string ``"null"``. It
    also represents a one-item list as its sole item because the serialized form
    contains no ``<sep/>`` token. Only these unambiguous, schema-supported cases
    are normalized; missing fields and other malformed values remain visible.
    """
    root = root_schema if root_schema is not None else schema_node
    resolved = resolve_schema_ref(root, schema_node)
    if not isinstance(resolved, dict):
        return prediction

    declared_types = schema_types(resolved, root)
    if prediction == "null" and "null" in declared_types:
        return None

    if "array" in declared_types:
        item_schema = resolved.get("items", {})
        if isinstance(prediction, list):
            return [
                normalize_prediction_to_schema(item, item_schema, root)
                for item in prediction
            ]
        if isinstance(prediction, dict):
            return [normalize_prediction_to_schema(prediction, item_schema, root)]
        return prediction

    if isinstance(prediction, dict) and (
        "object" in declared_types or isinstance(resolved.get("properties"), dict)
    ):
        properties = resolved.get("properties", {})
        return {
            key: normalize_prediction_to_schema(value, properties[key], root)
            if key in properties
            else value
            for key, value in prediction.items()
        }

    return prediction


def build_donut_compute_metrics(
    processor: Any,
    target_schema: Any,
    root_schema: Any,
) -> Any:
    """Create structured JSON metrics for generated Donut token sequences."""

    def decode_rows(token_rows: Any, *, generated: bool) -> list[Any]:
        if isinstance(token_rows, tuple):
            token_rows = token_rows[0]
        rows = token_rows.tolist() if hasattr(token_rows, "tolist") else token_rows
        pad_id = processor.tokenizer.pad_token_id
        restored_rows = [
            [pad_id if token_id == -100 else token_id for token_id in row]
            for row in rows
        ]
        sequences = processor.batch_decode(restored_rows)
        decoded: list[Any] = []
        for sequence in sequences:
            if generated:
                cleaned = clean_generated_sequence(sequence, processor)
            else:
                cleaned = sequence
                if processor.tokenizer.eos_token:
                    cleaned = cleaned.replace(processor.tokenizer.eos_token, "")
                if processor.tokenizer.pad_token:
                    cleaned = cleaned.replace(processor.tokenizer.pad_token, "")
                cleaned = cleaned.strip()
            parsed, parse_error = parse_generated_sequence(cleaned, processor)
            if parse_error:
                # A non-JSON string lets JsonEvaluator account for this as a parse
                # failure and as missing expected values.
                decoded.append(f"Donut parse error: {parse_error}")
                continue
            decoded.append(
                normalize_prediction_to_schema(parsed, target_schema, root_schema)
            )
        return decoded

    return make_compute_metrics(
        decode_predictions=lambda rows: decode_rows(rows, generated=True),
        decode_references=lambda rows: decode_rows(rows, generated=False),
        evaluator=JsonEvaluator(schema=target_schema),
    )


def generate_validation_preview_sample(
    *,
    torch: Any,
    image_module: Any,
    processor: Any,
    model: Any,
    example: dict[str, Any],
    task_start_token: str,
    max_length: int,
    target_schema: Any,
    root_schema: Any,
) -> dict[str, Any]:
    with image_module.open(example["image_path"]) as image:
        rgb_image = image.convert("RGB")
    device = next(model.parameters()).device
    pixel_values = processor(images=rgb_image, return_tensors="pt").pixel_values.to(device)
    decoder_input_ids = processor.tokenizer(
        task_start_token,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)
    generation_kwargs: dict[str, Any] = {
        "max_length": max_length,
        "num_beams": 1,
        "pad_token_id": processor.tokenizer.pad_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "use_cache": True,
        "return_dict_in_generate": True,
    }
    if processor.tokenizer.unk_token_id is not None:
        generation_kwargs["bad_words_ids"] = [[processor.tokenizer.unk_token_id]]

    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        **generation_kwargs,
    )
    raw_sequence = processor.batch_decode(outputs.sequences)[0]
    cleaned_sequence = clean_generated_sequence(raw_sequence, processor)
    raw_prediction, parse_error = parse_generated_sequence(cleaned_sequence, processor)
    prediction = normalize_prediction_to_schema(raw_prediction, target_schema, root_schema)
    differences = json_differences(example["gt_parse"], prediction)
    return {
        "id": example["id"],
        "image_path": example["image_path"],
        "annotation_path": example.get("annotation_path"),
        "ground_truth": example["gt_parse"],
        "raw_prediction": raw_prediction,
        "prediction": prediction,
        "exact_match": not differences,
        "differences": differences,
        "parse_error": parse_error,
        "raw_sequence": raw_sequence,
        "cleaned_sequence": cleaned_sequence,
    }


def render_validation_preview_html(payload: dict[str, Any]) -> str:
    def pretty(value: Any) -> str:
        return html.escape(json.dumps(value, ensure_ascii=False, indent=2))

    sample_sections: list[str] = []
    for sample in payload.get("samples", []):
        image_uri = Path(sample["image_path"]).resolve().as_uri()
        exact_match = "yes" if sample["exact_match"] else "no"
        parse_error = sample.get("parse_error") or "none"
        sample_sections.append(
            f"""
            <section class="sample">
              <h2>{html.escape(str(sample['id']))}</h2>
              <p><a href="{html.escape(image_uri, quote=True)}">Open source image</a>
                 &middot; exact match: <strong>{exact_match}</strong>
                 &middot; differences: {len(sample['differences'])}
                 &middot; parse error: {html.escape(parse_error)}</p>
              <div class="comparison">
                <div><h3>Ground truth</h3><pre>{pretty(sample['ground_truth'])}</pre></div>
                <div><h3>Prediction</h3><pre>{pretty(sample['prediction'])}</pre></div>
              </div>
              <details><summary>Structured differences</summary><pre>{pretty(sample['differences'])}</pre></details>
              <details><summary>Raw generated sequence</summary><pre>{html.escape(sample['raw_sequence'])}</pre></details>
            </section>
            """
        )

    error_section = ""
    if payload.get("error"):
        error_section = f"<p class=\"error\">{html.escape(str(payload['error']))}</p>"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Donut validation preview - step {payload['global_step']}</title>
  <style>
    body {{ margin: 2rem; color: #202124; font-family: system-ui, sans-serif; }}
    .comparison {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 1rem; }}
    .sample {{ border-top: 2px solid #dadce0; margin-top: 2rem; padding-top: 1rem; }}
    pre {{ background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px; overflow: auto; padding: 1rem; white-space: pre-wrap; }}
    .error {{ color: #b00020; font-weight: 600; }}
    @media (max-width: 900px) {{ .comparison {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>Donut validation preview</h1>
  <p>Step {payload['global_step']} &middot; epoch {payload.get('epoch')} &middot; status {html.escape(payload['status'])}</p>
  {error_section}
  {''.join(sample_sections)}
</body>
</html>
"""


def write_validation_preview(output_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    preview_dir = output_dir / "validation_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    stem = f"step_{int(payload['global_step']):08d}"
    json_path = preview_dir / f"{stem}.json"
    html_path = preview_dir / f"{stem}.html"
    latest_json_path = preview_dir / "latest.json"
    latest_html_path = preview_dir / "latest.html"
    rendered_html = render_validation_preview_html(payload)

    write_json(json_path, payload)
    write_json(latest_json_path, payload)
    html_path.write_text(rendered_html, encoding="utf-8")
    latest_html_path.write_text(rendered_html, encoding="utf-8")
    return {
        "json": str(json_path),
        "html": str(html_path),
        "latest_json": str(latest_json_path),
        "latest_html": str(latest_html_path),
    }


def build_validation_preview_callback(
    *,
    TrainerCallback: Any,
    torch: Any,
    image_module: Any,
    processor: Any,
    examples: list[dict[str, Any]],
    output_dir: Path,
    task_start_token: str,
    max_length: int,
    target_schema: Any,
    root_schema: Any,
) -> Any:
    class ValidationPreviewCallback(TrainerCallback):
        def __init__(self) -> None:
            self.last_preview_step = -1

        def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any) -> Any:
            current_logs = logs or {}
            global_step = int(state.global_step)
            if (
                not examples
                or "loss" not in current_logs
                or global_step <= 0
                or global_step == self.last_preview_step
                or not getattr(state, "is_world_process_zero", True)
            ):
                return control

            self.last_preview_step = global_step
            model = kwargs["model"]
            was_training = model.training
            payload: dict[str, Any] = {
                "status": "completed",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "global_step": global_step,
                "epoch": float(state.epoch) if state.epoch is not None else None,
                "training_log": current_logs,
                "task_start_token": task_start_token,
                "max_length": max_length,
                "samples": [],
            }
            try:
                model.eval()
                with torch.inference_mode():
                    payload["samples"] = [
                        generate_validation_preview_sample(
                            torch=torch,
                            image_module=image_module,
                            processor=processor,
                            model=model,
                            example=example,
                            task_start_token=task_start_token,
                            max_length=max_length,
                            target_schema=target_schema,
                            root_schema=root_schema,
                        )
                        for example in examples
                    ]
                payload["exact_matches"] = sum(
                    sample["exact_match"] for sample in payload["samples"]
                )
            except Exception as exc:
                payload["status"] = "failed"
                payload["error"] = f"{type(exc).__name__}: {exc}"
                print(
                    f"Validation preview failed at step {global_step}: {payload['error']}",
                    file=sys.stderr,
                )
            finally:
                if was_training:
                    model.train()

            paths = write_validation_preview(output_dir, payload)
            print(f"Saved validation preview for step {global_step} to {paths['html']}")
            return control

    return ValidationPreviewCallback()


@dataclass
class DonutBatchCollator:
    processor: Any
    image_module: Any
    max_length: int
    ignore_id: int = -100

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        images = []
        target_sequences = []

        for feature in features:
            with self.image_module.open(feature["image_path"]) as image:
                images.append(image.convert("RGB"))
            sequence = feature["target_sequence"]
            eos_token = self.processor.tokenizer.eos_token or ""
            target_sequences.append(sequence + eos_token)

        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values
        tokenized = self.processor.tokenizer(
            target_sequences,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = tokenized.input_ids
        labels[labels == self.processor.tokenizer.pad_token_id] = self.ignore_id

        return {"pixel_values": pixel_values, "labels": labels}


def add_special_tokens(processor: Any, model: Any, special_tokens: set[str]) -> int:
    ordered = sorted(token for token in special_tokens if token)
    if not ordered:
        return 0

    added = processor.tokenizer.add_special_tokens({"additional_special_tokens": ordered})
    if added > 0:
        model.decoder.resize_token_embeddings(len(processor.tokenizer))
    return added


def build_model_load_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    kwargs: dict[str, Any] = {"local_files_only": args.local_files_only}
    if args.cache_dir is not None:
        kwargs["cache_dir"] = str(args.cache_dir)
    return kwargs


def choose_precision_flags(args: argparse.Namespace) -> tuple[bool, bool]:
    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of --bf16 or --fp16.")
    return args.bf16, args.fp16


def validate_image_size_for_encoder(model: Any, image_size: tuple[int, int]) -> None:
    encoder_config = getattr(getattr(model, "encoder", None), "config", None)
    if encoder_config is None:
        return

    window_size = getattr(encoder_config, "window_size", None)
    patch_size = getattr(encoder_config, "patch_size", None)
    depths = getattr(encoder_config, "depths", None)
    if not isinstance(window_size, int) or not isinstance(patch_size, int) or not depths:
        return

    downscale_factor = patch_size * (2 ** (len(depths) - 1))
    minimum_side = window_size * downscale_factor
    height, width = image_size
    if height < minimum_side or width < minimum_side:
        raise ValueError(
            f"--image-size {height} {width} is too small for this Donut Swin encoder. "
            f"Both sides must be at least {minimum_side}px for window_size={window_size}. "
            "Use a larger debug size such as 640 480 or the default 1280 960."
        )


def get_decoder_max_position_embeddings(model: Any) -> int | None:
    decoder_config = getattr(getattr(model, "decoder", None), "config", None)
    value = getattr(decoder_config, "max_position_embeddings", None)
    return value if isinstance(value, int) else None


def resize_decoder_position_embeddings(model: Any, torch: Any, max_length: int) -> bool:
    current_max_length = get_decoder_max_position_embeddings(model)
    if current_max_length is None or max_length <= current_max_length:
        return False

    decoder = getattr(getattr(model, "decoder", None), "model", None)
    decoder_body = getattr(decoder, "decoder", None)
    old_embeddings = getattr(decoder_body, "embed_positions", None)
    if old_embeddings is None:
        raise ValueError(
            "Cannot resize decoder position embeddings for this model. "
            "Use --max-length no larger than the checkpoint decoder limit."
        )

    embedding_cls = type(old_embeddings)
    new_embeddings = embedding_cls(max_length, old_embeddings.embedding_dim)
    new_embeddings.to(device=old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)

    with torch.no_grad():
        copy_rows = min(old_embeddings.weight.shape[0], new_embeddings.weight.shape[0])
        new_embeddings.weight[:copy_rows].copy_(old_embeddings.weight[:copy_rows])
        if new_embeddings.weight.shape[0] > copy_rows:
            mean = old_embeddings.weight.mean().item()
            std = old_embeddings.weight.std().item()
            new_embeddings.weight[copy_rows:].normal_(mean=mean, std=std)

    decoder_body.embed_positions = new_embeddings
    model.decoder.config.max_position_embeddings = max_length
    if hasattr(model.config, "decoder"):
        model.config.decoder.max_position_embeddings = max_length
    return True


def summarize_token_lengths(
    examples: list[dict[str, Any]],
    processor: Any,
    max_length: int,
) -> dict[str, Any]:
    eos_token = processor.tokenizer.eos_token or ""
    lengths = [
        len(
            processor.tokenizer(
                example["target_sequence"] + eos_token,
                add_special_tokens=False,
            ).input_ids
        )
        for example in examples
    ]
    sorted_lengths = sorted(lengths)
    overlength = [
        {"id": example["id"], "target_tokens": length}
        for example, length in zip(examples, lengths, strict=True)
        if length > max_length
    ]
    p95_index = max(0, int(len(sorted_lengths) * 0.95) - 1)
    return {
        "min": sorted_lengths[0],
        "max": sorted_lengths[-1],
        "mean": sum(lengths) / len(lengths),
        "p50": sorted_lengths[len(sorted_lengths) // 2],
        "p95": sorted_lengths[p95_index],
        "over_max_length_count": len(overlength),
        "over_max_length_examples": overlength[:20],
    }


def validate_target_lengths(
    examples: list[dict[str, Any]],
    processor: Any,
    max_length: int,
) -> dict[str, Any]:
    summary = summarize_token_lengths(examples, processor, max_length)
    if summary["over_max_length_count"] > 0:
        example_ids = ", ".join(item["id"] for item in summary["over_max_length_examples"][:5])
        raise ValueError(
            f"{summary['over_max_length_count']} target sequences exceed --max-length={max_length}. "
            f"Longest target has {summary['max']} tokens. First overlength examples: {example_ids}. "
            "Increase --max-length or shorten the supervised target before training."
        )
    return summary


def describe_target_keys(example: dict[str, Any]) -> list[str]:
    gt_parse = example["gt_parse"]
    if isinstance(gt_parse, dict):
        return sorted(gt_parse.keys())
    return []


def print_dry_run_summary(
    dataset_root: Path,
    output_dir: Path,
    train_examples: list[dict[str, Any]],
    validation_examples: list[dict[str, Any]],
    train_split: str,
    validation_split: str,
    source_layout: str,
    annotation_target_key: str,
    schema_tokens: set[str],
    skeleton_tokens: set[str],
    data_tokens: set[str],
    target_shape_summary: dict[str, Any],
) -> None:
    max_train_chars = max(len(example["target_sequence"]) for example in train_examples)
    max_validation_chars = max(len(example["target_sequence"]) for example in validation_examples)

    print("Donut training dry run")
    print(f"  dataset_root: {dataset_root}")
    print(f"  output_dir: {output_dir}")
    print(f"  source_layout: {source_layout}")
    print(f"  annotation_target_key: {annotation_target_key}")
    print(f"  train_split: {train_split} ({len(train_examples)} examples)")
    print(f"  validation_split: {validation_split} ({len(validation_examples)} examples)")
    print(f"  train_target_keys: {describe_target_keys(train_examples[0])}")
    print(f"  validation_target_keys: {describe_target_keys(validation_examples[0])}")
    print(f"  max_train_target_chars: {max_train_chars}")
    print(f"  max_validation_target_chars: {max_validation_chars}")
    print(f"  schema_special_tokens: {len(schema_tokens)}")
    print(f"  skeleton_special_tokens: {len(skeleton_tokens)}")
    print(f"  data_special_tokens: {len(data_tokens)}")
    print(f"  skeleton_field_count: {target_shape_summary['skeleton_field_count']}")
    print(f"  dataset_field_count: {target_shape_summary['dataset_field_count']}")
    print(f"  fields_missing_from_dataset: {len(target_shape_summary['missing_from_dataset'])}")
    print(f"  fields_extra_in_dataset: {len(target_shape_summary['extra_in_dataset'])}")


def count_model_parameters(model: Any) -> dict[str, int]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return {"total": total, "trainable": trainable}


def build_training_arguments(
    Seq2SeqTrainingArguments: Any,
    args: argparse.Namespace,
    output_dir: Path,
    bf16: bool,
    fp16: bool,
    gradient_checkpointing: bool,
) -> Any:
    import inspect

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "save_strategy": "steps",
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "logging_steps": args.logging_steps,
        "save_total_limit": args.save_total_limit,
        "dataloader_num_workers": args.dataloader_num_workers,
        "remove_unused_columns": False,
        "predict_with_generate": args.predict_with_generate,
        "generation_max_length": args.max_length,
        "generation_num_beams": 1,
        "max_steps": args.max_steps,
        "bf16": bf16,
        "fp16": fp16,
        "gradient_checkpointing": gradient_checkpointing,
        "report_to": "none",
        "do_train": True,
        "do_eval": True,
        "load_best_model_at_end": True,
        "metric_for_best_model": (
            "eval_json_field_f1" if args.predict_with_generate else "eval_loss"
        ),
        "greater_is_better": bool(args.predict_with_generate),
        "seed": args.seed,
    }
    signature = inspect.signature(Seq2SeqTrainingArguments)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"

    return Seq2SeqTrainingArguments(**kwargs)


def build_trainer(Seq2SeqTrainer: Any, processor: Any, **kwargs: Any) -> Any:
    import inspect

    signature = inspect.signature(Seq2SeqTrainer.__init__)
    if "processing_class" in signature.parameters:
        kwargs["processing_class"] = processor
    else:
        kwargs["tokenizer"] = processor
    return Seq2SeqTrainer(**kwargs)


def trainer_state_to_dict(trainer: Any) -> dict[str, Any] | None:
    state = getattr(trainer, "state", None)
    if state is None:
        return None
    if hasattr(state, "to_json_string"):
        return json.loads(state.to_json_string())
    if hasattr(state, "__dict__"):
        return dict(state.__dict__)
    return None


def save_trainer_state(trainer: Any, output_dir: Path) -> dict[str, Any] | None:
    state = getattr(trainer, "state", None)
    if state is None:
        return None
    state_path = output_dir / "trainer_state.json"
    if hasattr(state, "save_to_json"):
        state.save_to_json(str(state_path))
        return trainer_state_to_dict(trainer)

    state_dict = trainer_state_to_dict(trainer)
    if state_dict is not None:
        write_json(state_path, state_dict)
    return state_dict


def main(
    argv: Sequence[str] | None = None,
    defaults: Mapping[str, Any] | None = None,
) -> int:
    args = parse_args(argv=argv, defaults=defaults)
    apply_checkpoint_policy(args)
    set_seed(args.seed)

    if args.validation_preview_samples < 0:
        raise ValueError("--validation-preview-samples must be greater than or equal to 0.")
    validation_preview_max_length = (
        args.max_length
        if args.validation_preview_max_length is None
        else args.validation_preview_max_length
    )
    if validation_preview_max_length <= 0:
        raise ValueError("--validation-preview-max-length must be greater than 0.")
    if validation_preview_max_length > args.max_length:
        raise ValueError(
            "--validation-preview-max-length cannot exceed --max-length because decoder position "
            "embeddings are sized from --max-length."
        )

    dataset_root = resolve_dataset_root(args)
    run = RunContext.create(
        pipeline_name="donut",
        runs_dir=args.runs_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        dataset_name=dataset_root.name,
        model_id=args.model_id,
    )
    output_dir = run.output_dir
    schema = load_json(args.schema_path)
    target_schema = select_schema_node_for_target(schema, args.annotation_target_key)
    target_skeleton = load_json(args.target_skeleton_path)
    train_examples, validation_examples, train_split, validation_split, source_layout = load_dataset_splits(
        dataset_root,
        args,
    )
    train_examples = add_target_sequences(train_examples, target_skeleton)
    validation_examples = add_target_sequences(validation_examples, target_skeleton)
    validation_preview_examples = select_validation_preview_examples(
        validation_examples,
        sample_count=args.validation_preview_samples,
        seed=args.seed,
    )
    target_shape_summary = summarize_target_shape(train_examples + validation_examples, target_skeleton)

    schema_tokens = collect_schema_tokens(target_schema, schema)
    skeleton_tokens = collect_field_tokens_from_gt_parse(target_skeleton)
    data_tokens: set[str] = set()
    for sample in train_examples + validation_examples:
        data_tokens.update(collect_field_tokens_from_gt_parse(sample["gt_parse"]))

    if args.dry_run:
        print_dry_run_summary(
            dataset_root=dataset_root,
            output_dir=output_dir,
            train_examples=train_examples,
            validation_examples=validation_examples,
            train_split=train_split,
            validation_split=validation_split,
            source_layout=source_layout,
            annotation_target_key=args.annotation_target_key,
            schema_tokens=schema_tokens,
            skeleton_tokens=skeleton_tokens,
            data_tokens=data_tokens,
            target_shape_summary=target_shape_summary,
        )
        print("  target_serialization: target_skeleton_order")
        if isinstance(target_skeleton, dict):
            print("  target_field_order: " + ", ".join(target_skeleton))
        print(f"  validation_preview_samples: {len(validation_preview_examples)}")
        if validation_preview_examples:
            print(
                "  validation_preview_sample_ids: "
                + ", ".join(str(example["id"]) for example in validation_preview_examples)
            )
            print(f"  validation_preview_max_length: {validation_preview_max_length}")
        return 0

    (
        torch,
        image_module,
        DonutProcessor,
        VisionEncoderDecoderModel,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
        TrainerCallback,
        Dataset,
    ) = load_runtime_dependencies()

    model_load_kwargs = build_model_load_kwargs(args)
    processor = DonutProcessor.from_pretrained(args.model_id, **model_load_kwargs)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_id, **model_load_kwargs)
    validate_image_size_for_encoder(model, args.image_size)

    processor.image_processor.size = {"height": args.image_size[0], "width": args.image_size[1]}
    processor.image_processor.do_align_long_axis = args.align_long_axis

    special_tokens = {
        args.task_start_token,
        "<sep/>",
        "<null/>",
    }
    special_tokens.update(schema_tokens)
    special_tokens.update(skeleton_tokens)
    special_tokens.update(data_tokens)
    added_special_tokens = add_special_tokens(processor, model, special_tokens)

    decoder_max_positions_before = get_decoder_max_position_embeddings(model)
    decoder_positions_resized = False
    if decoder_max_positions_before is not None and args.max_length > decoder_max_positions_before:
        if args.no_resize_decoder_position_embeddings:
            raise ValueError(
                f"--max-length={args.max_length} exceeds decoder max_position_embeddings="
                f"{decoder_max_positions_before}. Remove --no-resize-decoder-position-embeddings "
                "or choose a shorter --max-length."
            )
        decoder_positions_resized = resize_decoder_position_embeddings(model, torch, args.max_length)
    decoder_max_positions_after = get_decoder_max_position_embeddings(model)
    target_token_length_summary = validate_target_lengths(
        train_examples + validation_examples,
        processor,
        args.max_length,
    )

    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(args.task_start_token)
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.generation_config.max_length = args.max_length
    model.generation_config.early_stopping = False
    model.generation_config.no_repeat_ngram_size = 0
    model.generation_config.length_penalty = 1.0
    model.generation_config.num_beams = 1

    gradient_checkpointing = (
        True if args.gradient_checkpointing is None else args.gradient_checkpointing
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    output_dir.mkdir(parents=True, exist_ok=True)
    run_sections = {
        "dataset": {
            "dataset_root": str(args.dataset_root),
            "resolved_dataset_root": str(dataset_root),
            "source_layout": source_layout,
            "train_split": train_split,
            "validation_split": validation_split,
            "train_examples": len(train_examples),
            "validation_examples": len(validation_examples),
        },
        "target": {
            "annotation_target_key": args.annotation_target_key,
            "schema_path": str(args.schema_path),
            "target_skeleton_path": str(args.target_skeleton_path),
            "serialization": {
                "strategy": "target_skeleton_order",
                "top_level_field_order": (
                    list(target_skeleton) if isinstance(target_skeleton, dict) else []
                ),
                "unknown_fields": "alphabetical_after_template_fields",
            },
            "shape_summary": target_shape_summary,
            "schema_special_tokens": len(schema_tokens),
            "skeleton_special_tokens": len(skeleton_tokens),
            "data_special_tokens": len(data_tokens),
            "added_special_tokens": added_special_tokens,
        },
        "model": {
            "base_model_id": args.model_id,
            "task_start_token": args.task_start_token,
            "parameter_counts": count_model_parameters(model),
            "decoder_max_position_embeddings_before": decoder_max_positions_before,
            "decoder_max_position_embeddings_after": decoder_max_positions_after,
            "decoder_position_embeddings_resized": decoder_positions_resized,
        },
        "preprocessing": {
            "image_size": list(args.image_size),
            "align_long_axis": args.align_long_axis,
            "target_token_lengths": target_token_length_summary,
        },
        "training_parameters": namespace_to_dict(args),
        "validation_previews": {
            "enabled": bool(validation_preview_examples),
            "sample_count": len(validation_preview_examples),
            "sample_ids": [example["id"] for example in validation_preview_examples],
            "interval": "training_logging_steps",
            "max_length": validation_preview_max_length,
            "output_directory": str(output_dir / "validation_previews"),
        },
        "checkpoint_policy": {
            "retained": "best_and_last",
            "metric": (
                "eval_json_field_f1" if args.predict_with_generate else "eval_loss"
            ),
            "greater_is_better": bool(args.predict_with_generate),
            "save_total_limit": args.save_total_limit,
            "save_steps": args.save_steps,
            "eval_steps": args.eval_steps,
            "load_best_model_at_end": True,
        },
    }
    run.write_status("running", sections=run_sections)

    train_dataset = Dataset.from_list(build_trainer_dataset_records(train_examples))
    validation_dataset = Dataset.from_list(build_trainer_dataset_records(validation_examples))
    data_collator = DonutBatchCollator(
        processor=processor,
        image_module=image_module,
        max_length=args.max_length,
    )
    callbacks = []
    if validation_preview_examples:
        callbacks.append(
            build_validation_preview_callback(
                TrainerCallback=TrainerCallback,
                torch=torch,
                image_module=image_module,
                processor=processor,
                examples=validation_preview_examples,
                output_dir=output_dir,
                task_start_token=args.task_start_token,
                max_length=validation_preview_max_length,
                target_schema=target_schema,
                root_schema=schema,
            )
        )

    bf16, fp16 = choose_precision_flags(args)
    training_args = build_training_arguments(
        Seq2SeqTrainingArguments=Seq2SeqTrainingArguments,
        args=args,
        output_dir=output_dir,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=gradient_checkpointing,
    )

    trainer = build_trainer(
        Seq2SeqTrainer,
        processor=processor,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=(
            build_donut_compute_metrics(processor, target_schema, schema)
            if args.predict_with_generate
            else None
        ),
    )

    train_output = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer_state = save_trainer_state(trainer, output_dir)
    checkpoint_summary = prune_checkpoints_to_best_and_last(output_dir, state=trainer_state)
    if not checkpoint_summary["best"]["exists"] or not checkpoint_summary["last"]["exists"]:
        checkpoint_summary = summarize_checkpoints(output_dir, state=trainer_state)
    training_summary = summarize_training_history(trainer_state)
    plot_summary = generate_training_plots(output_dir)

    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    run.write_status(
        "completed",
        sections={
            **run_sections,
            "training_summary": training_summary,
            "checkpoints": checkpoint_summary,
            "plots": plot_summary,
        },
        metrics={
            "train": normalize_trainer_metrics(
                getattr(train_output, "metrics", {}),
                stage="train",
            )
        },
    )

    write_json(
        output_dir / "training_config.json",
        {
            "dataset_root": str(args.dataset_root),
            "resolved_dataset_root": str(dataset_root),
            "source_layout": source_layout,
            "train_split": train_split,
            "validation_split": validation_split,
            "model_id": args.model_id,
            "task_start_token": args.task_start_token,
            "schema_path": str(args.schema_path),
            "target_skeleton_path": str(args.target_skeleton_path),
            "annotation_target_key": args.annotation_target_key,
            "image_size": list(args.image_size),
            "align_long_axis": args.align_long_axis,
            "max_length": args.max_length,
            "decoder_max_position_embeddings_before": decoder_max_positions_before,
            "decoder_max_position_embeddings_after": decoder_max_positions_after,
            "decoder_position_embeddings_resized": decoder_positions_resized,
            "target_token_lengths": target_token_length_summary,
            "num_train_epochs": args.num_train_epochs,
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "gradient_checkpointing": gradient_checkpointing,
            "bf16": bf16,
            "fp16": fp16,
            "validation_preview_samples": len(validation_preview_examples),
            "validation_preview_sample_ids": [
                example["id"] for example in validation_preview_examples
            ],
            "validation_preview_max_length": validation_preview_max_length,
            "train_examples": len(train_examples),
            "validation_examples": len(validation_examples),
        },
    )

    print(f"Saved fine-tuned Donut model to {output_dir}")
    inference_example = validation_examples[0]
    inference_command = (
        "python3 src/Donut/run_donut_inference.py "
        f"--model-id {output_dir} "
        f"--task-prompt {args.task_start_token!r} "
        f"--image-path {inference_example['image_path']!r} "
        f"--annotation-target-key {args.annotation_target_key!r}"
    )
    if inference_example.get("annotation_path"):
        inference_command += f" --example-path {inference_example['annotation_path']!r}"
    print(f"Recommended next step: run inference with `{inference_command}`")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
