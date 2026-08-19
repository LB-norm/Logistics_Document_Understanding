"""Reusable Qwen vision-language LoRA/QLoRA fine-tuning logic.

The user-facing defaults live in :mod:`src.Qwen.run_qwen_training`.  Keeping
this module free of project-machine defaults makes the training path reusable
for other Qwen model sizes and compatible multimodal checkpoints.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import re
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "datasets" / "raw_data_20260527"
DEFAULT_SCHEMA_PATH = REPO_ROOT / "json_schema" / "content.schema.json"
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-2B"
DEFAULT_RUNS_DIR = REPO_ROOT / "runs" / "qwen"
DEFAULT_ANNOTATION_TARGET_KEY = "content"
DEFAULT_SYSTEM_PROMPT = (
    "You are an information extraction model for CMR delivery note scans. "
    "Return strict JSON only."
)
DEFAULT_USER_PROMPT = (
    "Extract all relevant document information into the target CMR/Lieferschein "
    "content JSON object. Use null for missing scalar values and [] for missing arrays."
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.Qwen.experiment_config import load_experiment_config
from src.eval_suite import JsonEvaluator
from src.utils.run_utils import RunContext, namespace_to_dict, normalize_trainer_metrics, write_json


def parse_args(
    argv: Sequence[str] | None = None,
    defaults: Mapping[str, Any] | None = None,
) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None)
    config_args, _ = config_parser.parse_known_args(raw_argv)
    experiment_config = (
        load_experiment_config(config_args.config)
        if config_args.config is not None
        else None
    )

    parser = argparse.ArgumentParser(
        description=(
            "LoRA/QLoRA fine-tune a Qwen-compatible vision-language model on "
            "a local document information extraction dataset."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional JSON experiment configuration. Explicit command-line arguments "
            "override values from the file."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=(
            "Dataset root. Supported layouts: the project data/datasets split folders "
            "with metadata.jsonl rows containing image/annotation paths, or a Qwen JSONL "
            "folder containing train.jsonl and validation.jsonl."
        ),
    )
    parser.add_argument(
        "--train-file",
        default="train.jsonl",
        help="Training JSONL file name relative to --dataset-root for Qwen JSONL datasets.",
    )
    parser.add_argument(
        "--validation-file",
        default="validation.jsonl",
        help="Validation JSONL file name relative to --dataset-root for Qwen JSONL datasets.",
    )
    parser.add_argument("--train-split", default="train", help="Training split directory name for project datasets.")
    parser.add_argument(
        "--validation-split",
        default=None,
        help="Validation split directory name for project datasets. If omitted, tries validation, val, then dev.",
    )
    parser.add_argument(
        "--annotation-target-key",
        default=DEFAULT_ANNOTATION_TARGET_KEY,
        help=(
            "Key inside project annotation JSON files to use as the assistant target. "
            "The default 'content' ignores annotation metadata. Use 'root' to train on the full JSON object."
        ),
    )
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
        help="Schema file referenced in dry-run summaries and documentation.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt used when converting project annotations into Qwen chat examples.",
    )
    parser.add_argument(
        "--user-prompt",
        default=DEFAULT_USER_PROMPT,
        help="User prompt used when converting project annotations into Qwen chat examples.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for checkpoints, metadata, and the final adapter. If omitted, "
            "a timestamped directory is created under --runs-dir."
        ),
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Parent directory for timestamped Qwen training runs.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional stable run folder name when --output-dir is omitted.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Base Hugging Face model id or local checkpoint path.",
    )
    parser.add_argument(
        "--model-class",
        default="auto",
        help=(
            "Transformers model loader class. 'auto' prefers AutoModelForMultimodalLM "
            "and falls back to AutoModelForImageTextToText; an explicit class name can "
            "be supplied for a model that needs its architecture-specific loader."
        ),
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=None,
        help="Optional lower bound for the processor image resolution budget.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Optional upper bound for the processor image resolution budget.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help=(
            "Optional sequence truncation length. Leave unset to avoid truncating image tokens. "
            "This is the recommended default for VLM SFT."
        ),
    )
    parser.add_argument("--num-train-epochs", type=float, default=3.0, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1, help="Train batch size per GPU.")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1, help="Eval batch size per GPU.")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
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
    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument("--dataloader-num-workers", type=int, default=2, help="PyTorch dataloader workers.")
    parser.add_argument(
        "--eval-strategy",
        choices=["steps", "epoch"],
        default="steps",
        help="Run teacher-forced validation every configured number of steps or once per epoch.",
    )
    parser.add_argument(
        "--save-strategy",
        choices=["steps", "epoch"],
        default="steps",
        help="Save resumable checkpoints every configured number of steps or once per epoch.",
    )
    parser.add_argument("--eval-steps", type=int, default=50, help="Evaluation interval when using step strategy.")
    parser.add_argument("--save-steps", type=int, default=50, help="Checkpoint interval when using step strategy.")
    parser.add_argument("--logging-steps", type=int, default=10, help="Logging interval.")
    parser.add_argument(
        "--validation-preview-samples",
        type=int,
        default=0,
        help=(
            "Generate JSON/HTML previews for this many fixed validation examples at each "
            "training logging step. Disabled by default in the reusable logic."
        ),
    )
    parser.add_argument(
        "--validation-preview-max-new-tokens",
        type=int,
        default=2048,
        help=(
            "Maximum number of answer tokens generated for each intermediate validation "
            "preview. This is separate from --max-length, which controls training inputs."
        ),
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="Maximum number of checkpoints to keep on disk.",
    )
    parser.add_argument("--max-steps", type=int, default=-1, help="Optional hard cap on optimizer steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--compute-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Training compute dtype and mixed precision mode.",
    )
    quantization_group = parser.add_mutually_exclusive_group()
    quantization_group.add_argument(
        "--load-in-4bit",
        dest="load_in_4bit",
        action="store_true",
        help="Load the base model in 4-bit and run QLoRA training.",
    )
    parser.add_argument(
        "--optim",
        default="auto",
        help=(
            "Trainer optimizer name. 'auto' selects paged_adamw_8bit for QLoRA and "
            "adamw_torch for regular LoRA/full-vision training."
        ),
    )
    quantization_group.add_argument(
        "--no-load-in-4bit",
        dest="load_in_4bit",
        action="store_false",
        help="Disable 4-bit loading and run regular LoRA.",
    )
    parser.set_defaults(load_in_4bit=True)
    parser.add_argument(
        "--bnb-4bit-quant-type",
        default="nf4",
        choices=["fp4", "nf4"],
        help="Quantization type for 4-bit loading.",
    )
    parser.add_argument(
        "--no-bnb-double-quant",
        action="store_true",
        help="Disable nested quantization for 4-bit loading.",
    )
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument(
        "--target-modules",
        default="all-linear",
        help=(
            "LoRA target modules. Use 'all-linear' for the recommended QLoRA setup or "
            "provide a comma-separated list such as q_proj,k_proj,v_proj,o_proj."
        ),
    )
    parser.add_argument(
        "--modules-to-save",
        default="",
        help=(
            "Comma-separated non-LoRA modules to train and save. Empty is recommended "
            "unless the tokenizer vocabulary was extended; embedding/lm_head training is "
            "very memory intensive for Qwen's large vocabulary."
        ),
    )
    parser.add_argument(
        "--vision-tuning",
        choices=["frozen", "lora", "full"],
        default="frozen",
        help=(
            "Vision encoder strategy: 'frozen' trains language-side LoRA only; 'lora' "
            "also trains LoRA adapters matched inside the vision encoder; 'full' unfreezes "
            "the vision encoder and therefore requires --no-load-in-4bit."
        ),
    )
    parser.add_argument(
        "--vision-module-names",
        default="visual,vision_tower,vision_model",
        help=(
            "Comma-separated attribute names used to locate the model's vision encoder. "
            "Override this for a compatible model with a different module name."
        ),
    )
    parser.add_argument(
        "--device-map",
        default="none",
        help="Device map passed to from_pretrained. The default 'none' lets Trainer handle placement.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="Attention implementation passed to from_pretrained when supported.",
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
        help="Allow missing model files to be downloaded from Hugging Face.",
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
        help="Parse and validate the dataset, print a summary, then exit before loading model dependencies.",
    )
    merged_defaults: dict[str, Any] = dict(defaults or {})
    if experiment_config is not None:
        merged_defaults.update(experiment_config.training)
    if merged_defaults:
        known_destinations = {action.dest for action in parser._actions}
        unknown_defaults = sorted(set(merged_defaults) - known_destinations)
        if unknown_defaults:
            raise ValueError("Unknown Qwen training default(s): " + ", ".join(unknown_defaults))
        parser.set_defaults(**merged_defaults)

    args = parser.parse_args(raw_argv)
    args.experiment_name = experiment_config.name if experiment_config else None
    args.experiment_description = experiment_config.description if experiment_config else None
    args.experiment_config_path = (
        str(experiment_config.path) if experiment_config is not None else None
    )
    return args


def load_runtime_dependencies(load_in_4bit: bool) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
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
        from datasets import Dataset
    except ImportError:
        missing.append("datasets")
        Dataset = None

    try:
        import transformers
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
    except ImportError:
        missing.append("transformers")
        transformers = None
        AutoProcessor = None
        BitsAndBytesConfig = None
        Trainer = None
        TrainingArguments = None

    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError:
        missing.append("peft")
        LoraConfig = None
        get_peft_model = None
        prepare_model_for_kbit_training = None

    if load_in_4bit:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            missing.append("bitsandbytes")

    if missing:
        missing_csv = ", ".join(missing)
        raise RuntimeError(
            "Missing runtime dependencies: "
            f"{missing_csv}. Install them before fine-tuning, for example: "
            "`pip install torch torchvision transformers datasets pillow peft bitsandbytes accelerate sentencepiece`."
        )

    return (
        torch,
        Image,
        Dataset,
        AutoProcessor,
        BitsAndBytesConfig,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        (LoraConfig, get_peft_model, prepare_model_for_kbit_training),
        transformers,
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


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_dataset_file(dataset_root: Path, relative_name: str) -> Path:
    dataset_file = dataset_root / relative_name
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    return dataset_file


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


def normalize_text_content(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def normalize_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [normalize_text_content(content)]

    if not isinstance(content, list):
        raise ValueError(
            "Each message 'content' must be either a string or a list of typed blocks "
            f"such as {{'type': 'text', 'text': '...'}}. Received: {content!r}"
        )

    normalized: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            normalized.append(normalize_text_content(item))
            continue

        if not isinstance(item, dict):
            raise ValueError(f"Invalid content block: {item!r}")

        if item.get("type") == "text":
            if "text" not in item:
                raise ValueError(f"Text content blocks require a 'text' field: {item!r}")
            normalized.append({"type": "text", "text": str(item["text"])})
            continue

        if item.get("type") == "image" or "image" in item or "image_url" in item:
            normalized.append({key: value for key, value in item.items()})
            if "type" not in normalized[-1]:
                normalized[-1]["type"] = "image"
            continue

        raise ValueError(f"Unsupported content block: {item!r}")

    return normalized


def resolve_image_paths(raw_record: dict[str, Any], dataset_root: Path) -> list[str]:
    raw_paths = None
    for key in ("images", "image_paths", "image_path", "image"):
        if key in raw_record:
            raw_paths = raw_record[key]
            break

    if raw_paths is None:
        raise ValueError(
            "Each record must include one of 'image', 'image_path', 'images', or 'image_paths'. "
            f"Record keys: {sorted(raw_record.keys())}"
        )

    if isinstance(raw_paths, str):
        paths = [raw_paths]
    elif isinstance(raw_paths, list) and all(isinstance(item, str) for item in raw_paths):
        paths = list(raw_paths)
    else:
        raise ValueError(f"Image path fields must be a string or list of strings. Received: {raw_paths!r}")

    resolved: list[str] = []
    for path_str in paths:
        path = Path(path_str)
        if not path.is_absolute():
            path = dataset_root / path
        if not path.exists():
            raise FileNotFoundError(f"Referenced image does not exist: {path}")
        resolved.append(str(path))
    return resolved


def count_image_placeholders(messages: list[dict[str, Any]]) -> int:
    count = 0
    for message in messages:
        for item in message["content"]:
            if item.get("type") == "image" or "image" in item or "image_url" in item:
                count += 1
    return count


def normalize_messages(messages: Any, image_count: int) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("Each dataset record must contain a non-empty 'messages' list.")

    normalized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError(f"Invalid message entry: {message!r}")
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported message role: {role!r}")
        normalized.append({"role": role, "content": normalize_content(message.get("content"))})

    if normalized[-1]["role"] != "assistant":
        raise ValueError("Each training example must end with a final assistant message containing the target JSON.")

    placeholder_count = count_image_placeholders(normalized)
    if placeholder_count == 0 and image_count > 0:
        last_user_index = None
        for index in range(len(normalized) - 1, -1, -1):
            if normalized[index]["role"] == "user":
                last_user_index = index
                break
        if last_user_index is None:
            raise ValueError("Records with images must contain at least one user message.")
        normalized[last_user_index]["content"] = (
            [{"type": "image"} for _ in range(image_count)] + normalized[last_user_index]["content"]
        )
        placeholder_count = image_count

    if placeholder_count != image_count:
        raise ValueError(
            "The number of image placeholders inside messages must match the number of image paths. "
            f"Found {placeholder_count} placeholders for {image_count} image files."
        )

    return normalized


def build_example_record(raw_record: dict[str, Any], dataset_root: Path) -> dict[str, Any]:
    image_paths = resolve_image_paths(raw_record, dataset_root)
    messages = normalize_messages(raw_record.get("messages"), image_count=len(image_paths))
    return {
        "id": raw_record.get("id"),
        "messages": messages,
        "image_paths": image_paths,
    }


def load_split(dataset_file: Path, dataset_root: Path) -> list[dict[str, Any]]:
    return [build_example_record(record, dataset_root) for record in read_jsonl(dataset_file)]


def build_project_messages(gt_parse: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": args.system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": args.user_prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(gt_parse, ensure_ascii=False, sort_keys=True),
                }
            ],
        },
    ]


def build_project_example_record(
    raw_record: dict[str, Any],
    dataset_root: Path,
    split_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if "image" not in raw_record:
        raise ValueError(f"Missing 'image' in project dataset metadata row: {raw_record}")
    if "annotation" not in raw_record:
        raise ValueError(f"Missing 'annotation' in project dataset metadata row: {raw_record}")

    image_path = resolve_existing_path(raw_record["image"], dataset_root, split_dir)
    annotation_path = resolve_existing_path(raw_record["annotation"], dataset_root, split_dir)
    annotation = load_json(annotation_path)
    gt_parse = extract_annotation_target(annotation, args.annotation_target_key)
    messages = normalize_messages(build_project_messages(gt_parse, args), image_count=1)

    return {
        "id": raw_record.get("id", image_path.stem),
        "messages": messages,
        "image_paths": [str(image_path)],
        "annotation_path": str(annotation_path),
        "target_keys": sorted(gt_parse.keys()) if isinstance(gt_parse, dict) else [],
    }


def load_project_split(
    dataset_root: Path,
    split_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    metadata_path = split_dir / "metadata.jsonl"
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.jsonl in split directory: {split_dir}")

    return [
        build_project_example_record(record, dataset_root, split_dir, args)
        for record in read_jsonl(metadata_path)
    ]


def choose_validation_split(dataset_root: Path, requested_split: str | None) -> str | None:
    if requested_split is not None:
        return requested_split

    for split_name in ("validation", "val", "dev"):
        if (dataset_root / split_name / "metadata.jsonl").exists():
            return split_name

    return None


def cap_examples(examples: list[dict[str, Any]], sample_limit: int | None) -> list[dict[str, Any]]:
    if sample_limit is None:
        return examples
    if sample_limit < 1:
        raise ValueError("Sample limits must be positive integers.")
    return examples[:sample_limit]


def load_dataset_splits(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, str, str, str]:
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    train_split_dir = dataset_root / args.train_split
    if (train_split_dir / "metadata.jsonl").exists():
        validation_split = choose_validation_split(dataset_root, args.validation_split)
        if validation_split is None:
            raise FileNotFoundError(
                "Could not find a validation split. Expected one of validation/, val/, or dev/ "
                "with metadata.jsonl, or pass --validation-split."
            )
        validation_split_dir = dataset_root / validation_split
        train_examples = load_project_split(dataset_root, train_split_dir, args)
        validation_examples = load_project_split(dataset_root, validation_split_dir, args)
        source_layout = "project_metadata_splits"
        train_source = args.train_split
        validation_source = validation_split
    else:
        train_file = resolve_dataset_file(dataset_root, args.train_file)
        validation_file = resolve_dataset_file(dataset_root, args.validation_file)
        train_examples = load_split(train_file, dataset_root)
        validation_examples = load_split(validation_file, dataset_root)
        source_layout = "qwen_jsonl"
        train_source = str(train_file)
        validation_source = str(validation_file)

    train_examples = cap_examples(train_examples, args.max_train_samples)
    validation_examples = cap_examples(validation_examples, args.max_validation_samples)

    if not train_examples:
        raise ValueError("Training split did not contain any examples.")
    if not validation_examples:
        raise ValueError("Validation split did not contain any examples.")

    return train_examples, validation_examples, str(dataset_root), train_source, validation_source, source_layout


def describe_target_keys(example: dict[str, Any]) -> list[str]:
    if "target_keys" in example:
        return example["target_keys"]

    assistant_message = example["messages"][-1]
    text_blocks = [
        item.get("text", "")
        for item in assistant_message["content"]
        if item.get("type") == "text"
    ]
    if not text_blocks:
        return []
    try:
        target = json.loads("".join(text_blocks))
    except json.JSONDecodeError:
        return []
    return sorted(target.keys()) if isinstance(target, dict) else []


def print_dry_run_summary(
    train_examples: list[dict[str, Any]],
    validation_examples: list[dict[str, Any]],
    resolved_dataset_root: str,
    train_source: str,
    validation_source: str,
    source_layout: str,
    args: argparse.Namespace,
) -> None:
    print("Qwen training dry run")
    print(f"  dataset_root: {resolved_dataset_root}")
    print(f"  source_layout: {source_layout}")
    print(f"  train_source: {train_source} ({len(train_examples)} examples)")
    print(f"  validation_source: {validation_source} ({len(validation_examples)} examples)")
    print(f"  annotation_target_key: {args.annotation_target_key}")
    print(f"  schema_path: {args.schema_path}")
    print(f"  train_images_first: {len(train_examples[0]['image_paths'])}")
    print(f"  validation_images_first: {len(validation_examples[0]['image_paths'])}")
    print(f"  train_target_keys: {describe_target_keys(train_examples[0])}")
    print(f"  validation_target_keys: {describe_target_keys(validation_examples[0])}")


def apply_chat_template_safely(processor: Any, messages: list[dict[str, Any]], add_generation_prompt: bool) -> str:
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


@dataclass
class QwenVisionDataCollator:
    processor: Any
    image_module: Any
    max_length: int | None
    image_token_ids: set[int]
    ignore_index: int = -100

    def _load_images(self, image_paths: list[str]) -> list[Any]:
        images: list[Any] = []
        for image_path in image_paths:
            with self.image_module.open(image_path) as image:
                images.append(image.convert("RGB"))
        return images

    def _processor_images(self, images_per_example: list[list[Any]]) -> list[Any]:
        if all(len(images) == 1 for images in images_per_example):
            return [images[0] for images in images_per_example]
        return images_per_example

    def _processor_kwargs(self, texts: list[str], images_per_example: list[list[Any]]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "text": texts,
            "images": self._processor_images(images_per_example),
            "padding": True,
            "return_tensors": "pt",
        }
        if self.max_length is not None:
            kwargs["max_length"] = self.max_length
            kwargs["truncation"] = True
        return kwargs

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        images_per_example = [self._load_images(feature["image_paths"]) for feature in features]
        full_texts = [
            apply_chat_template_safely(self.processor, feature["messages"], add_generation_prompt=False)
            for feature in features
        ]
        prompt_texts = [
            apply_chat_template_safely(self.processor, feature["messages"][:-1], add_generation_prompt=True)
            for feature in features
        ]

        batch = self.processor(**self._processor_kwargs(full_texts, images_per_example))
        prompt_batch = self.processor(**self._processor_kwargs(prompt_texts, images_per_example))

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = self.ignore_index

        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
        for row_index, prompt_length in enumerate(prompt_lengths):
            labels[row_index, :prompt_length] = self.ignore_index

        for token_id in self.image_token_ids:
            labels[labels == token_id] = self.ignore_index

        if not all((row != self.ignore_index).any().item() for row in labels):
            raise ValueError(
                "At least one batch element has no remaining assistant target tokens after masking. "
                "This usually means truncation removed the answer. Increase --max-length or leave it unset."
            )

        batch["labels"] = labels
        return batch


def select_validation_preview_examples(
    examples: list[dict[str, Any]], sample_count: int, seed: int
) -> list[dict[str, Any]]:
    """Choose one reproducible validation subset and keep it fixed for the run."""
    if sample_count <= 0:
        return []
    if sample_count >= len(examples):
        return list(examples)

    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    return [examples[index] for index in sorted(indices[:sample_count])]


def extract_assistant_target(example: dict[str, Any]) -> Any:
    """Parse the final assistant message as the structured preview ground truth."""
    assistant_message = example["messages"][-1]
    target_text = "".join(
        str(item.get("text", ""))
        for item in assistant_message["content"]
        if item.get("type") == "text"
    ).strip()
    if not target_text:
        raise ValueError(f"Validation example {example.get('id')!r} has no assistant target text.")
    try:
        return json.loads(target_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Validation example {example.get('id')!r} does not contain a valid JSON assistant target: {exc}"
        ) from exc


def json_differences(expected: Any, predicted: Any, path: str = "$") -> list[dict[str, Any]]:
    """Return compact, path-oriented differences for a validation preview."""
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


def _move_batch_to_device(batch: Any, device: Any) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _model_input_device(model: Any) -> Any:
    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if embeddings is not None and hasattr(embeddings, "weight"):
            return embeddings.weight.device
    return next(model.parameters()).device


def generate_validation_preview_sample(
    *,
    image_module: Any,
    processor: Any,
    model: Any,
    example: dict[str, Any],
    max_new_tokens: int,
    target_schema: Any,
) -> dict[str, Any]:
    """Generate one answer without teacher forcing and evaluate its JSON output."""
    images: list[Any] = []
    for image_path in example["image_paths"]:
        with image_module.open(image_path) as image:
            images.append(image.convert("RGB"))

    prompt = apply_chat_template_safely(
        processor,
        example["messages"][:-1],
        add_generation_prompt=True,
    )
    processor_images: list[Any] = [images[0]] if len(images) == 1 else [images]
    inputs = processor(
        text=[prompt],
        images=processor_images,
        padding=True,
        return_tensors="pt",
    )
    inputs = _move_batch_to_device(inputs, _model_input_device(model))
    prompt_tokens = int(inputs["input_ids"].shape[-1])

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
    }
    if processor.tokenizer.pad_token_id is not None:
        generation_kwargs["pad_token_id"] = processor.tokenizer.pad_token_id
    if processor.tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = processor.tokenizer.eos_token_id

    outputs = model.generate(**inputs, **generation_kwargs)
    sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
    generated_ids = sequences[:, prompt_tokens:]
    decoder = getattr(processor, "batch_decode", processor.tokenizer.batch_decode)
    raw_sequence = decoder(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    ground_truth = extract_assistant_target(example)
    evaluation = JsonEvaluator(schema=target_schema).evaluate(
        raw_sequence,
        ground_truth,
        sample_id=str(example.get("id")),
    )
    try:
        prediction: Any = json.loads(raw_sequence)
    except (json.JSONDecodeError, TypeError):
        prediction = raw_sequence
    differences = (
        json_differences(ground_truth, prediction)
        if evaluation.parse_valid
        else [
            {
                "kind": "parse_error",
                "path": "$",
                "ground_truth": ground_truth,
                "prediction": raw_sequence,
            }
        ]
    )
    return {
        "id": example.get("id"),
        "image_paths": example["image_paths"],
        "annotation_path": example.get("annotation_path"),
        "ground_truth": ground_truth,
        "prediction": prediction,
        "exact_match": evaluation.document_exact_match,
        "differences": differences,
        "parse_error": evaluation.parse_error,
        "schema_errors": evaluation.schema_errors,
        "metrics": evaluation.field_counts.metrics(),
        "raw_sequence": raw_sequence,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": int(generated_ids.shape[-1]),
    }


def render_validation_preview_html(payload: dict[str, Any]) -> str:
    def pretty(value: Any) -> str:
        return html.escape(json.dumps(value, ensure_ascii=False, indent=2))

    sample_sections: list[str] = []
    for sample in payload.get("samples", []):
        image_links = " &middot; ".join(
            f'<a href="{html.escape(Path(path).resolve().as_uri(), quote=True)}">Open image {index}</a>'
            for index, path in enumerate(sample["image_paths"], start=1)
        )
        exact_match = "yes" if sample["exact_match"] else "no"
        parse_error = sample.get("parse_error") or "none"
        schema_errors = sample.get("schema_errors") or []
        sample_sections.append(
            f"""
            <section class="sample">
              <h2>{html.escape(str(sample['id']))}</h2>
              <p>{image_links} &middot; exact match: <strong>{exact_match}</strong>
                 &middot; differences: {len(sample['differences'])}
                 &middot; parse error: {html.escape(parse_error)}
                 &middot; schema errors: {len(schema_errors)}</p>
              <p>Prompt tokens: {sample['prompt_tokens']} &middot; generated tokens: {sample['generated_tokens']}</p>
              <div class="comparison">
                <div><h3>Ground truth</h3><pre>{pretty(sample['ground_truth'])}</pre></div>
                <div><h3>Prediction</h3><pre>{pretty(sample['prediction'])}</pre></div>
              </div>
              <details><summary>Structured differences</summary><pre>{pretty(sample['differences'])}</pre></details>
              <details><summary>Schema errors</summary><pre>{pretty(schema_errors)}</pre></details>
              <details><summary>Raw generated sequence</summary><pre>{html.escape(sample['raw_sequence'])}</pre></details>
            </section>
            """
        )

    error_section = ""
    if payload.get("error"):
        error_section = f'<p class="error">{html.escape(str(payload["error"]))}</p>'
    summary = payload.get("evaluation", {}).get("summary", {})

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Qwen validation preview - step {payload['global_step']}</title>
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
  <h1>Qwen validation preview</h1>
  <p>Step {payload['global_step']} &middot; epoch {payload.get('epoch')} &middot; status {html.escape(payload['status'])}</p>
  <p>JSON parse rate: {summary.get('parse_rate', 'n/a')} &middot;
     schema-valid rate: {summary.get('schema_valid_rate', 'n/a')} &middot;
     field F1: {summary.get('field_f1', 'n/a')}</p>
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
    max_new_tokens: int,
    target_schema: Any,
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
                "max_new_tokens": max_new_tokens,
                "samples": [],
            }
            try:
                model.eval()
                with torch.inference_mode():
                    payload["samples"] = [
                        generate_validation_preview_sample(
                            image_module=image_module,
                            processor=processor,
                            model=model,
                            example=example,
                            max_new_tokens=max_new_tokens,
                            target_schema=target_schema,
                        )
                        for example in examples
                    ]
                report = JsonEvaluator(schema=target_schema).evaluate_batch(
                    [sample["raw_sequence"] for sample in payload["samples"]],
                    [sample["ground_truth"] for sample in payload["samples"]],
                    sample_ids=[str(sample["id"]) for sample in payload["samples"]],
                )
                payload["evaluation"] = report.to_dict(include_samples=False)
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


def parse_target_modules(raw_value: str) -> str | list[str]:
    if raw_value.strip() == "all-linear":
        return "all-linear"
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def resolve_lora_target_modules(
    model: Any,
    raw_value: str,
    vision_module_paths: list[str],
    vision_tuning: str,
    torch: Any,
) -> list[str]:
    """Resolve target names before PEFT so vision inclusion is intentional.

    PEFT's ``all-linear`` shortcut also covers a VLM's visual tower. Resolving
    actual module paths here lets frozen/full modes exclude the tower entirely,
    while the ``lora`` mode deliberately includes it.
    """
    requested = parse_target_modules(raw_value)
    requested_suffixes = None if requested == "all-linear" else requested
    output_embedding = (
        model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    )

    def is_inside_vision(module_name: str) -> bool:
        return any(
            module_name == root or module_name.startswith(root + ".")
            for root in vision_module_paths
        )

    def is_linear_like(module: Any) -> bool:
        return (
            isinstance(module, torch.nn.Linear)
            or module.__class__.__name__ == "Conv1D"
            or (
                hasattr(module, "in_features")
                and hasattr(module, "out_features")
                and hasattr(module, "weight")
            )
        )

    resolved: list[str] = []
    for module_name, module in model.named_modules():
        if not module_name or module is output_embedding or module_name.endswith("lm_head"):
            continue
        if vision_tuning != "lora" and is_inside_vision(module_name):
            continue
        if requested_suffixes is None:
            matches = is_linear_like(module)
        else:
            matches = any(
                module_name == suffix or module_name.endswith("." + suffix)
                for suffix in requested_suffixes
            )
        if matches:
            resolved.append(module_name)

    if not resolved:
        scope = "language and vision" if vision_tuning == "lora" else "language"
        raise RuntimeError(
            f"No {scope} modules matched --target-modules {raw_value!r}. "
            "Inspect model.named_modules() and provide compatible module suffixes."
        )
    return resolved


def parse_modules_to_save(raw_value: str) -> list[str] | None:
    modules = [item.strip() for item in raw_value.split(",") if item.strip()]
    return modules or None


def parse_vision_module_names(raw_value: str) -> list[str]:
    names = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not names:
        raise ValueError("--vision-module-names must contain at least one module attribute name.")
    return names


def validate_training_options(args: argparse.Namespace) -> None:
    if args.lora_r < 1:
        raise ValueError("--lora-r must be at least 1.")
    if args.lora_alpha < 1:
        raise ValueError("--lora-alpha must be at least 1.")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError("--lora-dropout must be in the range [0, 1).")
    if args.min_pixels is not None and args.min_pixels < 1:
        raise ValueError("--min-pixels must be positive.")
    if args.max_pixels is not None and args.max_pixels < 1:
        raise ValueError("--max-pixels must be positive.")
    if (
        args.min_pixels is not None
        and args.max_pixels is not None
        and args.min_pixels > args.max_pixels
    ):
        raise ValueError("--min-pixels cannot exceed --max-pixels.")
    if args.max_length is not None and args.max_length < 1:
        raise ValueError("--max-length must be positive when supplied.")
    if args.validation_preview_samples < 0:
        raise ValueError("--validation-preview-samples must be greater than or equal to 0.")
    if args.validation_preview_max_new_tokens < 1:
        raise ValueError("--validation-preview-max-new-tokens must be positive.")
    if args.validation_preview_samples and args.logging_steps < 1:
        raise ValueError("--logging-steps must be positive when validation previews are enabled.")
    if args.eval_strategy != args.save_strategy:
        raise ValueError(
            "--eval-strategy and --save-strategy must match so every candidate for "
            "best-model selection has a saved checkpoint."
        )
    if args.eval_strategy == "steps":
        if args.eval_steps < 1 or args.save_steps < 1:
            raise ValueError("--eval-steps and --save-steps must be positive.")
        if args.eval_steps != args.save_steps:
            raise ValueError(
                "--eval-steps and --save-steps must match so the best evaluated model "
                "is always checkpointed."
            )
    if args.save_total_limit < 2:
        raise ValueError(
            "--save-total-limit must be at least 2 so the best and last resumable "
            "checkpoints are both retained."
        )
    if args.vision_tuning == "full" and args.load_in_4bit:
        raise ValueError(
            "Full vision-encoder tuning is incompatible with 4-bit QLoRA weights. "
            "Use --vision-tuning lora, or combine --vision-tuning full with --no-load-in-4bit."
        )
    parse_vision_module_names(args.vision_module_names)


def select_model_loader(transformers_module: Any, requested_class: str) -> Any:
    if requested_class != "auto":
        loader = getattr(transformers_module, requested_class, None)
        if loader is None:
            raise RuntimeError(
                f"Transformers does not provide model class {requested_class!r}. "
                "Upgrade transformers or choose a valid --model-class."
            )
        return loader

    candidates = (
        "AutoModelForMultimodalLM",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "Qwen3_5ForConditionalGeneration",
    )
    for class_name in candidates:
        loader = getattr(transformers_module, class_name, None)
        if loader is not None:
            return loader
    raise RuntimeError(
        "No compatible multimodal auto-model loader was found in transformers. "
        "Install the transformers version required by the selected Qwen model."
    )


def find_vision_modules(model: Any, configured_names: list[str]) -> list[tuple[str, Any]]:
    name_set = set(configured_names)
    matches = [
        (name, module)
        for name, module in model.named_modules()
        if name and name.rsplit(".", 1)[-1] in name_set
    ]

    roots: list[tuple[str, Any]] = []
    for name, module in sorted(matches, key=lambda item: item[0].count(".")):
        if any(name == root_name or name.startswith(root_name + ".") for root_name, _ in roots):
            continue
        roots.append((name, module))
    return roots


def configure_vision_tuning(
    vision_modules: list[tuple[str, Any]],
    mode: str,
) -> dict[str, int | str | list[str]]:
    if not vision_modules:
        raise RuntimeError(
            "Could not locate the vision encoder. Set --vision-module-names to the relevant "
            "attribute name for this model before training, so its parameters are not tuned accidentally."
        )

    trainable_parameter_ids: set[int] = set()
    total_parameter_ids: set[int] = set()
    trainable_elements = 0
    total_elements = 0

    for _, module in vision_modules:
        parameter_names = [name for name, _ in module.named_parameters()]
        has_saved_module_copy = mode == "full" and any(
            "modules_to_save." in name for name in parameter_names
        )
        for parameter_name, parameter in module.named_parameters():
            parameter_id = id(parameter)
            if parameter_id not in total_parameter_ids:
                total_parameter_ids.add(parameter_id)
                total_elements += parameter.numel()

            if mode == "frozen":
                parameter.requires_grad = False
            elif mode == "lora":
                parameter.requires_grad = "lora_" in parameter_name
            elif mode == "full":
                # PEFT keeps an inactive original module beside the trainable,
                # checkpointed copy created by modules_to_save. Do not optimize
                # both copies when the full vision tower is being tuned.
                parameter.requires_grad = (
                    "modules_to_save." in parameter_name
                    if has_saved_module_copy
                    else True
                )
            else:
                raise ValueError(f"Unsupported vision tuning mode: {mode}")

            if parameter.requires_grad and parameter_id not in trainable_parameter_ids:
                trainable_parameter_ids.add(parameter_id)
                trainable_elements += parameter.numel()

    if mode == "lora" and trainable_elements == 0:
        raise RuntimeError(
            "--vision-tuning lora was requested, but --target-modules did not create any "
            "LoRA parameters inside the detected vision encoder. Use --target-modules all-linear "
            "or provide module names present in the vision tower."
        )

    return {
        "mode": mode,
        "module_paths": [name for name, _ in vision_modules],
        "trainable_parameters": trainable_elements,
        "total_parameters": total_elements,
    }


def count_trainable_parameters(model: Any) -> tuple[int, int]:
    trainable = 0
    total = 0
    seen: set[int] = set()
    for parameter in model.parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()
    return trainable, total


def resolve_dtype(args: argparse.Namespace, torch: Any) -> Any:
    return getattr(torch, args.compute_dtype)


def choose_precision_flags(args: argparse.Namespace) -> tuple[bool, bool]:
    return args.compute_dtype == "bfloat16", args.compute_dtype == "float16"


def resolve_load_in_4bit(args: argparse.Namespace) -> bool:
    return args.load_in_4bit


def resolve_gradient_checkpointing(args: argparse.Namespace) -> bool:
    return args.gradient_checkpointing


def resolve_optimizer(args: argparse.Namespace, load_in_4bit: bool) -> str:
    if args.optim != "auto":
        return args.optim
    return "paged_adamw_8bit" if load_in_4bit else "adamw_torch"


def build_model_load_kwargs(
    args: argparse.Namespace,
    torch: Any,
    BitsAndBytesConfig: Any,
    load_in_4bit: bool,
) -> dict[str, Any]:
    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    kwargs: dict[str, Any] = {
        "local_files_only": args.local_files_only,
        "attn_implementation": args.attn_implementation,
    }
    if args.cache_dir is not None:
        kwargs["cache_dir"] = str(args.cache_dir)

    if args.device_map != "none":
        kwargs["device_map"] = args.device_map

    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=not args.no_bnb_double_quant,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=resolve_dtype(args, torch),
        )
    else:
        kwargs["torch_dtype"] = resolve_dtype(args, torch)

    return kwargs


def build_processor_load_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"local_files_only": args.local_files_only}
    if args.cache_dir is not None:
        kwargs["cache_dir"] = str(args.cache_dir)
    if args.min_pixels is not None:
        kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None:
        kwargs["max_pixels"] = args.max_pixels
    return kwargs


def build_training_arguments(
    TrainingArguments: Any,
    args: argparse.Namespace,
    *,
    output_dir: Path,
    gradient_checkpointing: bool,
    bf16: bool,
    fp16: bool,
    load_in_4bit: bool,
) -> Any:
    """Build Trainer settings with guaranteed best/last checkpoint retention."""
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        # Transformers 5.4 removed the deprecated ``warmup_ratio`` argument.
        # ``warmup_steps`` accepts a float in [0, 1) with the same ratio
        # semantics, so keep our stable experiment-facing option and translate
        # it at the TrainingArguments boundary.
        warmup_steps=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        max_steps=args.max_steps,
        gradient_checkpointing=gradient_checkpointing,
        bf16=bf16,
        fp16=fp16,
        report_to="none",
        do_train=True,
        do_eval=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim=resolve_optimizer(args, load_in_4bit),
        seed=args.seed,
    )


def find_last_checkpoint(output_dir: Path) -> Path:
    """Return the highest-step resumable Trainer checkpoint."""
    candidates: list[tuple[int, Path]] = []
    if output_dir.is_dir():
        for path in output_dir.iterdir():
            match = re.fullmatch(r"checkpoint-(\d+)", path.name)
            if path.is_dir() and match:
                candidates.append((int(match.group(1)), path))
    if not candidates:
        raise RuntimeError(f"Training completed without a checkpoint in {output_dir}.")
    return max(candidates, key=lambda item: item[0])[1]


def copy_checkpoint_model_artifacts(source_dir: Path, destination_dir: Path) -> list[str]:
    """Copy model/adapter files from a resumable checkpoint, excluding optimizer state."""
    exact_names = {
        "README.md",
        "adapter_config.json",
        "config.json",
        "generation_config.json",
    }
    weight_prefixes = (
        "adapter_model.",
        "model.safetensors",
        "pytorch_model",
    )
    selected = [
        path
        for path in source_dir.iterdir()
        if path.is_file()
        and (path.name in exact_names or path.name.startswith(weight_prefixes))
    ]
    weight_files = [path for path in selected if path.name.startswith(weight_prefixes)]
    if not weight_files:
        raise RuntimeError(f"Checkpoint does not contain model weights: {source_dir}")

    destination_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for source_path in selected:
        destination_path = destination_dir / source_path.name
        shutil.copy2(source_path, destination_path)
        copied.append(source_path.name)
    return sorted(copied)


def save_best_and_last_model_artifacts(
    *,
    trainer: Any,
    processor: Any,
    output_dir: Path,
) -> dict[str, Any]:
    """Create stable model-only directories for the best and final checkpoints."""
    best_checkpoint_value = trainer.state.best_model_checkpoint
    if not best_checkpoint_value:
        raise RuntimeError(
            "Trainer did not identify a best checkpoint. Check evaluation and save settings."
        )
    best_checkpoint = Path(best_checkpoint_value).resolve()
    if not best_checkpoint.is_dir():
        raise RuntimeError(f"Best checkpoint was not retained: {best_checkpoint}")
    last_checkpoint = find_last_checkpoint(output_dir).resolve()

    best_model_dir = output_dir / "best_model"
    last_model_dir = output_dir / "last_model"
    trainer.save_model(str(best_model_dir))
    processor.save_pretrained(str(best_model_dir))

    if last_checkpoint == best_checkpoint:
        trainer.save_model(str(last_model_dir))
    else:
        copy_checkpoint_model_artifacts(last_checkpoint, last_model_dir)
    processor.save_pretrained(str(last_model_dir))

    return {
        "selection_metric": "eval_loss",
        "greater_is_better": False,
        "best_metric": trainer.state.best_metric,
        "best_checkpoint": str(best_checkpoint),
        "last_checkpoint": str(last_checkpoint),
        "best_model_dir": str(best_model_dir.resolve()),
        "last_model_dir": str(last_model_dir.resolve()),
    }


def main(
    argv: Sequence[str] | None = None,
    defaults: Mapping[str, Any] | None = None,
) -> int:
    args = parse_args(argv=argv, defaults=defaults)
    validate_training_options(args)
    set_seed(args.seed)

    train_examples, validation_examples, resolved_dataset_root, train_source, validation_source, source_layout = (
        load_dataset_splits(args)
    )
    validation_preview_examples = select_validation_preview_examples(
        validation_examples,
        sample_count=args.validation_preview_samples,
        seed=args.seed,
    )
    for preview_example in validation_preview_examples:
        extract_assistant_target(preview_example)

    if args.dry_run:
        print_dry_run_summary(
            train_examples=train_examples,
            validation_examples=validation_examples,
            resolved_dataset_root=resolved_dataset_root,
            train_source=train_source,
            validation_source=validation_source,
            source_layout=source_layout,
            args=args,
        )
        print(f"  model_id: {args.model_id}")
        print(f"  load_in_4bit: {args.load_in_4bit}")
        print(f"  vision_tuning: {args.vision_tuning}")
        print(f"  target_modules: {parse_target_modules(args.target_modules)}")
        print(f"  validation_preview_samples: {len(validation_preview_examples)}")
        if validation_preview_examples:
            print(
                "  validation_preview_sample_ids: "
                + ", ".join(str(example.get("id")) for example in validation_preview_examples)
            )
            print(
                "  validation_preview_max_new_tokens: "
                f"{args.validation_preview_max_new_tokens}"
            )
        return 0

    load_in_4bit = resolve_load_in_4bit(args)
    target_schema = load_json(args.schema_path.resolve()) if validation_preview_examples else None

    run_context = RunContext.create(
        pipeline_name="qwen",
        runs_dir=args.runs_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        dataset_name=args.dataset_root.resolve().name,
        model_id=args.model_id,
    )
    args.output_dir = run_context.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_context.write_status(
        "running",
        sections={
            "configuration": namespace_to_dict(args),
            "dataset": {
                "resolved_dataset_root": resolved_dataset_root,
                "source_layout": source_layout,
                "train_source": train_source,
                "validation_source": validation_source,
                "train_examples": len(train_examples),
                "validation_examples": len(validation_examples),
            },
            "validation_previews": {
                "enabled": bool(validation_preview_examples),
                "sample_count": len(validation_preview_examples),
                "sample_ids": [example.get("id") for example in validation_preview_examples],
                "interval": "training_logging_steps",
                "max_new_tokens": args.validation_preview_max_new_tokens,
                "output_directory": str(args.output_dir / "validation_previews"),
            },
            "checkpoint_policy": {
                "retained": "best_and_last",
                "selection_metric": "eval_loss",
                "greater_is_better": False,
                "eval_strategy": args.eval_strategy,
                "save_strategy": args.save_strategy,
                "save_total_limit": args.save_total_limit,
                "load_best_model_at_end": True,
            },
        },
    )

    try:
        (
            torch,
            image_module,
            Dataset,
            AutoProcessor,
            BitsAndBytesConfig,
            Trainer,
            TrainerCallback,
            TrainingArguments,
            peft_fns,
            transformers_module,
        ) = load_runtime_dependencies(load_in_4bit=load_in_4bit)
        LoraConfig, get_peft_model, prepare_model_for_kbit_training = peft_fns

        gradient_checkpointing = resolve_gradient_checkpointing(args)
        bf16, fp16 = choose_precision_flags(args)

        processor = AutoProcessor.from_pretrained(args.model_id, **build_processor_load_kwargs(args))
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.padding_side = "right"

        model_loader = select_model_loader(transformers_module, args.model_class)
        model = model_loader.from_pretrained(
            args.model_id,
            **build_model_load_kwargs(args, torch, BitsAndBytesConfig, load_in_4bit),
        )
        vision_modules = find_vision_modules(
            model,
            parse_vision_module_names(args.vision_module_names),
        )
        if not vision_modules:
            raise RuntimeError(
                "Could not locate the vision encoder. Set --vision-module-names to the relevant "
                "attribute name for this model before training."
            )

        model.config.use_cache = False
        if load_in_4bit:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=gradient_checkpointing,
            )
        elif gradient_checkpointing:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

        requested_modules_to_save = parse_modules_to_save(args.modules_to_save) or []
        effective_modules_to_save = list(requested_modules_to_save)
        if args.vision_tuning == "full":
            for vision_path, _ in vision_modules:
                vision_attribute = vision_path.rsplit(".", 1)[-1]
                if vision_attribute not in effective_modules_to_save:
                    effective_modules_to_save.append(vision_attribute)

        effective_target_modules = resolve_lora_target_modules(
            model=model,
            raw_value=args.target_modules,
            vision_module_paths=[path for path, _ in vision_modules],
            vision_tuning=args.vision_tuning,
            torch=torch,
        )

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=effective_target_modules,
            task_type="CAUSAL_LM",
            modules_to_save=effective_modules_to_save or None,
        )
        model = get_peft_model(model, peft_config)
        vision_modules = find_vision_modules(
            model,
            parse_vision_module_names(args.vision_module_names),
        )
        vision_summary = configure_vision_tuning(vision_modules, args.vision_tuning)
        trainable_parameters, total_parameters = count_trainable_parameters(model)
        if trainable_parameters == 0:
            raise RuntimeError("The selected LoRA and vision settings left no trainable parameters.")
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
        print(
            f"Vision tuning: {args.vision_tuning}; "
            f"{vision_summary['trainable_parameters']:,} / "
            f"{vision_summary['total_parameters']:,} vision parameters trainable."
        )

        image_token_ids = {
            getattr(model.config, "image_token_id", None),
            getattr(model.config, "video_token_id", None),
        }
        image_token_ids.discard(None)

        train_dataset = Dataset.from_list(train_examples)
        validation_dataset = Dataset.from_list(validation_examples)
        data_collator = QwenVisionDataCollator(
            processor=processor,
            image_module=image_module,
            max_length=args.max_length,
            image_token_ids=image_token_ids,
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
                    output_dir=args.output_dir,
                    max_new_tokens=args.validation_preview_max_new_tokens,
                    target_schema=target_schema,
                )
            )

        training_args = build_training_arguments(
            TrainingArguments,
            args,
            output_dir=args.output_dir,
            gradient_checkpointing=gradient_checkpointing,
            bf16=bf16,
            fp16=fp16,
            load_in_4bit=load_in_4bit,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )

        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        eval_metrics = trainer.evaluate()

        # With load_best_model_at_end enabled, the in-memory model is now the
        # best eval-loss checkpoint. Keep it at the run root for backwards
        # compatibility and also materialize explicit best/last directories.
        trainer.save_model(str(args.output_dir))
        trainer.save_state()
        processor.save_pretrained(str(args.output_dir))
        checkpoint_artifacts = save_best_and_last_model_artifacts(
            trainer=trainer,
            processor=processor,
            output_dir=args.output_dir,
        )

        resolved_config = namespace_to_dict(args)
        resolved_config.update(
            {
                "resolved_dataset_root": resolved_dataset_root,
                "source_layout": source_layout,
                "train_source": train_source,
                "validation_source": validation_source,
                "gradient_checkpointing": gradient_checkpointing,
                "load_in_4bit": load_in_4bit,
                "optimizer": resolve_optimizer(args, load_in_4bit),
                "requested_target_modules": parse_target_modules(args.target_modules),
                "target_modules": effective_target_modules,
                "requested_modules_to_save": requested_modules_to_save or None,
                "modules_to_save": effective_modules_to_save or None,
                "vision": vision_summary,
                "trainable_parameters": trainable_parameters,
                "total_parameters": total_parameters,
                "train_examples": len(train_examples),
                "validation_examples": len(validation_examples),
                "validation_preview_samples": len(validation_preview_examples),
                "validation_preview_sample_ids": [
                    example.get("id") for example in validation_preview_examples
                ],
                "checkpoint_policy": {
                    "retained": "best_and_last",
                    "eval_strategy": args.eval_strategy,
                    "save_strategy": args.save_strategy,
                    "save_total_limit": args.save_total_limit,
                    **checkpoint_artifacts,
                },
            }
        )
        write_json(args.output_dir / "training_config.json", resolved_config)

        normalized_metrics = {
            "train": normalize_trainer_metrics(train_result.metrics, "train"),
            "evaluation": normalize_trainer_metrics(eval_metrics, "eval"),
        }
        run_context.write_status(
            "completed",
            sections={
                "configuration": resolved_config,
                "dataset": {
                    "resolved_dataset_root": resolved_dataset_root,
                    "source_layout": source_layout,
                    "train_source": train_source,
                    "validation_source": validation_source,
                    "train_examples": len(train_examples),
                    "validation_examples": len(validation_examples),
                },
                "validation_previews": {
                    "enabled": bool(validation_preview_examples),
                    "sample_count": len(validation_preview_examples),
                    "sample_ids": [
                        example.get("id") for example in validation_preview_examples
                    ],
                    "interval": "training_logging_steps",
                    "max_new_tokens": args.validation_preview_max_new_tokens,
                    "output_directory": str(args.output_dir / "validation_previews"),
                },
                "checkpoint_policy": resolved_config["checkpoint_policy"],
            },
            metrics=normalized_metrics,
        )

        print(f"Saved best Qwen LoRA adapter to {args.output_dir}")
        print(f"Best model copy: {checkpoint_artifacts['best_model_dir']}")
        print(f"Last model copy: {checkpoint_artifacts['last_model_dir']}")
        print(
            "Recommended next step: run inference with "
            f"`python src/Qwen/run_inference.py --adapter-path {args.output_dir}`"
        )
        return 0
    except Exception as exc:
        run_context.write_status(
            "failed",
            sections={
                "configuration": namespace_to_dict(args),
                "validation_previews": {
                    "enabled": bool(validation_preview_examples),
                    "sample_count": len(validation_preview_examples),
                    "sample_ids": [
                        example.get("id") for example in validation_preview_examples
                    ],
                    "interval": "training_logging_steps",
                    "max_new_tokens": args.validation_preview_max_new_tokens,
                    "output_directory": str(args.output_dir / "validation_previews"),
                },
                "error": {"type": type(exc).__name__, "message": str(exc)},
            },
        )
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
