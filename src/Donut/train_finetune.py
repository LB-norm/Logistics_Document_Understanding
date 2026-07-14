from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
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

DEFAULT_SCHEMA_PATH = REPO_ROOT / "json_schema" / "content.schema.json"
DEFAULT_TARGET_SKELETON_PATH = REPO_ROOT / "json_schema" / "content.empty.json"
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "datasets" / "250_CMRS_240dpi_20260707"
DEFAULT_MODEL_ID = "naver-clova-ix/donut-base"
DEFAULT_RUNS_DIR = REPO_ROOT / "runs" / "donut"
DEFAULT_TASK_START_TOKEN = "<s_lieferschein>"
DEFAULT_ANNOTATION_TARGET_KEY = "content"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
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
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce activation memory.",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing.",
    )
    parser.add_argument("--dataloader-num-workers", type=int, default=4, help="PyTorch dataloader workers.")
    parser.add_argument("--eval-steps", type=int, default=250, help="Evaluation interval in optimizer steps.")
    parser.add_argument("--save-steps", type=int, default=250, help="Checkpoint save interval.")
    parser.add_argument("--logging-steps", type=int, default=25, help="Logging interval.")
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
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load Hugging Face model files only from the local cache.",
    )
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
        help="Enable generation during evaluation. Slower, but useful for inspecting outputs.",
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
    return parser.parse_args()


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


def load_runtime_dependencies() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
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
            VisionEncoderDecoderModel,
        )
    except ImportError:
        missing.append("transformers")
        DonutProcessor = None
        Seq2SeqTrainer = None
        Seq2SeqTrainingArguments = None
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

    return torch, Image, DonutProcessor, VisionEncoderDecoderModel, Seq2SeqTrainingArguments, Seq2SeqTrainer, Dataset


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
        "target_sequence": json_to_donut_tokens(gt_parse),
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
        "target_sequence": json_to_donut_tokens(gt_parse),
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
                "target_sequence": json_to_donut_tokens(gt_parse),
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


def json_to_donut_tokens(obj: Any, sort_keys: bool = True) -> str:
    if isinstance(obj, dict):
        keys = sorted(obj.keys()) if sort_keys else list(obj.keys())
        output = ""
        for key in keys:
            output += f"<s_{key}>"
            output += json_to_donut_tokens(obj[key], sort_keys=sort_keys)
            output += f"</s_{key}>"
        return output

    if isinstance(obj, list):
        return "<sep/>".join(json_to_donut_tokens(item, sort_keys=sort_keys) for item in obj)

    if obj is None:
        return "<null/>"

    if isinstance(obj, bool):
        return "true" if obj else "false"

    return str(obj)


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
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
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


def main() -> int:
    args = parse_args()
    apply_checkpoint_policy(args)
    set_seed(args.seed)

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
    train_examples, validation_examples, train_split, validation_split, source_layout = load_dataset_splits(
        dataset_root,
        args,
    )
    schema = load_json(args.schema_path)
    target_schema = select_schema_node_for_target(schema, args.annotation_target_key)
    target_skeleton = load_json(args.target_skeleton_path)
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
        return 0

    (
        torch,
        image_module,
        DonutProcessor,
        VisionEncoderDecoderModel,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
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

    if args.gradient_checkpointing and args.no_gradient_checkpointing:
        raise ValueError("Use either --gradient-checkpointing or --no-gradient-checkpointing, not both.")

    gradient_checkpointing = True
    if args.no_gradient_checkpointing:
        gradient_checkpointing = False
    elif args.gradient_checkpointing:
        gradient_checkpointing = True

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
        "checkpoint_policy": {
            "retained": "best_and_last",
            "metric": "eval_loss",
            "greater_is_better": False,
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
            "train_examples": len(train_examples),
            "validation_examples": len(validation_examples),
        },
    )

    print(f"Saved fine-tuned Donut model to {output_dir}")
    inference_example = validation_examples[0]
    inference_command = (
        "python3 src/Donut/run_inference.py "
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
