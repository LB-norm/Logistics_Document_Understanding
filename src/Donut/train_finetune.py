from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_PATH = Path(__file__).with_name("lieferschein.schema.json")
DEFAULT_EXAMPLE_PATH = Path(__file__).with_name("lieferschein.example.json")
DEFAULT_IMAGE_PATH = REPO_ROOT / "data" / "Lieferschein-Beispiel.png"
DEFAULT_MODEL_ID = "naver-clova-ix/donut-base"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "models" / "donut-lieferschein"
DEFAULT_TASK_START_TOKEN = "<s_lieferschein>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune Donut on a custom document parsing dataset that follows "
            "the metadata.jsonl / gt_parse format from the official Donut repository."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Dataset directory containing train/ and validation/ subdirectories. "
            "Each split must contain a metadata.jsonl file and referenced images. "
            "If omitted, a tiny local smoke-test dataset is created from the sample Lieferschein."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for checkpoints and the final fine-tuned model.",
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
    parser.add_argument("--max-length", type=int, default=768, help="Maximum decoder sequence length.")
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
        default=3,
        help="Maximum number of checkpoints to keep on disk.",
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
    return parser.parse_args()


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


def ensure_smoketest_dataset() -> Path:
    dataset_root = Path(tempfile.mkdtemp(prefix="donut_smoketest_", dir="/tmp"))
    example = load_json(DEFAULT_EXAMPLE_PATH)

    for split in ("train", "validation"):
        split_dir = dataset_root / split
        split_dir.mkdir(parents=True, exist_ok=True)

        target_image = split_dir / DEFAULT_IMAGE_PATH.name
        if not target_image.exists():
            shutil.copy2(DEFAULT_IMAGE_PATH, target_image)

        record = {
            "file_name": DEFAULT_IMAGE_PATH.name,
            "ground_truth": json.dumps({"gt_parse": example}, ensure_ascii=False),
        }
        metadata_path = split_dir / "metadata.jsonl"
        metadata_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    return dataset_root


def resolve_dataset_root(args: argparse.Namespace) -> Path:
    if args.dataset_root is not None:
        return args.dataset_root

    if not DEFAULT_IMAGE_PATH.exists():
        raise FileNotFoundError(f"Default smoke-test image not found: {DEFAULT_IMAGE_PATH}")
    if not DEFAULT_EXAMPLE_PATH.exists():
        raise FileNotFoundError(f"Default smoke-test example not found: {DEFAULT_EXAMPLE_PATH}")

    return ensure_smoketest_dataset()


def validate_split_dir(split_dir: Path) -> Path:
    metadata_path = split_dir / "metadata.jsonl"
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.jsonl in split directory: {split_dir}")
    return metadata_path


def collect_schema_tokens(schema: Any) -> set[str]:
    tokens: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, dict):
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


def build_example_record(raw_record: dict[str, Any], split_dir: Path) -> dict[str, Any]:
    if "file_name" not in raw_record:
        raise ValueError(f"Missing 'file_name' in {split_dir / 'metadata.jsonl'} record: {raw_record}")
    if "ground_truth" not in raw_record:
        raise ValueError(f"Missing 'ground_truth' in {split_dir / 'metadata.jsonl'} record: {raw_record}")

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
        "image_path": str(image_path),
        "gt_parse": gt_parse,
        "target_sequence": json_to_donut_tokens(gt_parse),
    }


def load_split(split_dir: Path) -> list[dict[str, Any]]:
    metadata_path = validate_split_dir(split_dir)
    return [build_example_record(record, split_dir) for record in read_jsonl(metadata_path)]


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


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    dataset_root = resolve_dataset_root(args)
    train_examples = load_split(dataset_root / "train")
    validation_examples = load_split(dataset_root / "validation")
    schema = load_json(args.schema_path)

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

    processor.image_processor.size = {"height": args.image_size[0], "width": args.image_size[1]}
    processor.image_processor.do_align_long_axis = False

    schema_tokens = collect_schema_tokens(schema)
    data_tokens: set[str] = set()
    for sample in train_examples + validation_examples:
        data_tokens.update(collect_field_tokens_from_gt_parse(sample["gt_parse"]))

    special_tokens = {
        args.task_start_token,
        "<sep/>",
        "<null/>",
    }
    special_tokens.update(schema_tokens)
    special_tokens.update(data_tokens)
    add_special_tokens(processor, model, special_tokens)

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

    train_dataset = Dataset.from_list(train_examples)
    validation_dataset = Dataset.from_list(validation_examples)
    data_collator = DonutBatchCollator(
        processor=processor,
        image_module=image_module,
        max_length=args.max_length,
    )

    bf16, fp16 = choose_precision_flags(args)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        predict_with_generate=args.predict_with_generate,
        generation_max_length=args.max_length,
        generation_num_beams=1,
        max_steps=args.max_steps,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=gradient_checkpointing,
        report_to="none",
        do_train=True,
        do_eval=True,
        load_best_model_at_end=False,
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        processing_class=processor,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))

    with (args.output_dir / "training_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset_root": str(args.dataset_root),
                "resolved_dataset_root": str(dataset_root),
                "model_id": args.model_id,
                "task_start_token": args.task_start_token,
                "schema_path": str(args.schema_path),
                "image_size": list(args.image_size),
                "max_length": args.max_length,
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
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved fine-tuned Donut model to {args.output_dir}")
    print(
        "Recommended next step: run inference with "
        f"`python3 src/Donut/run_inference.py --model-id {args.output_dir} --task-prompt {args.task_start_token!r}`"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
