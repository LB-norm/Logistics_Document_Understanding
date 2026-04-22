from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "qwen_lora_dataset"
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-27B"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "models" / "qwen-lieferschein-lora"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LoRA fine-tune Qwen3.5-27B on a local vision-language dataset for "
            "document information extraction."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=(
            "Dataset directory containing train.jsonl, validation.jsonl, and image files. "
            "See src/Qwen/README.md for the expected structure."
        ),
    )
    parser.add_argument(
        "--train-file",
        default="train.jsonl",
        help="Training JSONL file name relative to --dataset-root.",
    )
    parser.add_argument(
        "--validation-file",
        default="validation.jsonl",
        help="Validation JSONL file name relative to --dataset-root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for LoRA checkpoints and the final adapter.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Base Hugging Face model id or local checkpoint path.",
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
    parser.add_argument("--dataloader-num-workers", type=int, default=2, help="PyTorch dataloader workers.")
    parser.add_argument("--eval-steps", type=int, default=50, help="Evaluation interval in optimizer steps.")
    parser.add_argument("--save-steps", type=int, default=50, help="Checkpoint save interval.")
    parser.add_argument("--logging-steps", type=int, default=10, help="Logging interval.")
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
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the base model in 4-bit and run QLoRA training.",
    )
    parser.add_argument(
        "--no-load-in-4bit",
        action="store_true",
        help="Disable 4-bit loading and run regular LoRA.",
    )
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
        default="lm_head,embed_tokens",
        help="Comma-separated modules to keep trainable alongside LoRA adapters.",
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
    return parser.parse_args()


def load_runtime_dependencies() -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
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
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen3_5ForConditionalGeneration,
            Trainer,
            TrainingArguments,
        )
    except ImportError:
        missing.append("transformers")
        AutoProcessor = None
        BitsAndBytesConfig = None
        Qwen3_5ForConditionalGeneration = None
        Trainer = None
        TrainingArguments = None

    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError:
        missing.append("peft")
        LoraConfig = None
        get_peft_model = None
        prepare_model_for_kbit_training = None

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
        Qwen3_5ForConditionalGeneration,
        Trainer,
        TrainingArguments,
        (LoraConfig, get_peft_model, prepare_model_for_kbit_training),
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


def resolve_dataset_file(dataset_root: Path, relative_name: str) -> Path:
    dataset_file = dataset_root / relative_name
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    return dataset_file


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


def parse_target_modules(raw_value: str) -> str | list[str]:
    if raw_value.strip() == "all-linear":
        return "all-linear"
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def parse_modules_to_save(raw_value: str) -> list[str] | None:
    modules = [item.strip() for item in raw_value.split(",") if item.strip()]
    return modules or None


def resolve_dtype(args: argparse.Namespace, torch: Any) -> Any:
    return getattr(torch, args.compute_dtype)


def choose_precision_flags(args: argparse.Namespace) -> tuple[bool, bool]:
    return args.compute_dtype == "bfloat16", args.compute_dtype == "float16"


def resolve_load_in_4bit(args: argparse.Namespace) -> bool:
    if args.load_in_4bit and args.no_load_in_4bit:
        raise ValueError("Use either --load-in-4bit or --no-load-in-4bit, not both.")
    if args.no_load_in_4bit:
        return False
    if args.load_in_4bit:
        return True
    return True


def resolve_gradient_checkpointing(args: argparse.Namespace) -> bool:
    if args.gradient_checkpointing and args.no_gradient_checkpointing:
        raise ValueError("Use either --gradient-checkpointing or --no-gradient-checkpointing, not both.")
    if args.no_gradient_checkpointing:
        return False
    if args.gradient_checkpointing:
        return True
    return True


def build_model_load_kwargs(
    args: argparse.Namespace,
    torch: Any,
    BitsAndBytesConfig: Any,
    load_in_4bit: bool,
) -> dict[str, Any]:
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


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    train_file = resolve_dataset_file(args.dataset_root, args.train_file)
    validation_file = resolve_dataset_file(args.dataset_root, args.validation_file)
    train_examples = load_split(train_file, args.dataset_root)
    validation_examples = load_split(validation_file, args.dataset_root)

    (
        torch,
        image_module,
        Dataset,
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen3_5ForConditionalGeneration,
        Trainer,
        TrainingArguments,
        peft_fns,
    ) = load_runtime_dependencies()
    LoraConfig, get_peft_model, prepare_model_for_kbit_training = peft_fns

    load_in_4bit = resolve_load_in_4bit(args)
    gradient_checkpointing = resolve_gradient_checkpointing(args)
    bf16, fp16 = choose_precision_flags(args)

    processor = AutoProcessor.from_pretrained(args.model_id, **build_processor_load_kwargs(args))
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_id,
        **build_model_load_kwargs(args, torch, BitsAndBytesConfig, load_in_4bit),
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

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=parse_target_modules(args.target_modules),
        task_type="CAUSAL_LM",
        modules_to_save=parse_modules_to_save(args.modules_to_save),
    )
    model = get_peft_model(model, peft_config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

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

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
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
        max_steps=args.max_steps,
        gradient_checkpointing=gradient_checkpointing,
        bf16=bf16,
        fp16=fp16,
        report_to="none",
        do_train=True,
        do_eval=True,
        load_best_model_at_end=False,
        optim="paged_adamw_8bit" if load_in_4bit else "adamw_torch",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))

    with (args.output_dir / "training_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset_root": str(args.dataset_root),
                "train_file": str(train_file),
                "validation_file": str(validation_file),
                "model_id": args.model_id,
                "min_pixels": args.min_pixels,
                "max_pixels": args.max_pixels,
                "max_length": args.max_length,
                "num_train_epochs": args.num_train_epochs,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "per_device_eval_batch_size": args.per_device_eval_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "gradient_checkpointing": gradient_checkpointing,
                "load_in_4bit": load_in_4bit,
                "compute_dtype": args.compute_dtype,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "target_modules": parse_target_modules(args.target_modules),
                "modules_to_save": parse_modules_to_save(args.modules_to_save),
                "train_examples": len(train_examples),
                "validation_examples": len(validation_examples),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved Qwen LoRA adapter to {args.output_dir}")
    print(
        "Recommended next step: run inference with "
        f"`python3 src/Qwen/run_inference.py --adapter-path {args.output_dir}`"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
