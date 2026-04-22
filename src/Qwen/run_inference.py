from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_PATH = REPO_ROOT / "data" / "Lieferschein-Beispiel.png"
DEFAULT_SCHEMA_PATH = REPO_ROOT / "src" / "Donut" / "lieferschein.schema.json"
DEFAULT_EXAMPLE_PATH = REPO_ROOT / "src" / "Donut" / "lieferschein.example.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "output" / "qwen_lieferschein_inference.json"
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-27B"

PRESERVE_TEMPLATE_KEYS = {"document_type", "document_language"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference with Qwen3.5-27B on a document image and normalize the "
            "response into the project JSON skeleton."
        )
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=DEFAULT_IMAGE_PATH,
        help="Path to the input document image.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Base Hugging Face model id or local checkpoint path.",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="Optional LoRA adapter path produced by src/Qwen/train_finetune.py.",
    )
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
        help="JSON Schema describing the target output contract.",
    )
    parser.add_argument(
        "--example-path",
        type=Path,
        default=DEFAULT_EXAMPLE_PATH,
        help="Canonical example JSON used as the skeleton/template shape.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to save the inference result JSON.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=768,
        help="Upper bound for generated output tokens.",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Inference compute dtype.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the base model in 4-bit mode for lower VRAM usage.",
    )
    parser.add_argument(
        "--no-load-in-4bit",
        action="store_true",
        help="Disable 4-bit loading and load the model in the configured dtype.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map passed to from_pretrained. Use 'none' for a single-device load.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="Attention implementation passed to from_pretrained when supported.",
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
        "--local-files-only",
        action="store_true",
        help="Load model files only from the local Hugging Face cache.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_runtime_dependencies() -> tuple[Any, Any, Any, Any, Any]:
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
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
    except ImportError:
        missing.append("transformers")
        AutoProcessor = None
        BitsAndBytesConfig = None
        Qwen3_5ForConditionalGeneration = None

    if missing:
        missing_csv = ", ".join(missing)
        raise RuntimeError(
            "Missing runtime dependencies: "
            f"{missing_csv}. Install them before running inference, for example: "
            "`pip install torch torchvision transformers pillow bitsandbytes sentencepiece`."
        )

    return torch, Image, AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration


def load_image(image_path: Path, image_module: Any) -> Any:
    with image_module.open(image_path) as image:
        return image.convert("RGB")


def resolve_dtype(args: argparse.Namespace, torch: Any) -> Any:
    return getattr(torch, args.compute_dtype)


def resolve_load_in_4bit(args: argparse.Namespace) -> bool:
    if args.load_in_4bit and args.no_load_in_4bit:
        raise ValueError("Use either --load-in-4bit or --no-load-in-4bit, not both.")
    if args.no_load_in_4bit:
        return False
    if args.load_in_4bit:
        return True
    return True


def build_processor_load_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"local_files_only": args.local_files_only}
    if args.cache_dir is not None:
        kwargs["cache_dir"] = str(args.cache_dir)
    if args.min_pixels is not None:
        kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None:
        kwargs["max_pixels"] = args.max_pixels
    return kwargs


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
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=resolve_dtype(args, torch),
        )
    else:
        kwargs["torch_dtype"] = resolve_dtype(args, torch)
    return kwargs


def apply_chat_template_safely(processor: Any, messages: list[dict[str, Any]]) -> str:
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def extract_json_candidate(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None


def fill_from_template(template: Any, prediction: Any = None, key: str | None = None) -> Any:
    if isinstance(template, dict):
        source = prediction if isinstance(prediction, dict) else {}
        return {
            child_key: fill_from_template(child_template, source.get(child_key), child_key)
            for child_key, child_template in template.items()
        }

    if isinstance(template, list):
        if not template:
            return prediction if isinstance(prediction, list) else []
        item_template = template[0]
        if not isinstance(prediction, list):
            return []
        return [fill_from_template(item_template, item) for item in prediction]

    if prediction is not None:
        return prediction

    if key in PRESERVE_TEMPLATE_KEYS:
        return template

    return None


def get_model_device(model: Any) -> Any:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def move_batch_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def build_messages(schema: Any, example: Any) -> list[dict[str, Any]]:
    instruction = (
        "Extract the key information from this delivery note image and return exactly one JSON object. "
        "Do not add markdown fences, comments, or explanatory text. "
        "Use null for missing scalar values and [] for missing arrays.\n\n"
        "JSON Schema:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        "Example JSON shape:\n"
        f"{json.dumps(example, ensure_ascii=False, indent=2)}"
    )
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are an information extraction model for German delivery notes. "
                        "Return strict JSON only."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        },
    ]


def main() -> int:
    args = parse_args()

    if not args.image_path.exists():
        print(f"Input image not found: {args.image_path}", file=sys.stderr)
        return 1

    if not args.schema_path.exists():
        print(f"Schema file not found: {args.schema_path}", file=sys.stderr)
        return 1

    if not args.example_path.exists():
        print(f"Example file not found: {args.example_path}", file=sys.stderr)
        return 1

    schema = load_json(args.schema_path)
    template = load_json(args.example_path)

    torch, image_module, AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration = (
        load_runtime_dependencies()
    )

    load_in_4bit = resolve_load_in_4bit(args)
    processor = AutoProcessor.from_pretrained(args.model_id, **build_processor_load_kwargs(args))
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_id,
        **build_model_load_kwargs(args, torch, BitsAndBytesConfig, load_in_4bit),
    )
    if args.adapter_path is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError(
                "Loading a LoRA adapter requires `peft`. Install it with "
                "`pip install peft` and rerun inference."
            ) from exc
        model = PeftModel.from_pretrained(model, str(args.adapter_path))
    model.eval()

    image = load_image(args.image_path, image_module)
    messages = build_messages(schema, template)
    prompt_text = apply_chat_template_safely(processor, messages)
    inputs = processor(text=[prompt_text], images=[image], padding=True, return_tensors="pt")
    inputs = move_batch_to_device(inputs, get_model_device(model))

    notes: list[str] = []
    if args.adapter_path is None:
        notes.append(
            "No LoRA adapter was provided. The base Qwen3.5-27B model can run, but extraction quality will "
            "only be meaningful after task-specific fine-tuning."
        )

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    input_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, input_length:]
    raw_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    cleaned_text = strip_thinking(raw_text)
    json_candidate = extract_json_candidate(cleaned_text)

    parsed_prediction = None
    if json_candidate is None:
        notes.append("The model response did not contain a detectable JSON object.")
    else:
        try:
            parsed_prediction = json.loads(json_candidate)
        except json.JSONDecodeError as exc:
            notes.append(f"Failed to parse generated JSON: {exc}")

    guided_prediction = fill_from_template(template, parsed_prediction)

    result = {
        "model": {
            "model_id": args.model_id,
            "adapter_path": str(args.adapter_path) if args.adapter_path is not None else None,
            "max_new_tokens": args.max_new_tokens,
            "load_in_4bit": load_in_4bit,
        },
        "input": {
            "image_path": str(args.image_path),
            "schema_path": str(args.schema_path),
            "example_path": str(args.example_path),
        },
        "guidance": {
            "schema_title": schema.get("title"),
            "required_fields": schema.get("required", []),
        },
        "notes": notes,
        "generated": {
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "json_candidate": json_candidate,
            "raw_prediction": parsed_prediction,
            "guided_prediction": guided_prediction,
        },
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    print(f"Saved Qwen inference result to {args.output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
