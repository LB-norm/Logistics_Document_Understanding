from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_PATH = REPO_ROOT / "data" / "Lieferschein-Beispiel.png"
DEFAULT_SCHEMA_PATH = Path(__file__).with_name("lieferschein.schema.json")
DEFAULT_EXAMPLE_PATH = Path(__file__).with_name("lieferschein.example.json")
DEFAULT_OUTPUT_PATH = REPO_ROOT / "output" / "donut_lieferschein_inference.json"

# This checkpoint is a real Hugging Face Donut IE model and can be used as a
# runnable baseline. It is not trained for our Lieferschein schema.
DEFAULT_MODEL_ID = "naver-clova-ix/donut-base-finetuned-cord-v2"
DEFAULT_TASK_PROMPT = "<s_cord-v2>"

# Keep fixed metadata-style defaults from the example skeleton.
PRESERVE_TEMPLATE_KEYS = {"document_type", "document_language"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run CPU-based Donut inference on the sample Lieferschein image and "
            "normalize the result into the project output skeleton."
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
        help=(
            "Hugging Face model id or local checkpoint path. "
            "For a real Lieferschein extraction model, replace this with your own fine-tuned checkpoint."
        ),
    )
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Optional model revision on Hugging Face.",
    )
    parser.add_argument(
        "--task-prompt",
        default=DEFAULT_TASK_PROMPT,
        help=(
            "Decoder start prompt. For a future custom model this should typically be "
            "something like <s_lieferschein>."
        ),
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
        "--device",
        default="cpu",
        choices=["cpu"],
        help="Execution device. CPU is the only supported option in this skeleton.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=768,
        help="Upper bound for generated sequence length.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Beam width for generation. Use 1 for greedy decoding on CPU.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model files only from the local Hugging Face cache.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_runtime_dependencies() -> tuple[Any, Any, Any, Any]:
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
        from transformers import DonutProcessor, VisionEncoderDecoderModel
    except ImportError:
        missing.append("transformers")
        DonutProcessor = None
        VisionEncoderDecoderModel = None

    if missing:
        missing_csv = ", ".join(missing)
        raise RuntimeError(
            "Missing runtime dependencies: "
            f"{missing_csv}. Install them before running inference, for example: "
            "`pip install torch transformers pillow sentencepiece`."
        )

    return torch, Image, DonutProcessor, VisionEncoderDecoderModel


def load_image(image_path: Path, image_module: Any) -> Any:
    with image_module.open(image_path) as image:
        return image.convert("RGB")


def build_model_load_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"local_files_only": args.local_files_only}
    if args.cache_dir is not None:
        kwargs["cache_dir"] = str(args.cache_dir)
    if args.model_revision:
        kwargs["revision"] = args.model_revision
    return kwargs


def clean_sequence(sequence: str, processor: Any) -> str:
    cleaned = sequence
    if processor.tokenizer.eos_token:
        cleaned = cleaned.replace(processor.tokenizer.eos_token, "")
    if processor.tokenizer.pad_token:
        cleaned = cleaned.replace(processor.tokenizer.pad_token, "")
    return re.sub(r"<.*?>", "", cleaned, count=1).strip()


def parse_sequence_to_json(sequence: str, processor: Any) -> Any:
    if hasattr(processor, "token2json"):
        try:
            return processor.token2json(sequence)
        except Exception:
            pass
    return {"text_sequence": sequence}


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


def detect_prompt_warning(task_prompt: str, processor: Any) -> str | None:
    tokenizer = processor.tokenizer
    known_tokens = set(tokenizer.all_special_tokens)
    known_tokens.update(tokenizer.get_added_vocab().keys())

    if task_prompt in known_tokens:
        return None

    return (
        f"Task prompt {task_prompt!r} is not registered as a tokenizer special token in this checkpoint. "
        "Inference can still run, but a generic or differently fine-tuned model will not reliably follow the "
        "custom Lieferschein schema."
    )


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

    template = load_json(args.example_path)
    schema = load_json(args.schema_path)

    torch, image_module, DonutProcessor, VisionEncoderDecoderModel = load_runtime_dependencies()
    model_load_kwargs = build_model_load_kwargs(args)

    processor = DonutProcessor.from_pretrained(args.model_id, **model_load_kwargs)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_id, **model_load_kwargs)
    model.to(args.device)
    model.eval()

    image = load_image(args.image_path, image_module)
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(args.device)

    decoder_input_ids = processor.tokenizer(
        args.task_prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(args.device)

    max_supported_length = getattr(model.decoder.config, "max_position_embeddings", args.max_length)
    max_length = min(args.max_length, max_supported_length)

    generation_kwargs: dict[str, Any] = {
        "max_length": max_length,
        "num_beams": args.num_beams,
        "pad_token_id": processor.tokenizer.pad_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "use_cache": True,
        "return_dict_in_generate": True,
    }

    if processor.tokenizer.unk_token_id is not None:
        generation_kwargs["bad_words_ids"] = [[processor.tokenizer.unk_token_id]]

    notes: list[str] = []
    prompt_warning = detect_prompt_warning(args.task_prompt, processor)
    if prompt_warning:
        notes.append(prompt_warning)

    if args.task_prompt == DEFAULT_TASK_PROMPT and args.model_id == DEFAULT_MODEL_ID:
        notes.append(
            "This default checkpoint is fine-tuned for CORD receipt parsing, not for German delivery notes. "
            "It is useful as a runnable Donut baseline, but it will not reliably output the custom Lieferschein JSON "
            "until you fine-tune a checkpoint on this schema."
        )

    with torch.inference_mode():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            **generation_kwargs,
        )

    raw_sequence = processor.batch_decode(outputs.sequences)[0]
    cleaned_sequence = clean_sequence(raw_sequence, processor)
    raw_prediction = parse_sequence_to_json(cleaned_sequence, processor)
    guided_prediction = fill_from_template(template, raw_prediction)

    result = {
        "model": {
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "device": args.device,
            "task_prompt": args.task_prompt,
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
            "raw_sequence": raw_sequence,
            "cleaned_sequence": cleaned_sequence,
            "raw_prediction": raw_prediction,
            "guided_prediction": guided_prediction,
        },
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    print(f"Saved Donut inference result to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
