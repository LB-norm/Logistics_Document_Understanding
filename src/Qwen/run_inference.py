from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval_suite.schema import validate_json_schema

DEFAULT_SMALL_TEST_IMAGE_PATH = (
    REPO_ROOT
    / "data"
    / "small testing"
    / "3f3fdb18-c151-43dd-b54a-da34249241f6_CMR_page_1.jpg"
)
DEFAULT_IMAGE_PATH = DEFAULT_SMALL_TEST_IMAGE_PATH
DEFAULT_SCHEMA_PATH = REPO_ROOT / "json_schema" / "content.schema.json"
DEFAULT_TEMPLATE_PATH = REPO_ROOT / "json_schema" / "content.empty.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "output" / "qwen_lieferschein_inference.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "qwen_lieferschein_inference"
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-2B"
DEFAULT_ANNOTATION_TARGET_KEY = "root"
DEFAULT_SYSTEM_PROMPT = "You are an information extraction model for CMR delivery note scans. Return strict JSON only."
DEFAULT_USER_PROMPT = (
    "Extract all relevant document information into the target CMR/Lieferschein content JSON object. "
    "Use null for missing scalar values and [] for missing arrays."
)

PRESERVE_TEMPLATE_KEYS = {"document_type", "document_language"}


@dataclass
class InferenceRuntime:
    torch: Any
    image_module: Any
    processor: Any
    model: Any
    model_id: str
    processor_source: str


@dataclass
class ImageInferenceResult:
    image_path: Path
    prediction: Any
    raw_text: str
    cleaned_text: str
    json_candidate: str | None
    raw_prediction: Any
    notes: list[str]
    schema_errors: list[str]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference with a Qwen3.5 vision-language model on a document image and normalize the "
            "response into the project JSON skeleton."
        )
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--image-path",
        type=Path,
        default=None,
        help=(
            "Path to one input document image. If neither image option is supplied, the bundled "
            "small test image is used."
        ),
    )
    input_group.add_argument(
        "--image-paths",
        type=Path,
        nargs="+",
        default=None,
        metavar="IMAGE",
        help=(
            "Paths to multiple independent document images. The model is loaded once and one filled "
            "JSON file is written per image."
        ),
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help=(
            "Base Hugging Face model id or local checkpoint path. When an adapter is supplied, its "
            "adapter_config.json base_model_name_or_path is used by default."
        ),
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="Optional LoRA adapter path produced by src/Qwen/run_qwen_training.py.",
    )
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
        help="JSON Schema describing the target output contract.",
    )
    parser.add_argument(
        "--template-path",
        "--example-path",
        dest="template_path",
        type=Path,
        default=DEFAULT_TEMPLATE_PATH,
        help=(
            "Empty JSON template defining the final output shape. --example-path is retained as an "
            "alias for backwards compatibility."
        ),
    )
    parser.add_argument(
        "--annotation-target-key",
        default=DEFAULT_ANNOTATION_TARGET_KEY,
        help=(
            "Key inside the template JSON to use as the output shape. The default 'root' matches "
            "json_schema/content.empty.json; use 'content' for a wrapped annotation template."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to save the inference result JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for filled JSON files produced by --image-paths.",
    )
    parser.add_argument(
        "--diagnostics-path",
        type=Path,
        default=None,
        help=(
            "Optional diagnostics manifest path. Defaults to a sidecar JSON for one image and "
            "inference_manifest.jsonl inside --output-dir for multiple images."
        ),
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt used for generation. The default matches Qwen project fine-tuning.",
    )
    parser.add_argument(
        "--user-prompt",
        default=DEFAULT_USER_PROMPT,
        help="User prompt used for generation. The default matches Qwen project fine-tuning.",
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
    return parser.parse_args(argv)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_image_paths(args: argparse.Namespace) -> list[Path]:
    if args.image_paths is not None:
        return list(args.image_paths)
    if args.image_path is not None:
        return [args.image_path]
    return [DEFAULT_IMAGE_PATH]


def read_adapter_base_model(adapter_path: Path) -> str | None:
    config_path = adapter_path / "adapter_config.json"
    if not config_path.is_file():
        return None
    config = load_json(config_path)
    model_id = (
        config.get("base_model_name_or_path") if isinstance(config, dict) else None
    )
    return model_id if isinstance(model_id, str) and model_id.strip() else None


def resolve_model_id(args: argparse.Namespace) -> str:
    if args.model_id:
        return str(args.model_id)
    if args.adapter_path is not None:
        adapter_model_id = read_adapter_base_model(args.adapter_path)
        if adapter_model_id:
            return adapter_model_id
    return DEFAULT_MODEL_ID


def resolve_processor_source(args: argparse.Namespace, model_id: str) -> str:
    if args.adapter_path is not None:
        processor_config = args.adapter_path / "processor_config.json"
        if processor_config.is_file():
            return str(args.adapter_path)
    return model_id


def extract_json_target(obj: Any, target_key: str) -> Any:
    if target_key in {"", ".", "root"}:
        return obj

    target = obj
    for key in target_key.split("."):
        if not isinstance(target, dict) or key not in target:
            if target_key == DEFAULT_ANNOTATION_TARGET_KEY and isinstance(obj, dict):
                return obj
            raise KeyError(f"Target key {target_key!r} not found in JSON template.")
        target = target[key]

    if not isinstance(target, dict):
        raise TypeError(f"Target key {target_key!r} must resolve to a JSON object.")

    return target


def resolve_schema_ref(schema: dict[str, Any], node: Any) -> Any:
    while isinstance(node, dict) and "$ref" in node:
        ref = node["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/"):
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
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen3_5ForConditionalGeneration,
        )
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

    return (
        torch,
        Image,
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen3_5ForConditionalGeneration,
    )


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


def fill_from_template(
    template: Any, prediction: Any = None, key: str | None = None
) -> Any:
    if isinstance(template, dict):
        source = prediction if isinstance(prediction, dict) else {}
        return {
            child_key: fill_from_template(
                child_template, source.get(child_key), child_key
            )
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
    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if embeddings is not None and hasattr(embeddings, "weight"):
            return embeddings.weight.device
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


def build_messages(system_prompt: str, user_prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


def load_inference_runtime(args: argparse.Namespace) -> InferenceRuntime:
    (
        torch,
        image_module,
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen3_5ForConditionalGeneration,
    ) = load_runtime_dependencies()
    model_id = resolve_model_id(args)
    processor_source = resolve_processor_source(args, model_id)
    load_in_4bit = resolve_load_in_4bit(args)

    processor = AutoProcessor.from_pretrained(
        processor_source,
        **build_processor_load_kwargs(args),
    )
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id,
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

    return InferenceRuntime(
        torch=torch,
        image_module=image_module,
        processor=processor,
        model=model,
        model_id=model_id,
        processor_source=processor_source,
    )


def generate_image_prediction(
    runtime: InferenceRuntime,
    image_path: Path,
    template: Any,
    target_schema: Any,
    args: argparse.Namespace,
) -> ImageInferenceResult:
    image = load_image(image_path, runtime.image_module)
    messages = build_messages(args.system_prompt, args.user_prompt)
    prompt_text = apply_chat_template_safely(runtime.processor, messages)
    inputs = runtime.processor(
        text=[prompt_text], images=[image], padding=True, return_tensors="pt"
    )
    inputs = move_batch_to_device(inputs, get_model_device(runtime.model))

    notes: list[str] = []
    if args.adapter_path is None:
        notes.append(
            "No LoRA adapter was provided. The base Qwen3.5 model can run, but extraction quality will "
            "only be meaningful after task-specific fine-tuning."
        )

    with runtime.torch.inference_mode():
        outputs = runtime.model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
    input_length = inputs["input_ids"].shape[1]
    generated_ids = sequences[:, input_length:]
    raw_text = runtime.processor.batch_decode(
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
    if (
        parsed_prediction is not None
        and isinstance(template, dict)
        and not isinstance(parsed_prediction, dict)
    ):
        notes.append(
            "The model response JSON is not an object, so it cannot fill the object template."
        )
    if (
        isinstance(template, dict)
        and template
        and isinstance(parsed_prediction, dict)
        and not set(template).intersection(parsed_prediction)
    ):
        notes.append(
            "The model response object does not contain any top-level template fields."
        )

    prediction = fill_from_template(template, parsed_prediction)
    schema_errors = validate_json_schema(prediction, target_schema)
    if schema_errors:
        notes.append(
            f"The filled prediction has {len(schema_errors)} JSON Schema error(s)."
        )

    return ImageInferenceResult(
        image_path=image_path,
        prediction=prediction,
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        json_candidate=json_candidate,
        raw_prediction=parsed_prediction,
        notes=notes,
        schema_errors=schema_errors,
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def resolve_prediction_paths(
    args: argparse.Namespace, image_paths: list[Path]
) -> list[Path]:
    if len(image_paths) == 1:
        return [args.output_path]

    prediction_paths = [
        args.output_dir / f"{image_path.stem}.json" for image_path in image_paths
    ]
    duplicates = sorted(
        path.name for path in set(prediction_paths) if prediction_paths.count(path) > 1
    )
    if duplicates:
        duplicate_csv = ", ".join(duplicates)
        raise ValueError(
            "Multiple input images would overwrite the same prediction file: "
            f"{duplicate_csv}. Use images with unique stems."
        )
    return prediction_paths


def default_diagnostics_path(args: argparse.Namespace, image_count: int) -> Path:
    if args.diagnostics_path is not None:
        return args.diagnostics_path
    if image_count == 1:
        return args.output_path.with_suffix(
            args.output_path.suffix + ".diagnostics.json"
        )
    return args.output_dir / "inference_manifest.jsonl"


def result_status(result: ImageInferenceResult, template: Any) -> str:
    if result.raw_prediction is None:
        return "parse_error"
    if isinstance(template, dict) and not isinstance(result.raw_prediction, dict):
        return "parse_error"
    if (
        isinstance(template, dict)
        and template
        and isinstance(result.raw_prediction, dict)
        and not set(template).intersection(result.raw_prediction)
    ):
        return "template_error"
    if result.schema_errors:
        return "schema_error"
    return "ok"


def build_diagnostic_record(
    *,
    args: argparse.Namespace,
    runtime: InferenceRuntime,
    output_path: Path,
    result: ImageInferenceResult,
    template: Any,
) -> dict[str, Any]:
    return {
        "status": result_status(result, template),
        "image_path": str(result.image_path),
        "prediction_path": str(output_path),
        "model": {
            "model_id": runtime.model_id,
            "adapter_path": (
                str(args.adapter_path) if args.adapter_path is not None else None
            ),
            "processor_source": runtime.processor_source,
            "max_new_tokens": args.max_new_tokens,
            "load_in_4bit": resolve_load_in_4bit(args),
        },
        "raw_text": result.raw_text,
        "cleaned_text": result.cleaned_text,
        "json_candidate": result.json_candidate,
        "raw_prediction": result.raw_prediction,
        "schema_errors": result.schema_errors,
        "notes": result.notes,
    }


def write_diagnostics(
    path: Path, records: list[dict[str, Any]], multiple: bool
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not multiple:
        write_json(path, records[0])
        return
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def run_inference_on_images(
    *,
    args: argparse.Namespace,
    runtime: InferenceRuntime,
    image_paths: list[Path],
    prediction_paths: list[Path],
    template: Any,
    target_schema: Any,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for image_path, prediction_path in zip(image_paths, prediction_paths, strict=True):
        try:
            result = generate_image_prediction(
                runtime, image_path, template, target_schema, args
            )
            write_json(prediction_path, result.prediction)
            record = build_diagnostic_record(
                args=args,
                runtime=runtime,
                output_path=prediction_path,
                result=result,
                template=template,
            )
            print(f"Saved filled Qwen prediction to {prediction_path}")
        except Exception as exc:
            record = {
                "status": "inference_error",
                "image_path": str(image_path),
                "prediction_path": None,
                "model": {
                    "model_id": runtime.model_id,
                    "adapter_path": (
                        str(args.adapter_path)
                        if args.adapter_path is not None
                        else None
                    ),
                    "processor_source": runtime.processor_source,
                },
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"Inference failed for {image_path}: {exc}", file=sys.stderr)
        records.append(record)
    return records


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    image_paths = resolve_image_paths(args)

    missing_images = [path for path in image_paths if not path.is_file()]
    if missing_images:
        for image_path in missing_images:
            print(f"Input image not found: {image_path}", file=sys.stderr)
        return 1
    if not args.schema_path.is_file():
        print(f"Schema file not found: {args.schema_path}", file=sys.stderr)
        return 1
    if not args.template_path.is_file():
        print(f"Template file not found: {args.template_path}", file=sys.stderr)
        return 1
    if args.adapter_path is not None and not args.adapter_path.is_dir():
        print(f"Adapter directory not found: {args.adapter_path}", file=sys.stderr)
        return 1

    schema = load_json(args.schema_path)
    template = extract_json_target(
        load_json(args.template_path), args.annotation_target_key
    )
    target_schema = select_schema_node_for_target(schema, args.annotation_target_key)
    template_schema_errors = validate_json_schema(template, target_schema)
    if template_schema_errors:
        print(
            f"Template does not satisfy the selected JSON Schema ({len(template_schema_errors)} error(s)):",
            file=sys.stderr,
        )
        for error in template_schema_errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    prediction_paths = resolve_prediction_paths(args, image_paths)
    diagnostics_path = default_diagnostics_path(args, len(image_paths))

    runtime = load_inference_runtime(args)
    records = run_inference_on_images(
        args=args,
        runtime=runtime,
        image_paths=image_paths,
        prediction_paths=prediction_paths,
        template=template,
        target_schema=target_schema,
    )
    write_diagnostics(diagnostics_path, records, multiple=len(image_paths) > 1)
    print(f"Saved inference diagnostics to {diagnostics_path}")

    failed = sum(record["status"] != "ok" for record in records)
    if failed:
        print(
            f"Inference completed with {failed} unsuccessful result(s).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
