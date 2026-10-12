"""Prepare split-preserving OCR evidence and text-only LLM fine-tuning records."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

from src.PP_parser.ocr import OCRExtractor, save_ocr, write_json
from src.PP_parser.prepare_finetune import load_project_split, write_jsonl
from src.Qwen.prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT, build_ocr_messages

REPO_ROOT = Path(__file__).resolve().parents[2]


def ensure_output_directory(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        if not overwrite or not output_dir.is_dir():
            raise FileExistsError(f"Output directory is not empty: {output_dir}. Use --overwrite to replace artifacts.")
    output_dir.mkdir(parents=True, exist_ok=True)


def load_dataset_examples(
    dataset_root: Path, splits: Sequence[str], annotation_target_key: str, limit: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    if not splits or len(set(splits)) != len(splits):
        raise ValueError("Specify at least one split, without duplicates")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    examples = {}
    for split in splits:
        if not split or Path(split).name != split or split in {".", ".."}:
            raise ValueError("Split names must be single directory names")
        rows = load_project_split(dataset_root, split, annotation_target_key)
        ids = [row["id"] for row in rows]
        if any(not isinstance(value, str) or not value for value in ids):
            raise ValueError(f"Split {split} has an invalid sample ID")
        if len(set(ids)) != len(ids):
            raise ValueError(f"Split {split} has duplicate sample IDs")
        if not rows:
            raise ValueError(f"Split {split} contains no examples")
        examples[split] = rows[:limit] if limit is not None else rows
    return examples


def build_training_record(
    example: dict[str, Any], document: dict[str, Any], template: Any, schema: Any,
) -> dict[str, Any]:
    prompts = build_ocr_messages(DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT, document["text"], template, schema)
    # Plain-string content is a portable chat/SFT format, with no image tokens.
    messages = [{"role": message["role"], "content": message["content"][0]["text"]} for message in prompts]
    messages.append({"role": "assistant", "content": json.dumps(example["target"], ensure_ascii=False)})
    return {"id": example["id"], "text": document["text"], "target": example["target"], "messages": messages}


def prepare_ocr_dataset(
    extractor: OCRExtractor,
    dataset_root: Path,
    output_dir: Path,
    *,
    splits: Sequence[str] = ("train", "val"),
    annotation_target_key: str = "content",
    overwrite: bool = False,
    dry_run: bool = False,
    limit: int | None = None,
    template: Any = None,
    schema: Any = None,
) -> dict[str, Any]:
    """Keep source splits/targets; omit failed or empty OCR from the SFT JSONL.

    Output metadata paths are relative to output_dir. Source image/annotation
    paths are absolute provenance links. Original images and labels are untouched.
    A dry run validates all metadata/targets without loading OCR or writing files.
    """
    dataset_root, output_dir = Path(dataset_root).resolve(), Path(output_dir).resolve()
    if dataset_root == output_dir or dataset_root in output_dir.parents or output_dir in dataset_root.parents:
        raise ValueError("Source dataset and output directories must not overlap")
    examples = load_dataset_examples(dataset_root, splits, annotation_target_key, limit)
    if template is None:
        template = json.loads((REPO_ROOT / "json_schema/content.empty.json").read_text(encoding="utf-8"))
    if schema is None:
        schema = json.loads((REPO_ROOT / "json_schema/content.schema.json").read_text(encoding="utf-8"))
    summary = {
        "format_version": 1,
        "dataset_root": str(dataset_root), "output_dir": str(output_dir),
        "annotation_target_key": annotation_target_key, "dry_run": dry_run,
        "settings": {**extractor.config.pipeline_settings(), "min_confidence": extractor.config.min_confidence,
                     "pdf_dpi": extractor.config.pdf_dpi},
        "splits": {},
    }
    if dry_run:
        summary["splits"] = {split: {"selected": len(rows)} for split, rows in examples.items()}
        return summary
    ensure_output_directory(output_dir, overwrite=overwrite)
    write_json(output_dir / "schema.json", schema)
    write_json(output_dir / "template.json", template)
    for split, rows in examples.items():
        metadata, training = [], []
        for index, example in enumerate(rows):
            record = {
                "id": example["id"], "source_split": split,
                "source_image": str(example["image_path"]),
                "source_annotation": str(example["annotation_path"]),
            }
            try:
                document = extractor.extract(example["image_path"], source_id=example["id"])
                json_path, text_path = save_ocr(document, output_dir / split / "ocr", f"{index:06d}_{example['image_path'].stem}")
                record.update({
                    "status": "ok" if document["has_text"] else "empty_ocr",
                    "ocr_json": json_path.relative_to(output_dir).as_posix(),
                    "ocr_text": text_path.relative_to(output_dir).as_posix(),
                })
                if document["has_text"]:
                    training.append({
                        **build_training_record(example, document, template, schema),
                        "source_split": split, "ocr_json": record["ocr_json"], "ocr_text": record["ocr_text"],
                    })
            except Exception as exc:
                record.update({"status": "ocr_error", "error": f"{type(exc).__name__}: {exc}"})
                print(f"OCR failed for {example['id']}: {record['error']}", file=sys.stderr)
            metadata.append(record)
            print(f"{split} {index + 1}/{len(rows)} {record['status']}: {example['id']}", flush=True)
        write_jsonl(output_dir / split / "metadata.jsonl", metadata)
        write_jsonl(output_dir / f"{split}.jsonl", training)
        summary["splits"][split] = {
            "selected": len(rows), "ok": len(training),
            "empty_ocr": sum(row["status"] == "empty_ocr" for row in metadata),
            "ocr_error": sum(row["status"] == "ocr_error" for row in metadata),
            "metadata": f"{split}/metadata.jsonl", "training_jsonl": f"{split}.jsonl",
        }
    write_json(output_dir / "preparation_summary.json", summary)
    return summary
