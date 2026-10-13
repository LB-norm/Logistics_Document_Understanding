"""Prepare split-preserving, text-only SFT records from saved OCR and layouts."""
from __future__ import annotations

from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

from src.eval_suite.schema import validate_json_schema
from src.Qwen.prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT, build_fusion_messages

from .pipeline import FusionConfig, FusionPipeline, MODE_NAMES, to_llm_text


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            records.append(row)
    if not records:
        raise ValueError(f"No records in {path}")
    return records


def _index(records: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    indexed = {}
    for row in records:
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in indexed:
            raise ValueError(f"{name}: sample IDs must be nonempty strings and unique")
        indexed[identifier] = row
    return indexed


def _linked_path(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Expected a nonempty artifact path")
    path = Path(value)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _project_path(root: Path, split: str, value: Any) -> Path:
    # Project metadata permits dataset-root-relative and split-relative paths.
    if isinstance(value, str) and not Path(value).is_absolute() and not (root / value).is_file():
        return _linked_path(root / split, value)
    return _linked_path(root, value)


def _relative(path: Path, output: Path) -> str:
    return Path(os.path.relpath(path, output)).as_posix()


def _target(annotation: Any, key: str) -> Any:
    if key == "root":
        return annotation
    for part in key.split("."):
        if not isinstance(annotation, dict) or part not in annotation:
            raise ValueError(f"Annotation is missing target key {key!r}")
        annotation = annotation[part]
    return annotation


def _same_filename(value: Any, image: Path, name: str) -> None:
    if not isinstance(value, str) or Path(value).name != image.name:
        raise ValueError(f"{name} does not match source image {image.name}")


def _schema_check(value: Any, schema: Any, name: str) -> None:
    errors = validate_json_schema(value, schema)
    if errors:
        raise ValueError(f"{name} fails schema validation: {'; '.join(errors[:5])}")


def _length_summary(values: list[int]) -> dict[str, int | float]:
    return {"min": min(values), "mean": round(sum(values) / len(values), 2), "max": max(values)}


def prepare_fusion_dataset(
    dataset_root: Path,
    output_dir: Path,
    *,
    ocr_dir: Path | None = None,
    layout_dir: Path | None = None,
    splits: Sequence[str] = ("train", "val"),
    modes: Sequence[int | str] = (1, 2, 3),
    min_overlap: float = 0.1,
    allow_layout_rescale: bool = False,
    annotation_target_key: str | None = None,
    schema_path: Path | None = None,
    template_path: Path | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate all requested data before publishing any training files.

    Source IDs, targets and split membership must agree exactly. This builder
    handles one source raster per sample; multi-page documents require an explicit
    page manifest rather than inferring page associations from list positions.
    No model, image decoder, tokenizer or ML package is loaded.
    """
    dataset_root, output_dir = Path(dataset_root).resolve(), Path(output_dir).resolve()
    ocr_dir = (Path(ocr_dir) if ocr_dir is not None else dataset_root / "ocr_v6_cmr").resolve()
    layout_dir = (Path(layout_dir) if layout_dir is not None
                  else dataset_root / "yolo_v11s/layout_detections").resolve()
    if not splits or len(set(splits)) != len(splits):
        raise ValueError("Specify at least one split, without duplicates")
    for split in splits:
        if not split or Path(split).name != split or split in {".", ".."}:
            raise ValueError("Split names must be single directory names")
    configs = [FusionConfig(mode, min_overlap, allow_layout_rescale) for mode in modes]
    if not configs or len({config.mode for config in configs}) != len(configs):
        raise ValueError("Specify at least one fusion mode, without duplicates or equivalent aliases")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer")
    protected = [ocr_dir, layout_dir, *((dataset_root / split).resolve() for split in splits)]
    if (output_dir == dataset_root or output_dir in dataset_root.parents
            or any(output_dir == path or output_dir in path.parents or path in output_dir.parents
                   for path in protected)):
        raise ValueError("Output directory must not overlap OCR, layout or source split directories")
    if output_dir.exists() and (not output_dir.is_dir()
                                or (any(output_dir.iterdir()) and not overwrite and not dry_run)):
        raise FileExistsError(f"Output directory is not empty: {output_dir}; use --overwrite")
    generated = [Path(name) for name in ("schema.json", "template.json", "preparation_summary.json")]
    generated.extend(Path(MODE_NAMES[c.mode]) / f"{split}.jsonl" for c in configs for split in splits)
    for relative in generated:
        destination = output_dir / relative
        if (output_dir not in destination.resolve().parents or destination.is_dir()
                or destination.is_symlink()
                or any(parent.is_symlink() for parent in destination.parents
                       if parent != output_dir and output_dir in parent.parents)):
            raise ValueError(f"Unsafe or invalid generated output path: {destination}")

    schema_path = Path(schema_path).resolve() if schema_path is not None else ocr_dir / "schema.json"
    template_path = Path(template_path).resolve() if template_path is not None else ocr_dir / "template.json"
    for path in (schema_path, template_path):
        if output_dir == path or output_dir in path.parents:
            raise ValueError("Schema and template must be outside the output directory")
    schema, template = _json(schema_path), _json(template_path)
    _schema_check(template, schema, "Output template")
    if annotation_target_key is None:
        preparation = ocr_dir / "preparation_summary.json"
        annotation_target_key = (_json(preparation).get("annotation_target_key", "content")
                                 if preparation.is_file() else "content")
    if not isinstance(annotation_target_key, str) or not annotation_target_key:
        raise ValueError("annotation_target_key must be a nonempty string")

    # Index layouts once. Exact split membership and unique filenames make the
    # filename fallback safe even when OCR provenance points to another machine.
    prepared = {}
    seen_ids: set[str] = set()
    seen_images: set[Path] = set()
    for split in splits:
        training = _jsonl(ocr_dir / f"{split}.jsonl")
        training_by_id = _index(training, f"{split} OCR training")
        metadata = _index(_jsonl(ocr_dir / split / "metadata.jsonl"), f"{split} OCR metadata")
        source = _index(_jsonl(dataset_root / split / "metadata.jsonl"), f"{split} source metadata")
        if set(training_by_id) != set(metadata) or set(metadata) != set(source):
            raise ValueError(f"{split}: source, OCR metadata and OCR training sample IDs must match exactly")
        if seen_ids.intersection(source):
            raise ValueError(f"{split}: sample IDs occur in multiple splits")
        seen_ids.update(source)
        layout_path = layout_dir / f"{split}.json"
        predictions = _json(layout_path)
        if not isinstance(predictions, list):
            raise ValueError(f"{layout_path}: expected a YOLO prediction list")
        layouts = {}
        for prediction in predictions:
            if not isinstance(prediction, dict) or not isinstance(prediction.get("image"), str):
                raise ValueError(f"{layout_path}: each prediction requires an image path")
            filename = Path(prediction["image"]).name
            if not filename or filename in layouts:
                raise ValueError(f"{layout_path}: ambiguous or duplicate image filename {filename!r}")
            layouts[filename] = prediction
        images = {identifier: _project_path(dataset_root, split, row.get("image"))
                  for identifier, row in source.items()}
        filenames = [image.name for image in images.values()]
        if len(set(filenames)) != len(filenames):
            raise ValueError(f"{split}: source image filenames must be unique")
        if len(set(images.values())) != len(images) or seen_images.intersection(images.values()):
            raise ValueError(f"{split}: source images are reused by multiple samples or splits")
        seen_images.update(images.values())
        if set(filenames) != set(layouts):
            raise ValueError(f"{split}: YOLO images must exactly match the source split images")
        prepared[split] = (training[:limit], metadata, source, images, layouts, layout_path)

    summary: dict[str, Any] = {
        "format_version": 1, "representation": "ocr_layout_fusion_sft", "dry_run": dry_run,
        "sources": {"dataset_root": _relative(dataset_root, output_dir),
                    "ocr_dir": _relative(ocr_dir, output_dir), "layout_dir": _relative(layout_dir, output_dir)},
        "settings": {"modes": [MODE_NAMES[c.mode] for c in configs], "min_overlap": min_overlap,
                     "allow_layout_rescale": allow_layout_rescale,
                     "annotation_target_key": annotation_target_key, "limit_per_split": limit,
                     "truncated": False},
        "length_unit": "characters and UTF-8 bytes; excludes chat-template tokens; not token counts",
        "schema": "schema.json", "template": "template.json", "splits": {},
    }
    # Staging avoids partial datasets on late pairing/geometry/target errors, and
    # streams rows rather than holding all three complete datasets in memory.
    with ExitStack() as stack:
        stage = None
        if not dry_run:
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            stage = Path(stack.enter_context(tempfile.TemporaryDirectory(
                prefix=".fusion-sft-", dir=output_dir.parent)))
        for split, (training, metadata, source, images, layouts, layout_path) in prepared.items():
            split_summary: dict[str, Any] = {"examples": len(training), "modes": {}}
            summary["splits"][split] = split_summary
            handles = {}
            lengths = {}
            for config in configs:
                mode = MODE_NAMES[config.mode]
                lengths[mode] = {key: [] for key in ("fusion_characters", "prompt_characters",
                                                    "target_characters", "conversation_characters",
                                                    "conversation_utf8_bytes")}
                split_summary["modes"][mode] = {
                    "training_jsonl": f"{mode}/{split}.jsonl", "ocr_lines": 0,
                    "layout_regions": 0, "unassigned_ocr_lines": None if config.mode == 1 else 0,
                    "ambiguous_ocr_lines": None if config.mode == 1 else 0,
                }
                if stage is not None:
                    path = stage / mode / f"{split}.jsonl"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    handles[mode] = stack.enter_context(path.open("w", encoding="utf-8"))
            for row in training:
                identifier = row["id"]
                try:
                    meta, original, image = metadata[identifier], source[identifier], images[identifier]
                    if meta.get("status") != "ok":
                        raise ValueError("OCR status must be ok; failed/empty samples cannot be silently dropped")
                    if row.get("source_split") != split or meta.get("source_split") != split:
                        raise ValueError("OCR source_split differs from the requested split")
                    _same_filename(meta.get("source_image"), image, "OCR metadata source_image")
                    ocr_path = _linked_path(ocr_dir, row.get("ocr_json"))
                    if ocr_path != _linked_path(ocr_dir, meta.get("ocr_json")):
                        raise ValueError("OCR metadata and training artifact links disagree")
                    document = _json(ocr_path)
                    if document.get("has_text") is False:
                        raise ValueError("OCR document is marked as empty")
                    if document.get("source_id") != identifier:
                        raise ValueError("OCR document source_id differs from the sample ID")
                    _same_filename(document.get("source_path"), image, "OCR document source_path")
                    if len(document.get("pages", [])) != 1:
                        raise ValueError("Batch SFT requires one raster page per sample; use an explicit page manifest for multi-page documents")
                    annotation_path = _project_path(dataset_root, split, original.get("annotation"))
                    if any(output_dir in path.parents for path in (ocr_path, annotation_path, image)):
                        raise ValueError("Source artifacts must be outside the output directory")
                    target = _target(_json(annotation_path), annotation_target_key)
                    if "target" not in row or row["target"] != target:
                        raise ValueError("OCR target differs from the source annotation")
                    assistant = row.get("messages", [])[-1:]
                    if (not assistant or assistant[0].get("role") != "assistant"
                            or json.loads(assistant[0]["content"]) != target):
                        raise ValueError("OCR assistant answer differs from the source annotation")
                    _schema_check(target, schema, "Target")
                    for config in configs:
                        mode = MODE_NAMES[config.mode]
                        fused = FusionPipeline(config).fuse(document, layouts[image.name])
                        prompts = build_fusion_messages(DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT,
                                                        fused, template, schema)
                        messages = [{"role": p["role"], "content": p["content"][0]["text"]}
                                    for p in prompts]
                        messages.append({"role": "assistant", "content": json.dumps(
                            target, ensure_ascii=False, allow_nan=False)})
                        prompt_chars = sum(len(p["content"]) for p in messages[:-1])
                        sizes = {"fusion_characters": len(to_llm_text(fused)),
                                 "prompt_characters": prompt_chars,
                                 "target_characters": len(messages[-1]["content"]),
                                 "conversation_characters": sum(len(p["content"]) for p in messages),
                                 "conversation_utf8_bytes": sum(len(p["content"].encode("utf-8")) for p in messages)}
                        for key, value in sizes.items():
                            lengths[mode][key].append(value)
                        record = {
                            "id": identifier, "source_split": split, "fusion_mode": mode,
                            "messages": messages, "target": target, "lengths": sizes,
                            "provenance": {"source_image": _relative(image, output_dir),
                                           "source_annotation": _relative(annotation_path, output_dir),
                                           "ocr_json": _relative(ocr_path, output_dir),
                                           "layout_json": _relative(layout_path, output_dir),
                                           "layout_image": layouts[image.name]["image"]},
                        }
                        encoded = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                        if stage is not None:
                            handles[mode].write(encoded + "\n")
                        stats = split_summary["modes"][mode]
                        for page in fused["pages"]:
                            stats["ocr_lines"] += len(page["ocr"]["lines"])
                            stats["layout_regions"] += len(page["layout"]["detections"])
                            if config.mode != 1:
                                stats["unassigned_ocr_lines"] += len(page["unassigned_ocr_ids"])
                                stats["ambiguous_ocr_lines"] += sum(len(m["candidates"]) > 1 for m in page["memberships"])
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    raise ValueError(f"{split}/{identifier}: {exc}") from exc
            for mode, values in lengths.items():
                split_summary["modes"][mode]["lengths"] = {key: _length_summary(v) for key, v in values.items()}
            for handle in handles.values():
                handle.close()
        if stage is not None:
            for name, data in (("schema.json", schema), ("template.json", template),
                               ("preparation_summary.json", summary)):
                (stage / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                          encoding="utf-8")
            output_dir.mkdir(parents=True, exist_ok=True)
            # Replace only generated files. Unknown/stale files are never deleted;
            # preparation_summary.json declares the current dataset membership.
            files = sorted(stage.rglob("*"))
            for path in files:
                if path.is_file() and path.name != "preparation_summary.json":
                    destination = output_dir / path.relative_to(stage)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    path.replace(destination)
            (stage / "preparation_summary.json").replace(output_dir / "preparation_summary.json")
    return summary
