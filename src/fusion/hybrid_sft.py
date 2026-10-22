"""Pair existing compact hard-fusion evidence with the original full-page images."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.Qwen.prompts import (
    DEFAULT_SYSTEM_PROMPT, HYBRID_REPRESENTATION, HYBRID_EVIDENCE_MODES,
    build_hybrid_messages,
)
from src.fusion.compact_sft import COMPACT_MARKER, DEFAULT_ROOT
from src.fusion.dataset import _schema_check
from src.fusion.pipeline import to_llm_text


def prepare_hybrid_dataset(source: Path, output: Path, *, dry_run: bool = False,
                           system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                           evidence_modes: Sequence[str] = ("hard",)) -> dict[str, Any]:
    """Validate every pairing and target before atomically publishing a new dataset.

    Source provenance is relative to the compact dataset root, never hard/.
    Images are referenced, not copied or resized. Only train/val are prepared.
    """
    source, output = source.resolve(), output.resolve()
    evidence_modes = tuple(evidence_modes)
    if not evidence_modes or len(set(evidence_modes)) != len(evidence_modes) or any(mode not in HYBRID_EVIDENCE_MODES for mode in evidence_modes):
        raise ValueError("Choose distinct evidence modes from ocr, layout, hard")
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Output must be separate from the source dataset")
    if output.exists():
        raise ValueError(f"Output already exists; choose a new directory: {output}")
    settings = json.loads((source / "preparation_summary.json").read_text(encoding="utf-8"))
    if settings.get("representation") != "ocr_layout_fusion_compact_v1":
        raise ValueError("Expected a compact fusion dataset")
    schema = json.loads((source / "schema.json").read_text(encoding="utf-8"))
    template = json.loads((source / "template.json").read_text(encoding="utf-8"))
    _schema_check(template, schema, "Template")
    summary = {
        "format_version": 1, "representation": HYBRID_REPRESENTATION,
        "source_dataset": os.path.relpath(source, output), "decimals": settings["decimals"],
        "fusion_mode": "hard", "schema": "schema.json", "template": "template.json",
        "evidence_modes": list(evidence_modes),
        "system_prompt": system_prompt, "user_prompt": None,
        "truncated": False, "images_copied": False, "splits": {},
    }
    seen_ids, seen_images = set(), set()
    prepared = {}
    for split in ("train", "val"):
        expected = settings["splits"][split]["examples"]
        rows = {mode: [] for mode in evidence_modes}
        for line in (source / "hard" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("fusion_mode") != "hard" or row.get("source_split") != split:
                raise ValueError("Record mode or split disagrees with its file")
            if row.get("representation") != "ocr_layout_fusion_compact_v1":
                raise ValueError("Expected compact fusion records")
            identifier = row["id"]
            if not isinstance(identifier, str) or not identifier or identifier in seen_ids:
                raise ValueError("Sample IDs must be nonempty and unique across splits")
            seen_ids.add(identifier)
            original = row["messages"]
            if [m["role"] for m in original] != ["system", "user", "assistant"]:
                raise ValueError("Expected system/user/assistant messages")
            if any(not isinstance(m["content"], str) for m in original):
                raise ValueError("Expected plain-string compact source conversations")
            user, marker, serialized = original[1]["content"].partition(COMPACT_MARKER)
            if not marker:
                raise ValueError("Missing compact evidence marker")
            user = user.rstrip()
            if summary["user_prompt"] is None:
                summary["user_prompt"] = user
            elif user != summary["user_prompt"]:
                raise ValueError("All examples must share the same task instructions")
            document = json.loads(serialized)
            if document["mode"] != "hard" or len(document["pages"]) != 1:
                raise ValueError("Hybrid training requires one hard-fusion page per image")
            if document.get("coordinates") != "normalized_0_1":
                raise ValueError("Expected normalized compact coordinates")
            target = json.loads(original[2]["content"])
            if target != row["target"]:
                raise ValueError("Assistant answer differs from target")
            _schema_check(target, schema, f"Target {identifier}")
            image = (source / row["provenance"]["source_image"]).resolve()
            if not image.is_file() or image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
                raise ValueError(f"Missing source raster image for {identifier}: {image}")
            if image in seen_images:
                raise ValueError("An image is reused within or across splits")
            seen_images.add(image)
            result = deepcopy(row)
            result["representation"] = HYBRID_REPRESENTATION
            result["image"] = os.path.relpath(image, output)
            # Keep artifact provenance portable when the dataset folder moves with its parent.
            for key in ("source_image", "source_annotation", "ocr_json", "layout_json"):
                if key in result["provenance"]:
                    result["provenance"][key] = os.path.relpath(
                        (source / row["provenance"][key]).resolve(), output)
            result.pop("lengths", None)  # Source text-only lengths no longer describe this input.
            for mode in evidence_modes:
                variant = deepcopy(result)
                variant["evidence_mode"] = mode
                variant["messages"] = build_hybrid_messages(
                    system_prompt, user, document, template, schema, compact=True, evidence_mode=mode)
                variant["messages"].append(deepcopy(original[2]))
                rows[mode].append(variant)
        count = len(rows[evidence_modes[0]])
        if not count or count != expected:
            raise ValueError(f"Unexpected sample count for {split}: {count} versus {expected}")
        prepared[split] = rows
        summary["splits"][split] = {"examples": count,
            "modes": {mode: {"training_jsonl": f"{mode}/{split}.jsonl", "examples": count} for mode in evidence_modes}}
        if evidence_modes == ("hard",):
            summary["splits"][split]["training_jsonl"] = f"hard/{split}.jsonl"
    if dry_run:
        return summary
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".fusion-hybrid-", dir=output.parent))
    try:
        for name in ("schema.json", "template.json"):
            shutil.copyfile(source / name, stage / name)
        for mode in evidence_modes:
            (stage / mode).mkdir()
            for split, variants in prepared.items():
                (stage / mode / f"{split}.jsonl").write_text(
                    "".join(to_llm_text(row) + "\n" for row in variants[mode]), encoding="utf-8")
        (stage / "preparation_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_ROOT / "fusion_sft_compact")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "fusion_sft_hybrid")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--evidence-modes", nargs="+", choices=HYBRID_EVIDENCE_MODES, default=["hard"],
                        help="Prepare image+OCR, image+layout, and/or image+hard-fusion evidence from aligned source records.")
    args = parser.parse_args(argv)
    try:
        summary = prepare_hybrid_dataset(args.input_dir, args.output_dir,
                                         dry_run=args.dry_run, system_prompt=args.system_prompt,
                                         evidence_modes=args.evidence_modes)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
