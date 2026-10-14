"""Convert existing fusion SFT conversations to a compact, text-only dataset."""
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

from src.fusion.compact import COMPACT_INPUT_INSTRUCTIONS, compact_document
from src.fusion.dataset import _length_summary, _schema_check
from src.fusion.pipeline import to_llm_text

INPUT_MARKER = "OCR/layout representation JSON:\n"
COMPACT_MARKER = "Compact OCR/layout representation JSON:\n"
SYSTEM_MARKER = "\nThe input is OCR/layout representation JSON."
DEFAULT_ROOT = REPO_ROOT / "data/datasets/250_CMRS_240dpi_20260707"


def convert_record(row: dict[str, Any], source: Path, output: Path,
                   decimals: int = 4) -> dict[str, Any]:
    """Preserve sample IDs, targets, answer strings and source artifact references."""
    messages = row["messages"]
    if [item["role"] for item in messages] != ["system", "user", "assistant"]:
        raise ValueError("Expected exactly system/user/assistant messages")
    if any(not isinstance(item["content"], str) for item in messages):
        raise ValueError("Expected plain-string message content")
    prefix, separator, serialized = messages[1]["content"].partition(INPUT_MARKER)
    system_prefix, system_separator, _ = messages[0]["content"].partition(SYSTEM_MARKER)
    if not separator or not system_separator:
        raise ValueError("Expected the original fusion prompt contract")
    document = json.loads(serialized)
    if document["mode"] != row["fusion_mode"]:
        raise ValueError("Record and representation fusion modes disagree")
    if json.loads(messages[2]["content"]) != row["target"]:
        raise ValueError("Assistant answer differs from target")
    compact = to_llm_text(compact_document(document, decimals))
    result = deepcopy(row)
    result["representation"] = "ocr_layout_fusion_compact_v1"
    result["messages"][0]["content"] = system_prefix + COMPACT_INPUT_INSTRUCTIONS
    result["messages"][1]["content"] = prefix + COMPACT_MARKER + compact
    # Provenance stays outside the model input and is relative to the new root.
    for key in ("source_image", "source_annotation", "ocr_json", "layout_json"):
        if key in result.get("provenance", {}):
            artifact = (source / row["provenance"][key]).resolve()
            result["provenance"][key] = os.path.relpath(artifact, output)
    content = [item["content"] for item in result["messages"]]
    result["lengths"] = {
        "fusion_characters": len(compact), "prompt_characters": sum(map(len, content[:2])),
        "target_characters": len(content[2]), "conversation_characters": sum(map(len, content)),
        "conversation_utf8_bytes": sum(len(text.encode("utf-8")) for text in content),
    }
    return result


def compact_dataset(source: Path, output: Path, decimals: int = 4) -> dict[str, Any]:
    """Validate and stage all source splits before publishing a new directory."""
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents:
        raise ValueError("Output must be separate from the source dataset")
    if output.exists():
        raise ValueError(f"Output already exists; choose a new directory: {output}")
    schema = json.loads((source / "schema.json").read_text(encoding="utf-8"))
    input_summary = json.loads((source / "preparation_summary.json").read_text(encoding="utf-8"))
    summary: dict[str, Any] = {
        "format_version": 1, "representation": "ocr_layout_fusion_compact_v1",
        "source_dataset": os.path.relpath(source, output), "decimals": decimals,
        "truncated": False, "schema": "schema.json", "template": "template.json",
        "length_unit": "characters and UTF-8 bytes before the chat template; not token counts",
        "splits": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".fusion-compact-", dir=output.parent))
    try:
        for name in ("schema.json", "template.json"):
            shutil.copyfile(source / name, stage / name)
        seen_ids: dict[str, set[str]] = {}
        for split, split_info in input_summary["splits"].items():
            summary["splits"][split] = {"examples": split_info["examples"], "modes": {}}
            aligned_ids = None
            aligned_targets = None
            for mode, info in split_info["modes"].items():
                if Path(split).name != split or mode not in {"separate", "hard", "soft"}:
                    raise ValueError("Invalid split or mode in preparation summary")
                rows = []
                original_sizes = []
                for line in (source / mode / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row["fusion_mode"] != mode or row["source_split"] != split:
                        raise ValueError("Record mode or split disagrees with its file")
                    converted = convert_record(row, source, output, decimals)
                    _schema_check(converted["target"], schema, f"Target {row['id']}")
                    rows.append(converted)
                    original_sizes.append(sum(len(item["content"]) for item in row["messages"]))
                ids = [row["id"] for row in rows]
                targets = [row["target"] for row in rows]
                if not rows or len(rows) != split_info["examples"]:
                    raise ValueError(f"Unexpected sample count for {split}/{mode}")
                if any(not isinstance(identifier, str) or not identifier for identifier in ids):
                    raise ValueError("Sample IDs must be nonempty strings")
                previously_seen = seen_ids.setdefault(mode, set())
                if len(set(ids)) != len(ids) or previously_seen.intersection(ids):
                    raise ValueError("Duplicate sample IDs within or across splits")
                previously_seen.update(ids)
                if aligned_ids is None:
                    aligned_ids, aligned_targets = ids, targets
                elif ids != aligned_ids or targets != aligned_targets:
                    raise ValueError("Fusion modes must preserve the same sample order and targets")
                relative = Path(mode) / f"{split}.jsonl"
                (stage / relative).parent.mkdir(parents=True, exist_ok=True)
                (stage / relative).write_text(
                    "".join(to_llm_text(row) + "\n" for row in rows), encoding="utf-8")
                stats = {
                    "training_jsonl": relative.as_posix(), "examples": len(rows),
                    "lengths": {key: _length_summary([row["lengths"][key] for row in rows])
                                for key in rows[0]["lengths"]},
                    "original_conversation_characters": _length_summary(original_sizes),
                }
                for key in ("ocr_lines", "layout_regions", "unassigned_ocr_lines", "ambiguous_ocr_lines"):
                    if key in info:
                        stats[key] = info[key]
                summary["splits"][split]["modes"][mode] = stats
        (stage / "preparation_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_ROOT / "fusion_sft")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "fusion_sft_compact")
    parser.add_argument("--decimals", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        summary = compact_dataset(args.input_dir, args.output_dir, args.decimals)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
