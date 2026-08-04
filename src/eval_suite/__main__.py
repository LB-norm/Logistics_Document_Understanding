"""Command-line entry point for evaluating saved JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evaluator import JsonEvaluator


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            records.append(record)
    return records


def _select(value: Any, key_path: str, *, source: str) -> Any:
    if key_path in {"", ".", "root"}:
        return value
    selected = value
    for key in key_path.split("."):
        if not isinstance(selected, dict) or key not in selected:
            raise KeyError(f"{source} does not contain key path {key_path!r}")
        selected = selected[key]
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare predicted JSON with annotated ground-truth JSON."
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--pairs",
        type=Path,
        help=(
            "JSONL file with prediction and ground_truth fields and an optional "
            "sample_id field. A prediction may itself be a JSON string."
        ),
    )
    inputs.add_argument("--prediction", type=Path, help="One predicted JSON file.")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        help="Ground-truth JSON file; required with --prediction.",
    )
    parser.add_argument("--schema", type=Path, help="Optional target JSON Schema.")
    parser.add_argument(
        "--prediction-key",
        default="root",
        help="Optional dotted path to the predicted object inside each input (default: root).",
    )
    parser.add_argument(
        "--ground-truth-key",
        default="root",
        help="Optional dotted path such as 'content' inside each annotation (default: root).",
    )
    parser.add_argument("--output", type=Path, help="Optional report output path.")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Omit per-sample records from the report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.prediction is not None and args.ground_truth is None:
        raise ValueError("--ground-truth is required with --prediction")
    schema = _load_json(args.schema) if args.schema else None
    evaluator = JsonEvaluator(schema=schema)

    if args.pairs:
        records = _load_jsonl(args.pairs)
        for index, record in enumerate(records, start=1):
            missing = {"prediction", "ground_truth"} - record.keys()
            if missing:
                raise ValueError(
                    f"Pair record {index} is missing: {', '.join(sorted(missing))}"
                )
        report = evaluator.evaluate_batch(
            [
                _select(record["prediction"], args.prediction_key, source="prediction")
                for record in records
            ],
            [
                _select(record["ground_truth"], args.ground_truth_key, source="ground truth")
                for record in records
            ],
            sample_ids=[str(record.get("sample_id", index)) for index, record in enumerate(records)],
        )
    else:
        report = evaluator.evaluate_batch(
            [
                _select(
                    _load_json(args.prediction),
                    args.prediction_key,
                    source=str(args.prediction),
                )
            ],
            [
                _select(
                    _load_json(args.ground_truth),
                    args.ground_truth_key,
                    source=str(args.ground_truth),
                )
            ],
            sample_ids=[args.prediction.stem],
        )

    rendered = json.dumps(
        report.to_dict(include_samples=not args.summary_only),
        ensure_ascii=False,
        indent=2,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
