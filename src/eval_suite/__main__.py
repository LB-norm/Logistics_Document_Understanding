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


def _pair_value(
    record: dict[str, Any],
    *,
    value_key: str,
    path_key: str,
    key_path: str,
    manifest_path: Path,
    record_number: int,
) -> Any:
    """Load an embedded pair value or a JSON file referenced by the manifest."""
    present = [key for key in (value_key, path_key) if key in record]
    if len(present) != 1:
        raise ValueError(
            f"Pair record {record_number} must contain exactly one of "
            f"{value_key!r} or {path_key!r}."
        )
    if value_key in record:
        value = record[value_key]
        source = f"{manifest_path}:{record_number}:{value_key}"
    else:
        raw_path = record[path_key]
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(
                f"Pair record {record_number} field {path_key!r} must be a path string."
            )
        path = Path(raw_path)
        if not path.is_absolute():
            path = manifest_path.parent / path
        value = _load_json(path)
        source = str(path)
    return _select(value, key_path, source=source)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare predicted JSON with annotated ground-truth JSON."
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--pairs",
        type=Path,
        help=(
            "JSONL file with embedded prediction/ground_truth values or "
            "prediction_path/ground_truth_path references and an optional sample_id."
        ),
    )
    inputs.add_argument(
        "--testset-pairs",
        type=Path,
        help=(
            "JSONL manifest for held-out evaluation. Every row must identify its "
            "subset as 'default' or 'challenge'; both subsets must be present."
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

    pairs_path = args.testset_pairs or args.pairs
    if pairs_path:
        records = _load_jsonl(pairs_path)
        predictions = [
            _pair_value(
                record,
                value_key="prediction",
                path_key="prediction_path",
                key_path=args.prediction_key,
                manifest_path=pairs_path,
                record_number=index,
            )
            for index, record in enumerate(records, start=1)
        ]
        ground_truths = [
            _pair_value(
                record,
                value_key="ground_truth",
                path_key="ground_truth_path",
                key_path=args.ground_truth_key,
                manifest_path=pairs_path,
                record_number=index,
            )
            for index, record in enumerate(records, start=1)
        ]
        sample_ids = [
            str(record.get("sample_id", index))
            for index, record in enumerate(records, start=1)
        ]
        if args.testset_pairs:
            missing_subset = [
                index
                for index, record in enumerate(records, start=1)
                if "subset" not in record
            ]
            if missing_subset:
                raise ValueError(
                    "Every test-set pair requires a subset; missing in record(s): "
                    + ", ".join(map(str, missing_subset))
                )
            report = evaluator.evaluate_testset(
                predictions,
                ground_truths,
                subset_labels=[record["subset"] for record in records],
                sample_ids=sample_ids,
            )
        else:
            report = evaluator.evaluate_batch(
                predictions,
                ground_truths,
                sample_ids=sample_ids,
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
