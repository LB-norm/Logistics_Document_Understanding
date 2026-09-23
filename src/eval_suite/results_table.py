"""Compare all model-output directories in an experiment folder.

The command in this module is the folder-level counterpart to
``python -m src.eval_suite --predictions ...``.  It evaluates every immediate
subdirectory as one model/run and writes comparison tables containing overall,
default-subset, challenge-subset, and challenge-minus-default metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.Qwen.run_inference import (
    extract_json_fragment,
    fill_from_template,
    recover_complete_top_level_fields,
    strip_thinking,
)

from .__main__ import load_testset_directory, sample_id_from_path
from .evaluator import JsonEvaluator, TestsetEvaluationReport
from .paths import DEFAULT_TESTSET_PATH, REPO_ROOT


DEFAULT_SCHEMA_PATH = REPO_ROOT / "json_schema" / "content.schema.json"
METRIC_NAMES = (
    "parse_rate",
    "schema_valid_rate",
    "document_exact_match_rate",
    "field_precision",
    "field_recall",
    "field_f1",
    "value_similarity",
)
SUBSET_NAMES = ("overall", "default", "challenge")


@dataclass(frozen=True)
class RunEvaluation:
    """Evaluation result for one model/run directory."""

    run_name: str
    run_directory: Path
    report: TestsetEvaluationReport
    manifest_records: dict[str, dict[str, Any]] | None = None


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_manifest_records(run_directory: Path) -> dict[str, dict[str, Any]] | None:
    """Load inference diagnostics keyed by the same sample IDs as predictions."""

    manifest_path = run_directory / "inference_manifest.jsonl"
    if not manifest_path.is_file():
        return None

    records: dict[str, dict[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(
                    f"{manifest_path}:{line_number} must contain a JSON object"
                )
            image_path = record.get("image_path")
            if not isinstance(image_path, str):
                raise ValueError(
                    f"{manifest_path}:{line_number} must contain an image_path"
                )
            sample_id = sample_id_from_path(Path(image_path))
            if sample_id in records:
                raise ValueError(
                    f"Duplicate sample {sample_id!r} in {manifest_path}"
                )
            records[sample_id] = record
    return records


def _recover_manifest_predictions(
    predictions: list[Any],
    sample_ids: Sequence[str],
    manifest_records: dict[str, dict[str, Any]] | None,
) -> list[Any]:
    """Recover complete top-level fields from legacy all-null fallbacks.

    Recovery is intentionally limited to raw generations already marked as
    parse failures. The original prediction object supplies the target shape.
    Historical prediction and manifest files are never modified.
    """

    if manifest_records is None:
        return predictions

    recovered_predictions = list(predictions)
    for index, (prediction, sample_id) in enumerate(
        zip(predictions, sample_ids, strict=True)
    ):
        record = manifest_records.get(sample_id)
        if record is None or record.get("status") != "parse_error":
            continue
        raw_text = record.get("raw_text")
        if not isinstance(raw_text, str):
            continue
        fragment, _ = extract_json_fragment(strip_thinking(raw_text))
        if fragment is None:
            continue
        recovered, recovered_fields = recover_complete_top_level_fields(
            fragment, prediction
        )
        if recovered_fields:
            recovered_predictions[index] = fill_from_template(prediction, recovered)
    return recovered_predictions


def discover_run_directories(input_folder: Path) -> list[Path]:
    """Return the immediate model/run directories below ``input_folder``.

    A directory qualifies when it contains JSON files.  If the input directory
    itself contains predictions but no qualifying children, it is treated as a
    single run.  Generated comparison tables therefore do not affect discovery.
    """

    input_folder = input_folder.resolve()
    if not input_folder.is_dir():
        raise NotADirectoryError(f"Results folder not found: {input_folder}")

    child_runs = sorted(
        (
            child
            for child in input_folder.iterdir()
            if child.is_dir() and any(child.rglob("*.json"))
        ),
        key=lambda path: path.name.casefold(),
    )
    if child_runs:
        return child_runs
    if any(input_folder.glob("*.json")):
        return [input_folder]
    raise ValueError(
        f"No model/run directories containing JSON predictions found in "
        f"{input_folder}"
    )


def evaluate_results_folder(
    input_folder: Path,
    *,
    testset_path: Path = DEFAULT_TESTSET_PATH,
    schema_path: Path | None = DEFAULT_SCHEMA_PATH,
    prediction_key: str = "root",
    ground_truth_key: str = "content",
) -> list[RunEvaluation]:
    """Evaluate each model/run in an experiment folder against its annotations."""

    schema = _load_json(schema_path) if schema_path is not None else None
    evaluator = JsonEvaluator(schema=schema)
    evaluations: list[RunEvaluation] = []

    for run_directory in discover_run_directories(input_folder):
        manifest_records = _load_manifest_records(run_directory)
        predictions, ground_truths, sample_ids, subset_labels = (
            load_testset_directory(
                run_directory,
                testset_path,
                prediction_key=prediction_key,
                ground_truth_key=ground_truth_key,
            )
        )
        predictions = _recover_manifest_predictions(
            predictions, sample_ids, manifest_records
        )
        report = evaluator.evaluate_testset(
            predictions,
            ground_truths,
            subset_labels=subset_labels,
            sample_ids=sample_ids,
        )
        evaluations.append(
            RunEvaluation(
                run_name=run_directory.name,
                run_directory=run_directory,
                report=report,
                manifest_records=manifest_records,
            )
        )

    return sorted(
        evaluations,
        key=lambda result: (
            -result.report.overall.summary()["field_f1"],
            result.run_name.casefold(),
        ),
    )


def _summary_for(result: RunEvaluation, subset: str) -> dict[str, Any]:
    if subset == "overall":
        report = result.report.overall
    else:
        report = result.report.subsets[subset]
    summary = report.summary()

    if result.manifest_records is None:
        return summary

    records: list[dict[str, Any]] = []
    for sample in report.samples:
        if sample.sample_id not in result.manifest_records:
            raise ValueError(
                f"Inference manifest in {result.run_directory} has no record for "
                f"sample {sample.sample_id!r}"
            )
        records.append(result.manifest_records[sample.sample_id])

    # Prediction files contain normalized JSON and are therefore parseable even
    # when raw generation failed. Raw contract metrics must come from diagnostics.
    summary["parse_rate"] = sum(
        record.get("status") not in {"parse_error", "inference_error"}
        for record in records
    ) / len(records)
    summary["schema_valid_rate"] = sum(
        record.get("status") == "ok" for record in records
    ) / len(records)
    return summary


def _comparison_for(result: RunEvaluation) -> dict[str, float | None]:
    default = _summary_for(result, "default")
    challenge = _summary_for(result, "challenge")
    return {
        metric: (
            float(challenge[metric]) - float(default[metric])
            if challenge[metric] is not None and default[metric] is not None
            else None
        )
        for metric in METRIC_NAMES
    }


def flatten_result(result: RunEvaluation) -> dict[str, str | int | float | None]:
    """Convert one result into a flat row suitable for CSV/dataframe loading."""

    row: dict[str, str | int | float | None] = {
        "run": result.run_name,
        "run_directory": str(result.run_directory),
    }
    for subset in SUBSET_NAMES:
        summary = _summary_for(result, subset)
        row[f"{subset}_samples"] = summary["samples"]
        for metric in METRIC_NAMES:
            row[f"{subset}_{metric}"] = summary[metric]
    for metric, value in _comparison_for(result).items():
        row[f"challenge_minus_default_{metric}"] = value
    return row


def _format_csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6f}"
    return "" if value is None else value


def write_csv(evaluations: Sequence[RunEvaluation], output_path: Path) -> None:
    """Write the complete flat comparison table as CSV."""

    rows = [flatten_result(result) for result in evaluations]
    if not rows:
        raise ValueError("Cannot write an empty results table.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {key: _format_csv_value(value) for key, value in row.items()}
            for row in rows
        )


def _markdown_value(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:+.4f}" if signed else f"{value:.4f}"
    return str(value)


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    alignment = "| " + " | ".join(
        [":---"] + ["---:"] * (len(headers) - 1)
    ) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, alignment, *body])


def write_markdown(
    evaluations: Sequence[RunEvaluation],
    output_path: Path,
    *,
    input_folder: Path,
) -> None:
    """Write readable overall/subset/gap tables as Markdown."""

    metric_headers = [metric.replace("_", " ").title() for metric in METRIC_NAMES]
    sections = [
        "# Evaluation results",
        "",
        f"Input folder: `{input_folder.resolve()}`  ",
        f"Runs: {len(evaluations)}  ",
        "Rows are ranked by overall field F1 (highest first).",
        "Raw parse/schema rates come from inference diagnostics; field metrics use "
        "conservatively recovered complete top-level fields when available.",
    ]

    for subset in SUBSET_NAMES:
        sections.extend(
            [
                "",
                f"## {subset.title()}",
                "",
                _markdown_table(
                    ["Run", "Samples", *metric_headers],
                    (
                        [
                            result.run_name.replace("|", "\\|"),
                            str(_summary_for(result, subset)["samples"]),
                            *[
                                _markdown_value(_summary_for(result, subset)[metric])
                                for metric in METRIC_NAMES
                            ],
                        ]
                        for result in evaluations
                    ),
                ),
            ]
        )

    sections.extend(
        [
            "",
            "## Challenge minus default",
            "",
            "Negative values indicate lower performance on challenge files.",
            "",
            _markdown_table(
                ["Run", *metric_headers],
                (
                    [
                        result.run_name.replace("|", "\\|"),
                        *[
                            _markdown_value(
                                _comparison_for(result)[metric], signed=True
                            )
                            for metric in METRIC_NAMES
                        ],
                    ]
                    for result in evaluations
                ),
            ),
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections) + "\n", encoding="utf-8")


def write_results_tables(
    evaluations: Sequence[RunEvaluation],
    *,
    input_folder: Path,
    output_directory: Path | None = None,
    output_stem: str = "evaluation_table",
) -> tuple[Path, Path]:
    """Write CSV and Markdown tables and return their paths."""

    target = output_directory or input_folder
    csv_path = target / f"{output_stem}.csv"
    markdown_path = target / f"{output_stem}.md"
    write_csv(evaluations, csv_path)
    write_markdown(evaluations, markdown_path, input_folder=input_folder)
    return csv_path, markdown_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every model/run directory in a results folder and create "
            "overall, default, and challenge comparison tables."
        )
    )
    parser.add_argument(
        "input_folder",
        type=Path,
        help="Folder whose immediate subdirectories contain model predictions.",
    )
    parser.add_argument(
        "--testset-path",
        type=Path,
        default=DEFAULT_TESTSET_PATH,
        help=f"Test-set root (default: {DEFAULT_TESTSET_PATH}).",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
        help=f"JSON schema used for schema-valid rate (default: {DEFAULT_SCHEMA_PATH}).",
    )
    parser.add_argument(
        "--no-schema",
        action="store_true",
        help="Do not calculate schema validity.",
    )
    parser.add_argument(
        "--prediction-key",
        default="root",
        help="Dotted path to the predicted object (default: root).",
    )
    parser.add_argument(
        "--ground-truth-key",
        default="content",
        help="Dotted path inside annotation files (default: content).",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="Output directory (default: the input folder).",
    )
    parser.add_argument(
        "--output-stem",
        default="evaluation_table",
        help="Output filename without extension (default: evaluation_table).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    schema_path = None if args.no_schema else args.schema
    evaluations = evaluate_results_folder(
        args.input_folder,
        testset_path=args.testset_path,
        schema_path=schema_path,
        prediction_key=args.prediction_key,
        ground_truth_key=args.ground_truth_key,
    )
    csv_path, markdown_path = write_results_tables(
        evaluations,
        input_folder=args.input_folder,
        output_directory=args.output_directory,
        output_stem=args.output_stem,
    )
    print(f"Evaluated {len(evaluations)} run(s).")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
