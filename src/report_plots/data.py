"""Read evaluation snapshots and inference diagnostics without changing artifacts."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from src.eval_suite.paths import DEFAULT_TESTSET_PATH, REPO_ROOT

METRICS = ("field_f1", "field_precision", "field_recall", "value_similarity",
           "parse_rate", "schema_valid_rate", "document_exact_match_rate")
SUBSETS = ("overall", "default", "challenge")
MEMORY_KINDS = ("process", "torch_allocated", "torch_reserved", "paddle_allocated", "paddle_reserved")


def sample_id(value: str) -> str:
    """Match the eval suite's annotation/image IDs, including Windows paths."""
    stem = Path(value.replace("\\", "/")).stem.removeprefix("gt_")
    return re.sub(r"_\d+dpi$", "", stem)


def number(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{context}: boolean is not a measurement")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: expected a finite number, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{context}: expected a finite number, got {value!r}")
    return result


@dataclass
class Measurement:
    sample_id: str
    status: str
    subset: str | None
    seconds: float | None
    generation_seconds: float | None
    stages: dict[str, float]
    memory: dict[str, float]
    devices: tuple[str, ...] = ()

    @property
    def completed(self) -> bool:
        # Parse/schema failures still completed inference; infrastructure failures did not.
        return self.status in {"ok", "parse_error", "schema_error", "template_error", "template_mismatch"}


@dataclass
class RunResult:
    key: str
    label: str
    group: str
    metrics: dict[str, dict[str, float]]
    counts: dict[str, int]
    measurements: list[Measurement] = field(default_factory=list)
    fields: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    samples: dict[str, dict[str, Any]] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def metric(self, metric: str, subset: str = "overall") -> float:
        try:
            return self.metrics[subset][metric]
        except KeyError as exc:
            raise ValueError(f"{self.key}: missing {subset}.{metric}") from exc


def _subset_ids(testset_path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for subset in SUBSETS[1:]:
        for path in (testset_path / "annotations" / "ground_truths" / subset).glob("*.json"):
            key = sample_id(path.name)
            if key in result:
                raise ValueError(f"Duplicate annotation ID: {key}")
            result[key] = subset
    return result


def _memory(vram: dict[str, Any]) -> dict[str, float]:
    result = {}

    def peak(block: dict[str, Any], kind: str, name: str) -> None:
        if not block.get("available", False):
            return
        values = []
        for device in block.get("devices", []):
            raw = device.get(kind + "_bytes")
            if raw is not None:
                value = number(raw, context=name) / 1024**3
            elif device.get(kind + "_gib") is not None:
                value = number(device[kind + "_gib"], context=name)
            else:
                continue
            if value < 0:
                raise ValueError(f"{name}: negative VRAM")
            values.append(value)
        if values:
            # Max per-device peak: never sum peaks that occurred at different times.
            result[name] = max(values)

    peak(vram, "peak_used", "process")
    for framework in ("torch", "paddle"):
        block = vram.get("framework_peaks", {}).get(framework, {})
        if not block and framework in str(vram.get("source", "")).lower():
            block = vram
        for kind in ("allocated", "reserved"):
            peak(block, "peak_" + kind, framework + "_" + kind)
    return result


def read_manifest(path: Path, subsets: Mapping[str, str]) -> list[Measurement]:
    result = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            image = record.get("image_path", record.get("ocr_text_path"))
            if not image:
                raise ValueError(f"{path}:{line_number}: missing image_path/ocr_text_path")
            key = sample_id(image)
            if key in seen:
                raise ValueError(f"{path}:{line_number}: duplicate sample {key}")
            seen.add(key)
            stages = {}
            for name, stage in record.get("stages", {}).items():
                if stage.get("inference_seconds") is not None:
                    stages[name] = number(stage["inference_seconds"], context=f"{path}:{name}")
            seconds = record.get("inference_seconds")
            generation = stages.get("generation", record.get("generation", {}).get("inference_seconds"))
            vram = record.get("vram", {})
            item = Measurement(
                key, record.get("status", "unknown"), subsets.get(key),
                number(seconds, context=str(path)) if seconds is not None else None,
                number(generation, context=str(path)) if generation is not None else None,
                stages, _memory(vram), tuple(d.get("device", "unknown") for d in vram.get("devices", [])),
            )
            if any(v < 0 for v in [*stages.values(), *[x for x in (item.seconds, item.generation_seconds) if x is not None]]):
                raise ValueError(f"{path}:{line_number}: negative inference time")
            result.append(item)
    return result


def _resolve_run(raw: str, table: Path, run_name: str) -> Path | None:
    # Prefer a sibling so a relocated snapshot cannot accidentally use its old source run.
    candidates = [table.parent / run_name]
    if raw:
        path = Path(raw.replace("\\", "/"))
        candidates.extend([path if path.is_absolute() else table.parent / path,
                           REPO_ROOT / path])
        for marker in ("results", "output", "runs"):
            if marker in path.parts:
                candidates.append(REPO_ROOT.joinpath(*path.parts[path.parts.index(marker):]))
    return next((p.resolve() for p in candidates if p.is_dir()), None)


def _attach_details(run: RunResult, report: dict[str, Any], source: str) -> None:
    blocks = {"overall": report.get("overall", {})}
    blocks.update(report.get("subsets", {}))
    for subset, block in blocks.items():
        summary = block.get("summary", {})
        # A stale field report must not silently accompany a newer ranking table.
        for metric, expected in run.metrics.get(subset, {}).items():
            if metric in {"parse_rate", "schema_valid_rate"}:
                continue  # Saved reports may score template-filled JSON instead of raw generation.
            if metric in summary and abs(number(summary[metric], context=source) - expected) > 2e-5:
                raise ValueError(f"{run.key}: {source} disagrees with table for {subset}.{metric}")
        if subset in run.counts and summary.get("samples") != run.counts[subset]:
            raise ValueError(f"{run.key}: {source} has a different {subset} sample count")
        run.fields[subset] = block.get("field_breakdown", {})
        for name, values in run.fields[subset].items():
            score = number(values.get("f1"), context=f"{source}:{name}.f1")
            support = number(values.get("support"), context=f"{source}:{name}.support")
            if not 0 <= score <= 1 or support < 0 or not support.is_integer():
                raise ValueError(f"{source}:{name}: invalid field F1/support")
    for sample in report.get("samples", []):
        key = sample.get("sample_id")
        if not key or key in run.samples:
            raise ValueError(f"{source}: missing or duplicate sample_id {key!r}")
        run.samples[key] = sample
    for item in run.measurements:
        subset = run.samples.get(item.sample_id, {}).get("subset")
        if item.subset is not None and subset is not None and item.subset != subset:
            raise ValueError(f"{run.key}: conflicting subset for {item.sample_id}")
        item.subset = subset or item.subset
    run.sources.append(source)


def load_results(
    source: str | Path, *, namespace: str = "", labels: Mapping[str, str] | None = None,
    groups: Mapping[str, str] | None = None, run_directories: Mapping[str, str | Path] | None = None,
    reports: Mapping[str, str | Path] | None = None, details: bool = False,
    testset_path: str | Path = DEFAULT_TESTSET_PATH, schema_path: str | Path | None = None,
    prediction_key: str = "root", ground_truth_key: str = "content",
) -> list[RunResult]:
    """Load a comparison CSV or evaluate a prediction folder in memory.

    A folder's existing evaluation_table.csv is authoritative. ``details=True``
    computes field/sample reports only for runs without an explicit saved report.
    No source artifacts or evaluation tables are written.
    """
    source = Path(source).resolve()
    testset_path = Path(testset_path)
    table = source / "evaluation_table.csv" if source.is_dir() else source
    evaluations = {}
    if table.is_file():
        with table.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    elif source.is_dir():
        from src.eval_suite.results_table import evaluate_results_folder, flatten_result, DEFAULT_SCHEMA_PATH
        evaluated = evaluate_results_folder(source, testset_path=testset_path,
                                           schema_path=Path(schema_path) if schema_path else DEFAULT_SCHEMA_PATH,
                                           prediction_key=prediction_key, ground_truth_key=ground_truth_key)
        evaluations = {r.run_name: r for r in evaluated}
        rows = [flatten_result(r) for r in evaluated]
    else:
        raise FileNotFoundError(source)
    if not rows:
        raise ValueError(f"{source}: no evaluation rows")
    subsets = _subset_ids(testset_path)
    result = []
    seen = set()
    for row in rows:
        name = row.get("run")
        if not isinstance(name, str) or not name.strip() or name in seen:
            raise ValueError(f"{source}: missing or duplicate run {name!r}")
        seen.add(name)
        key = f"{namespace}::{name}" if namespace else name
        metrics = {}
        counts = {}
        for subset in SUBSETS:
            metrics[subset] = {}
            for metric in METRICS:
                raw = row.get(f"{subset}_{metric}")
                if raw is not None and raw != "":
                    value = number(raw, context=f"{key}:{subset}.{metric}")
                    if not 0 <= value <= 1:
                        raise ValueError(f"{key}:{subset}.{metric}: expected fraction in [0, 1]")
                    metrics[subset][metric] = value
            raw_count = row.get(f"{subset}_samples")
            if raw_count is not None and raw_count != "":
                count = number(raw_count, context=key)
                if count < 1 or not count.is_integer():
                    raise ValueError(f"{key}: sample count must be a positive integer")
                counts[subset] = int(count)
        run = RunResult(key, (labels or {}).get(name, name), (groups or {}).get(name, "Models"),
                        metrics, counts, sources=[str(table if table.is_file() else source)])
        explicit = (run_directories or {}).get(name)
        directory = Path(explicit).resolve() if explicit is not None else _resolve_run(str(row.get("run_directory", "")), table, name)
        if explicit is not None and not directory.is_dir():
            raise FileNotFoundError(directory)
        manifest = directory / "inference_manifest.jsonl" if directory else None
        if manifest is not None and manifest.is_file():
            run.measurements = read_manifest(manifest, subsets)
            if "overall" in counts and len(run.measurements) != counts["overall"]:
                raise ValueError(f"{key}: manifest count differs from evaluation table")
            run.sources.append(str(manifest))
            if run.measurements and all(not item.stages for item in run.measurements):
                run.notes.append("Top-level timing is native model inference; no pipeline stage breakdown is recorded.")
        else:
            run.notes.append("No inference manifest found; resource plots require one.")
        report_path = (reports or {}).get(name)
        if report_path is not None:
            path = Path(report_path).resolve()
            _attach_details(run, json.loads(path.read_text(encoding="utf-8")), str(path))
        elif details or name in evaluations:
            from src.eval_suite.results_table import evaluate_results_folder, DEFAULT_SCHEMA_PATH
            if directory is None:
                raise ValueError(f"{key}: details require a local prediction directory or saved report")
            evaluation = evaluations.get(name)
            if evaluation is None:
                evaluated = evaluate_results_folder(directory, testset_path=testset_path,
                                                   schema_path=Path(schema_path) if schema_path else DEFAULT_SCHEMA_PATH,
                                                   prediction_key=prediction_key, ground_truth_key=ground_truth_key)
                if len(evaluated) != 1:
                    raise ValueError(f"{key}: expected one prediction run at {directory}")
                evaluation = evaluated[0]
            _attach_details(run, evaluation.report.to_dict(), f"computed:{directory}")
        result.append(run)
    return result


def load_config(path: str | Path, *, details: bool = False) -> tuple[list[RunResult], dict[str, Any]]:
    """Read a figure selection file; relative paths are relative to that file."""
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    sources = config.get("sources", [])
    if not sources:
        raise ValueError("Plot config requires a nonempty sources list")

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else path.parent / candidate

    runs = []
    for entry in sources:
        options = dict(entry)
        source = resolve(options.pop("path"))
        for name in ("run_directories", "reports"):
            if name in options:
                options[name] = {key: resolve(value) for key, value in options[name].items()}
        for name in ("testset_path", "schema_path"):
            if name in options:
                options[name] = resolve(options[name])
        options["details"] = details or options.get("details", False)
        runs.extend(load_results(source, **options))
    if len({r.key for r in runs}) != len(runs):
        raise ValueError("Duplicate run keys across sources; give each source a namespace")
    by_key = {r.key: r for r in runs}
    for name, attr in (("labels", "label"), ("groups", "group")):
        for key, value in config.get(name, {}).items():
            if key not in by_key:
                raise ValueError(f"Unknown run key in {name}: {key}")
            setattr(by_key[key], attr, value)
    selection = config.get("runs")
    if selection is not None:
        if len(set(selection)) != len(selection) or any(key not in by_key for key in selection):
            raise ValueError("Config runs must contain unique, known run keys")
        runs = [by_key[key] for key in selection]
    if not runs:
        raise ValueError("No runs selected")
    return runs, config
