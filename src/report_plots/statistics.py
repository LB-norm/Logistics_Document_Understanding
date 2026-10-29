"""Explicit selection and aggregation rules used by the figure recipes."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .data import Measurement, RunResult, SUBSETS, number


@dataclass
class RuntimeSelection:
    samples: list[Measurement]
    failures: int
    missing: int
    warmup_removed: int


def runtime_samples(run: RunResult, *, subset: str = "overall", measure: str = "total",
                    drop_first: int = 0, memory_kind: str | None = None) -> RuntimeSelection:
    if subset not in SUBSETS or measure not in {"total", "generation"}:
        raise ValueError("Unknown subset or timing measure")
    if drop_first < 0:
        raise ValueError("drop_first must be nonnegative")
    if not run.measurements:
        raise ValueError(f"{run.key}: no inference measurements")
    if subset != "overall" and any(s.subset is None for s in run.measurements):
        raise ValueError(f"{run.key}: subset timing requires annotations or a report with subset labels")
    removed = min(drop_first, len(run.measurements))
    items = [s for s in run.measurements[drop_first:] if subset == "overall" or s.subset == subset]
    failures = sum(not s.completed for s in items)
    completed = [s for s in items if s.completed]

    def available(s: Measurement) -> bool:
        if memory_kind is not None:
            return memory_kind in s.memory
        return (s.seconds if measure == "total" else s.generation_seconds) is not None

    samples = [s for s in completed if available(s)]
    if not samples:
        raise ValueError(f"{run.key}: no completed samples with {memory_kind or measure} measurements")
    return RuntimeSelection(samples, failures, len(completed) - len(samples), removed)


def timing_values(selection: RuntimeSelection, measure: str) -> list[float]:
    return [float(s.seconds if measure == "total" else s.generation_seconds) for s in selection.samples]


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot summarize an empty measurement set")
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Measurements must be finite and nonnegative")
    return {"n": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95, method="linear")), "max": float(values.max())}


def mean_stages(selection: RuntimeSelection) -> dict[str, float]:
    """Mean wall times over identical completed documents, plus measured overhead."""
    samples = selection.samples
    if any(not s.stages or s.seconds is None for s in samples):
        raise ValueError("Stage plots require top-level total time and stage wall times for every selected sample")
    names = sorted({name for s in samples for name in s.stages})
    if "other" in names:
        raise ValueError("Stage name 'other' is reserved for pipeline overhead")
    overhead = []
    for sample in samples:
        difference = float(sample.seconds) - sum(sample.stages.values())
        if difference < -max(1e-5, float(sample.seconds) * .001):
            raise ValueError(f"{sample.sample_id}: stage times exceed pipeline time; cannot stack overlapping stages")
        overhead.append(max(0., difference))
    result = {name: sum(s.stages.get(name, 0.) for s in samples) / len(samples) for name in names}
    result["other"] = float(np.mean(overhead))
    return result


def pareto_indices(cost: Sequence[float], quality: Sequence[float]) -> list[int]:
    """Nondominated points for minimum cost and maximum quality; ties are retained."""
    if len(cost) != len(quality):
        raise ValueError("Cost and quality must have equal lengths")
    if any(not math.isfinite(x) for x in [*cost, *quality]):
        raise ValueError("Pareto coordinates must be finite")
    return [i for i, (x, y) in enumerate(zip(cost, quality))
            if not any(a <= x and b >= y and (a < x or b > y)
                       for a, b in zip(cost, quality))]


def _sample_counts(run: RunResult, subset: str) -> dict[str, list[float]]:
    if not run.samples:
        raise ValueError(f"{run.key}: paired uncertainty requires sample reports; load with details=True")
    result = {}
    for key, sample in run.samples.items():
        if subset != "overall" and sample.get("subset") != subset:
            continue
        metrics = sample.get("metrics", {})
        try:
            counts = [number(metrics[k], context=f"{run.key}:{key}:{k}")
                      for k in ("true_positives", "false_positives", "false_negatives")]
        except KeyError as exc:
            raise ValueError(f"{run.key}: sample report lacks field counts") from exc
        if any(x < 0 or not x.is_integer() for x in counts):
            raise ValueError(f"{run.key}: invalid per-document field counts")
        result[key] = counts
    return result


def paired_bootstrap(baseline: RunResult, other: RunResult, *, subset: str = "overall",
                     resamples: int = 5000, seed: int = 42, confidence: float = .95) -> dict:
    """Percentile interval for the difference in MICRO F1, resampling paired documents.

    This measures uncertainty over the observed documents, not variation across
    training seeds. The same document draw is used for both runs in every replicate.
    """
    if subset not in SUBSETS or resamples < 100 or not 0 < confidence < 1:
        raise ValueError("Need a valid subset, at least 100 resamples, and confidence in (0, 1)")
    left, right = _sample_counts(baseline, subset), _sample_counts(other, subset)
    if not left or left.keys() != right.keys():
        raise ValueError("Paired bootstrap requires identical, nonempty document ID sets")
    if any(baseline.samples[k].get("subset") != other.samples[k].get("subset") for k in left):
        raise ValueError("Paired documents have conflicting subset labels")
    ids = sorted(left)
    a, b = np.asarray([left[k] for k in ids]), np.asarray([right[k] for k in ids])

    def f1(counts):
        tp, fp, fn = np.moveaxis(counts, -1, 0)
        denominator = 2 * tp + fp + fn
        return np.divide(2 * tp, denominator, out=np.ones_like(tp, dtype=float), where=denominator != 0)

    for run, counts in ((baseline, a), (other, b)):
        if abs(float(f1(counts.sum(axis=0))) - run.metric("field_f1", subset)) > 2e-5:
            raise ValueError(f"{run.key}: sample field counts disagree with aggregate micro F1")
    rng = np.random.default_rng(seed)
    draws = []
    # Bound intermediate memory for larger datasets/resample counts.
    for start in range(0, resamples, 256):
        indices = rng.integers(0, len(ids), size=(min(256, resamples-start), len(ids)))
        draws.extend((f1(b[indices].sum(axis=1)) - f1(a[indices].sum(axis=1))).tolist())
    alpha = (1 - confidence) / 2
    low, high = np.quantile(draws, [alpha, 1-alpha], method="linear")
    return {"difference": float(f1(b.sum(axis=0)) - f1(a.sum(axis=0))),
            "low": float(low), "high": float(high), "confidence": confidence,
            "resamples": resamples, "seed": seed, "n": len(ids), "sample_ids": ids}
