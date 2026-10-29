"""Publication-oriented Matplotlib recipes. Nothing is rendered on import."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
from typing import Any, Sequence

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
import numpy as np

from .data import MEMORY_KINDS, METRICS, RunResult
from .statistics import (mean_stages, paired_bootstrap, pareto_indices,
                         runtime_samples, summarize, timing_values)

PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#444444")
METRIC_LABELS = {"field_f1": "Field F1", "field_precision": "Field precision",
                 "field_recall": "Field recall", "value_similarity": "Value similarity",
                 "parse_rate": "JSON parse rate", "schema_valid_rate": "Schema validity",
                 "document_exact_match_rate": "Document exact match"}
RC = {"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
      "axes.labelsize": 10, "legend.fontsize": 9, "axes.spines.top": False,
      "axes.spines.right": False, "axes.linewidth": .7, "grid.color": "#DDDDDD",
      "grid.linewidth": .6, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
      "savefig.facecolor": "white", "figure.facecolor": "white", "axes.facecolor": "white"}


@dataclass(frozen=True)
class PlotStyle:
    width: float = 7.2
    height: float | None = None
    font_size: float = 10
    label_width: int = 36

    def __post_init__(self):
        if (not np.isfinite(self.width) or self.width <= 0 or
            (self.height is not None and (not np.isfinite(self.height) or self.height <= 0)) or
            not np.isfinite(self.font_size) or self.font_size <= 0 or self.label_width < 1):
            raise ValueError("Figure dimensions, font size, and label width must be positive")


@dataclass
class Plot:
    figure: Figure
    recipe: str
    runs: Sequence[RunResult]
    data: Any
    options: dict[str, Any]

    def save(self, stem: str | Path, *, formats: Sequence[str] = ("pdf", "svg", "png"),
             dpi: int = 300) -> list[Path]:
        """Save vector/raster figures plus the exact plotted values and provenance."""
        if not formats or any(fmt not in {"pdf", "svg", "png"} for fmt in formats):
            raise ValueError("Formats must be a nonempty selection of pdf, svg, png")
        if dpi < 1:
            raise ValueError("dpi must be positive")
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        paths = []
        with mpl.rc_context(RC):
            for fmt in dict.fromkeys(formats):
                path = Path(str(stem) + "." + fmt)
                self.figure.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight")
                paths.append(path)
        metadata = {"recipe": self.recipe, "options": self.options, "data": self.data,
                    "runs": [{"key": r.key, "label": r.label, "group": r.group,
                              "sources": r.sources, "notes": r.notes, "sample_counts": r.counts}
                             for r in self.runs],
                    "versions": {"matplotlib": mpl.__version__, "numpy": np.__version__}}
        path = Path(str(stem) + ".data.json")
        path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return [*paths, path]


def _check(runs: Sequence[RunResult]) -> None:
    if not runs or len({r.key for r in runs}) != len(runs):
        raise ValueError("Need a nonempty selection of runs with unique keys")


def _color(key: str) -> str:
    return PALETTE[int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % len(PALETTE)]


def _labels(runs, style, counts=None):
    return [textwrap.fill(r.label, style.label_width, break_long_words=False) +
            (f" (n={counts[i]})" if counts is not None else "") for i, r in enumerate(runs)]


def _frame(runs, title, xlabel, *, style=None, rows=None, horizontal=True):
    _check(runs)
    style = style or PlotStyle()
    height = style.height or max(3.2, 1.5 + .55 * (rows if rows is not None else len(runs)))
    with mpl.rc_context({**RC, "font.size": style.font_size,
                         "axes.labelsize": style.font_size, "axes.titlesize": style.font_size + 2}):
        fig = Figure(figsize=(style.width, height), layout="constrained")
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        ax.set_title(title, loc="left", pad=12)
        ax.set_xlabel(xlabel)
        ax.set_axisbelow(True)
        ax.grid(axis="x" if horizontal else "y")
        if horizontal:
            ax.set_yticks(range(len(runs)), _labels(runs, style))
            ax.invert_yaxis()
    return fig, ax, style


def _plot(fig, recipe, runs, data, style, **options):
    # Artists added after frame construction inherit the caller's rcParams;
    # explicitly style axes and legends so per-figure settings stay isolated.
    for ax in fig.axes:
        ax.tick_params(labelsize=style.font_size)
        ax.xaxis.label.set_fontsize(style.font_size)
        ax.yaxis.label.set_fontsize(style.font_size)
        ax.set_title(ax.get_title(loc="left"), loc="left", fontsize=style.font_size+2)
        if ax.get_legend() is not None:
            for text in ax.get_legend().get_texts():
                text.set_fontsize(style.font_size-1)
    for text in fig.findobj(match=mpl.text.Text):
        text.set_fontfamily("DejaVu Sans")
    return Plot(fig, recipe, runs, data, {"style": asdict(style), **options})


def _baseline(runs, key):
    matches = [r for r in runs if r.key == key]
    if len(matches) != 1:
        raise ValueError(f"Baseline must identify one selected run: {key!r}")
    return matches[0]


def plot_quality(runs: Sequence[RunResult], *, metrics: Sequence[str] = ("field_f1",),
                 subset: str = "overall", style: PlotStyle | None = None) -> Plot:
    if not metrics or len(set(metrics)) != len(metrics) or any(m not in METRICS for m in metrics):
        raise ValueError("Choose unique metrics from report_plots.data.METRICS")
    fig, ax, style = _frame(runs, f"Extraction quality · {subset}", "Score (%)", style=style)
    data = []
    height = .75 / len(metrics)
    for j, metric in enumerate(metrics):
        values = [100 * r.metric(metric, subset) for r in runs]
        positions = np.arange(len(runs)) + (j - (len(metrics)-1)/2) * height
        bars = ax.barh(positions, values, height=height * .92, color=PALETTE[j % len(PALETTE)], label=METRIC_LABELS[metric])
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=style.font_size-1)
        data.extend({"run": r.key, "metric": metric, "score": value/100} for r, value in zip(runs, values))
    ax.set_xlim(0, 108)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    if len(metrics) > 1:
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.15), ncol=min(3, len(metrics)), frameon=False)
    return _plot(fig, "quality", runs, data, style, subset=subset, metrics=list(metrics))


def plot_subsets(runs: Sequence[RunResult], *, metric: str = "field_f1", style=None) -> Plot:
    fig, ax, style = _frame(runs, "Default and challenge documents", f"{METRIC_LABELS[metric]} (%)", style=style)
    data = []
    for i, run in enumerate(runs):
        a, b = [100 * run.metric(metric, subset) for subset in ("default", "challenge")]
        ax.plot([a, b], [i, i], color="#BBBBBB", lw=2)
        ax.scatter(a, i, color=PALETTE[0], marker="o", s=42, label="Default" if i == 0 else None, zorder=3)
        ax.scatter(b, i, color=PALETTE[1], marker="s", s=38, label="Challenge" if i == 0 else None, zorder=3)
        data.append({"run": run.key, "default": a/100, "challenge": b/100, "gap_pp": b-a})
    ax.set_xlim(0, 103)
    ax.legend(frameon=False, loc="lower right")
    return _plot(fig, "subsets", runs, data, style, metric=metric)


def plot_deltas(runs: Sequence[RunResult], *, baseline: str, metric="field_f1", subset="overall", style=None) -> Plot:
    ref = _baseline(runs, baseline)
    others = [r for r in runs if r.key != baseline]
    fig, ax, style = _frame(others, f"Change relative to {ref.label}", f"{METRIC_LABELS[metric]} difference (percentage points)", style=style)
    values = [100 * (r.metric(metric, subset) - ref.metric(metric, subset)) for r in others]
    bars = ax.barh(range(len(others)), values, height=.6, color=[PALETTE[2] if x >= 0 else PALETTE[1] for x in values])
    ax.bar_label(bars, labels=[f"{x:+.2f}" for x in values], padding=3)
    ax.axvline(0, color="#555555", lw=.8)
    span = max([abs(x) for x in values] + [.5]) * 1.3
    ax.set_xlim(-span, span)
    return _plot(fig, "deltas", runs, [{"run": r.key, "delta_pp": x} for r, x in zip(others, values)], style,
                 baseline=baseline, metric=metric, subset=subset)


def _resource_data(runs, subset, measure, drop_first, memory_kind=None):
    result = []
    for run in runs:
        selected = runtime_samples(run, subset=subset, measure=measure, drop_first=drop_first, memory_kind=memory_kind)
        values = ([s.memory[memory_kind] for s in selected.samples] if memory_kind is not None
                  else timing_values(selected, measure))
        result.append({"run": run.key, "values": values, "sample_ids": [s.sample_id for s in selected.samples],
                       "summary": summarize(values), "excluded_failures": selected.failures,
                       "missing_measurements": selected.missing, "warmup_removed": selected.warmup_removed,
                       "devices": sorted({d for s in selected.samples for d in s.devices})})
    return result


def _boxes(runs, *, recipe, title, xlabel, subset, measure="total", drop_first=0, memory_kind=None, style=None):
    data = _resource_data(runs, subset, measure, drop_first, memory_kind)
    fig, ax, style = _frame(runs, title, xlabel, style=style)
    values = [row["values"] for row in data]
    orientation = ({"orientation": "horizontal"} if "orientation" in inspect.signature(ax.boxplot).parameters
                   else {"vert": False})  # Matplotlib 3.8/3.9 compatibility.
    boxes = ax.boxplot(values, positions=range(len(runs)), **orientation, patch_artist=True,
                       widths=.5, showfliers=False, manage_ticks=False,
                       medianprops={"color": "#111111", "linewidth": 1.5},
                       whiskerprops={"color": "#666666"}, capprops={"color": "#666666"})
    for i, (run, box, row) in enumerate(zip(runs, boxes["boxes"], data)):
        color = _color(run.key)
        box.set_facecolor(color)
        box.set_alpha(.3)
        # Deterministic jitter keeps export reproducible without changing global RNG state.
        offsets = .11 * np.sin(np.arange(len(row["values"])) * 2.4)
        ax.scatter(row["values"], i + offsets, s=13, color=color, alpha=.65, zorder=3)
        ax.scatter(row["summary"]["p95"], i, marker="D", s=30, color="#222222", zorder=4,
                   label="95th percentile" if i == 0 else None)
    ax.set_yticks(range(len(runs)), _labels(runs, style, [len(v) for v in values]))
    ax.set_xlim(left=0)
    ax.legend(frameon=False)
    excluded = sum(row["excluded_failures"] for row in data)
    missing = sum(row["missing_measurements"] for row in data)
    ax.text(0, -.18, f"Completed inference; {excluded} failed, {missing} unmeasured excluded. Boxes: median/IQR; whiskers: 1.5×IQR.",
            transform=ax.transAxes, fontsize=max(6, style.font_size-2), va="top")
    return _plot(fig, recipe, runs, data, style, subset=subset, measure=measure,
                 drop_first=drop_first, memory_kind=memory_kind)


def plot_latency(runs: Sequence[RunResult], *, subset="overall", measure="total", drop_first=0, style=None) -> Plot:
    name = "Total inference" if measure == "total" else "VLM generation"
    return _boxes(runs, recipe="latency", title=f"{name} latency · {subset}", xlabel=f"{name} latency (s/document)",
                  subset=subset, measure=measure, drop_first=drop_first, style=style)


def plot_memory(runs: Sequence[RunResult], *, kind="process", subset="overall", drop_first=0, style=None) -> Plot:
    if kind not in MEMORY_KINDS:
        raise ValueError(f"Unknown memory kind {kind}")
    name = "Sampled process" if kind == "process" else kind.replace("_", " ").title()
    return _boxes(runs, recipe="memory", title=f"{name} GPU memory · {subset}",
                  xlabel=f"{name} peak (GiB, maximum per device)", subset=subset,
                  memory_kind=kind, drop_first=drop_first, style=style)


def plot_ecdf(runs: Sequence[RunResult], *, subset="overall", measure="total", drop_first=0, style=None) -> Plot:
    data = _resource_data(runs, subset, measure, drop_first)
    name = "Total inference" if measure == "total" else "VLM generation"
    fig, ax, style = _frame(runs, f"{name} latency distribution · {subset}", "Latency (s/document)",
                            style=style, rows=5, horizontal=False)
    for i, (run, row) in enumerate(zip(runs, data)):
        values = sorted(row["values"])
        ax.step([0, *values], [0, *np.arange(1, len(values)+1)/len(values)], where="post",
                label=f"{run.label} (n={len(values)})", color=PALETTE[i % len(PALETTE)],
                linestyle=("-", "--", "-.", ":")[(i // len(PALETTE)) % 4], lw=1.7)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("Documents completed within latency")
    ax.legend(frameon=False, loc="lower right")
    return _plot(fig, "ecdf", runs, data, style, subset=subset, measure=measure, drop_first=drop_first)


def plot_stages(runs: Sequence[RunResult], *, subset="overall", drop_first=0, style=None) -> Plot:
    data = []
    for run in runs:
        selection = runtime_samples(run, subset=subset, drop_first=drop_first)
        data.append({"run": run.key, "mean_seconds": mean_stages(selection), "n": len(selection.samples),
                     "sample_ids": [s.sample_id for s in selection.samples],
                     "excluded_failures": selection.failures, "missing_measurements": selection.missing,
                     "warmup_removed": selection.warmup_removed})
    fig, ax, style = _frame(runs, f"Pipeline stage costs · {subset}", "Mean inference time (s/document)", style=style)
    names = {k for row in data for k in row["mean_seconds"]}
    order = [n for n in ("ocr", "yolo", "fusion", "generation", "other") if n in names]
    order += sorted(names-set(order))
    left = np.zeros(len(runs))
    for i, stage in enumerate(order):
        values = [row["mean_seconds"].get(stage, 0) for row in data]
        ax.barh(range(len(runs)), values, left=left, height=.6, label=stage.upper() if stage in {"ocr", "yolo"} else stage.title(),
                color=PALETTE[i % len(PALETTE)])
        left += values
    ax.set_xlim(left=0)
    ax.set_yticks(range(len(runs)), _labels(runs, style, [row["n"] for row in data]))
    ax.legend(frameon=False, ncol=min(3, len(order)), loc="upper center", bbox_to_anchor=(.5, -.15))
    return _plot(fig, "stages", runs, data, style, subset=subset, drop_first=drop_first,
                 aggregation="Mean stage wall times over identical completed documents; other = total minus stages")


def plot_tradeoff(runs: Sequence[RunResult], *, resource="latency", statistic="median", metric="field_f1",
                  subset="overall", measure="total", memory_kind="process", drop_first=0, style=None) -> Plot:
    if resource not in {"latency", "memory"} or statistic not in {"median", "mean", "p95", "max"}:
        raise ValueError("Choose latency/memory and mean/median/p95/max")
    data = _resource_data(runs, subset, measure, drop_first, memory_kind if resource == "memory" else None)
    x = [row["summary"][statistic] for row in data]
    y = [100 * r.metric(metric, subset) for r in runs]
    frontier = pareto_indices(x, y)
    label = f"{statistic.title()} {'total inference' if measure == 'total' else 'VLM generation'} latency (s/document)"
    if resource == "memory":
        label = f"{statistic.title()} {memory_kind.replace('_', ' ')} peak memory (GiB, maximum per device)"
    fig, ax, style = _frame(runs, f"Quality versus {'latency' if resource == 'latency' else 'memory'} · {subset}", label,
                            style=style, rows=5, horizontal=False)
    groups = list(dict.fromkeys(r.group for r in runs))
    marked = set()
    for i, (run, a, b) in enumerate(zip(runs, x, y)):
        group = groups.index(run.group)
        ax.scatter(a, b, s=60, color=PALETTE[group % len(PALETTE)],
                   marker=("o", "s", "^", "D", "v")[group % 5], edgecolors="white", linewidths=.7,
                   label=run.group if run.group not in marked else None, zorder=3)
        marked.add(run.group)
        ax.annotate(textwrap.fill(run.label, style.label_width), (a, b), xytext=(6, 5),
                    textcoords="offset points", fontsize=style.font_size-1)
        data[i].update(quality=b/100, resource_value=a, pareto=i in frontier)
    order = sorted(frontier, key=lambda i: (x[i], y[i]))
    ax.plot([x[i] for i in order], [y[i] for i in order], color="#777777", linestyle="--", lw=1,
            label="Observed Pareto frontier")
    ax.set_ylabel(f"{METRIC_LABELS[metric]} (%)")
    ax.set_xlim(0, max(x+[1]) * 1.35)
    ax.set_ylim(max(0, min(y)-8), min(101, max(y)+10))
    ax.legend(frameon=False, loc="lower right")
    return _plot(fig, "tradeoff", runs, data, style, subset=subset, resource=resource, statistic=statistic,
                 metric=metric, measure=measure, memory_kind=memory_kind, drop_first=drop_first,
                 note="Quality uses all evaluated samples; resources use completed measured samples. Hardware is not normalized.")


def plot_fields(runs: Sequence[RunResult], *, subset="overall", min_support=1, fields=None, style=None) -> Plot:
    if min_support < 1:
        raise ValueError("min_support must be positive")
    for run in runs:
        if not run.fields.get(subset):
            raise ValueError(f"{run.key}: field heatmap needs a detailed report; load with details=True")
    names = list(fields) if fields is not None else sorted({name for r in runs for name, values in r.fields[subset].items()
                                                          if values.get("support", 0) >= min_support})
    if not names or len(set(names)) != len(names):
        raise ValueError("Need a nonempty selection of unique fields")
    if fields is not None and any(not any(name in r.fields[subset] for r in runs) for name in names):
        raise ValueError("Unknown field name")
    fig, ax, style = _frame(runs, f"Field extraction quality · {subset}", "", style=style, rows=len(names), horizontal=False)
    matrix = np.full((len(names), len(runs)), np.nan)
    support = []
    for i, name in enumerate(names):
        row = []
        for j, run in enumerate(runs):
            values = run.fields[subset].get(name, {})
            count = values.get("support", 0)
            row.append(count)
            if count >= min_support:
                matrix[i, j] = 100 * values["f1"]
        support.append(row)
    cmap = mpl.colormaps["cividis"].copy()
    cmap.set_bad("#EEEEEE")
    artist = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", vmin=0, vmax=100, cmap=cmap)
    ax.grid(False)
    ax.set_yticks(range(len(names)), [textwrap.fill(n, 46, break_long_words=False) for n in names])
    ax.set_xticks(range(len(runs)), _labels(runs, style), rotation=40, ha="right")
    fig.colorbar(artist, ax=ax, label="Field F1 (%)", fraction=.035, pad=.02)
    if len(names) * len(runs) <= 160:
        for i in range(len(names)):
            for j in range(len(runs)):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i,j]:.0f}", ha="center", va="center", fontsize=style.font_size-2,
                            color="white" if matrix[i,j] < 45 else "#111111")
    data = {"fields": names, "runs": [r.key for r in runs], "support": support,
            "f1": [[None if not np.isfinite(v) else float(v/100) for v in row] for row in matrix]}
    return _plot(fig, "fields", runs, data, style, subset=subset, min_support=min_support,
                 note="Grey cells are missing or below support threshold; array indices follow eval-suite grouping.")


def plot_paired(runs: Sequence[RunResult], *, baseline: str, subset="overall", resamples=5000,
                seed=42, confidence=.95, style=None) -> Plot:
    ref = _baseline(runs, baseline)
    others = [r for r in runs if r.key != baseline]
    fig, ax, style = _frame(others, f"Paired comparison relative to {ref.label}", "Micro field F1 difference (percentage points)", style=style)
    data = [{"run": r.key, **paired_bootstrap(ref, r, subset=subset, resamples=resamples, seed=seed, confidence=confidence)}
            for r in others]
    for i, row in enumerate(data):
        # Draw intervals directly: a percentile interval need not contain the point estimate.
        ax.plot([row["low"]*100, row["high"]*100], [i, i], color=PALETTE[0], lw=2)
        ax.scatter(row["difference"]*100, i, color=PALETTE[0], s=35, zorder=3)
    ax.axvline(0, color="#555555", linestyle="--", lw=1)
    ax.text(0, -.18, f"{confidence:.0%} paired document-bootstrap intervals; {resamples:,} resamples. Training-seed variance is not measured.",
            transform=ax.transAxes, fontsize=max(6, style.font_size-2), va="top")
    return _plot(fig, "paired", runs, data, style, baseline=baseline, subset=subset,
                 resamples=resamples, seed=seed, confidence=confidence)
