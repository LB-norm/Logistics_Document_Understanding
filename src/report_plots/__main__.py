"""Generate scientific report figures: python -m src.report_plots --help."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from .data import MEMORY_KINDS, METRICS, SUBSETS, load_config, load_results

RECIPES = ("quality", "subsets", "deltas", "latency", "ecdf", "stages", "tradeoff", "memory", "fields", "paired")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("sources", nargs="*", type=Path, help="Comparison CSVs or prediction folders; use a config for namespaces/labels")
    parser.add_argument("--config", type=Path, help="JSON sources, run selection, labels, groups, and style; paths relative to config")
    parser.add_argument("--plots", nargs="+", choices=RECIPES, default=["quality", "subsets", "latency", "tradeoff"])
    parser.add_argument("--run", action="append", help="Select a run key, repeat to set a custom order")
    parser.add_argument("--baseline", help="Selected run key for deltas/paired")
    parser.add_argument("--subset", choices=SUBSETS, default="overall")
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=["field_f1"], help="Quality bars; first metric also used by subsets/deltas/tradeoff")
    parser.add_argument("--measure", choices=["total", "generation"], default="total")
    parser.add_argument("--drop-first", type=int, default=0, help="Discard first N manifest records per run before filtering subsets/failures; affects resource plots only")
    parser.add_argument("--memory-kind", choices=MEMORY_KINDS, default="process")
    parser.add_argument("--resource", choices=["latency", "memory"], default="latency", help="Trade-off x axis")
    parser.add_argument("--statistic", choices=["median", "mean", "p95", "max"], default="median")
    parser.add_argument("--details", action="store_true", help="Compute detailed field/sample reports in memory when none are configured")
    parser.add_argument("--testset-path", type=Path, help="Override test set for direct sources (config sources have their own options)")
    parser.add_argument("--schema", type=Path, help="Override schema for evaluations of direct sources")
    parser.add_argument("--min-support", type=int, default=1, help="Minimum annotated field support in heatmaps")
    parser.add_argument("--resamples", type=int, default=5000, help="Paired bootstrap document draws")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence", type=float, default=.95)
    parser.add_argument("--width", type=float, help="Figure width in inches; overrides config style")
    parser.add_argument("--height", type=float, help="Figure height in inches; otherwise adapts to row count")
    parser.add_argument("--font-size", type=float)
    parser.add_argument("--output-dir", type=Path, default=Path("output/report_plots"))
    parser.add_argument("--formats", nargs="+", choices=["pdf", "svg", "png"], default=["pdf", "svg", "png"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--list-runs", action="store_true", help="Inspect selected run keys and measurement counts without generating figures")
    args = parser.parse_args(argv)
    if bool(args.sources) == bool(args.config):
        parser.error("Provide direct sources or --config, exclusively")
    if args.drop_first < 0 or args.dpi < 1 or args.min_support < 1:
        parser.error("drop-first must be nonnegative; dpi/min-support must be positive")
    if not args.list_runs and any(p in args.plots for p in ("deltas", "paired")) and not args.baseline:
        parser.error("deltas and paired require --baseline")
    if args.config and (args.testset_path or args.schema):
        parser.error("Put testset_path/schema_path in config source entries when using --config")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        # Detailed figures require reports, but listing run IDs never triggers evaluation.
        details = args.details or (not args.list_runs and any(p in args.plots for p in ("fields", "paired")))
        config = {}
        if args.config:
            runs, config = load_config(args.config, details=details)
        else:
            options = {"details": details}
            if args.testset_path:
                options["testset_path"] = args.testset_path
            if args.schema:
                options["schema_path"] = args.schema
            runs = [r for source in args.sources for r in load_results(source, **options)]
        if len({r.key for r in runs}) != len(runs):
            raise ValueError("Duplicate run keys across sources; use config source namespaces")
        if args.run:
            by_key = {r.key: r for r in runs}
            if len(set(args.run)) != len(args.run) or any(k not in by_key for k in args.run):
                raise ValueError("--run must specify unique known run keys; inspect --list-runs")
            runs = [by_key[k] for k in args.run]
        if args.list_runs:
            for run in runs:
                print(f"{run.key}\t{run.label}\t{run.counts.get('overall', '?')} evaluated\t{len(run.measurements)} measured")
            return 0

        from . import plots
        style_options = dict(config.get("style", {}))
        for key in ("width", "height", "font_size"):
            if getattr(args, key) is not None:
                style_options[key] = getattr(args, key)
        style = plots.PlotStyle(**style_options)
        shared = {"subset": args.subset, "style": style}
        runtime = {**shared, "drop_first": args.drop_first}
        metric = args.metrics[0]
        figures = []
        for recipe in dict.fromkeys(args.plots):
            if recipe == "quality":
                figure = plots.plot_quality(runs, metrics=args.metrics, **shared)
            elif recipe == "subsets":
                figure = plots.plot_subsets(runs, metric=metric, style=style)
            elif recipe == "deltas":
                figure = plots.plot_deltas(runs, baseline=args.baseline, metric=metric, **shared)
            elif recipe in {"latency", "ecdf"}:
                figure = getattr(plots, "plot_" + recipe)(runs, measure=args.measure, **runtime)
            elif recipe == "stages":
                figure = plots.plot_stages(runs, **runtime)
            elif recipe == "memory":
                figure = plots.plot_memory(runs, kind=args.memory_kind, **runtime)
            elif recipe == "tradeoff":
                figure = plots.plot_tradeoff(runs, resource=args.resource, statistic=args.statistic,
                                             metric=metric, measure=args.measure, memory_kind=args.memory_kind, **runtime)
            elif recipe == "fields":
                figure = plots.plot_fields(runs, min_support=args.min_support, **shared)
            elif recipe == "paired":
                figure = plots.plot_paired(runs, baseline=args.baseline, resamples=args.resamples,
                                           seed=args.seed, confidence=args.confidence, **shared)
            suffix = f"{recipe}_{args.subset}"
            if recipe in {"quality", "subsets", "deltas"}:
                suffix += "_" + "_".join(args.metrics if recipe == "quality" else [metric])
            if recipe in {"latency", "ecdf"}:
                suffix += "_" + args.measure
            if recipe == "memory":
                suffix += "_" + args.memory_kind
            if recipe == "tradeoff":
                suffix += f"_{args.resource}_{args.statistic}_{args.measure if args.resource == 'latency' else args.memory_kind}"
            figures.append((suffix, figure))
        # Validate every requested recipe before creating output files.
        for suffix, figure in figures:
            for path in figure.save(args.output_dir / suffix, formats=args.formats, dpi=args.dpi):
                print(path)
        return 0
    except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
