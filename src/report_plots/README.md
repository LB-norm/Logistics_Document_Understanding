# Scientific report plots

Reusable Matplotlib figures for extraction experiments, with a CLI and Python API.
Reads existing evaluation CSVs, prediction folders, detailed evaluation reports,
and inference manifests. No model inference, training, source-file changes, or
figure generation happens on import. Dependencies (`matplotlib`, `numpy`) are
already in the repository's `requirements.txt`; no Seaborn, Pandas, LaTeX, GPU,
or display server is required.

## Figure designs

| CLI recipe | Report question | Design |
| --- | --- | --- |
| `quality` | Which model extracts most accurately? | Horizontal bars for F1 or grouped precision/recall/F1; full percentage axis, readable model labels. Also supports parse/schema/exact-match rates and value similarity. |
| `subsets` | How much do irregular documents hurt? | Paired dots for default and challenge scores on each model row. |
| `deltas` | How does an ablation change the baseline? | Signed differences in percentage points, centered on zero. |
| `latency` | What is typical latency, and how variable is it? | Boxplots with every document overlaid and an explicit p95 marker. Total inference or VLM generation can be selected. |
| `ecdf` | What fraction finishes within a latency budget? | Empirical cumulative latency distributions, with document counts in the legend. |
| `stages` | Where does pipeline time go? | Stacked mean OCR/layout/fusion/generation wall times and measured overhead. |
| `tradeoff` | Which models balance quality and resource cost? | Quality versus latency or VRAM, with named points, experiment groups, and an observed Pareto frontier. |
| `memory` | What GPU memory does each model require? | Per-document peak distributions; process measurements and framework allocator measurements stay distinct. |
| `fields` | Which document fields remain difficult? | Field-by-model F1 heatmap, with a fixed 0–100% scale and grey missing/low-support cells. |
| `paired` | How uncertain is a model's F1 improvement? | Paired document-bootstrap intervals for the difference in micro F1. |

Start with overall quality, subset gaps, latency distributions, and one trade-off
figure. Use field heatmaps and stage costs to explain the results; use baseline
deltas and paired intervals for specific experimental claims. Avoid putting every
model and every metric into a single panel. The API exposes each Matplotlib
figure/axis for final annotation and layout adjustments.

Defaults use a white background, restrained grid, DejaVu Sans, and an accessible
palette. Scatter groups also use different marker shapes; ECDFs add line styles
when the palette cycles. Long names can be replaced with explicit report labels.
Charts preserve caller/config order, making ablation ordering consistent across
figures. Sort/select the returned runs in Python when a ranked order is preferred.

## CLI

To reproduce the curated 14-figure comparison of the concluded End-to-End,
parser + LLM, and Hybrid experiments:

```bash
python -m src.report_plots.experiment_suite
```

This writes individual PDF/SVG/PNG figures, exact figure data, fresh evaluation
snapshots, an input audit with source hashes, a multi-page PDF atlas, an HTML
gallery, and a ZIP bundle to `output/report_figures/experiment_suite`. Override
the destination with `--output-dir`. It evaluates all 39 current run artifacts,
including all 15 parser variants and the recovered Hybrid image20 sample, without
modifying source predictions, manifests, or historical comparison tables.
The gallery labels differences in model size and inference campaign; resource
comparisons use the recent same-GPU measurements rather than historical timings.

Run from the repository root, in the project's Python environment. These commands
are examples for later use; creating the module does not create report figures.

Inspect data availability without generating figures:

```bash
python -m src.report_plots results/New/Hybrid --list-runs
python -m src.report_plots --config src/report_plots/examples/hybrid.json --list-runs
```

Generate a report selection using the supplied readable hybrid labels:

```bash
python -m src.report_plots \
  --config src/report_plots/examples/hybrid.json \
  --plots quality subsets latency ecdf stages tradeoff memory \
  --output-dir output/report_figures/hybrid
```

Precision/recall/F1, challenge-only scores, and VLM generation latency:

```bash
python -m src.report_plots results/New/Hybrid \
  --plots quality latency \
  --metrics field_precision field_recall field_f1 \
  --subset challenge --measure generation \
  --output-dir output/report_figures/hybrid_challenge
```

Compare dropout against its hard-fusion baseline:

```bash
python -m src.report_plots \
  --config src/report_plots/examples/hybrid.json \
  --plots deltas paired \
  --baseline qwen35-9b-hybrid-input-hard-nf4-r16-chunked \
  --resamples 5000 --seed 42 \
  --output-dir output/report_figures/hybrid_deltas
```

The `fields` and `paired` recipes automatically request detailed evaluation
reports. If none are supplied, the eval suite computes them from the local
predictions and annotations in memory. It performs no inference and writes no
evaluation files. Other recipes use the existing CSV and manifest alone.

```bash
python -m src.report_plots --config src/report_plots/examples/hybrid.json \
  --plots fields --min-support 3 --formats pdf svg \
  --width 9 --output-dir output/report_figures/fields

python -m src.report_plots --config src/report_plots/examples/hybrid.json \
  --plots tradeoff --resource memory --memory-kind process --statistic max \
  --output-dir output/report_figures/memory_tradeoff
```

Use `--run KEY` repeatedly to restrict a figure or set its order. Use `--width`,
`--height`, and `--font-size` for journal/thesis dimensions, in inches and points.
`--formats pdf svg png` and `--dpi 300` are defaults. PDF is useful for LaTeX,
SVG for later vector edits, and PNG for slides/previews. Fonts are embedded as
TrueType in PDF; SVG retains editable text and requires the font on the reader's
system. The constrained layout and tight export bounds accommodate tick labels.

Each export also writes `<figure>.data.json` containing the plotted numeric
values, document IDs where relevant, exclusions, source paths, options, and
Matplotlib/NumPy versions. Output names encode the recipe and principal options.
Repeating the same output name replaces its previous exports; choose separate
output folders for different selections, warmup exclusions, or baselines.

## Sources and selection files

A source can be an `evaluation_table.csv`, another CSV with the same columns, or
a prediction folder. A folder's existing `evaluation_table.csv` is authoritative.
Without a table, the existing eval suite evaluates that folder in memory, using
the default test set and schema. Failed/missing inference outputs are not
fabricated: the eval suite's normal completeness checks still apply.

For several experiments, use a JSON config. All paths in it are relative to the
config file. Source namespaces make repeated run names unambiguous:

```json
{
  "sources": [
    {"path": "../results/New/Hybrid", "namespace": "hybrid"},
    {"path": "../results/New/Parser_based", "namespace": "parser"}
  ],
  "runs": [
    "hybrid::qwen35-9b-hybrid-input-hard-nf4-r16-chunked",
    "parser::qwen35-9b-fusion-hard-l1-bf16-r16-lr5e5"
  ],
  "labels": {
    "hybrid::qwen35-9b-hybrid-input-hard-nf4-r16-chunked": "Hybrid hard fusion",
    "parser::qwen35-9b-fusion-hard-l1-bf16-r16-lr5e5": "Text-only hard fusion"
  },
  "groups": {
    "hybrid::qwen35-9b-hybrid-input-hard-nf4-r16-chunked": "Hybrid",
    "parser::qwen35-9b-fusion-hard-l1-bf16-r16-lr5e5": "Parser"
  },
  "style": {"width": 7.2, "font_size": 10, "label_width": 30}
}
```

Each source entry also accepts `labels`, `groups`, `run_directories`, `reports`,
`details`, `testset_path`, `schema_path`, `prediction_key`, and `ground_truth_key`.
The latter two select dotted JSON paths for wrapped predictions/annotations,
using the eval suite's conventions. Source-local mappings use the original
run name; top-level labels/groups/runs use namespaced keys. An explicit
`run_directories` mapping resolves remote/relocated measurements; a `reports`
mapping points to saved `TestsetEvaluationReport.to_dict()` JSON files:

```json
{
  "sources": [{
    "path": "comparison.csv",
    "run_directories": {"model-a": "predictions/model-a"},
    "reports": {"model-a": "reports/model-a.json"}
  }]
}
```

For ordinary copied results, sibling prediction folders are preferred over old
absolute CSV paths. Recorded repository-relative locations are also resolved
locally. If no manifest is found, quality plots still work; resource plots fail
with a message identifying the missing run. Explicitly configured missing paths,
duplicate run/document IDs, manifest/table count mismatches, and nonfinite or
out-of-range scores are errors. Detailed report extraction scores and sample
counts must match their CSV snapshot, preventing stale field reports from being
silently combined with newer results. Parse/schema rates remain the CSV's raw
generation rates; some detailed reports instead describe template-filled JSON.

## Python API

```python
from src.report_plots import load_config, load_results
from src.report_plots.plots import PlotStyle, plot_quality, plot_tradeoff, plot_fields

runs, config = load_config("src/report_plots/examples/hybrid.json")
style = PlotStyle(width=7.2, font_size=10)

quality = plot_quality(runs, metrics=("field_f1", "field_recall"), style=style)
quality.figure.axes[0].set_title("Hybrid extraction results", loc="left")
quality.save("output/report_figures/quality", formats=("pdf", "png"))

tradeoff = plot_tradeoff(runs, resource="latency", statistic="p95", style=style)
tradeoff.save("output/report_figures/p95_tradeoff")

detailed = load_results("results/New/Hybrid", details=True)
fields = plot_fields(detailed, min_support=3,
                     fields=["senderInformation.senderNameCompany",
                             "itemList[].grossWeightInKg.supplyChainConsignmentItemGrossWeight"])
```

Other functions are `plot_subsets`, `plot_deltas`, `plot_latency`, `plot_ecdf`,
`plot_stages`, `plot_memory`, and `plot_paired`. Every recipe returns a `Plot` with
`.figure`, `.data`, `.options`, and `.save(...)`. Figures use an Agg canvas and do
not register pyplot windows or modify global style/RNG settings. Data loaders
can be imported independently of Matplotlib. Standard `Figure` methods remain
available for panel composition or custom export.

## Measurement and statistical conventions

- Quality uses the eval suite's existing **micro** field scores. Rates are stored
  as fractions and displayed as percentages; differences use percentage points.
  The plotting module does not change numeric normalization or reinterpret
  notation-related parse/schema flags.
- Resource distributions include completed `ok`, `parse_error`, `schema_error`,
  and `template_error` generations. Infrastructure failures and unknown statuses
  are excluded, with counts saved in sidecars. A malformed output still consumed
  inference time. Missing timings/VRAM stay missing, never zero.
- `--measure total` uses top-level `inference_seconds`; `generation` uses the
  generation stage's wall time or its native diagnostic time. A standalone
  model's top-level time may cover less work than a complete pipeline's time.
  Loading/saving are normally excluded by the inference trackers. Check hardware,
  scope, precision, image budget, and software versions before comparing runs;
  the module does not claim that different machines are normalized.
- Warmup is retained by default. `--drop-first N` explicitly removes the first N
  manifest records **before** subset and completion filtering, for resource plots
  only. It does not change quality scores. Reruns may replace individual records,
  so manifest order alone cannot reconstruct every process's first invocation.
- Median and p95 are empirical document statistics; p95 uses NumPy's linear
  quantile interpolation. Boxplots show median/IQR and 1.5×IQR whiskers, with raw
  observations and p95 separately. These are not confidence intervals. No
  concurrency/batched throughput is inferred from per-document latency.
- Stage bars use the same completed sample set for every stage of a run. Missing
  stages in a pipeline are absent work; recorded stage times must fit inside the
  total. `other` is the remaining measured wall time. Native and wrapper timing
  are never stacked together. Independent stage memory peaks are never summed.
- `process` memory requires sampled `peak_used` data. `torch_allocated`,
  `torch_reserved`, `paddle_allocated`, and `paddle_reserved` are explicit,
  separate choices. Each observation is the **maximum per-device peak**, in GiB;
  it is not a simultaneous multi-GPU total. CPU/unavailable readings are omitted.
  Sampling may miss short peaks and includes resident models/caches/contexts.
- Trade-off quality uses the complete evaluated set, while resources use measured
  completed documents; sidecars expose both populations. Pareto points minimize
  the selected resource and maximize quality. The frontier is descriptive of
  the selected runs; it does not establish statistical dominance.
- `paired` requires identical document IDs and subset assignments. Each bootstrap
  draw resamples documents with replacement and recomputes micro F1 from summed
  TP/FP/FN for both runs using the same draw. Intervals are percentile intervals
  (95% by default), reproducible with the given seed. They reflect document
  sampling uncertainty, not training-seed variation or a corrected multiple-test
  significance claim. `paired` always measures field F1.

## Verification

```bash
python -m unittest discover -s tests -p 'test_report_plots.py' -v
```

Tests use synthetic data and temporary exports. They check scientific aggregation,
failure handling, memory measurement provenance, paired bootstrap semantics,
source validation, all figure recipes, and PDF/SVG/PNG exports. They do not render
the experiment's real report figures.
