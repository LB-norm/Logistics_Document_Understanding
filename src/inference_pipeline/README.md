# Unified document inference

Run one of the existing end-to-end models, or compose PP-OCRv6, YOLO layout
detection, spatial fusion, and a downstream Qwen LLM. Models load once and are
reused for sequential page-image inference. The pipeline calls the existing model
loading/generation functions; their defaults and JSON normalization remain in
their respective modules.

| Mode | Model selection | Execution |
| --- | --- | --- |
| `end-to-end` | Qwen base model, Qwen fine-tuned adapter/merged checkpoint, or Donut checkpoint | Image → existing model inference → content JSON |
| `parser` | Qwen base model or fine-tuned downstream Qwen checkpoint/adapter, plus YOLO weights and fusion mode | Image → PP-OCR + YOLO → fusion → text-only Qwen generation → content JSON |

Choose models using a Hugging Face ID or local path. Qwen adapter directories can
be selected with `--adapter-path` or supplied directly as `--model-id`; the native
adapter metadata resolves the base model and saved processor. An explicit
`--model-id` together with `--adapter-path` selects the base model explicitly.
Complete merged Qwen checkpoints use `--model-id`. No hardcoded catalog needs
updating when additional fine-tuned models become available.

## Commands

Run from the repository root. End-to-end Qwen base model:

```bash
python -m src.inference_pipeline \
  --mode end-to-end \
  --model-type qwen \
  --model-id Qwen/Qwen3.5-2B \
  --images data/example.png \
  --output-dir output/pipeline/qwen-base
```

End-to-end Qwen fine-tuned adapter, with its matching base model inferred:

```bash
python -m src.inference_pipeline \
  --mode end-to-end \
  --adapter-path runs/qwen/my-experiment/best_model \
  --images data/first.png data/second.png \
  --output-dir output/pipeline/qwen-finetuned
```

End-to-end Donut checkpoint:

```bash
python -m src.inference_pipeline \
  --mode end-to-end \
  --model-type donut \
  --model-id runs/donut/my-experiment \
  --images data/example.png \
  --task-prompt '<s_lieferschein>' \
  --schema-path json_schema/content.schema.json \
  --example-path json_schema/content.empty.json \
  --annotation-target-key root \
  --output-dir output/pipeline/donut
```

Those Donut template/schema/prompt overrides match the project's CMR contract;
without overrides its existing inference defaults still apply. The current native
Donut inference supports CPU only.

Parser mode with a base downstream LLM:

```bash
python -m src.inference_pipeline \
  --mode parser \
  --model-id Qwen/Qwen3.5-9B \
  --fusion-mode 3 \
  --yolo-model-dir runs/yolo_v11s/cmr/best_model \
  --images data/example.png \
  --output-dir output/pipeline/parser-soft
```

For a trained downstream LLM, replace `--model-id` with a merged checkpoint or use
`--adapter-path runs/qwen/parser-experiment/best_model`. `--fusion-mode` is required
in parser mode and accepts `1`/`separate`, `2`/`hard`, or `3`/`soft`.
[Fusion semantics](../fusion/README.md) are unchanged. The complete representation
is passed in memory to the existing Qwen generation path with text tokens only;
the LLM receives no page pixels. The prompt explains memberships, soft geometric
weights, repeated OCR IDs, and unassigned text. An empty OCR result stops that
image with an error rather than generating from layout metadata alone.

## Settings and validation

Model-specific flags are forwarded to the selected model's existing argument
parser. For example, append `--resolution high --max-new-tokens 4096` for Qwen or
`--max-length 1024 --num-beams 2` for Donut. Without overrides, Qwen retains its
native resolution, quantization, dtype, device map, prompt, template, schema, and
generation settings. Donut retains its native processor and generation settings.

Use these commands to inspect the wrapper or native model flags:

```bash
python -m src.inference_pipeline --help
python -m src.inference_pipeline --model-help --model-type qwen
python -m src.inference_pipeline --model-help --model-type donut
```

YOLO retains its saved image size, NMS configuration, automatic device selection,
and default detection threshold. PP-OCR retains `OCRConfig()` defaults, including
CPU inference and medium model size. Optional wrapper settings include
`--yolo-device`, `--yolo-threshold`, `--ocr-device`, `--ocr-model-size`,
`--ocr-min-confidence`, `--ocr-detection-model-dir`,
`--ocr-recognition-model-dir`, and `--min-overlap`. For Paddle GPU inference, add
`--ocr-device gpu:0` with the corresponding Paddle GPU installation.

Append `--dry-run` to validate input files, output path collisions, selected
configuration, model-native arguments, schema/template, adapter directory, and
YOLO weights/config paths. It loads no model weights and writes no files; it does
not verify remote model availability or the content of weight files.

Inputs are page images with unique stems within a command. Render PDFs to page
images before using this entry point so every mode is evaluated per image, with
OCR and YOLO receiving the same raster and coordinate frame. The standalone OCR
and fusion modules continue to support multi-page documents.

## Python API

```python
from src.inference_pipeline import InferencePipeline, PipelineConfig, run_inference

pipeline = InferencePipeline(PipelineConfig(
    mode="parser",
    model_id="Qwen/Qwen3.5-9B",
    fusion_mode=3,
    yolo_model_dir="runs/yolo_v11s/cmr/best_model",
))
result = pipeline.predict("data/example.png")
generated_json = result["prediction"]
seconds = result["inference_seconds"]
vram = result["vram"]

# The same instance reuses models on subsequent images.
second = pipeline.predict("data/another.png")

# Convenience function: one shared model load for the entire sequence.
results = run_inference(
    ["data/first.png", "data/second.png"],
    PipelineConfig(mode="end-to-end", model_id="Qwen/Qwen3.5-2B"),
)
```

`PipelineConfig.model_args` is a tuple of native CLI flags, such as
`("--resolution", "high")`. `ocr_config` accepts an existing `OCRConfig` object.
`pipeline.load()` can explicitly initialize models before prediction. Native
input/output/model-selection flags belong to `PipelineConfig` rather than
`model_args`.

Results contain `prediction`, `status`, source/model provenance,
`inference_seconds`, `vram`, `stages`, and generation diagnostics. Parser results
also contain `fusion`. Initialization/preflight failures raise exceptions;
per-image execution failures return `status: "inference_error"`, an `error`,
`prediction: null`, elapsed time, and available measurements. Batch processing
continues after an image failure. Parse/schema/template problems retain generated
predictions with a non-`ok` status.

## Artifacts and measurements

Each command writes `<image-stem>.json` containing only the final template-filled
prediction and `inference_manifest.jsonl` containing one diagnostic record per
image. Parser mode also writes `<image-stem>.fusion.json`. The manifest records
artifact paths, raw generation diagnostics, total time, per-stage time, and VRAM.
Records are flushed as each image finishes. Existing output artifacts are replaced
on a new run; choose separate directories when comparing models/modes. Exit status
is nonzero if any image fails generation, parsing, template matching, or schema checks.

`inference_seconds` covers the whole per-image execution, including image loading,
OCR/layout inference, fusion, LLM generation, and JSON normalization/diagnostics.
It excludes model initialization and output-file writing. Native GPU stage
trackers synchronize GPU work before returning. `stages` also contains each step's
wall time, its existing `native_inference_seconds`, and its allocator VRAM metrics
when available. Fusion has timing only.

The top-level `vram` records a **sampled peak of GPU memory for the current
process**, queried through `nvidia-smi` about every 100 ms. This includes both
Paddle and PyTorch, resident model weights, caches, and CUDA contexts. Devices use
GPU UUIDs, with `peak_used_bytes` and `peak_used_gib`. Sampling can miss allocations
shorter than a poll, and monitoring adds some execution overhead. Timed execution
excludes the initial/final GPU probes and sampler shutdown. No extra pip package
is required for the sampler.

`vram.framework_peaks` retains the original PyTorch and Paddle allocator peaks
separately, including allocated/reserved bytes and GiB. These are maxima across
the recorded stages, with framework-local device indices. Independent peaks are
not summed to invent a simultaneous total. Without process GPU readings, parser
mode marks `combined_peak_available: false` and keeps those framework metrics;
end-to-end mode can fall back to its single framework's allocator peak. CPU runs
mark VRAM unavailable. All memory measurements refer to the current process,
rather than total GPU board usage, and include any other resident models owned
by that process.

## Tests

Tests use fake model runtimes and actual native preprocessing/generation adapters,
so they run without downloading weights:

```bash
python -m unittest discover -s tests -p 'test_inference_pipeline.py' -v
python -m unittest discover -s tests -p 'test_inference_tracking.py' -v
```
