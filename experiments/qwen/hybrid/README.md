# Qwen3.5-9B image + hard-fusion experiments

The current stage-one campaign is the
[three-run NF4 input comparison](../input_comparison/README.md). This directory
retains the earlier four-run visual-adaptation/parser-dropout proposal separately;
it is not the stage-one queue.

Four independent runs combine the original full-page image with the winning compact
hard OCR/layout representation. The two factors are visual adaptation and parser
dropout. Every run starts from `Qwen/Qwen3.5-9B`; existing adapters are not merged.

| ID | Language backbone | All vision blocks + merger | Training parser dropout |
| --- | --- | --- | ---: |
| H1 | LoRA | Frozen | 0% |
| H2 | LoRA | LoRA | 0% |
| H3 | LoRA | Frozen | 20% |
| H4 | LoRA | LoRA | 20% |

Shared settings: unquantized BF16, medium resolution (2.8 MP maximum), standard
multimodal loss, LoRA rank 16 / alpha 32 / dropout 0.05, peak LR `5e-5`, AdamW
with 32-bit optimizer states (`adamw_torch`), physical batch size 1, accumulation 8,
10 epochs, cosine schedule, 5% warmup, weight decay 0.01, clipping norm 1, SDPA,
gradient checkpointing, and seed 42. Visual LoRA covers all discovered vision
blocks (27 for this checkpoint) and the merger; other visual inputs stay frozen.
No quantization or optimizer changes accompany the two experimental factors.

Parser dropout samples a Bernoulli decision each time a training example is
collated. It removes the entire dedicated OCR/layout evidence block, keeping the
image, task instructions, schema, skeleton and assistant target. Thus 20% is an
expected frequency, not an exact quota per epoch. Trainer's seeded Python RNG
controls the draws. Only training records carry dropout metadata; teacher-forced
validation, generation previews and normal inference always retain both inputs.

The compact evidence retains all OCR lines, polygons/confidences, detected region
labels/boxes/confidences and hard memberships, including unassigned text. The image
uses the whole page, with the processor applying its normal aspect-preserving
pixel budget. Do not independently crop/rotate images after producing parser data.
OCR and detector artifacts must come from the training-split-fitted parser, without
using validation/test annotations to construct evidence.

## Prepare and validate

Run from the repository root, using the project environment. On a remote machine,
transfer the original dataset, compact evidence and referenced source images
together. Provenance paths are relative to the compact dataset root.

```bash
# Validate all pairings and targets without writing files:
python -m src.fusion.hybrid_sft --dry-run

# Build the hybrid dataset (does not rerun OCR/YOLO or copy images):
python -m src.fusion.hybrid_sft

# Validate the four configurations:
python src/Qwen/run_qwen_experiment_queue.py experiments/qwen/hybrid/queue.json --list

# Validate every example in every configuration, without model dependencies:
python src/Qwen/run_qwen_experiment_queue.py experiments/qwen/hybrid/queue.json \
  --continue-on-error -- --dry-run
```

The builder reads `data/datasets/250_CMRS_240dpi_20260707/fusion_sft_compact`
and writes `fusion_sft_hybrid/` next to it:

```text
fusion_sft_hybrid/
  schema.json
  template.json
  preparation_summary.json
  hard/train.jsonl
  hard/val.jsonl
```

Override `--input-dir` and `--output-dir` to prepare another location. The builder
refuses an existing output directory, validates both splits before publishing,
checks unique IDs and images across splits, preserves targets and assistant answer
strings, and uses existing source-image provenance. It deliberately prepares only
train/val and only hard fusion. Source images are referenced, not copied or resized.

Check actual image + parser + complete-answer lengths without loading model weights:

```bash
python src/Qwen/run_qwen_training.py \
  --config experiments/qwen/hybrid/h1_frozen.json \
  --dry-run --check-multimodal-lengths
```

This requires cached/downloadable processor and model configuration files, Pillow
and PyTorch. The four runs have identical full-input lengths, so one length preflight
is sufficient. Hybrid training automatically runs it over all selected train/val
examples before loading model weights and saves `multimodal_sequence_lengths.json`.
The processor expands visual tokens at the configured resolution. Context checks
include the complete assistant answer and reject oversized inputs without
truncation, including when an explicit `--max-length` is supplied. Standard
multimodal loss can require substantially more memory than the parser campaign's
text-only chunked loss; measure training peaks on the target GPU.

For a short GPU smoke run in separate outputs, retaining the ten-epoch scheduler:

```bash
python src/Qwen/run_qwen_experiment_queue.py experiments/qwen/hybrid/queue.json \
  --continue-on-error -- --runs-dir runs/qwen-hybrid-smoke \
  --stop-after-steps 2 --validation-preview-samples 0
```

These smoke outputs are separate from the real campaign. They still run the full
data preflight and an epoch evaluation after the early training stop.

## Launch training

```bash
bash experiments/qwen/hybrid/run_queue.sh
```

Runs go to `runs/qwen-hybrid/<run-name>/`; `queue.log` is appended in the parent.
Each experiment uses a fresh Python process. The launcher continues after OOMs or
other failures and returns nonzero if any run failed. Manual interruption stops it.
`QWEN_PYTHON` chooses the interpreter; `QWEN_HYBRID_RUNS_DIR` chooses the output root.
Extra arguments override settings for every run:

```bash
bash experiments/qwen/hybrid/run_queue.sh \
  --dataset-root /mnt/datasets/250_CMRS_240dpi_20260707/fusion_sft_hybrid \
  --cache-dir /mnt/model-cache
```

Use Python's queue CLI with `--start-at qwen35-9b-hybrid-h3-bf16-r16` to skip
preceding completed runs. The lowest epoch validation loss selects `best_model/`.
Only model/adapter snapshots are retained; these configs do not save resumable
optimizer state. Run metadata and previews record the original hybrid examples.

Each retained adapter includes `hybrid_contract.json`, containing the exact shared
system/task instructions, schema, skeleton, fusion mode and compact rounding
precision. Hybrid inference automatically uses this contract, including when the
adapter is moved away from its original run or dataset.

## Validation and test inference

Infer every completed run on validation, then select the configuration using
extraction metrics rather than the two training previews:

```bash
python experiments/qwen/hybrid/run_testset.py --split val
python experiments/qwen/hybrid/run_testset.py --split test
```

The launcher defaults to GPU PP-OCRv6 medium models under
`output/ocr_v6_model_cache/official_models/` and the existing layout detector under
`runs/yolo_v11s/cmr_100epochs/best_model`. Override `--ocr-models-dir`, `--ocr-device`,
`--yolo-model-dir`, `--runs-dir`, `--dataset-root`, or `--output-dir` as needed.
`QWEN_HYBRID_YOLO_MODEL_DIR` also overrides the detector. Use `--dry-run` to check
paths/contracts without loading models. `--run NAME` selects one completed model
and can be repeated. Additional flags go to the existing inference CLI.

Images come from the selected split's metadata, or from its `images/` tree when
metadata is absent/empty (as in the existing test split). Each model runs in a fresh
process. Predictions, full fusion audit exports, per-stage timings, VRAM and raw
generation diagnostics go to `output/qwen-hybrid/<split>/<run-name>/`. Existing
outputs for that destination are replaced. A failed run does not prevent later
models from being attempted; the final status is nonzero if any failed.

Run the proposed image-only diagnostic with the same trained hybrid model:

```bash
python experiments/qwen/hybrid/run_testset.py --split val --omit-parser-evidence
```

Outputs go to `<split>/without-parser/<run-name>/`. OCR/layout still run and are
saved for auditing, but Qwen receives only the image and hybrid task contract;
therefore this diagnostic's pipeline timing still includes parser work.

Direct unified inference is also supported:

```bash
python -m src.inference_pipeline --mode hybrid \
  --model-id runs/qwen-hybrid/qwen35-9b-hybrid-h2-bf16-r16/best_model \
  --yolo-model-dir runs/yolo_v11s/cmr_100epochs/best_model \
  --no-load-in-4bit --resolution medium \
  --images /path/to/page.png
```

`hybrid` defaults to hard fusion and rejects other fusion modes. If OCR returns
no text, hybrid inference can still extract from the image. Parser-only inference
retains its existing rejection of empty OCR.

To reuse previously exported full fusion JSON without rerunning OCR/YOLO:

```bash
python src/Qwen/run_inference.py --hybrid \
  --adapter-path runs/qwen-hybrid/qwen35-9b-hybrid-h2-bf16-r16/best_model \
  --image-path /path/to/page.png --fusion-json-paths /path/to/page.fusion.json \
  --no-load-in-4bit --resolution medium --output-path output/hybrid-page.json
```

For multiple images, use `--image-paths` and supply exactly one full fusion JSON per
image in matching order. `--hybrid-contract /path/to/hybrid_contract.json` explicitly
selects a saved contract. Normal inference never applies parser dropout. Report
overall/default/challenge F1, recall, schema validity, latency and peak VRAM, plus
which image-only errors the hybrid fixes or introduces. Keep historical baselines'
optimizer differences in mind when interpreting comparisons; use matched controls
for a strict causal estimate. Prior test results informed this design, so further
results on that same test set remain exploratory.
