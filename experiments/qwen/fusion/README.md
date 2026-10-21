# Qwen3.5 9B OCR/layout fusion experiments

This campaign runs the same five training configurations on each of the
`separate`, `hard`, and `soft` fusion representations: **15 runs** in total.
Every run starts independently from `Qwen/Qwen3.5-9B` and uses text-only inputs.
The model's vision weights are loaded but remain frozen; no images are processed.
All 15 configs use the compact dataset and chunked answer-position loss with
256 supervised positions per chunk. All runs use a 10-epoch screening schedule;
there is no 75-step comparison limit.

| ID | Base weights | Language tuning | Peak learning rate |
| --- | --- | --- | ---: |
| Q1 | NF4 with double quantization, BF16 compute | LoRA rank 16 | `5e-5` |
| L1 | BF16 | LoRA rank 16 | `5e-5` |
| L2 | BF16 | LoRA rank 16 | `1e-5` |
| F1 | BF16 | Full language backbone and LM head | `1e-6` |
| F2 | BF16 | Full language backbone and LM head | `5e-6` |

Q1/L1 compare base-weight precision at the same learning rate. L1/L2 compare
adapter learning rates. F1/F2 compare full language learning rates. Full-tuning
runs ignore the LoRA settings retained in their JSON files. These are screening
configurations, not claims about optimal hyperparameters.

All configs explicitly use:

- `input_mode: text`, frozen vision blocks and merger, and untruncated complete
  conversations (`max_length: null`). The text pipeline checks token lengths
  against the checkpoint's language context and tokenizer before loading weights.
- `adamw_torch`: AdamW with 32-bit optimizer states, including Q1. Optimizer state
  quantization is not another varying factor in this comparison.
- Physical train/evaluation batch size 1 and gradient accumulation 8, giving an
  effective training batch size of 8 on a single GPU.
- Ten epochs, cosine decay, 5% warmup, weight decay `0.01`, and gradient clipping
  norm `1.0`.
- LoRA rank 16, alpha 32, dropout `0.05`, all matching language-side linear
  modules, and no additional `modules_to_save`, where adapters are used.
- Gradient checkpointing, SDPA attention, seed 42, and zero data-loader workers.
- Teacher-forced validation and checkpointing once per epoch; two fixed
  validation generation previews every 50 optimizer steps, with a 2048-token
  generation budget capped by remaining context space.

## Training budget

The ten-epoch budget halves the epoch-dependent work of the original twenty-epoch
campaign. The eight completed image-input runs currently available in `runs/`
all achieved their lowest epoch validation loss between epochs 3 and 5, with
higher losses afterwards. This supports a shorter screening budget, but those
runs do not establish convergence for the text-only fusion data or its lower
learning rates. Check the new validation curves and generated extraction
metrics before deciding whether selected configurations need longer training.

The cosine schedule and 5% warmup are calculated over the new ten-epoch horizon;
this is a complete shorter schedule, rather than interrupting a twenty-epoch
schedule halfway through. Compare all fifteen runs with the same epoch budget.
Loading and final evaluation add fixed overhead, so wall-clock time will be
approximately half rather than exactly half. Retained model sizes are unchanged.
If a selected run warrants twenty epochs, start a new independent run with
`--num-train-epochs 20` and a fresh output directory: the model-only checkpoints
do not contain the state needed to resume the original training trajectory.

## Dataset and paths

Prepare the data with the [batch fusion SFT builder](../../../src/fusion/README.md#batch-sft-dataset-preparation),
then generate the [compact dataset](../../../src/fusion/README.md#compact-sft-dataset):

```bash
python -m src.fusion.compact_sft
```

The configs default to the shared root:

```text
data/datasets/250_CMRS_240dpi_20260707/fusion_sft_compact/
  separate/train.jsonl   separate/val.jsonl
  hard/train.jsonl       hard/val.jsonl
  soft/train.jsonl       soft/val.jsonl
```

Each config selects its mode through `train_file` and `validation_file`. Therefore
a shared `--dataset-root` override must point to **fusion_sft_compact itself**, and keeps
all three modes distinct. Paths in training configs are relative to the repository
working directory; paths inside the queue are relative to `queue.json`.

The schema defaults explicitly to `json_schema/content.schema.json`, matching the
current prepared targets. For another target contract, pass `--schema-path` with
the corresponding prepared schema.

## Validate and launch

Run from the repository root with the project environment activated, or replace
`python` with `.venv/bin/python`.

List the enabled experiments and validate their training settings:

```bash
python src/Qwen/run_qwen_experiment_queue.py experiments/qwen/fusion/queue.json --list
```

Validate all 15 configs against all their existing train/validation examples,
without loading model dependencies or writing run outputs:

```bash
python src/Qwen/run_qwen_experiment_queue.py experiments/qwen/fusion/queue.json \
  -- --dry-run
```

Also check actual token lengths, using tokenizer/processor/config files without
loading model weights (these files may be downloaded if not cached):

```bash
python src/Qwen/run_qwen_experiment_queue.py experiments/qwen/fusion/queue.json \
  -- --dry-run --check-text-lengths
```

Launch the full campaign inside a `tmux` session:

```bash
bash experiments/qwen/fusion/run_queue.sh
```

The launcher always passes `--continue-on-error`, so an OOM or another nonzero
experiment exit is recorded and the next experiment starts in a fresh process.
All GPU allocations owned by the terminated child process are released before
the next process starts. Manual interruption still stops the queue.
Runs and the appended queue log go to `runs/qwen-fusion-compact-chunked/`, separate
from earlier standard-loss and short comparison runs. The final queue exit code
is 1 if any experiments failed, even after it attempted all 15; otherwise it is 0.

The script uses the active environment, or the repository's `.venv` when no
environment is active. Override the interpreter with `QWEN_PYTHON` or the output
root with `QWEN_FUSION_RUNS_DIR`. Additional script arguments are shared training
overrides; for example `bash experiments/qwen/fusion/run_queue.sh --dry-run`
validates all 15 datasets, and `--cache-dir /mnt/model-cache` selects a model cache.

For a remote dataset/model cache, use the same shared overrides for every entry:

```bash
python src/Qwen/run_qwen_experiment_queue.py experiments/qwen/fusion/queue.json \
  --continue-on-error \
  -- \
  --dataset-root /mnt/datasets/250_CMRS_240dpi_20260707/fusion_sft_compact \
  --runs-dir /mnt/experiments/qwen-fusion-compact-chunked \
  --cache-dir /mnt/model-cache
```

All nine adapter runs are attempted before any full-tuning run starts:

- Runs 1–3: Q1 QLoRA, in separate/hard/soft order.
- Runs 4–6: L1 LoRA, in separate/hard/soft order.
- Runs 7–9: L2 LoRA, in separate/hard/soft order.
- Runs 10–12: F1 full tuning, in separate/hard/soft order.
- Runs 13–15: F2 full tuning, in separate/hard/soft order.

Each experiment runs in a fresh process and has a unique run name.
The script continues after failures. When invoking Python directly, include
`--continue-on-error` for that behavior. To skip preceding completed entries,
invoke Python with `--start-at` before the `--`
separator, for example:

```bash
python src/Qwen/run_qwen_experiment_queue.py experiments/qwen/fusion/queue.json \
  --continue-on-error \
  --start-at qwen35-9b-fusion-hard-l1-bf16-r16-lr5e5 \
  -- --runs-dir runs/qwen-fusion-compact-chunked
```

## Outputs and comparison

Each run selects the checkpoint with the lowest validation `eval_loss` and
retains its model/adapter in `best_model/`, following the existing experiment
campaign's model-only policy (`save_only_model: true`, `save_total_limit: 1`).
These checkpoints do not contain optimizer state and cannot resume training.
Finalization renames the winning checkpoint rather than copying its weights,
and deletes the remaining epoch checkpoints. Successful runs have no separate
`last_model/` or additional weight export in the run root.
The normal training config, run metadata, VRAM measurements, token-length report
and validation previews are retained. Use `--save-resume-state --save-total-limit 2`
after `--` if resumable best/last checkpoints are required instead.

After training, generate predictions on the same complete validation split for
each retained model and compare field F1, precision/recall, JSON parse rate,
schema validity and document exact match. The two training previews are diagnostics,
not the full validation ranking. Compare fusion modes within each recipe and
select each mode's best configuration using validation results. Keep the held-out
test set for final reporting. Repeat close contenders with additional shared seeds
before drawing conclusions; this queue is the initial seed-42 screening pass.

## Test-set inference for the 14 completed runs

Run the compact-fusion inference launcher from the repository root:

```bash
.venv/bin/python -u experiments/qwen/fusion/run_testset.py
```

The launcher expects 14 completed `best_model/` directories under
`runs/qwen-fusion-compact-chunked/` and 30 PNG test images under
`data/datasets/250_CMRS_240dpi_20260707/test/images/`. It skips runs without a
`best_model/`, reuses the saved compact training prompt contract and rounding
precision, and preserves each run's NF4/BF16 loading settings. Each model runs
in a separate process; later models are attempted even if an earlier one fails.

Predictions and `inference_manifest.jsonl` diagnostics are written to
`output/qwen-fusion-compact-chunked-test/<run-name>/`. Existing results for the
same run are replaced. A nonzero exit status means at least one run reported
an inference, parsing, schema, or initialization error; inspect its diagnostics
and console output.

The default YOLO checkpoint is `runs/yolo_v11s/cmr_100epochs/best_model`.
To select another layout checkpoint, set `QWEN_FUSION_YOLO_MODEL_DIR`:

```bash
QWEN_FUSION_YOLO_MODEL_DIR=runs/yolo_v11s/cmr_50epochs/best_model \
  .venv/bin/python -u experiments/qwen/fusion/run_testset.py
```

The launcher uses the local PP-OCRv6 medium detection and recognition weights
under `output/ocr_v6_model_cache/official_models/`, plus the existing local Qwen
cache. Its Python source lives alongside the tracked experiment files so it can
be transferred with the repository.

## Recovering after a disk-space failure

The shell launcher continues after failures, including a full disk. To stop on
the first failure and free space before attempting later experiments, invoke
Python directly without `--continue-on-error`:

```bash
python src/Qwen/run_qwen_experiment_queue.py experiments/qwen/fusion/queue.json \
  -- --runs-dir runs/qwen-fusion-compact-chunked
```

Completed runs do not need to be retrained. After freeing space, restart at the
first failed experiment using its name from `queue.log` or the console output.
For example, if the first full-tuning experiment failed:

```bash
python src/Qwen/run_qwen_experiment_queue.py experiments/qwen/fusion/queue.json \
  --start-at qwen35-9b-fusion-separate-f1-bf16-full-lr1e6 \
  -- --runs-dir runs/qwen-fusion-compact-chunked
```

Repeat any original dataset/cache overrides after `--`. The failed experiment
starts training from the base model; `--start-at` skips earlier queue entries
and does not resume an epoch. Before retrying, remove or move aside that failed
experiment's output directory to avoid mixing partial checkpoints with the new
attempt. Preserve the completed experiments' directories.

Best-only retention applies after successful completion. During training, the
Trainer may retain both the best and latest checkpoints, with a third snapshot
temporarily present while saving. Failed runs can leave those snapshots or
partial files behind. Existing matching checkpoints put the final 15-run weights
at about 115 GB, with roughly 150–160 GB needed during checkpoint saves,
excluding the shared base-model cache and datasets. Keeping only the best still
requires about 18.8 GB for each completed full-tuning run.
