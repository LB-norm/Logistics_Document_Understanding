# Qwen experiment configurations

This directory stores portable, version-controlled Qwen experiment definitions.
The training code remains in `src/Qwen`; these JSON files describe only what is
different for a particular run.

For short 27B NF4/INT8 capacity probes at two image resolutions, see the
separate [memory-test queue](memory_tests/README.md).

The current queue runs the model-size screening campaign for the official Qwen3.5
4B, 9B, and 27B checkpoints sequentially with the same frozen-vision NF4 QLoRA
rank-16 recipe. Training sequence truncation is disabled. The non-quantized BF16
LoRA configurations and the 35B-A3B configuration remain available for reference
but are excluded from `queue.json`.

## Run one experiment

From the repository root:

```bash
python src/Qwen/run_qwen_training.py \
  --config experiments/qwen/qwen35_4b_qlora_r16.json \
  --dataset-root /mnt/datasets/250_CMRS_240dpi_20260707
```

Explicit command-line arguments override values in both the JSON file and the
project launcher defaults. Paths that depend on the machine, especially
`--dataset-root`, `--runs-dir`, and `--cache-dir`, should normally be supplied on
the command line instead of committed in an experiment file.

## Inspect or run a queue

Validate the queue and show its entries without starting anything:

```bash
python src/Qwen/run_qwen_experiment_queue.py \
  experiments/qwen/queue.json \
  --list
```

Run all enabled experiments sequentially, forwarding shared machine-specific
arguments after `--`:

```bash
python src/Qwen/run_qwen_experiment_queue.py \
  experiments/qwen/queue.json \
  -- \
  --dataset-root /mnt/datasets/250_CMRS_240dpi_20260707 \
  --runs-dir /mnt/experiments/qwen \
  --cache-dir /mnt/model-cache
```

Each experiment starts in a fresh Python process. This avoids retaining a model
or CUDA allocations when the next queued experiment starts. The queue stops on
the first failure by default. Use `--continue-on-error` to attempt later entries,
or `--start-at EXPERIMENT_NAME` to restart from a specific active entry.

Before leaving a long remote run unattended, use `--list` and verify the dataset,
run-output, and model-cache paths passed to the queue command.

## Peak VRAM logging

Both QLoRA and non-quantized LoRA runs automatically write `vram_usage.json` in
the run directory. It is refreshed at training log steps, evaluations, training
end, and completion, with the latest peaks also printed to the console (and
`queue.log` when using `tee`). Completed runs include the same summary under
`vram` in `run_metadata.json`. On a caught training failure, the code also tries
to save the peaks reached before failure.

Each visible CUDA device has `peak_allocated_bytes`, `peak_reserved_bytes`, and
equivalent `_gib` fields. Peaks are tracked continuously from before model
loading, including training, validation previews, and final evaluation. Reading
the counters does not reset them, so short allocation spikes between log steps
are retained. Resuming training starts a new measurement window.

These are the current process's PyTorch allocator peaks: allocated memory holds
tensors; reserved memory also includes the allocator's cached blocks. They
exclude CUDA context overhead, allocations outside PyTorch, and other processes,
so they do not represent the total board usage reported by `nvidia-smi` or an
exact minimum GPU capacity. Without CUDA, the summary reports `available: false`.

## Current model-size screening

The three queued configurations differ only in their `model_id`, experiment name,
and description. Their controlled training recipe is:

- 4-bit NF4 QLoRA with BF16 compute
- frozen vision encoder and language-side `all-linear` LoRA targets
- LoRA rank 16, alpha 32, and dropout 0.05
- physical batch size 1 with 8 gradient accumulation steps
- ten epochs with a maximum learning rate of `5e-5`
- cosine learning-rate decay with 5% of optimizer steps used for warmup
- AdamW weight decay of `0.01`
- `medium` resolution (2.8 MP maximum), untruncated training sequences, and seed 42
- 2048-token generation budget for validation previews
- evaluation and checkpointing after every epoch, retaining the best and last model

## Remote launch checklist

Run a dataset-only validation of every queued experiment before using the GPU:

```bash
python src/Qwen/run_qwen_experiment_queue.py \
  experiments/qwen/queue.json \
  -- \
  --dataset-root /absolute/path/to/dataset \
  --runs-dir /absolute/path/to/qwen-screening-runs \
  --cache-dir /absolute/path/to/model-cache \
  --dry-run \
  --max-train-samples 2 \
  --max-validation-samples 2
```

Then start the real queue inside `tmux`, omitting all three dry-run arguments:

```bash
tmux new -s qwen-screening

cd /absolute/path/to/Masterstudienarbeit
source .venv/bin/activate
mkdir -p /absolute/path/to/qwen-screening-runs

python src/Qwen/run_qwen_experiment_queue.py \
  experiments/qwen/queue.json \
  -- \
  --dataset-root /absolute/path/to/dataset \
  --runs-dir /absolute/path/to/qwen-screening-runs \
  --cache-dir /absolute/path/to/model-cache \
  2>&1 | tee /absolute/path/to/qwen-screening-runs/queue.log
```

Detach with `Ctrl-b`, then `d`, and reconnect later with
`tmux attach -t qwen-screening`. The queue stops on the first failed experiment.
After resolving a failure, `--start-at EXPERIMENT_NAME` can skip the completed
entries.

## Experiment format

```json
{
  "format_version": 1,
  "name": "unique-portable-run-name",
  "description": "What this experiment changes and why.",
  "training": {
    "model_id": "Qwen/Qwen3.5-9B",
    "vision_tuning": "frozen",
    "num_train_epochs": 10.0,
    "learning_rate": 0.00005,
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "save_total_limit": 2,
    "seed": 42
  }
}
```

`name` becomes `run_name` unless the training object explicitly supplies one.
Every key in `training` must match a command-line option from
`run_qwen_training.py`, written with underscores instead of hyphens. Unknown keys
are rejected before the model is loaded.

The project defaults evaluate and save once per epoch, select the lowest validation
loss, and retain both the best and final checkpoint. Completed runs additionally expose
model-only `best_model/` and `last_model/` adapter directories. Experiment configs cannot
set `save_total_limit` below 2 or use mismatched evaluation and save strategies.

## Queue format

Queue paths are resolved relative to the queue file:

```json
{
  "format_version": 1,
  "name": "qwen-experiment-queue",
  "description": "Sequential runs for one GPU.",
  "experiments": [
    {"config": "baseline.json", "enabled": true},
    {"config": "vision_lora.json", "enabled": false}
  ]
}
```
