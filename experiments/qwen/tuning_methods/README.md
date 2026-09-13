# Qwen3.5 9B tuning-method comparison

This queue contains the seven new experiments L1 through F4. Q1 is deliberately
not repeated: `../qwen35_9b_qlora_r16.json` already provides the frozen-vision
NF4 QLoRA baseline.

| Order | ID | Base weights | Text | Vision blocks | Merger | Optimizer |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | L1 | BF16 | LoRA | Frozen | Frozen | Paged 8-bit AdamW |
| 2 | Q2 | NF4, BF16 compute | LoRA | LoRA, all 27 | LoRA | Paged 8-bit AdamW |
| 3 | L2 | BF16 | LoRA | LoRA, all 27 | LoRA | Paged 8-bit AdamW |
| 4 | F1 | BF16 | Full | Frozen | Frozen | Paged 8-bit AdamW |
| 5 | F2 | BF16 | Full | Frozen | Full | Paged 8-bit AdamW |
| 6 | F3 | BF16 | Full | Full, last 9/27 | Full | Paged 8-bit AdamW |
| 7 | F4 | BF16 | Full | Full, 27/27 | Full | Paged 8-bit AdamW |

All runs use the paged 8-bit AdamW optimizer. The model parameters and compute
remain BF16 in the unquantized runs. F4 also trains the visual patch and
positional embeddings, making it a genuinely complete full-model run. F1--F3
leave those non-block visual components frozen.

Apart from the requested tuning method, the configs retain the Q1 controlled
recipe: Qwen3.5 9B, medium resolution, rank 16/alpha 32/dropout 0.05 for adapter
runs, effective batch size 8, ten epochs, cosine decay, 5% warmup, and seed 42.
The adapter runs use a `5e-5` maximum learning rate; the full-tuning runs use
`1e-6`.

## Full-tuning learning-rate repeat

`queue_full_lr5e6.json` repeats F1 through F4 with a `5e-6` maximum learning
rate. All other controlled settings remain unchanged. The repeat configs and
run names use an `-lr5e6` suffix so the completed `1e-6` experiment definitions
and output directories remain intact.

Inspect the repeat queue with:

```bash
python src/Qwen/run_qwen_experiment_queue.py \
  experiments/qwen/tuning_methods/queue_full_lr5e6.json \
  --list
```

Launch it by replacing the queue path in the commands below with
`experiments/qwen/tuning_methods/queue_full_lr5e6.json`.

## Validate and launch

From the repository root, inspect the resolved queue without loading a model:

```bash
python src/Qwen/run_qwen_experiment_queue.py \
  experiments/qwen/tuning_methods/queue.json \
  --list
```

Run a dataset-only preflight before reserving a GPU:

```bash
python src/Qwen/run_qwen_experiment_queue.py \
  experiments/qwen/tuning_methods/queue.json \
  -- \
  --dataset-root /absolute/path/to/dataset \
  --runs-dir /absolute/path/to/qwen35-9b-tuning-runs \
  --cache-dir /absolute/path/to/model-cache \
  --dry-run \
  --max-train-samples 2 \
  --max-validation-samples 2
```

For the real run, omit the three dry-run options and use a persistent terminal:

```bash
set -o pipefail
python src/Qwen/run_qwen_experiment_queue.py \
  experiments/qwen/tuning_methods/queue.json \
  -- \
  --dataset-root /absolute/path/to/dataset \
  --runs-dir /absolute/path/to/qwen35-9b-tuning-runs \
  --cache-dir /absolute/path/to/model-cache \
  2>&1 | tee /absolute/path/to/qwen35-9b-tuning-runs/queue.log
```

The root `experiments/qwen/queue.json` mirrors this queue for convenience. Each
entry runs in a fresh process, and the queue stops at the first failure. Resume
at an experiment with, for example,
`--start-at qwen35-9b-f3-full-last9`.

The launcher verifies the loaded visual structure before training. F3 must find
at least 9 blocks, while F4 explicitly requests 27; the recorded
`training_config.json` lists selected zero-based block indexes and trainable
parameter counts for text, blocks, merger, and remaining visual components.

All seven runs use model-only epoch snapshots: optimizer, scheduler, scaler,
and RNG states are not written. After training, the best snapshot is renamed to
`best_model/`. Every other epoch snapshot is removed, including a distinct
final snapshot, and no full-model copy is written at the run root. These runs
therefore cannot resume from their retained artifacts.
