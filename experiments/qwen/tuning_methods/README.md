# Qwen3.5 9B tuning-method comparison

This queue contains the seven new experiments L1 through F4. Q1 is deliberately
not repeated: `../qwen35_9b_qlora_r16.json` already provides the frozen-vision
NF4 QLoRA baseline.

| Order | ID | Base weights | Text | Vision blocks | Merger | Optimizer |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | L1 | BF16 | LoRA | Frozen | Frozen | AdamW |
| 2 | Q2 | NF4, BF16 compute | LoRA | LoRA, all 27 | LoRA | Paged 8-bit AdamW |
| 3 | L2 | BF16 | LoRA | LoRA, all 27 | LoRA | AdamW |
| 4 | F1 | BF16 | Full | Frozen | Frozen | Paged 8-bit AdamW |
| 5 | F2 | BF16 | Full | Frozen | Full | Paged 8-bit AdamW |
| 6 | F3 | BF16 | Full | Full, last 9/27 | Full | Paged 8-bit AdamW |
| 7 | F4 | BF16 | Full | Full, 27/27 | Full | Paged 8-bit AdamW |

The full-fine-tuning runs use an 8-bit optimizer to reduce optimizer-state
memory; the model parameters and compute remain BF16. F4 also trains the visual
patch and positional embeddings, making it a genuinely complete full-model run.
F1--F3 leave those non-block visual components frozen.

Apart from the requested tuning method and the optimizer needed by that method,
the configs retain the Q1 controlled recipe: Qwen3.5 9B, medium resolution,
rank 16/alpha 32/dropout 0.05 for adapter runs, effective batch size 8, ten
epochs, `5e-5` maximum learning rate, cosine decay, 5% warmup, and seed 42.

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

Full-model checkpoints are much larger than adapter checkpoints. With best and
last resumable checkpoints plus the two exported model directories, reserve
substantial disk space before starting F1--F4.
