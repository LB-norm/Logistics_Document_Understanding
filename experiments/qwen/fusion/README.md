# Qwen3.5 9B OCR/layout fusion experiments

This campaign runs the same five training configurations on each of the
`separate`, `hard`, and `soft` fusion representations: **15 runs** in total.
Every run starts independently from `Qwen/Qwen3.5-9B` and uses text-only inputs.
The model's vision weights are loaded but remain frozen; no images are processed.

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

## Dataset and paths

Prepare the data with the [batch fusion SFT builder](../../../src/fusion/README.md#batch-sft-dataset-preparation).
The configs default to the shared root:

```text
data/datasets/250_CMRS_240dpi_20260707/fusion_sft/
  separate/train.jsonl   separate/val.jsonl
  hard/train.jsonl       hard/val.jsonl
  soft/train.jsonl       soft/val.jsonl
```

Each config selects its mode through `train_file` and `validation_file`. Therefore
a shared `--dataset-root` override must point to **fusion_sft itself**, and keeps
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

Launch the campaign:

```bash
python src/Qwen/run_qwen_experiment_queue.py experiments/qwen/fusion/queue.json \
  -- --runs-dir runs/qwen-fusion
```

For a remote dataset/model cache, use the same shared overrides for every entry:

```bash
python src/Qwen/run_qwen_experiment_queue.py experiments/qwen/fusion/queue.json \
  -- \
  --dataset-root /mnt/datasets/250_CMRS_240dpi_20260707/fusion_sft \
  --runs-dir /mnt/experiments/qwen-fusion \
  --cache-dir /mnt/model-cache
```

The queue runs Q1 for separate/hard/soft first, then L1, L2, F1 and F2 in the same
mode order. Each experiment runs in a fresh process and has a unique run name.
The queue stops on the first failure; `--continue-on-error` attempts subsequent
entries. To skip preceding completed entries, put `--start-at` before the `--`
separator, for example:

```bash
python src/Qwen/run_qwen_experiment_queue.py experiments/qwen/fusion/queue.json \
  --start-at qwen35-9b-fusion-hard-l1-bf16-r16-lr5e5 \
  -- --runs-dir runs/qwen-fusion
```

## Outputs and comparison

Each run selects the checkpoint with the lowest validation `eval_loss` and
retains its model/adapter in `best_model/`, following the existing experiment
campaign's model-only policy (`save_only_model: true`, `save_total_limit: 1`).
These checkpoints do not contain optimizer state and cannot resume training.
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
