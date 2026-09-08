# Qwen3.5 27B memory probes

Four short runs compare base-weight quantization and image pixel budgets. All
runs use `Qwen/Qwen3.5-27B` and start in separate Python processes. The configs
are separate from the model-size screening queue.

| Order | Base-weight format | Maximum image pixels | Square-equivalent budget |
| --- | --- | ---: | --- |
| 1 | NF4 QLoRA, double quantization | 1,048,576 | 1024 x 1024 (current default) |
| 2 | NF4 QLoRA, double quantization | 4,194,304 | 2048 x 2048 |
| 3 | LLM.int8() + LoRA | 1,048,576 | 1024 x 1024 |
| 4 | LLM.int8() + LoRA | 4,194,304 | 2048 x 2048 |

These are pixel caps, not forced square dimensions. The processor preserves
aspect ratio and rounds to patch-compatible dimensions. The higher budget
allows four times as many pixels; small source images may not use the full cap.

## Why INT8

The 8-bit cases use bitsandbytes `LLM.int8()` (`load_in_8bit=True`), with the
standard outlier threshold of 6.0 and no FP32 CPU weight offload or retained
FP16 weight copy. The frozen quantized base is prepared for k-bit training,
and LoRA adapters are trained. BF16 remains the Trainer mixed-precision setting;
the INT8 backend handles its own mixed-precision/outlier computation.

This is distinct from an 8-bit optimizer: **all four cases use the same
`paged_adamw_8bit` optimizer**. FP8 is a possible separate experiment, but it
would also change the quantization backend and hardware compatibility, so it
is not included in this comparison. See the official
[bitsandbytes quantization documentation](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
and [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization).

## Fixed workload

- 10 optimizer steps, batch size 1, gradient accumulation 8: normally 80 training
  examples per run, sampled from the same full training split with seed 42.
- Frozen vision encoder; language-side LoRA rank 16, alpha 32, dropout 0.05.
- BF16 mixed precision, gradient checkpointing, SDPA, learning rate `1e-4`.
- No sequence truncation. JSON target length therefore contributes to memory.
- Log, evaluate, and save at steps 5 and 10. Evaluation uses the first 8
  validation examples; 2 fixed previews allow up to 2048 generated tokens.
- The same checkpoint behavior as normal training, including best/last copies.
  Allow disk space for adapters and optimizer checkpoints from four runs.

The `max_steps` limit takes precedence over `num_train_epochs`. These are
capacity probes, not model-quality experiments. They measure the peak for the
samples actually processed, not the worst possible sample in the whole dataset.

## Run remotely

Use the updated training code and these configs on the remote machine, with the
project's training dependencies installed. From the repository root, activate
the remote Python environment and run:

```bash
python src/Qwen/run_qwen_experiment_queue.py \
  experiments/qwen/memory_tests/queue.json --list
```

Launch all four tests, replacing the machine-specific paths below. Use a fresh
`--runs-dir` for each campaign: experiment names are fixed, so reusing a directory
can mix new results with previous checkpoints. `CUDA_VISIBLE_DEVICES=0` selects
one GPU; change the index if needed. Leave that GPU free of other workloads for
a comparable measurement.

```bash
mkdir -p /mnt/experiments/qwen27b-memory-01
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python src/Qwen/run_qwen_experiment_queue.py \
  experiments/qwen/memory_tests/queue.json \
  --continue-on-error \
  -- \
  --dataset-root /mnt/datasets/250_CMRS_240dpi_20260707 \
  --runs-dir /mnt/experiments/qwen27b-memory-01 \
  --cache-dir /mnt/model-cache \
  2>&1 | tee /mnt/experiments/qwen27b-memory-01/queue.log
```

`--continue-on-error` attempts the remaining tests after any failed process,
including OOM. A fresh process releases the previous run's GPU allocations.
The queue exits with code 1 after finishing if any test failed; this is expected
when a memory probe fails. An interrupted queue can be restarted with
`--start-at qwen35-27b-memory-03-int8-default`, for example.

For a dataset-only preflight, append `--dry-run` to the forwarded training
arguments before the shell redirection. This checks dataset/config loading but
does not test CUDA, model loading, or quantization kernels.

## Read the results

Each experiment has its own directory under `--runs-dir`:

- `vram_usage.json`: latest per-device PyTorch peaks, in bytes and GiB, updated
  while training and after a caught failure.
- `run_metadata.json`: `status` (`completed` or `failed`), final `vram` summary,
  and `error.type` / `error.message` on failure.
- `validation_previews/`: generated examples or preview failure details.
- The campaign's `queue.log`: output from all four processes, including peaks
  and full error traces.

Compare `vram.devices[0].peak_allocated_gib` and `peak_reserved_gib` in each
run's metadata. Allocated is live tensor memory; reserved also includes cached
allocator blocks. Peaks include model loading, training, previews, evaluation,
and final saving, and retain spikes between log updates. They are **PyTorch
allocator peaks**, not total board usage or an exact minimum GPU capacity:
CUDA contexts, external allocations, and other processes are excluded.

A failed allocation is not included in the allocated peak. For OOM cases, read
the exception's requested allocation as well. A preview OOM also fails the run.
Other failures (missing dependencies, unsupported kernels, invalid data) do not
establish a VRAM limit. If the OS kills the process, only the last persisted
snapshot may survive and metadata can remain `running`.

The default NF4 run is the expected baseline, but success still depends on the
remote environment and sampled sequence lengths. If all four pass, this matrix
has not reached the limit. Repeat with a larger `max_pixels` budget in both
high-resolution configs, or run a full epoch to exercise more target lengths.
