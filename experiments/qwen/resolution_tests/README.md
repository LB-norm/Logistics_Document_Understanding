# Qwen3.5 9B image-resolution study

This queue trains the missing image-resolution variants for the Qwen3.5 9B
model. The existing `qwen35-9b-qlora-r16-screening-untruncated` run supplies the
`medium` baseline, so it is deliberately not queued again.

| Queue order | Training resolution | Maximum image pixels |
| --- | --- | ---: |
| 1 | `low` | 1,400,000 |
| 2 | `high` | 4,200,000 |
| 3 | `native` | 5,600,000 |

All three runs otherwise copy the existing 9B baseline exactly: ten epochs,
frozen vision encoder, NF4 QLoRA rank 16, BF16 compute, batch size 1 with eight
gradient-accumulation steps, cosine decay with 5% warmup, and seed 42. The
resolution values are upper pixel budgets; aspect ratio is preserved and images
smaller than a budget are not enlarged to fill it.

Validate the queue from the repository root:

```bash
python src/Qwen/run_qwen_experiment_queue.py \
  experiments/qwen/resolution_tests/queue.json \
  --list
```

Launch it with the machine-specific paths forwarded to all three runs:

```bash
python src/Qwen/run_qwen_experiment_queue.py \
  experiments/qwen/resolution_tests/queue.json \
  -- \
  --dataset-root /absolute/path/to/dataset \
  --runs-dir /absolute/path/to/qwen-resolution-runs \
  --cache-dir /absolute/path/to/model-cache
```

For the later inference comparison, pass `--resolution low`, `medium`, `high`,
or `native` to `src/Qwen/run_inference.py`. Keep both the trained adapter and the
inference resolution in the result identity: this supports a full 4-by-4 matrix
that separates training-resolution effects from inference-resolution effects.
