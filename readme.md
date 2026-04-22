# Logistics Document Understanding

This project explores multiple document information extraction pipelines for scanned real-world documents. The goal is to convert input document images into structured JSON outputs.

## Donut Fine-Tuning

The Donut pipeline lives in [src/Donut](src/Donut). It supports:

- schema definition for Lieferschein extraction
- inference with a Hugging Face Donut checkpoint
- task-specific fine-tuning on custom labeled data
- a built-in smoke-test dataset when no dataset path is provided

### Dataset Format

The fine-tuning script follows the standard Donut dataset layout:

```text
dataset_root/
├── train/
│   ├── metadata.jsonl
│   └── <image files>
└── validation/
    ├── metadata.jsonl
    └── <image files>
```

Each `metadata.jsonl` row must contain:

```json
{
  "file_name": "example.png",
  "ground_truth": "{\"gt_parse\": {\"document_type\": \"lieferschein\", \"delivery_note_number\": \"2017042708\"}}"
}
```

Notes:

- `file_name` is relative to the split directory
- `ground_truth` can be a JSON string, matching Donut conventions
- `gt_parse` must match the target extraction structure you want Donut to learn
- for this project, the reference schema is [src/Donut/lieferschein.schema.json](src/Donut/lieferschein.schema.json)

### Smoke Test

If `--dataset-root` is omitted, the training script creates a tiny local smoke-test dataset from the sample Lieferschein and example JSON.

Run a one-step smoke test with the repo venv:

```bash
./.venv/bin/python src/Donut/train_finetune.py \
  --model-id naver-clova-ix/donut-base-finetuned-cord-v2 \
  --local-files-only \
  --output-dir models/donut-lieferschein-smoketest \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --dataloader-num-workers 0 \
  --max-steps 1 \
  --num-train-epochs 1
```

### Recommended Training Commands

Recommended starting point for a 96 GB GPU:

```bash
./.venv/bin/python src/Donut/train_finetune.py \
  --dataset-root /path/to/lieferschein_dataset \
  --model-id naver-clova-ix/donut-base \
  --output-dir models/donut-lieferschein \
  --task-start-token "<s_lieferschein>" \
  --image-size 1280 960 \
  --max-length 768 \
  --per-device-train-batch-size 2 \
  --per-device-eval-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --num-train-epochs 10 \
  --learning-rate 3e-5 \
  --bf16
```

This is conservative for 96 GB VRAM. You can likely increase batch size, image size, or sequence length if the dataset requires it.

### Lower-Memory Variants

If you have less GPU memory available, reduce memory pressure in this order:

1. lower `--per-device-train-batch-size`
2. increase `--gradient-accumulation-steps`
3. reduce `--image-size`
4. reduce `--max-length`
5. keep gradient checkpointing enabled

Examples:

`24 GB` class GPU:

```bash
./.venv/bin/python src/Donut/train_finetune.py \
  --dataset-root /path/to/lieferschein_dataset \
  --model-id naver-clova-ix/donut-base \
  --output-dir models/donut-lieferschein \
  --task-start-token "<s_lieferschein>" \
  --image-size 960 720 \
  --max-length 512 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 10 \
  --learning-rate 3e-5 \
  --bf16
```

`12-16 GB` class GPU:

```bash
./.venv/bin/python src/Donut/train_finetune.py \
  --dataset-root /path/to/lieferschein_dataset \
  --model-id naver-clova-ix/donut-base \
  --output-dir models/donut-lieferschein \
  --task-start-token "<s_lieferschein>" \
  --image-size 768 576 \
  --max-length 384 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --num-train-epochs 10 \
  --learning-rate 3e-5 \
  --bf16
```

CPU-only debugging:

```bash
./.venv/bin/python src/Donut/train_finetune.py \
  --model-id naver-clova-ix/donut-base-finetuned-cord-v2 \
  --local-files-only \
  --max-steps 1 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --dataloader-num-workers 0
```

CPU training is only suitable for smoke tests and debugging.

### Inference After Fine-Tuning

Run inference with a fine-tuned checkpoint:

```bash
./.venv/bin/python src/Donut/run_inference.py \
  --model-id models/donut-lieferschein \
  --task-prompt "<s_lieferschein>"
```

### Practical Notes

- a schema file helps define and validate the target output, but Donut still needs fine-tuning to learn that schema
- the default public receipt checkpoint is only a runnable baseline, not a correct Lieferschein model
- if CUDA is not visible to PyTorch, training will fall back to CPU and become very slow
- if you use `--local-files-only`, the model must already exist in the local Hugging Face cache
