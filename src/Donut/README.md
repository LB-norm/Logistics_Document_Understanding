# Donut Fine-Tuning Pipeline

This folder contains the Donut training and inference scripts for the CMR/Lieferschein information extraction task.

## Supported Dataset Layouts

The main project dataset under `data/datasets/raw_data_20260527` is supported directly:

```text
dataset_root/
  train/
    metadata.jsonl
    images/...
    annotations/...
  val/
    metadata.jsonl
    images/...
    annotations/...
  test/
    metadata.jsonl
    images/...
    annotations/...
```

Each metadata row points to the image and annotation file:

```json
{
  "id": "cmr_dachser__example_page_1",
  "image": "train/images/cmr_dachser/example_page_1.jpg",
  "annotation": "train/annotations/cmr_dachser/example_page_1_0.json"
}
```

The annotation files contain:

```json
{
  "content": {
    "senderInformation": {},
    "consigneeInformation": {},
    "itemList": []
  },
  "metadata": {
    "prompt": "Extract all relevant information from the document.",
    "method": "zero_shot_prompting"
  }
}
```

By default, training uses only `annotation["content"]` as the Donut `gt_parse` target. Annotation `metadata` is ignored.

The trainer also still accepts:

- official Donut split folders with `metadata.jsonl` rows containing `file_name` and `ground_truth`
- a flat folder with image/json pairs, such as `data/small testing`

For flat folders, a single example is reused for validation so the Trainer can run a smoke test.

## Dry Run

Validate dataset parsing without loading the model:

```bash
python3 src/Donut/train_finetune.py \
  --dataset-root data/datasets/raw_data_20260527 \
  --dry-run
```

Quick one-example dry run:

```bash
python3 src/Donut/train_finetune.py \
  --dataset-root "data/small testing" \
  --dry-run
```

## Training

Small local smoke train from the copied one-example folder:

```bash
python3 src/Donut/train_finetune.py \
  --dataset-root "data/small testing" \
  --model-id models/donut-lieferschein-smoketest-cpucheck \
  --local-files-only \
  --output-dir /tmp/donut-train-pipeline-smoke \
  --max-steps 1 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --dataloader-num-workers 0 \
  --max-length 128 \
  --image-size 640 480 \
  --no-gradient-checkpointing
```

Full dataset training:

```bash
python3 src/Donut/train_finetune.py \
  --dataset-root data/datasets/raw_data_20260527 \
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

The script automatically uses `val/metadata.jsonl` when `validation/metadata.jsonl` is not present. Pass `--validation-split` to override that.

## Target Shape

The target JSON is the CMR content object defined by `src/Donut/lieferschein.schema.json`, not the annotation wrapper. With the default settings, the model learns to generate fields such as:

- `senderInformation`
- `consigneeInformation`
- `carrierInformation`
- `itemList`
- `goodsReceived`
- `referenceIdentificationNumber`

Use `--annotation-target-key root` only if you intentionally want to train on the complete annotation object including metadata.

## Inference

After training, run inference with the fine-tuned checkpoint:

```bash
python3 src/Donut/run_inference.py \
  --model-id models/donut-lieferschein \
  --task-prompt "<s_lieferschein>"
```
