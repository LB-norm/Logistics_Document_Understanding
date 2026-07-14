# Donut Fine-Tuning Pipeline

This folder contains the Donut training and inference scripts for the CMR/Lieferschein information extraction task.

## Supported Dataset Layouts

The current default project dataset under `data/datasets/250_CMRS_240dpi_20260707` is supported directly:

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
python3 src/Donut/train_finetune.py --dry-run
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
  --max-length 1024 \
  --image-size 640 480 \
  --no-gradient-checkpointing
```

Recommended full dataset training on an NVIDIA RTX 3080 Ti:

```bash
source .venv/bin/activate

python3 src/Donut/train_finetune.py --dry-run

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 src/Donut/train_finetune.py \
  --dataset-root data/datasets/250_CMRS_240dpi_20260707 \
  --model-id naver-clova-ix/donut-base \
  --task-start-token "<s_lieferschein>" \
  --schema-path json_schema/content.schema.json \
  --target-skeleton-path json_schema/content.empty.json \
  --image-size 1280 960 \
  --max-length 1024 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 10 \
  --learning-rate 3e-5 \
  --warmup-steps 50 \
  --eval-steps 25 \
  --save-steps 25 \
  --save-total-limit 2 \
  --logging-steps 5 \
  --dataloader-num-workers 4 \
  --fp16
```

When `--output-dir` is omitted, the trainer creates a timestamped run folder under `runs/donut/`. Each run contains the best and last checkpoints, final model weights, `training_config.json`, `trainer_state.json`, plots under `plots/`, and `run_metadata.json` with dataset, target skeleton, model, parameter, duration, checkpoint, plot, and metric details. Run folder creation and normalized Trainer metric serialization are handled by the shared utilities in `src/utils/run_utils.py`, so the same metadata format can be reused by Qwen and later evaluation pipelines.

The Donut trainer tracks the best checkpoint by lowest `eval_loss`, loads that best checkpoint before saving the root model, and prunes checkpoint folders to best plus last. Keep `--save-steps` equal to `--eval-steps`; the script will override mismatched values to make checkpoint selection exact.

Generate or regenerate the training plots for an existing run:

```bash
python3 -m src.utils.training_plots runs/donut/<run-name>
```

If the 3080 Ti runs out of memory, keep `--max-length 1024` and reduce the image size first:

```bash
--image-size 960 720
```

or, more aggressively:

```bash
--image-size 768 576
```

Resume an interrupted run from a saved checkpoint:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 src/Donut/train_finetune.py \
  --output-dir runs/donut/<run-name> \
  --resume-from-checkpoint runs/donut/<run-name>/checkpoint-<step> \
  --fp16
```

The script automatically uses `val/metadata.jsonl` when `validation/metadata.jsonl` is not present. Pass `--validation-split` to override that.

## Target Shape

The target JSON is the CMR content object defined by `json_schema/content.empty.json` and `json_schema/content.schema.json`, not the annotation wrapper. With the default settings, the model learns to generate fields such as:

- `senderInformation`
- `consigneeInformation`
- `carrierInformation`
- `itemList`
- `goodsReceived`
- `referenceIdentificationNumber`

Use `--annotation-target-key root` only if you intentionally want to train on the complete annotation object including metadata.

The trainer adds field names from the schema, empty skeleton, and dataset annotations as Donut special tokens. It validates tokenized target lengths after adding those tokens and fails before training if any label sequence would be truncated. If `--max-length` exceeds the base Donut decoder position limit, decoder position embeddings are extended automatically unless `--no-resize-decoder-position-embeddings` is passed.

## Inference

After training, run inference with the fine-tuned checkpoint:

```bash
python3 src/Donut/run_inference.py \
  --model-id runs/donut/<run-name> \
  --task-prompt "<s_lieferschein>" \
  --image-path data/datasets/250_CMRS_240dpi_20260707/val/images/images/<image-file>.png \
  --schema-path json_schema/content.schema.json
```
