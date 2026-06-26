# PP Parser / PaddleOCR-VL

This folder currently contains PaddleOCR-VL/PP-Structure inference experiments and a
dataset preparation script for PaddleOCR-VL fine-tuning.

## Current State

- `PP_OCR_VL.py` is a minimal inference smoke script.
- `PPStructureV3_parser.py` is an inference experiment and currently contains
  unresolved Git conflict markers.
- `PP_vis.py` saves PaddleOCR visualization images.
- `prepare_finetune.py` converts the existing project image/annotation pairs into
  ERNIEKit SFT JSONL for PaddleOCR-VL VLM fine-tuning.

PaddleOCR's current recommendation is to fine-tune the PaddleOCR-VL VLM component
with ERNIEKit. Fine-tuning the layout analysis and ranking models is not currently
supported by PaddleOCR.

## Prepare SFT Data

The script accepts the same project dataset layout used by Donut and Qwen:

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

Each metadata row must contain `image` and `annotation`. By default,
`annotation["content"]` becomes the supervised answer.

Dry run:

```bash
python3 src/PP_parser/prepare_finetune.py \
  --dataset-root data/datasets/raw_data_20260527 \
  --dry-run
```

Write a prepared dataset:

```bash
python3 src/PP_parser/prepare_finetune.py \
  --dataset-root data/datasets/raw_data_20260527 \
  --output-dir data/pp_ocr_vl_sft \
  --overwrite
```

Output:

```text
data/pp_ocr_vl_sft/
  train.jsonl
  validation.jsonl
  test.jsonl
  train_manifest.jsonl
  validation_manifest.jsonl
  test_manifest.jsonl
  images/...
  preparation_summary.json
  run_erniekit_train.sh
```

The training JSONL records follow the ERNIEKit VL SFT shape:

```json
{
  "image_info": [
    {
      "image_url": "./images/train/...",
      "matched_text_index": 0
    }
  ],
  "text_info": [
    {
      "text": "Extract the CMR/Lieferschein information from this document image. Return only the target JSON object, without markdown or commentary.",
      "tag": "mask"
    },
    {
      "text": "{\"senderInformation\":{...}}",
      "tag": "no_mask"
    }
  ]
}
```

## ERNIEKit Training Handoff

Install ERNIEKit in a CUDA 12+ environment, download the base PaddleOCR-VL model,
then run the generated command template from the ERNIE repository root:

```bash
bash /path/to/this/repo/data/pp_ocr_vl_sft/run_erniekit_train.sh
```

The generated script uses:

- `examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml`
- `model_name_or_path=PaddlePaddle/PaddleOCR-VL`
- `train_dataset_path=<prepared train.jsonl>`
- estimated `max_steps` and `warmup_steps`

Adjust `packing_size`, `gradient_accumulation_steps`, `max_seq_len`, and
`learning_rate` in the ERNIEKit config or by command-line override based on GPU
memory and dataset size.

## Annotation Requirement

Your current final JSON annotations are sufficient only for this direct
image-to-final-JSON SFT formulation. That trains the VLM component to answer the
prompt with the final target JSON.

They are not sufficient to supervise PaddleOCR-VL's internal parser stages:

- no layout element boxes/classes
- no reading-order labels
- no cropped element-level OCR targets
- no table/formula/chart intermediate targets
- no alignment between final JSON fields and source text regions

If the goal is a parser-style pipeline that first reads/layout-parses the page and
then maps evidence to fields, add intermediate annotations or generated reviewed
pseudo-labels. At minimum, store OCR text spans with bounding boxes and link each
final JSON field to one or more spans. For tables/items, link rows and cells to
field paths such as `itemList[0].grossWeightInKg.supplyChainConsignmentItemGrossWeight`.

## Sources

- PaddleOCR-VL usage tutorial:
  https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html
- ERNIEKit PaddleOCR-VL SFT guide:
  https://github.com/PaddlePaddle/ERNIE/blob/release/v1.4/docs/paddleocr_vl_sft.md
