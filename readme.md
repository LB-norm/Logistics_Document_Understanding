# Logistics Document Understanding

Experimental pipelines for extracting structured information from scanned logistics documents, with a focus on German delivery notes (*Lieferscheine*) and CMR documents. Each pipeline converts a document image into JSON that follows the supplied delivery-note schema.

## Included Pipelines

| Pipeline | Purpose | Documentation |
| --- | --- | --- |
| Donut | Fine-tune and run the Donut vision encoder-decoder model. | [Donut pipeline](src/Donut/README.md) |
| Qwen | LoRA/QLoRA fine-tune and run Qwen3.5 vision-language models. | [Qwen pipeline](src/Qwen/README.md) |
| PaddleOCR-VL | Run PaddleOCR-VL and prepare supervised fine-tuning data for ERNIEKit. | [PaddleOCR-VL pipeline](src/PP_parser/README.md) |
| Dataset utilities | Create deterministic train/validation/test splits from image and annotation pairs. | [dataset_utils.py](src/utils/dataset_utils.py) |

The target output contract is defined by [json_schema/content.empty.json](json_schema/content.empty.json) and [json_schema/content.schema.json](json_schema/content.schema.json).

## What Is Not Included

Datasets, trained checkpoints, generated documents, inference output, and private research material are intentionally excluded from version control. Provide your own data and model checkpoints when running the pipelines.

## Setup

Use Python 3.10 or later. Python 3.12 is the development environment used for this repository.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` installs the general Python dependencies. For GPU execution, install PyTorch and PaddlePaddle wheels that match your CUDA version. The synthetic document generator additionally requires Poppler's `pdftoppm` executable and the system libraries required by WeasyPrint.

## Dataset Format

Donut, Qwen, and the PaddleOCR-VL preparation script accept the same project dataset layout:

```text
dataset_root/
├── train/
│   ├── metadata.jsonl
│   ├── images/...
│   └── annotations/...
├── val/
│   ├── metadata.jsonl
│   ├── images/...
│   └── annotations/...
└── test/                         # optional
    ├── metadata.jsonl
    ├── images/...
    └── annotations/...
```

Each line in `metadata.jsonl` identifies an image and its annotation file:

```json
{
  "id": "cmr_example_page_1",
  "image": "train/images/cmr_example_page_1.jpg",
  "annotation": "train/annotations/cmr_example_page_1.json"
}
```

By default, the scripts use `annotation["content"]` as the supervised target and ignore annotation metadata. The Donut and Qwen trainers also support their respective native metadata formats; see their pipeline documentation for details.

## Quick Start: Donut

First validate that the dataset can be read. This does not load a model:

```bash
python src/Donut/train_finetune.py --dry-run
```

Fine-tune Donut on the dataset:

```bash
python src/Donut/train_finetune.py \
  --dataset-root data/datasets/250_CMRS_240dpi_20260707 \
  --model-id naver-clova-ix/donut-base \
  --task-start-token "<s_lieferschein>" \
  --schema-path json_schema/content.schema.json \
  --target-skeleton-path json_schema/content.empty.json \
  --image-size 1280 960 \
  --max-length 1024 \
  --per-device-train-batch-size 2 \
  --per-device-eval-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --num-train-epochs 10 \
  --learning-rate 3e-5 \
  --bf16
```

When `--output-dir` is omitted, the trainer creates a timestamped run folder under `runs/donut/` containing the best and last checkpoints, final weights, `training_config.json`, `trainer_state.json`, plots, and `run_metadata.json`. The run-folder and normalized metric helpers live in `src/utils/run_utils.py` so Qwen and later evaluation pipelines can use the same metadata shape. The command is a starting point for a high-memory GPU. Reduce batch size or image size for smaller GPUs. CPU training is intended only for parsing checks and smoke tests.

Training plots can be generated again from any run folder with:

```bash
python3 -m src.utils.training_plots runs/donut/<run-name>
```

Run a fine-tuned checkpoint on an image:

```bash
python src/Donut/run_inference.py \
  --image-path /path/to/document.jpg \
  --model-id runs/donut/<run-name> \
  --task-prompt "<s_lieferschein>" \
  --schema-path json_schema/content.schema.json \
  --example-path /path/to/example_annotation.json \
  --output-path output/result.json
```

## Other Workflows

- Use [src/Qwen/README.md](src/Qwen/README.md) for Qwen dataset formats, LoRA/QLoRA training, and inference.
- Use [src/PP_parser/README.md](src/PP_parser/README.md) to run PaddleOCR-VL or create ERNIEKit SFT data. PaddleOCR-VL fine-tuning itself requires an ERNIEKit environment.
- Use [src/dataset_gen](src/dataset_gen) to generate synthetic delivery notes and [src/utils/dataset_utils.py](src/utils/dataset_utils.py) to build dataset splits.

## Notes

- The schema describes the desired output contract; it does not make a base model capable of extracting delivery-note fields without appropriate fine-tuning.
- Donut training validates tokenized target lengths after adding schema/skeleton special tokens and refuses to train if labels would be truncated.
- The public Donut receipt checkpoint is useful as a runnable baseline, not as a delivery-note model.
- `--local-files-only` requires all referenced model files to be present in the local Hugging Face cache.
