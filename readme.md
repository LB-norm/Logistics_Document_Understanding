# Logistics Document Understanding

Training and inference code for extracting structured JSON from scanned logistics
documents. The current dataset and schema cover CMR documents and German delivery notes.

## Pipelines

| Pipeline | Use | Documentation |
| --- | --- | --- |
| Donut | Full fine-tuning of the Donut vision encoder-decoder model | [Donut](src/Donut/README.md) |
| Qwen | LoRA/QLoRA fine-tuning of Qwen vision-language models | [Qwen](src/Qwen/README.md) |
| PaddleOCR-VL | Inference and ERNIEKit data preparation | [PaddleOCR-VL](src/PP_parser/README.md) |
| Dataset tools | Dataset generation, auditing, and deterministic splits | [Dataset utilities](src/utils/dataset_utils.py) |

The output format is defined by
[content.empty.json](json_schema/content.empty.json) and
[content.schema.json](json_schema/content.schema.json).

Datasets, downloaded models, training runs, and inference output are ignored by Git.

## Setup

Python 3.12 is used for development. Create a virtual environment and install the
dependencies:

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install CUDA-specific PyTorch or PaddlePaddle wheels separately when the versions from
`requirements.txt` do not match the local CUDA setup. The synthetic document generator
also needs Poppler's `pdftoppm` executable and the system dependencies required by
WeasyPrint.

## Dataset layout

Donut and Qwen use the same project dataset layout:

```text
dataset_root/
|-- train/
|   |-- metadata.jsonl
|   |-- images/
|   `-- annotations/
|-- val/
|   |-- metadata.jsonl
|   |-- images/
|   `-- annotations/
`-- test/                         # optional
    |-- metadata.jsonl
    |-- images/
    `-- annotations/
```

Each line in `metadata.jsonl` links an image to its annotation:

```json
{
  "id": "cmr_example_page_1",
  "image": "train/images/cmr_example_page_1.jpg",
  "annotation": "train/annotations/cmr_example_page_1.json"
}
```

Both trainers use `annotation["content"]` as the target by default. Annotation metadata is
not included in the training target.

## Start Donut training

Check the dataset first:

```powershell
.\.venv\Scripts\python.exe src\Donut\run_donut_training.py --dry-run
```

Start the configured run on an NVIDIA GPU:

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
.\.venv\Scripts\python.exe src\Donut\run_donut_training.py --fp16
```

The defaults are in `DEFAULT_TRAINING_CONFIG` in
[run_donut_training.py](src/Donut/run_donut_training.py). Command-line arguments override
individual values. Donut uses `local_files_only=True` by default; add
`--no-local-files-only` when the base checkpoint is not cached yet.

Runs are written to `runs/donut/<run-name>/`. Open
`validation_previews/latest.html` during training to inspect current predictions.

## Start Qwen QLoRA training

Check the dataset without loading the model:

```powershell
.\.venv\Scripts\python.exe src\Qwen\run_qwen_training.py --dry-run
```

Start the default `Qwen/Qwen3.5-2B` QLoRA run:

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
.\.venv\Scripts\python.exe src\Qwen\run_qwen_training.py
```

The default configuration targets a 12 GB GPU: NF4 4-bit loading, batch size 1,
gradient accumulation, gradient checkpointing, language-side LoRA, and a frozen vision
encoder. The defaults are in
[run_qwen_training.py](src/Qwen/run_qwen_training.py).

Runs are written to `runs/qwen/<run-name>/`. Open
`validation_previews/latest.html` during training to compare generated JSON with the fixed
validation examples.

See the [Qwen documentation](src/Qwen/README.md) for vision-side LoRA, larger models,
memory settings, and inference. See the [Donut documentation](src/Donut/README.md) for
generation-based validation, checkpoint selection, and inference.

## Tests

Run the test suite from the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
