# PP-OCRv6 baseline

The OCR stage uses **PP-OCRv6 medium detection and recognition** by default.
It provides reusable text and geometry for `image → OCR (+ separate layout) → LLM → JSON`,
and prepares OCR/ground-truth pairs from existing project datasets.

## Setup

```bash
.venv/bin/python -m pip install 'paddleocr>=3.7,<4' 'paddlepaddle>=3.2' Pillow numpy PyMuPDF
```

The CPU Paddle wheel is sufficient for CPU inference. For `--device gpu:0`, install
an appropriate `paddlepaddle-gpu` wheel for your CUDA environment instead of the CPU
wheel. Model weights download on first inference. Local models can be supplied via
`--text-detection-model-dir` and `--text-recognition-model-dir`; choose the matching
`--model-size medium|small|tiny` (default: medium). Each tier uses an explicit v6
model pair, independent of language-based selection. The recognizer is multilingual;
there is no `--lang` switch. Medium/small support 50 languages, including German,
English, French, Czech and Polish. Cyrillic and Arabic need other recognition models.

If the default Hugging Face download stalls, Paddle's official host can be selected
with `PADDLE_PDX_MODEL_SOURCE=bos` in the command environment. To keep weights in the
project, set `PADDLE_PDX_CACHE_HOME` to a writable cache directory under `output/`.

## Use inside a multipart pipeline

```python
from pathlib import Path
from src.PP_parser import OCRConfig, OCRExtractor, save_ocr

ocr = OCRExtractor(OCRConfig(device="gpu:0"))  # construct once; weights load lazily
result = ocr.extract("document.png")
text = result["text"]                       # pass to your text-only LLM
spans = result["pages"][0]["lines"]         # combine with your layout detections

# Also accepts a PIL image or a uint8 OpenCV BGR/grayscale ndarray:
# result = ocr.extract(rectified_image, source_id="cmr-001")
# Saving is optional and separate from inference:
# save_ocr(result, Path("output/ocr"), "cmr-001")
```

Extraction does not invoke an LLM or write files. Reuse one extractor sequentially
across documents. Each line has an ID (unique within its document), index, text,
recognition confidence, a four-point polygon, and an axis-aligned `bbox` in
`[xmin, ymin, xmax, ymax]` form. Pages carry `page_number`, `width`, `height`, and
`coordinate_space: "input_image_pixels"`. IDs follow `p0001-l000000`, preserving
Paddle recognition order; they are stable for an identical extraction, not across
changed models or thresholds. This order is not a semantic reading-order guarantee.

To combine these outputs with YOLO detections before passing them to an LLM, use
the [OCR/layout fusion pipeline](../fusion/README.md). It supports separate outputs,
hard spatial assignment, and soft spatial assignment with ambiguous memberships.

PIL images use RGB; arrays use BGR or grayscale. Raster pixels are not resized by
this adapter. Paddle's detector may internally resize, and the effective detection
parameters are recorded on each page. Returned geometry refers to the input page.
Document rotation, unwarping and text-line orientation classification are disabled.
Run OCR and layout on the **same page image**, including any shared rectification.
If one stage uses different preprocessing, map its coordinates using the saved transform.

PDFs are rendered explicitly by PyMuPDF at `--pdf-dpi 240`; coordinates refer to the
rendered page pixels, not PDF points. Multiple pages remain separate and text includes
page markers. Multi-frame raster files are rejected; supply individual pages or a PDF.
The document JSON has `format_version: 2`, source provenance, settings, actual package
versions and inference time (including image loading/rendering, excluding model startup).
Both the document and each page record `inference_seconds` and `vram` with peak
allocated/reserved Paddle CUDA memory in bytes and GiB. Page timing measures OCR
and result normalization; document timing also includes image loading/rendering.
Document VRAM uses the maximum per-device page peak. CPU runs mark VRAM
unavailable. PaddleOCR-VL records these metrics in `inference_summary.json`.
`--min-confidence` defaults to zero; empty OCR is retained and reported as a failure.

## Standalone OCR and downstream inference

```bash
.venv/bin/python src/PP_parser/run_ocr.py \
  --input-paths "data/small testing/3f3fdb18-c151-43dd-b54a-da34249241f6_CMR_page_1.jpg" \
  --output-dir output/pp_ocr_v6 --device cpu

.venv/bin/python src/Qwen/run_inference.py \
  --ocr-text-paths output/pp_ocr_v6/3f3fdb18-c151-43dd-b54a-da34249241f6_CMR_page_1.ocr.txt \
  --output-path output/pp_ocr_v6/prediction.json
```

Multiple image/PDF paths reuse one OCR instance. Each input produces `<stem>.ocr.json`
and `<stem>.ocr.txt`; `manifest.json` records successes, empty OCR and errors. Stems
must be unique. A nonzero exit status indicates failed or empty OCR. Use `--dry-run`
to validate input paths without loading OCR. Existing output requires `--overwrite`.

The Qwen text path uses only text tokens, with the project schema/skeleton and the
shared OCR prompt. Its default is `Qwen/Qwen3.5-9B`; the full multimodal checkpoint
is still loaded, including vision weights. The input contains no visual evidence.

## Prepare train/validation data for downstream LLM fine-tuning

```bash
# Validate all source metadata and annotations without running OCR:
.venv/bin/python src/PP_parser/run_ocr.py \
  --dataset-root data/datasets/250_CMRS_240dpi_20260707 \
  --output-dir data/ocr_v6_cmr --dry-run

# Run OCR on existing train/val splits:
.venv/bin/python src/PP_parser/run_ocr.py \
  --dataset-root data/datasets/250_CMRS_240dpi_20260707 \
  --output-dir data/ocr_v6_cmr --splits train val --device gpu:0
```

`--splits` defaults to `train val`; specify `train validation` or another existing
split name as needed. The source split membership and metadata order are preserved.
Paths in source metadata can be absolute, dataset-root-relative or split-relative.
`annotation["content"]` supplies the assistant answer by default; use
`--annotation-target-key` for another dotted key or `root`. Use `--schema-path` and
`--template-path` matching that target. `--limit 2` selects the first two examples
per split for a smoke run. Missing paths/labels and duplicate IDs are rejected before
OCR or output writes. Source and output directories must not overlap.

```text
data/ocr_v6_cmr/
  train.jsonl
  val.jsonl
  schema.json
  template.json
  preparation_summary.json
  train/
    metadata.jsonl
    ocr/000000_<image-stem>.ocr.json
    ocr/000000_<image-stem>.ocr.txt
  val/
    metadata.jsonl
    ocr/...
```

Each split JSONL contains `id`, `source_split`, `text`, `target`, `messages`, and
relative links to its OCR artifacts. `messages` is a text-only chat record with
system/user/assistant roles and plain-string content. System/user prompts match the
existing OCR inference contract, including the supplied schema/skeleton; only the
assistant message contains the final JSON target. The OCR input is generated from
pixels, never ground-truth labels. Full geometric evidence is kept in `.ocr.json`
for future layout fusion rather than exposed to the text-only baseline.

These records can be used with the [Qwen text-only training path](../Qwen/README.md#text-only-ocrlayout-training):

```bash
python src/Qwen/run_qwen_training.py --input-mode text --dataset-root data/ocr_v6_cmr
```

It tokenizes the saved `messages`, supervises assistant tokens, and preserves the
OCR inference prompt contract. The preparation command itself does not train an LLM.

To prepare the three OCR/layout fusion variants from these artifacts and saved
YOLO predictions, use the [batch fusion SFT builder](../fusion/README.md#batch-sft-dataset-preparation):

```bash
python -m src.fusion.prepare_sft --dataset-root data/datasets/250_CMRS_240dpi_20260707
```

It defaults to OCR in `<dataset-root>/ocr_v6_cmr`; use `--ocr-dir data/ocr_v6_cmr`
for the output location in the OCR preparation example above.

Split metadata preserves every selected example, including `ok`, `empty_ocr`, and
`ocr_error` statuses, with absolute source paths for provenance. Empty/failed OCR
is excluded from the fine-tuning JSONL and counted in the summary; the command
returns nonzero if any occurred. No source images or labels are copied or modified.
`--overwrite` rewrites current outputs without deleting the directory; old unreferenced
artifacts may remain. Use current metadata/JSONL membership, not a directory glob.

## Verification

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_pp_ocr.py' -v
```

Tests cover the API, input color/coordinate contract, PDF rendering, lazy v6
initialization, split/target preservation, error handling, CLI execution, and the
text-only downstream inference path. Most tests inject a deterministic OCR backend;
they verify integration behavior without downloading weights or asserting OCR accuracy.

Sources: [official OCR API](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html),
[PP-OCRv6 technical report](https://arxiv.org/abs/2606.13108).

## PaddleOCR-VL experiments

This folder currently contains PaddleOCR-VL/PP-Structure inference experiments and a
dataset preparation script for PaddleOCR-VL fine-tuning.

`PPStructureV3_parser.py` uses the same project resolution presets as Qwen:
`low` (1.4 MP), `medium` (2.8 MP, default), `high` (4.2 MP), and `native`
(5.6 MP). Select one with `--resolution`; each preset is an upper pixel budget.

## Current State

- `run_ocr.py` is the plain PP-OCR baseline described above.
- `PPStructureV3_parser.py` is a PaddleOCR-VL inference experiment.
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
  --dataset-root data/datasets/250_CMRS_240dpi_20260707 \
  --dry-run
```

Write a prepared dataset:

```bash
python3 src/PP_parser/prepare_finetune.py \
  --dataset-root data/datasets/250_CMRS_240dpi_20260707 \
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
