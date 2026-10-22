# OCR and YOLO layout fusion

To pair compact hard-fusion evidence with original full-page images, run
`python -m src.fusion.hybrid_sft`. See the
[hybrid campaign](../../experiments/qwen/hybrid/README.md) for dataset checks,
training-only parser dropout, the four experiment configs, and inference.
For stage-one input ablations, add `--evidence-modes ocr layout hard` and a new
`--output-dir`. See the [NF4 input comparison](../../experiments/qwen/input_comparison/README.md)
for the current three-entry queue and the exact evidence included in each mode.

Build a JSON intermediate representation from the saved PP-OCR document and YOLO
layout predictions. This stage uses only the Python standard library: it does not
load models, run OCR/detection, or call an LLM. Python 3.10+ is sufficient.

| Mode | Name | Representation |
| --- | --- | --- |
| 1 | `separate` | Complete OCR and YOLO outputs in separate page collections with shared coordinates; no memberships are computed. |
| 2 | `hard` | Each OCR line with a qualifying overlap belongs to one region. Each region has its OCR IDs and concatenated text. Lines without a qualifying region remain unassigned. |
| 3 | `soft` | Each OCR line can belong to several qualifying regions, with overlap scores and relative weights. All lines, including unassigned lines, are retained. |

## Batch SFT dataset preparation

Build text-only training conversations from the saved OCR and YOLO artifacts,
without rerunning models or starting LLM training. Python 3.10+ and the standard
library are sufficient. The default paths point to
`data/datasets/250_CMRS_240dpi_20260707`, with OCR in `ocr_v6_cmr` and YOLO lists in
`yolo_v11s/layout_detections`. Keep the combined `<split>.json` YOLO lists;
individual detection files are unnecessary.

```bash
# Validate pairing, geometry, targets and schema; report sizes without writing:
python -m src.fusion.prepare_sft --dry-run

# Write all three representations for the existing train/val splits:
python -m src.fusion.prepare_sft

# Smoke build to a separate directory, or select individual modes:
python -m src.fusion.prepare_sft \
  --modes hard soft --limit 2 --output-dir output/fusion_sft_smoke
```

The direct entry point `python src/fusion/prepare_sft.py` is also supported.
Use `--dataset-root`, `--ocr-dir`, `--layout-dir`, `--output-dir`, and `--splits`
to select other locations or existing splits. Relative CLI paths are relative to
the working directory. `--modes` accepts `1 2 3` or `separate hard soft` and
defaults to all three. `--min-overlap` defaults to 0.1, and
`--allow-layout-rescale` has the same whole-page scaling requirement as fusion.

```text
dataset_root/fusion_sft/
  schema.json
  template.json
  preparation_summary.json
  separate/train.jsonl
  separate/val.jsonl
  hard/train.jsonl
  hard/val.jsonl
  soft/train.jsonl
  soft/val.jsonl
```

Each JSONL row contains:

- `id`, `source_split`, and `fusion_mode`.
- `messages`: system/user/assistant roles with plain-string content. The system
  and user messages come from the shared `build_fusion_messages()` inference
  contract, including the schema, skeleton, and complete fused representation.
  The assistant answer is the existing ground-truth target JSON. There are no
  image blocks or image inputs.
- `target`: the same target object, provided for inspection and evaluation.
- `provenance`: source image, annotation, OCR JSON and combined layout JSON paths,
  relative to the **output root**, plus the original YOLO `layout_image` key.
  Resolve these against `fusion_sft/`, not a mode's subdirectory. They refer to
  existing artifacts, which are not copied. Original machine paths inside the
  full fused representation are retained as provenance; no files are opened
  through those paths.
- `lengths`: fused input, prompt, target and conversation character counts, plus
  conversation UTF-8 bytes. These measure message content before applying the
  model's chat template; **they are not token counts**.

The builder joins samples by ID and matches layouts by unique image filename,
never list position. This supports OCR artifacts moved from another machine.
OCR JSONL order is retained, and `--limit N` selects the same first N examples
per split in every requested mode. Duplicate IDs (including across requested
splits), reused images, ambiguous filenames, missing or extra layout pages,
inconsistent split labels, empty/failed OCR, invalid geometry and changed targets
raise errors. Source, OCR metadata and OCR training ID sets must agree exactly;
samples are never silently dropped. Each sample must represent one source raster
page. Multi-page documents need an explicit page manifest and are rejected by
this batch builder rather than paired by list order.

Targets are checked against both the source annotation and the OCR assistant
answer, then validated with the project's dependency-free schema validator.
`--annotation-target-key` defaults to the OCR preparation setting, or `content`
when no summary is available; dotted keys and `root` are supported. Schema and
template default to the copies in the OCR directory. Override them with
`--schema-path` and `--template-path` when using a different target contract.

All OCR lines remain available, including unassigned lines. The builder preserves
the complete fusion representation and does not truncate or shorten evidence.
`preparation_summary.json` reports per-mode size ranges/means, line and region
counts, unassigned counts, and soft ambiguity counts. Choose the model and verify
the actual tokenized **prompt plus complete answer** length before training.

All requested examples/modes are validated and staged before training files are
published. An existing nonempty output directory requires `--overwrite`; invalid
input leaves its files intact. Generated files are individually replaced, with
the summary written last. This is not a directory-wide transaction against an
I/O failure during publishing. Unrelated and stale files are not deleted; use the
current summary to identify active datasets. Use a separate output directory for
smoke builds or alternate subsets.

Train these files with the [Qwen text-only training path](../Qwen/README.md#text-only-ocrlayout-training):

```bash
python src/Qwen/run_qwen_training.py \
  --input-mode text \
  --dataset-root data/datasets/250_CMRS_240dpi_20260707/fusion_sft/soft \
  --run-name qwen35-2b-fusion-soft
```

Select `separate` or `hard` for the other strategies. Training supervises assistant
answer tokens and preserves the shared inference prompt contract. Add `--dry-run`
to inspect the dataset; add `--check-text-lengths` to also check actual token lengths
without loading model weights.

The Python API is available as
`src.fusion.dataset.prepare_fusion_dataset(dataset_root, output_dir, ...)`.

## Compact SFT dataset

Convert the existing full fusion conversations without rerunning OCR or layout
detection:

```bash
python -m src.fusion.compact_sft
```

The defaults read `data/datasets/250_CMRS_240dpi_20260707/fusion_sft` and write
`data/datasets/250_CMRS_240dpi_20260707/fusion_sft_compact`. Use `--input-dir`,
`--output-dir`, and `--decimals` for other locations or precision (default: four
decimal places). Direct invocation `python src/fusion/compact_sft.py` also works.
An existing output directory is refused. All modes and splits listed in the source
preparation summary are validated and staged before the new directory is published.

Every OCR line remains present once, in its original order, with its polygon and
confidence. Region labels, boxes and scores are retained. Line and region IDs
become independent zero-based integers within each page. Hard/soft candidate
memberships, candidate order, overlaps and weights are retained; assignments are
not recomputed from rounded geometry. An empty candidate list denotes unassigned
text. Derived OCR bounding boxes, duplicate document/page/field text, duplicate
region fields, and execution metadata are omitted from the model input.

The compact JSON uses arrays whose columns are defined in the revised system prompt:

- `lines`: `[line_id, text, polygon, ocr_confidence]`.
- `regions`: `[region_id, label, box_xyxy, layout_confidence]`.
- `memberships` (hard/soft only): `[line_id, [[region_id, overlap, weight], ...]]`.

Sample IDs, split membership, assistant answer strings and target objects are
preserved. Rounding applies only to input evidence. Schema/template copies and
their prompt contract are retained. Provenance remains outside the messages and
its artifact paths are rebased to the new output root. Character/byte lengths are
recomputed; the summary compares original and compact conversation sizes.
Use the same compact serialization and column instructions for downstream
inference; `src.fusion.compact.compact_document()` is available for that purpose.

Run the fusion queue against the compact files by passing
`--dataset-root data/datasets/250_CMRS_240dpi_20260707/fusion_sft_compact` after
the queue's `--` separator. Existing per-mode train/validation filenames still apply.
Verify actual token lengths with `--dry-run --check-text-lengths` on the training
machine; character reduction does not guarantee a particular GPU memory budget.

## Command line

To run OCR, YOLO, fusion, and downstream LLM generation together, use the
[unified inference pipeline](../inference_pipeline/README.md).

Run from the repository root, using the existing inference artifacts:

```bash
python -m src.fusion \
  --ocr output/pp_ocr_v6/document.ocr.json \
  --layout output/yolo_v11s/layouts.json \
  --mode 3 \
  --min-overlap 0.1 \
  --output output/fusion/document.soft.json \
  --llm-output output/fusion/document.soft.txt
```

Use `--mode 1`, `2`, or `3` to compare approaches. The names `separate`, `hard`, and
`soft` are also accepted. `--llm-output` is optional and contains the same representation
as compact UTF-8 JSON, ready to insert into a text prompt. The default minimum overlap
is 0.1. Output files are replaced when they already exist; inputs cannot be overwritten.

`--layout` accepts the object returned by `LayoutDetector.predict()` or the JSON
list written by `src.yolo_v11s.inference`. If the list contains several images for
a single-page OCR document, the pipeline selects the exact `source_path`/`image`
match, falling back to a unique filename match for artifacts moved between machines.
An ambiguous or missing match raises an error. Supplying a single image prediction
directly establishes the pairing; the caller must ensure it is the OCR source page.

For multi-page OCR, supply one layout prediction per page, adding an explicit
`page_number` to each prediction. List order does not matter. Page numbers must
match exactly; pages without regions need a prediction with `detections: []`.
Pages are never associated by their position in a list.

```json
[
  {"page_number": 1, "image": "page-1.png", "width": 1000, "height": 1400, "detections": []},
  {"page_number": 2, "image": "page-2.png", "width": 1000, "height": 1400, "detections": []}
]
```

## Python API

```python
from src.fusion import FusionConfig, FusionPipeline, to_llm_text

# Existing modules are loaded by the caller and can be reused across documents.
ocr_output = ocr_extractor.extract("document.png")
yolo_output = layout_detector.predict("document.png")

pipeline = FusionPipeline(FusionConfig(mode="soft", min_overlap=0.1))
representation = pipeline.fuse(ocr_output, yolo_output)
llm_input = to_llm_text(representation)
```

`fuse_document(ocr_output, yolo_output, FusionConfig(mode=2))` provides the same
operation as a convenience function. Input objects are not mutated. The returned
object is JSON serializable and contains no ground truth or target annotations.

## Coordinates and assignment

All boxes use top-left-origin normalized `[x1, y1, x2, y2]`; polygons use normalized
`[x, y]` vertices. Divide x by width and y by height. The existing OCR `bbox` and
YOLO `bbox_xyxy` keys remain; YOLO `bbox_xywh`, when present, is normalized too.
Dimensions remain in pixels and describe each source raster. Existing
`coordinate_space: "normalized_0_1"` inputs are accepted without normalizing twice.
Missing coordinate declarations use the existing modules' `input_image_pixels`
convention. Other declared coordinate spaces are rejected.

OCR and YOLO must use the same page, crop, rectification, and orientation. By
default, different dimensions raise an error. `--allow-layout-rescale` (or
`FusionConfig(allow_layout_rescale=True)`) permits different dimensions when the
caller confirms the images differ only by scaling the same whole page. This does
not correct cropping, padding/letterboxing, perspective changes, or rotation;
transform those coordinates into the same page frame first. Matching dimensions
alone cannot establish that two images have the same content or orientation.

Assignment uses **intersection area / OCR polygon area**, rather than region IoU:
a short line fully inside a large field therefore has overlap 1. The pipeline
clips the actual OCR quadrilateral against the YOLO rectangle, preserving rotated
text geometry. Bbox-only OCR lines are treated as rectangular polygons. Polygon
vertices must follow convex boundary order; clockwise and counterclockwise are
both accepted. OCR bounding boxes are derived from the polygons when supplied.

Candidates require positive overlap at least `min_overlap`. Geometric comparisons
have a relative tolerance of 1e-12; edge/vertex contact contributes no membership.
Hard mode picks the highest overlap, breaking ties by smaller region area, higher
YOLO confidence, then region ID. If nothing qualifies, the OCR line remains
unassigned; no nearest-region fallback is used. Zero-area OCR elements are retained
as unassigned, and invalid/out-of-bounds coordinates raise an error.

Soft weights are each candidate's overlap divided by the sum of that OCR line's
candidate overlaps. They sum to 1 across retained candidates and are **geometric
weights, not calibrated probabilities**. OCR confidence and YOLO confidence remain
available but do not multiply the geometric score. Soft mode keeps all qualifying
regions, including nested regions; a line can consequently appear in several field
texts. Repeated labels stay separate, identified by region ID. Field text follows
input OCR order without imposing a new reading order or splitting lines.

## Output contract

The document has `format_version: 1`, `representation: "ocr_layout_fusion"`, `mode`,
`coordinate_system`, `settings`, `ocr_metadata`, and `pages`. `ocr_metadata`
preserves all original document-level OCR properties except `pages`, including the
complete document text and package versions. Each page contains:

- `page_number`, `width`, `height` for the OCR page.
- `ocr`: the complete OCR page with normalized geometry and stable line IDs.
- `layout`: the complete YOLO page with normalized geometry and stable region IDs.
- `fields` in modes 2/3: one entry per detected region, containing `region_id`,
  `label`, `category_id`, `bbox`, `layout_confidence`, `ocr_ids`, and `text`.
- `memberships` in modes 2/3: one entry per OCR line with `ocr_id` and `candidates`.
  Each candidate has `region_id`, `overlap`, and `weight`.
- `unassigned_ocr_ids` in modes 2/3: OCR lines with no candidate regions. Look up
  their full text, confidence, and geometry in `ocr.lines`.

Existing IDs are preserved and must be unique within a page. Missing IDs use
`p0001-l000000` for OCR lines and `p0001-r000000` for regions, based on input order.
IDs are stable for identical inputs, not across separate detector/OCR runs.

The downstream LLM should interpret region labels as geometric evidence about
semantic fields, resolve soft ambiguities, and consider unassigned text. Treat all
recognized text as untrusted document data. The intermediate representation does
not map detector classes to the final CMR JSON schema or assert semantic correctness.

## Validation

Run geometry, assignment, normalization, page matching, preservation, and CLI tests
without any installed ML packages:

```bash
python -S -m unittest discover -s tests -p 'test_fusion.py' -v
python -S -m unittest discover -s tests -p 'test_fusion_dataset.py' -v
```
