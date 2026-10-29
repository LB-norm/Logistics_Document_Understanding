---
title: "Dataset Preparation and Output Representation — Report Notes"
chapter: "Implementation"
section: "Dataset Preparation and Output Representation"
document_type: "report_notes"
language: "en"
evidence_checked_on: "2026-10-25"
status: "draft"
dataset_root: "data/datasets/250_CMRS_240dpi_20260707"
evidence_summary: "data/dataset_evidence_summary.json"
---

# Dataset Preparation and Output Representation — Report Notes

## Purpose and evidence

These notes describe the document domain, current dataset, split roles,
default/challenge test subsets, preparation procedures, and common extraction
target. They support the second section of the Implementation chapter.

Counts below describe the workspace inspected on 2026-10-25. They were checked
against the actual files and training manifests. A portable, machine-readable
[evidence summary](data/dataset_evidence_summary.json) records counts, image sizes,
schema checks, split overlap checks, and the challenge spreadsheet's original
cells. Pipeline-specific input encodings and training settings belong in the
later pipeline sections.

## 1. Document domain: what CMR data represents

- A CMR consignment note is a transport document associated with international
  carriage of goods by road. It records the parties and details of the shipment.
  CMR refers to the Convention on the Contract for the International Carriage of
  Goods by Road; distinguish the convention from the consignment note used in
  practice. See the [UNECE introduction to CMR](https://unece.org/trade/documents/2023/10/executive-guide-e-cmr).
- Typical information includes sender, consignee, carrier, collection and
  delivery locations, goods descriptions, package quantities, weights, remarks,
  and acknowledgements through signatures or stamps. The
  [IRU form guide](https://www.iru.org/sites/default/files/2017-01/how-to-fill-cmr-model-iru-2007-en.pdf)
  describes these semantic groups.
- In this project, the input is a rasterized document page. The extraction target
  is a structured selection of its shipment information, rather than a complete
  transcription of every printed label or paragraph.
- The data combines form structure with variable values. Information can appear
  as printed text, handwriting, or stamps, and some fields are blank or illegible.
  The challenge spreadsheet explicitly identifies handwriting, poor quality,
  changed layouts, and rotated pages.
- Document understanding therefore requires transcription, semantic assignment,
  and handling of missing or unreadable information. A value should be assigned
  according to its meaning, rather than only the physical box containing it.
- The code supports CMR and delivery-note extraction, while the inspected dataset
  is the CMR collection identified below. Do not imply a separately measured
  delivery-note dataset without additional evidence.

Repository sources: [shared extraction instructions](../../src/Qwen/prompts.py),
[target template](../../json_schema/content.empty.json).

## 2. Dataset identity, composition, and counting unit

### 2.1 Primary dataset

- Dataset root: `data/datasets/250_CMRS_240dpi_20260707`.
- Raw project folders contain DACHSER-labelled CMR material, PDFs, PNG renderings,
  and JSON annotations. They also contain older annotation collections and
  alternate-resolution material; these are not additional examples in the active
  dataset counts.
- The primary training/validation collection comprises 250 page-image/annotation
  pairs: 200 training pages and 50 validation pages.
- The test folder adds 30 pages, giving 280 current pairs across all three splits.
- Use **pages** or **page-image/annotation pairs** as the counting unit. Filenames
  contain source identifiers and page numbers; several pages may originate from
  one source file. The counts do not establish 280 independent shipments.

### 2.2 Current split inventory

| Partition | Page images | Content annotations | Nonempty metadata records | Role |
| --- | ---: | ---: | ---: | --- |
| Training | 200 | 200 | 200 | Parameter optimization |
| Validation | 50 | 50 | 50 | Monitoring, checkpoint selection, and experiment comparison |
| Test: default | 15 | 15 | Not separately recorded in test metadata | Reference performance on the default subset |
| Test: challenge | 15 | 15 | Not separately recorded in test metadata | Performance on explicitly challenging cases |
| Total | 280 | 280 | 250 | Complete current extraction collection |

The test `metadata.jsonl` exists but is empty. Test evaluation discovers annotations
from the default/challenge directories and pairs outputs by normalized filenames;
the empty metadata file does not mean that the test set is empty.

Sources: [training manifest](../../data/datasets/250_CMRS_240dpi_20260707/train/metadata.jsonl),
[validation manifest](../../data/datasets/250_CMRS_240dpi_20260707/val/metadata.jsonl),
[test discovery implementation](../../src/eval_suite/__main__.py),
[measured inventory](data/dataset_evidence_summary.json).

### 2.3 Historical summary discrepancy

- The active root's `dataset_summary.json` records an earlier state: 247 pairs,
  split into 198 training and 49 validation examples, with no test examples.
- The raw-data copy's summary records 250 pairs with a 200/50 split and seed 42.
- Current train/validation manifests and actual image/annotation counts agree on
  200/50; the current test directories contain the additional 30 pairs.
- Use the current counts in the report. Do not describe the old root summary as
  a complete inventory of the final dataset or infer that test pages came from
  the same random split.

Sources: [older active-root summary](../../data/datasets/250_CMRS_240dpi_20260707/dataset_summary.json),
[raw-data summary](../../data/raw_data/250_CMRS_240dpi_20260707/dataset_summary.json).

## 3. On-disk structure and sample identification

```text
250_CMRS_240dpi_20260707/
├── train/
│   ├── metadata.jsonl
│   ├── images/images/*.png
│   └── annotations/ground_truths/gt_*.json
├── val/
│   ├── metadata.jsonl
│   ├── images/images/*.png
│   └── annotations/ground_truths/gt_*.json
├── test/
│   ├── metadata.jsonl                  # Empty in the inspected dataset
│   ├── images/*.png
│   └── annotations/ground_truths/
│       ├── default/gt_*.json            # 15 annotations
│       └── challenge/
│           ├── gt_*.json               # 15 annotations
│           └── Testset flags.xlsx      # Flags for all 15 challenge pages
├── Layout_dataset/result.json          # Separate COCO layout annotations
├── ocr_v6_cmr/                          # Derived OCR evidence and conversations
├── yolo_v11s/layout_detections/         # Predicted training/validation layouts
├── fusion_sft/                          # Derived parser conversations
├── fusion_sft_compact/                  # Compact parser conversations
├── fusion_sft_hybrid/                   # Derived image + parser conversations
└── fusion_sft_input_comparison/         # OCR/layout/hard hybrid variants
```

Training/validation metadata records contain a stable sample ID, image and
annotation paths relative to the dataset root, and source-path provenance.
The repeated `images/images` directory results from preserving the source
subdirectory when copying images into a split.

Illustrative metadata record, using a fictitious source identifier:

```json
{
  "id": "images__example-source_CMR_page_1_240dpi",
  "dataset": "images",
  "image": "train/images/images/example-source_CMR_page_1_240dpi.png",
  "annotation": "train/annotations/ground_truths/gt_example-source_CMR_page_1.json",
  "source_image": "images/example-source_CMR_page_1_240dpi.png",
  "source_annotation": "ground_truths/gt_example-source_CMR_page_1.json"
}
```

The PDF-rendering utility records page numbers and rendering DPI in output
filenames. Evaluation removes a leading `gt_` and a trailing DPI suffix to match
annotation and prediction filenames. Sample matching uses identifiers, not file
ordering.

Sources: [dataset copying and ID construction](../../src/utils/dataset_utils.py),
[evaluation filename matching](../../src/eval_suite/__main__.py).

## 4. Split roles and independence

- Training examples supply the supervision used to update model parameters.
- Validation examples support checkpoint selection and model/configuration
  comparison without becoming training answers.
- Test pages provide a common comparison set for final extraction outputs.
  Default and challenge are two subsets of that test partition, not two training
  datasets.
- The split utility sorts candidate images, pairs images with annotations,
  deterministically shuffles samples using a seed, and partitions the shuffled
  list by ratios. It copies files and writes provenance metadata.
- The historical raw summary records seed 42 and an 80/20 train/validation split.
  The current utility's generic defaults are 80/10/10; do not substitute those
  defaults for the recorded dataset configuration.
- The utility splits samples rather than grouping all pages from one source.
  Inspection found no repeated normalized page IDs or byte-identical PNGs
  between train, validation, and test. However, filename source prefixes overlap
  for two sources between training and validation, and one source between
  validation and test. No such overlap was found between training and test.
- These findings support page-level separation. They do not establish strict
  source-document or shipment-level separation. Filename prefixes are evidence
  of shared source files, not a verified shipment identity.
- Different pages with the same source prefix also occur across the default and
  challenge test subsets; those subsets should not be described as independent
  source populations.
- Some historical test results informed later hybrid experiment design, as the
  hybrid experiment documentation acknowledges. Repeated comparisons on that
  same test set are exploratory rather than a wholly untouched final assessment.

Sources: [split implementation](../../src/utils/dataset_utils.py),
[overlap check results](data/dataset_evidence_summary.json),
[hybrid comparison notes](../../experiments/qwen/input_comparison/README.md).

## 5. Default and challenge test subsets

### 5.1 Purpose and interpretation

- Both subsets use the same extraction task and target schema.
- The default subset provides the reference group. The challenge subset contains
  pages marked for difficult visual or semantic conditions.
- This separation allows the report to distinguish routine extraction performance
  from robustness to quality degradation, handwriting, rotation, and variation
  in content or layout.
- This is the intended interpretation of the directory organization and flag
  names. The repository does not provide a formal sampling protocol, objective
  difficulty thresholds, or quantitative definitions of content/layout novelty.
- Do not equate default with perfectly clean or handwriting-free. The spreadsheet
  labels only challenge pages and does not establish an absence of difficulties
  in the default subset.
- Do not equate `new_content` or `new_layout` with a rigorously established
  out-of-distribution benchmark. Treat them as curator-provided descriptors.
- Preserve separate default/challenge results as well as overall results. The
  evaluation implementation already supports this grouping; metric definitions
  belong in the evaluation section.

Source: [default/challenge evaluation documentation](../../src/eval_suite/README.md#default-and-challenge-test-subsets).

### 5.2 Located challenge spreadsheet

The table was found at:

`data/datasets/250_CMRS_240dpi_20260707/test/annotations/ground_truths/challenge/Testset flags.xlsx`

Workbook sheet: `Tabelle1`. Columns: `ID`, `bad_quality`, `handwriting`,
`new_content`, `new_layout`, and `rotated`.

| Challenge flag | Pages flagged in the spreadsheet | Interpretation for discussion |
| --- | ---: | --- |
| `bad_quality` | 3 | Poor visual quality that can hinder transcription |
| `handwriting` | 8 | Handwritten content, ranging from a date or a few lines to a complete form |
| `new_content` | 10 | Content described by the curator as new; novelty criteria are not specified |
| `new_layout` | 9 | Layout described by the curator as new; novelty criteria are not specified |
| `rotated` | 4 | Rotation/orientation variation; angles are not specified in the spreadsheet |

Counting rule: a cell beginning with `True` counts as present, including entries
such as `True (1 Datum)` and `True (komplett)`. Categories overlap: a single page
may have several flags. The counts must not be added to obtain a document count.

**Coverage:** the updated spreadsheet contains 15 unique page rows, matching all
15 challenge annotation filenames after removing the optional `gt_` prefix.
The page `2a39a028-6acb-423c-84c4-2fe2b13c279f_CMR_page_3` is marked for new
content and a new layout, with the other three flags set to false. The counts
above cover the complete challenge subset.

Source: [original spreadsheet](../../data/datasets/250_CMRS_240dpi_20260707/test/annotations/ground_truths/challenge/Testset%20flags.xlsx),
[portable extraction of original cells and counts](data/dataset_evidence_summary.json).

### 5.3 Per-page challenge overview

IDs below omit the annotation prefix `gt_`. Handwriting comments translate the
spreadsheet's parenthetical notes; they do not add a new visual assessment.

| Page ID | Poor quality | Handwriting | New content | New layout | Rotated |
| --- | --- | --- | --- | --- | --- |
| `0abf783e-33ba-4d6e-9fa7-dc5fbea72e7b_CMR_page_1` | Yes | One date | Yes | Yes | No |
| `0b9ee203-f25a-4b3c-b050-87f643fddd29_CMR_page_1` | Yes | No | No | No | No |
| `2f1d4af2-540c-424c-9c43-3ff968fdf40f_CMR_page_2` | Yes | Four lines | No | No | No |
| `0c685d4c-f534-4e0f-bf32-30e5d12232af_CMR_page_2` | No | Complete form | No | No | No |
| `0cad5e73-d240-4344-9e56-e5ce1aacae2b_CMR_page_1` | No | Complete form | Yes | Yes | No |
| `0c71a3b2-b344-460e-9a52-b61eec9f308f_CMR_page_2` | No | No | Yes | No | Yes |
| `2a216cce-6809-4b72-be7f-2f762986b907_CMR_page_4` | No | No | Yes | Yes | No |
| `00f86a3c-8ac2-4b02-a2b1-d6634ff09696_CMR_page_1` | No | One line | Yes | Yes | No |
| `0a88f0e9-7c2c-4b32-b2fb-409732ce6bc3_CMR_page_1` | No | One line | Yes | Yes | No |
| `2ebb730b-dec9-427b-a03f-afe8778a76db_CMR_page_1` | No | No | Yes | Yes | No |
| `0df99b72-9ed1-43a2-ade3-e924f78219cf_CMR_page_1` | No | Two lines | Yes | Yes | Yes |
| `00a8666b-a105-4b5d-86ed-39c537e68543_CMR_page_1` | No | No | No | No | Yes |
| `00a8666b-a105-4b5d-86ed-39c537e68543_CMR_page_2` | No | No | No | No | Yes |
| `0a5252d6-cbdb-4369-88d5-1ea96f26f09c_CMR_page_1` | No | Yes; extent unspecified | Yes | Yes | No |
| `2a39a028-6acb-423c-84c4-2fe2b13c279f_CMR_page_3` | No | No | Yes | Yes | No |

## 6. Data preparation and preprocessing

### 6.1 Page rendering and image characteristics

- The dataset is labelled as a 240-DPI collection. The rendering utility converts
  PDFs into page PNGs with a default setting of 240 DPI, using PyMuPDF or Poppler's
  `pdftoppm`. The PyMuPDF path scales PDF coordinates by `dpi / 72`.
- The inspected 280 images are RGB PNGs. The most frequent dimensions are
  1985 × 2807 pixels: 181 of the 250 training/validation images have this size.
  This corresponds to approximately 5.57 megapixels per page.
- Dimensions vary. Rotated and unusually large images are present; one training
  image is 5667 × 7887 pixels and one test image is 5757 × 6970 pixels.
  A DPI filename suffix is not proof that every image has identical dimensions
  or was produced by exactly the same rendering history.
- Source rendering resolution and model input resolution are separate concepts.
  Each model processor prepares the stored page for its own input requirements.
  Explain the concrete resizing policies in the relevant pipeline sections.

Sources: [PDF rendering utility](../../src/utils/dataset_utils.py),
[image-size inventory](data/dataset_evidence_summary.json).

### 6.2 Pairing and split construction

- Preparation locates complete image/JSON pairs using supported naming patterns,
  including `gt_` prefixes and DPI suffixes.
- Unmatched images/annotations are reported in the generated summary.
- Copied samples retain their source-relative paths and receive metadata records.
- The splitting/copying routine does not itself denoise, rectify, or resize images.
- Use the actual dataset manifests to establish the final sample selection and
  counts, rather than assuming every raw file was included.

Source: [pairing, copying, and split utilities](../../src/utils/dataset_utils.py).

### 6.3 Optional geometry and annotation preparation

- The repository provides perspective correction, deskewing, and optional
  orientation correction. These are available utilities, not evidence that all
  extraction images were rectified.
- For parser evidence, OCR and layout boxes must refer to the same page geometry.
  A geometric transform requires corresponding updates to coordinates/labels.
- Annotation-review tools check schema structure, suspicious formats, repeated
  entity consistency, outliers, and potential label errors. They create a manual
  review workspace rather than automatically declaring all flagged values wrong.
- Separate normalization utilities can add evidence-supported country suffixes
  to route locations or remove zero-only decimal suffixes from gross weights.
  Describe these as implemented tools unless applied-change records establish
  which corrections were actually made for the reported dataset version.

Sources: [rectification utilities](../../src/preprocessing/README.md),
[annotation audit workflow](../../src/utils/ANNOTATION_AUDIT.md),
[location normalization](../../src/utils/normalize_location_labels.py),
[weight normalization](../../src/utils/normalize_gross_weights.py).

### 6.4 Derived inputs for the three pipelines

- Image-based models use the source page and its content annotation.
- Parser preparation produces OCR evidence, predicted layout evidence, and
  conversational training files for separate/hard/soft representations.
- Compact parser files reduce serialization overhead while retaining evidence.
- Hybrid preparation associates page images with selected parser evidence.
- The inspected preparation summaries retain 200 training and 50 validation
  examples in these variants; the OCR summary reports successful nonempty OCR
  for all 250 selected pages.
- These are alternate representations of the same examples, not additional
  independent training documents. Their target meaning remains the same.
- Explain representation geometry, compression, prompts, and sequence handling
  in the parser and hybrid sections.

Sources: [OCR preparation summary](../../data/datasets/250_CMRS_240dpi_20260707/ocr_v6_cmr/preparation_summary.json),
[fusion preparation](../../src/fusion/README.md),
[hybrid preparation](../../experiments/qwen/input_comparison/README.md).

## 7. Ground-truth JSON representation

### 7.1 Annotation wrapper and supervised content

All inspected annotations contain `content` and `metadata` objects. Training uses
`annotation["content"]` by default. Annotation metadata is provenance and is not
part of the expected extraction answer.

An illustrative wrapper, with the content object abbreviated:

```json
{
  "content": {
    "referenceIdentificationNumber": "EXAMPLE-123"
  },
  "metadata": {
    "prompt": "Extract all relevant information from the document.",
    "method": "zero_shot_prompting"
  }
}
```

The example omits required content fields for readability and is not a complete
schema-valid target. The inspected metadata contains the shown prompting/method
labels. Those labels alone do not establish how much human review or correction
the final annotations received. Verify annotation provenance before claiming
fully manual annotation, a specific annotator count, or inter-annotator agreement.

Sources: [Qwen dataset loading](../../src/Qwen/qwen_finetune_logic.py),
[Donut dataset and target handling](../../src/Donut/README.md).

### 7.2 Common semantic structure

The output contract is defined by:

- [content.empty.json](../../json_schema/content.empty.json): target skeleton and
  example field structure.
- [content.schema.json](../../json_schema/content.schema.json): required keys,
  nested structures, allowed types, and array-item definitions.

The target has 14 top-level entries:

| JSON entry | Represented information |
| --- | --- |
| `senderInformation` | Sender company, address, country, and phone |
| `consigneeInformation` | Recipient company, address, and country |
| `takingOverTheGoods` | Collection/takeover place and the project's date field, `logisticsDateOfArrival` |
| `deliveryOfTheGoods` | Delivery location |
| `sendersInstructions` | Transport instructions |
| `carrierInformation` | Carrier company, address, country, and vehicle registration |
| `successiveCarrierInformation` | Subsequent carrier details |
| `carrierReservationsAndObservations` | Carrier remarks or reservations |
| `documentsHandedToCarrier` | References or remarks concerning accompanying documents |
| `itemList` | Repeated goods entries |
| `signatureOrStampOfTheSender` | Information in the sender acknowledgement/stamp area |
| `signatureOrStampOfTheCarrier` | Information in the carrier acknowledgement/stamp area, including a date |
| `goodsReceived` | Consignee receipt/stamp information, including a date |
| `referenceIdentificationNumber` | Document/shipment reference identifier |

These are the project's selected extraction fields, not a claim to reproduce all
CMR particulars or an officially standardized e-CMR message. In particular,
retain project field names accurately rather than silently changing their meaning.

### 7.3 Repeated goods and optional information

- `itemList` is an array of objects, allowing multiple goods entries.
- Each item groups markings, package count, packing type, goods description, and
  gross weight in kilograms.
- Across the current collection, item lists range from zero to eight entries.
  Training has 176 one-item pages and 24 multi-item pages. Validation includes
  one empty list, 42 one-item pages, and seven multi-item pages. Test includes
  27 one-item pages and three multi-item pages.
- The skeleton contains one empty item as a structural example; this does not
  impose a one-item limit. The extraction instruction uses `[]` when no items
  are present.
- With one item, the skeleton contains 49 scalar leaf slots. Actual target size
  varies with item count and text length.

### 7.4 Value types, missingness, and unreadability

| Representation | Meaning or constraint |
| --- | --- |
| String value | Extracted text; preserve meaningful text and identifier formatting |
| Number or integer | Allowed for selected quantitative fields, alongside strings and null according to the schema |
| `null` | No information is supplied for the field |
| `"<unreadable>"` | Information is visibly present but cannot be reliably transcribed |
| `[]` | No entries in a repeated collection |

- Most leaves permit strings or null. Package quantities additionally permit
  integers; gross weights additionally permit numbers. Not every field accepts
  every JSON type.
- Required keys and nullable values are compatible: the structure remains
  complete even when a field has no value.
- Country fields are nested objects with a `value` key. The schema does not
  enforce a particular ISO code vocabulary or a normalized date format.
- The schema rejects additional object properties and requires the defined
  fields. It checks output structure; it does not establish that extracted
  values match the page.
- The current prompt instructs semantic assignment, discourages guessing, and
  distinguishes absent information from visible but illegible information.
- A text-only parser model cannot directly inspect visual illegibility; its
  prompt explicitly cautions against inferring unreadability from OCR omissions.
  Explain this modality-specific limitation in the parser section.

Sources: [JSON schema](../../json_schema/content.schema.json),
[shared and modality-specific instructions](../../src/Qwen/prompts.py).

### 7.5 Structural validation and target serialization

- All 200 training, 50 validation, and 30 test content annotations passed the
  project's dependency-free schema validator during this inspection.
- This establishes structural consistency, not complete annotation correctness.
- Qwen and Donut share the semantic target. Their training serialization differs:
  Qwen uses JSON assistant answers, while Donut uses its structured special-token
  serialization and converts generated output back into the content structure.
  Develop those details in the end-to-end section.
- Keep the final extraction target distinct from OCR/layout intermediate JSON,
  layout boxes, and inference diagnostics.

Sources: [validation evidence](data/dataset_evidence_summary.json),
[schema validator](../../src/eval_suite/schema.py),
[Donut target serialization](../../src/Donut/README.md#target-serialization).

## 8. Separate layout annotations

- Content annotations supervise the final extraction; layout annotations supervise
  region detection. These are different annotation tasks.
- The inspected `Layout_dataset/result.json` is a Label Studio COCO export with
  36 images, 533 bounding-box annotations, and 15 region categories.
- Region categories largely follow content groups, with `carrierLicensePlate`
  represented as an additional separate category.
- COCO annotations provide image identifiers/dimensions, categories, and bounding
  boxes. They do not replace the structured content ground truth.
- The export's 36 images do not establish the exact detector training/validation
  split. Report the detector's actual split and preparation from its run
  configuration in the Parser-based Pipeline section.

Sources: [COCO export](../../data/datasets/250_CMRS_240dpi_20260707/Layout_dataset/result.json),
[layout export inventory](data/dataset_evidence_summary.json),
[YOLO dataset interface](../../src/yolo_v11s/README.md).

## 9. Scope for the report and remaining details to establish

- Develop pipeline-specific prompts, representation construction, resizing,
  training, and inference in sections three to five.
- Develop test pairing, scoring, subset aggregation, latency, and memory
  measurement implementation in section six.
- Before making stronger dataset claims, establish the acquisition and selection
  procedure, annotation creation/review history, and exact criteria for default
  and challenge membership.
- Preserve relevant dataset evidence with the report. Most dataset artifacts and
  run outputs are ignored by Git; links to them require the original workspace.
  The companion JSON preserves the inspected aggregate evidence and challenge
  table values without copying the source images or content annotations.
