# Evaluation Suite Plan

This folder should contain the future evaluation suite for comparing document
understanding pipelines on logistics document scans. The suite should evaluate
two related but separate questions:

1. How well does a complete pipeline produce the final structured JSON?
2. How useful and correct is an intermediate parser output on its own?

The second question matters because parser-style systems such as PaddleOCR-VL
can produce layout blocks, markdown, tables, OCR text, and bounding boxes before
any final JSON extraction step happens.

## Current State

There is no standalone evaluation suite yet.

Existing code contains training-time evaluation settings for Donut and Qwen, plus
some saved sample outputs, but there is no shared benchmark runner, metric
implementation, prediction registry, or aggregate report generator.

The current dataset annotations provide final JSON labels under
`annotation["content"]`. They do not provide parser-level ground truth such as
OCR spans, bounding boxes, layout classes, table cells, reading order, or links
from JSON fields to source regions.

## Proposed Folder Structure

The future `src/eval` folder could be structured like this:

```text
src/eval/
  evaluation_plan.md
  run_evaluation.py
  config.py
  dataset.py
  normalization.py
  schemas.py
  report.py
  metrics/
    final_json.py
    parser_output.py
    text_matching.py
    aggregation.py
  adapters/
    base.py
    donut.py
    qwen.py
    paddleocr_vl.py
    saved_outputs.py
```

This is only a planning target. Do not create these files until implementation
starts.

## Core Concepts

### Dataset Samples

The evaluator should load samples from the existing dataset layout:

```text
data/datasets/<dataset_name>/
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

Each loaded sample should expose:

- sample id
- split
- image path
- annotation path
- target JSON, normally `annotation["content"]`
- optional metadata such as source dataset and source image path

The evaluator should default to `test`, but support any split for debugging.

### Prediction Records

Every pipeline should be converted into a common prediction record:

```json
{
  "sample_id": "...",
  "pipeline": "paddleocr_vl",
  "status": "ok",
  "raw_output": {},
  "parser_output": {},
  "predicted_json": {},
  "normalized_json": {},
  "runtime_seconds": 1.23,
  "errors": []
}
```

Not every pipeline will fill every field. For example:

- Donut may produce `raw_output`, `predicted_json`, and `normalized_json`.
- Qwen may produce raw generated text plus parsed JSON.
- PaddleOCR-VL may produce `parser_output`, markdown, layout blocks, and no final
  JSON unless paired with a downstream extraction step.

### Run Artifacts

Evaluation runs should be reproducible and inspectable. A run folder could look
like this:

```text
evaluation_runs/<run_id>/
  run_config.json
  predictions/
    <sample_id>.json
  parser_outputs/
    <sample_id>.json
    <sample_id>.md
  metrics/
    sample_metrics.jsonl
    aggregate_metrics.json
  reports/
    summary.md
    field_breakdown.csv
```

The run config should record model ids, checkpoint paths, prompts, schema path,
dataset root, split, timestamp, and evaluator version.

## Final JSON Evaluation

Final JSON evaluation compares the predicted extraction result against
`annotation["content"]`.

### Validity Metrics

- JSON parse rate: percentage of samples where the model output can be parsed.
- Object rate: percentage of parsed outputs that are JSON objects.
- Schema validity: percentage of predictions that validate against the target
  schema or target content schema.
- Required field presence: percentage of required fields present at each nesting
  level.
- Extra field rate: fields produced by the model but absent from the schema.

### Field-Level Metrics

Flatten nested JSON paths into field paths such as:

```text
senderInformation.senderNameCompany
itemList[0].grossWeightInKg.supplyChainConsignmentItemGrossWeight
```

Then compute:

- exact match
- normalized exact match
- fuzzy string similarity
- numeric match with tolerance
- date/time normalized match
- null accuracy
- missing-field rate
- hallucinated-value rate, especially when target is null

Normalization should be explicit and configurable. Useful normalizations:

- trim whitespace
- collapse repeated whitespace
- case folding for selected text fields
- normalize Unicode punctuation where safe
- normalize German date variants
- normalize decimal separators
- normalize phone numbers lightly
- treat integer and float equivalents as equal

### Section-Level Metrics

Aggregate field scores by business section:

- sender information
- consignee information
- taking over the goods
- delivery of the goods
- sender instructions
- carrier information
- successive carrier information
- reservations and observations
- documents handed to carrier
- item list
- charges
- goods received
- reference identification number

This is more interpretable than one global score.

### Item List Metrics

`itemList` needs special handling because row order and row grouping can vary.

Possible approach:

1. Match predicted rows to target rows using a weighted similarity over key item
   fields.
2. Score fields only after row matching.
3. Report row precision, row recall, and row field accuracy.

Important item fields:

- shipping marks
- package quantity
- package type
- cargo identification
- gross weight
- volume

### Aggregate Scores

Useful aggregate reports:

- macro field accuracy
- micro field accuracy
- weighted business score
- schema validity rate
- parse failure rate
- per-section score table
- worst fields by accuracy
- best fields by accuracy
- sample-level score distribution

The aggregate score should not hide parse failures. Parse rate, schema validity,
and field accuracy should always be reported separately.

## Parser Output Evaluation

Parser output evaluation should be separate from final JSON evaluation.

For PaddleOCR-VL, parser output can include:

- markdown text
- parsed layout blocks in `parsing_res_list`
- block labels
- block bounding boxes
- layout detection boxes and confidence scores
- recognized table HTML/markdown

### Current Feasible Parser Metrics

Because the current annotation files do not include parser-level ground truth, the
first parser evaluation should be evidence based.

The key question is:

Can the parser output expose the textual evidence needed to recover the annotated
JSON fields?

Metrics:

- value coverage: percentage of non-null annotation values found in parser text
- fuzzy value coverage: same, but tolerant to OCR errors
- critical field coverage: coverage for reference number, sender, consignee,
  dates, package quantities, weights, locations
- table value coverage: coverage for item-list values
- parser empty-output rate
- parser block count statistics
- table block count statistics
- average text length
- duplicate text or repeated-token ratio
- runtime and failure rate
- layout confidence summaries when available

This does not prove layout correctness. It measures whether the parser output is
useful as evidence for final extraction.

### Parser Metrics Requiring New Ground Truth

For a stronger parser benchmark, manually annotate a subset of documents with
parser-level labels.

Possible parser ground truth format:

```json
{
  "blocks": [
    {
      "id": "b1",
      "label": "table",
      "bbox": [154, 54, 2373, 3284],
      "text": "...",
      "reading_order": 1
    }
  ],
  "tables": [
    {
      "bbox": [154, 54, 2373, 3284],
      "cells": [
        {
          "row": 0,
          "col": 0,
          "text": "Exemplar für Absender",
          "bbox": [154, 54, 400, 90]
        }
      ]
    }
  ],
  "field_evidence": {
    "senderInformation.senderNameCompany": ["b1"],
    "referenceIdentificationNumber": ["b7"]
  }
}
```

With this, evaluate:

- layout detection precision, recall, and F1
- block label precision, recall, and F1
- bounding-box IoU
- detection mAP at IoU thresholds
- reading-order correlation
- OCR character error rate
- OCR word error rate
- table cell precision, recall, and structure accuracy
- field evidence recall

This should be added only after the final-JSON evaluator and evidence-based parser
evaluator exist.

## Adapter Design

Each pipeline should have an adapter that converts its native output into the
common prediction record.

Adapters should avoid duplicating metric logic. Their job is only:

- run or load a prediction
- extract raw generated text or parser output
- parse final JSON if available
- record runtime and errors

Possible adapters:

- Donut checkpoint adapter
- Qwen checkpoint/LoRA adapter
- PaddleOCR-VL parser adapter
- saved-output adapter for evaluating existing result files without rerunning
  models

The saved-output adapter is important because some models are heavy and expensive
to rerun.

## Implementation Order

Recommended rollout:

1. Define the common prediction record and run artifact layout.
2. Implement dataset loading for the existing metadata format.
3. Implement JSON flattening and normalization.
4. Implement final JSON metrics.
5. Implement saved-output evaluation so existing predictions can be scored.
6. Add Donut and Qwen adapters.
7. Add PaddleOCR-VL parser-output adapter.
8. Implement evidence-based parser metrics.
9. Add aggregate reports and per-field breakdowns.
10. Later, add parser-level annotation support and layout/OCR/table metrics.

## Open Decisions

- Should evaluation compare against `annotation["content"]` only, or sometimes the
  full annotation wrapper?
- Are the current annotations accepted as gold labels, or are they pseudo-labels
  that need manual review first?
- Should train/val/test be regenerated with document-level grouping to avoid
  multi-page leakage?
- Which fields should be weighted most heavily for the final thesis score?
- How strict should string normalization be for company names, countries, phone
  numbers, and dates?
- Should parser evidence metrics count a value as covered if only a partial fuzzy
  match is present?
- How many documents should receive parser-level annotations for a strong parser
  benchmark?

## Initial Recommendation

Build the final JSON evaluator first, because it directly measures the project
goal. In parallel, keep parser outputs as first-class artifacts.

Then add an evidence-based parser evaluator that checks whether parser markdown,
text blocks, and table outputs contain the annotated target values. This provides
a useful parser-only comparison before investing time in manual parser-level
ground truth.
