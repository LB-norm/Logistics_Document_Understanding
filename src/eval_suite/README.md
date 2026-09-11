# Structured JSON Evaluation Suite

This module evaluates the final task outcome: whether a document model produced
the correct structured JSON. It is independent of Donut, Qwen, or any other
model implementation.

## Metrics

The aggregate report intentionally contains only a small set of complementary
metrics:

- `parse_rate`: fraction of outputs that are valid JSON (already parsed Python
  objects count as valid).
- `schema_valid_rate`: fraction conforming to the supplied target schema.
- `document_exact_match_rate`: fraction exactly equal to the complete annotation.
- `field_precision`, `field_recall`, and `field_f1`: micro-averaged extraction
  quality over populated leaf fields. A wrong value contributes one false
  positive and one false negative. Predictions in annotated-null fields count
  as false positives; null fields do not inflate recall.
- `value_similarity`: average normalized character similarity over populated
  ground-truth fields. Missing values score zero. This separates near OCR errors
  from completely wrong values.

Field metrics use conservative normalization by default: Unicode NFKC, trimmed
and collapsed whitespace, case folding, numeric equivalence for numbers and
unambiguous numeric strings. Punctuation and leading-zero strings are retained.
Whole-document exact match always uses the original JSON values.

The report also includes per-field breakdowns. Array indices are grouped as
`itemList[]...`, making item fields comparable across documents without adding
more headline metrics.

## Python API

```python
import json
from src.eval_suite import JsonEvaluator

schema = json.load(open("json_schema/content.schema.json", encoding="utf-8"))
evaluator = JsonEvaluator(schema=schema)

report = evaluator.evaluate_batch(predictions, annotations, sample_ids=ids)
print(report.summary())

# Flat numeric values suitable for Trainer logging / compute_metrics:
trainer_metrics = report.training_metrics()
```

For one-off calls, `evaluate_json(...)` and `evaluate_batch(...)` return plain
dictionaries.

Training frameworks can use `make_compute_metrics(...)`; model-specific parsing
stays in two decoder functions:

```python
from src.eval_suite import JsonEvaluator, make_compute_metrics

compute_metrics = make_compute_metrics(
    decode_predictions=decode_model_outputs_to_json,
    decode_references=decode_label_tokens_to_json,
    evaluator=JsonEvaluator(schema=schema),
)
```

Autoregressive models must generate predictions during evaluation for these
metrics. Cross-entropy can still be logged alongside them, but it should not be
used as a proxy for extraction quality.

## Command line

Evaluate two files:

```powershell
python -m src.eval_suite `
  --prediction prediction.json `
  --ground-truth annotation.json `
  --ground-truth-key content `
  --schema json_schema/content.schema.json
```

Use `--prediction-key generated.guided_prediction` (or another dotted path) when
the model's saved inference artifact wraps the final prediction in metadata.

For a batch, provide JSONL records containing `sample_id`, `prediction`, and
`ground_truth`:

```powershell
python -m src.eval_suite --pairs evaluation_pairs.jsonl --schema json_schema/content.schema.json
```

`prediction_path` and `ground_truth_path` may be used instead of embedding the
JSON values. Relative paths are resolved from the directory containing the JSONL
manifest.

## Default and challenge test subsets

The default held-out test set is
`data/datasets/250_CMRS_240dpi_20260707/test`. Its ground-truth annotations are
discovered under `annotations/ground_truths/default` and
`annotations/ground_truths/challenge`.

Evaluate a model output folder against all 30 test annotations with:

```powershell
python -m src.eval_suite `
  --predictions output/qwen/<model-output-folder> `
  --schema json_schema/content.schema.json `
  --output output/qwen/<model-output-folder>/testset_evaluation.json
```

Prediction JSON files are found recursively. They are paired by document name,
so an inference output such as
`000d4526-..._CMR_page_3_240dpi.json` matches
`gt_000d4526-..._CMR_page_3.json`. All annotated samples must have exactly one
prediction. The annotation key defaults to `content` in this mode; use
`--prediction-key` for wrapped model outputs and `--testset-path` to override the
dataset location.

Alternatively, held-out model evaluation can use an explicit manifest so that
every prediction is traceable to one annotation and one distribution. Each
record must use exactly one of `prediction`/`prediction_path`, exactly one of
`ground_truth`/`ground_truth_path`, and set `subset` to either `default` or
`challenge`. Both subsets must occur in the manifest.

```jsonl
{"sample_id":"regular-001","subset":"default","prediction_path":"predictions/default/regular-001.json","ground_truth_path":"annotations/default/regular-001.json"}
{"sample_id":"irregular-001","subset":"challenge","prediction_path":"predictions/challenge/irregular-001.json","ground_truth_path":"annotations/challenge/irregular-001.json"}
```

Evaluate the manifest with:

```powershell
python -m src.eval_suite `
  --testset-pairs testset_pairs.jsonl `
  --ground-truth-key content `
  --schema json_schema/content.schema.json `
  --output testset_evaluation.json
```

For plain Qwen output files, the default `--prediction-key root` is appropriate.
For a wrapped Donut inference artifact, add the dotted path that reaches its
content object, for example `--prediction-key generated.guided_prediction` (or
`generated.guided_prediction.content` for artifacts whose template contains a
top-level `content` wrapper).

The test-set report contains:

- `overall`: metrics and field breakdown over the complete test set;
- `subsets.default` and `subsets.challenge`: independently aggregated metrics
  and field breakdowns;
- `comparison.metrics`: challenge minus default for each headline metric, so a
  negative field-F1 value denotes degradation on irregular documents;
- `samples`: traceable per-document outcomes with their subset labels (omitted
  with `--summary-only`).

The same behavior is available through `JsonEvaluator.evaluate_testset(...)` or
the `evaluate_testset(...)` convenience function.
