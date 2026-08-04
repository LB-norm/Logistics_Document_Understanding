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
