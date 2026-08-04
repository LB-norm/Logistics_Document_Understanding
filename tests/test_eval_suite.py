from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from src.eval_suite import (
    JsonEvaluator,
    NormalizationConfig,
    make_compute_metrics,
    validate_json_schema,
)


class JsonEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["sender", "reference", "items"],
            "properties": {
                "sender": {"type": ["string", "null"]},
                "reference": {"type": ["string", "null"]},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["quantity", "description"],
                        "properties": {
                            "quantity": {"type": ["integer", "string", "null"]},
                            "description": {"type": ["string", "null"]},
                        },
                    },
                },
            },
        }

    def test_perfect_prediction_scores_one(self) -> None:
        truth = {
            "sender": "ACME",
            "reference": None,
            "items": [{"quantity": 2, "description": "Steel"}],
        }
        report = JsonEvaluator(schema=self.schema).evaluate_batch([truth], [truth])
        summary = report.summary()

        self.assertEqual(summary["parse_rate"], 1.0)
        self.assertEqual(summary["schema_valid_rate"], 1.0)
        self.assertEqual(summary["document_exact_match_rate"], 1.0)
        self.assertEqual(summary["field_f1"], 1.0)
        self.assertEqual(summary["value_similarity"], 1.0)

    def test_wrong_missing_and_hallucinated_values_affect_field_f1(self) -> None:
        truth = {
            "sender": "ACME",
            "reference": None,
            "items": [{"quantity": 2, "description": "Steel"}],
        }
        prediction = {
            "sender": "Other",             # FP + FN
            "reference": "invented",       # FP on an annotated-null field
            "items": [{"quantity": 2, "description": None}],  # TP + FN
        }
        summary = JsonEvaluator(schema=self.schema).evaluate_batch(
            [prediction], [truth]
        ).summary()

        self.assertEqual(summary["field_counts"]["true_positives"], 1)
        self.assertEqual(summary["field_counts"]["false_positives"], 2)
        self.assertEqual(summary["field_counts"]["false_negatives"], 2)
        self.assertAlmostEqual(summary["field_precision"], 1 / 3)
        self.assertAlmostEqual(summary["field_recall"], 1 / 3)
        self.assertAlmostEqual(summary["field_f1"], 1 / 3)

    def test_empty_ground_truth_fields_do_not_inflate_recall(self) -> None:
        truth = {"sender": None, "reference": None, "items": []}
        prediction = {"sender": None, "reference": None, "items": []}
        summary = JsonEvaluator(schema=self.schema).evaluate_batch(
            [prediction], [truth]
        ).summary()

        self.assertEqual(summary["field_counts"]["true_positives"], 0)
        self.assertEqual(summary["field_counts"]["empty_correct"], 2)
        self.assertEqual(summary["field_f1"], 1.0)

    def test_normalization_is_conservative_and_exact_match_remains_strict(self) -> None:
        truth = {"sender": "  MÜLLER   GmbH ", "reference": "0012", "items": []}
        prediction = {"sender": "müller gmbh", "reference": 12, "items": []}
        report = JsonEvaluator().evaluate_batch([prediction], [truth])

        self.assertEqual(report.summary()["field_recall"], 0.5)
        self.assertEqual(report.summary()["document_exact_match_rate"], 0.0)

    def test_safe_numeric_strings_match_numbers(self) -> None:
        evaluator = JsonEvaluator()
        report = evaluator.evaluate_batch(
            [{"quantity": "26"}], [{"quantity": 26}]
        )
        self.assertEqual(report.summary()["field_f1"], 1.0)
        self.assertEqual(report.summary()["document_exact_match_rate"], 0.0)

    def test_strict_exact_match_distinguishes_booleans_and_integers(self) -> None:
        report = JsonEvaluator().evaluate_batch([{"value": True}], [{"value": 1}])
        self.assertEqual(report.summary()["document_exact_match_rate"], 0.0)
        self.assertEqual(report.summary()["field_f1"], 0.0)

    def test_invalid_json_counts_as_parse_failure_and_all_values_missing(self) -> None:
        truth = {"sender": "ACME", "reference": None, "items": []}
        report = JsonEvaluator(schema=self.schema).evaluate_batch(
            ["not json"], [truth]
        )
        summary = report.summary()

        self.assertEqual(summary["parse_rate"], 0.0)
        self.assertEqual(summary["schema_valid_rate"], 0.0)
        self.assertEqual(summary["field_recall"], 0.0)
        self.assertIn("JSONDecodeError", report.samples[0].parse_error)

    def test_schema_validation_reports_nested_errors(self) -> None:
        invalid = {
            "sender": "ACME",
            "reference": None,
            "items": [{"quantity": True}],
            "extra": "value",
        }
        errors = validate_json_schema(invalid, self.schema)

        self.assertTrue(any("description" in error and "required" in error for error in errors))
        self.assertTrue(any("quantity" in error and "expected type" in error for error in errors))
        self.assertTrue(any("extra" in error for error in errors))

    def test_array_indices_are_grouped_in_field_breakdown(self) -> None:
        truth = {"items": [{"quantity": 1}, {"quantity": 2}]}
        report = JsonEvaluator().evaluate_batch([truth], [truth])

        self.assertEqual(report.field_breakdown()["items[].quantity"]["support"], 2)

    def test_compute_metrics_adapter_returns_flat_numbers(self) -> None:
        adapter = make_compute_metrics(
            decode_predictions=lambda values: [json.loads(value) for value in values],
            decode_references=lambda values: values,
            evaluator=JsonEvaluator(schema=self.schema),
        )
        truth = {"sender": "ACME", "reference": None, "items": []}
        metrics = adapter(
            SimpleNamespace(predictions=[json.dumps(truth)], label_ids=[truth])
        )

        self.assertEqual(metrics["json_field_f1"], 1.0)
        self.assertEqual(metrics["json_schema_valid_rate"], 1.0)

    def test_custom_case_sensitive_normalization(self) -> None:
        evaluator = JsonEvaluator(
            normalization=NormalizationConfig(case_sensitive=True)
        )
        summary = evaluator.evaluate_batch(
            [{"sender": "acme"}], [{"sender": "ACME"}]
        ).summary()
        self.assertEqual(summary["field_f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
