from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.eval_suite import (
    JsonEvaluator,
    NormalizationConfig,
    evaluate_testset,
    make_compute_metrics,
    validate_json_schema,
)
from src.eval_suite.__main__ import main as eval_main


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

    def test_testset_report_separates_default_and_challenge(self) -> None:
        report = JsonEvaluator().evaluate_testset(
            [{"value": "correct"}, {"value": "wrong"}],
            [{"value": "correct"}, {"value": "correct"}],
            subset_labels=["default", "challenge"],
            sample_ids=["regular-1", "irregular-1"],
        )
        rendered = report.to_dict()

        self.assertEqual(rendered["overall"]["summary"]["samples"], 2)
        self.assertEqual(rendered["subsets"]["default"]["summary"]["field_f1"], 1.0)
        self.assertEqual(rendered["subsets"]["challenge"]["summary"]["field_f1"], 0.0)
        self.assertEqual(
            rendered["comparison"]["metrics"]["field_f1"], -1.0
        )
        self.assertEqual(
            [sample["subset"] for sample in rendered["samples"]],
            ["default", "challenge"],
        )

    def test_testset_requires_both_known_subsets(self) -> None:
        evaluator = JsonEvaluator()
        with self.assertRaisesRegex(ValueError, "Unknown test subset"):
            evaluator.evaluate_testset(
                [{"value": 1}, {"value": 1}],
                [{"value": 1}, {"value": 1}],
                subset_labels=["default", "hard"],
            )
        with self.assertRaisesRegex(ValueError, "missing: challenge"):
            evaluator.evaluate_testset(
                [{"value": 1}],
                [{"value": 1}],
                subset_labels=["default"],
            )

    def test_testset_convenience_function(self) -> None:
        rendered = evaluate_testset(
            [{"value": 1}, {"value": 1}],
            [{"value": 1}, {"value": 1}],
            subset_labels=["DEFAULT", " challenge "],
        )
        self.assertEqual(rendered["subsets"]["default"]["summary"]["samples"], 1)
        self.assertEqual(rendered["subsets"]["challenge"]["summary"]["samples"], 1)

    def test_testset_cli_loads_relative_prediction_and_annotation_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            prediction_default = root / "prediction-default.json"
            prediction_challenge = root / "prediction-challenge.json"
            annotation_default = root / "annotation-default.json"
            annotation_challenge = root / "annotation-challenge.json"
            manifest = root / "testset.jsonl"
            output = root / "report.json"

            prediction_default.write_text('{"value": "A"}', encoding="utf-8")
            prediction_challenge.write_text('{"value": "B"}', encoding="utf-8")
            annotation_default.write_text(
                '{"content": {"value": "A"}}', encoding="utf-8"
            )
            annotation_challenge.write_text(
                '{"content": {"value": "A"}}', encoding="utf-8"
            )
            rows = [
                {
                    "sample_id": "default-1",
                    "subset": "default",
                    "prediction_path": prediction_default.name,
                    "ground_truth_path": annotation_default.name,
                },
                {
                    "sample_id": "challenge-1",
                    "subset": "challenge",
                    "prediction_path": prediction_challenge.name,
                    "ground_truth_path": annotation_challenge.name,
                },
            ]
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            argv = [
                "eval_suite",
                "--testset-pairs",
                str(manifest),
                "--ground-truth-key",
                "content",
                "--output",
                str(output),
            ]
            with patch.object(sys, "argv", argv):
                self.assertEqual(eval_main(), 0)

            rendered = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                rendered["subsets"]["default"]["summary"]["field_f1"], 1.0
            )
            self.assertEqual(
                rendered["subsets"]["challenge"]["summary"]["field_f1"], 0.0
            )

    def test_testset_cli_discovers_default_annotations_for_prediction_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            testset = root / "test"
            predictions = root / "output" / "model" / "test"
            default_annotations = (
                testset / "annotations" / "ground_truths" / "default"
            )
            challenge_annotations = (
                testset / "annotations" / "ground_truths" / "challenge"
            )
            default_annotations.mkdir(parents=True)
            challenge_annotations.mkdir(parents=True)
            predictions.mkdir(parents=True)

            (default_annotations / "gt_default-document.json").write_text(
                '{"content": {"value": "A"}}', encoding="utf-8"
            )
            (challenge_annotations / "gt_challenge-document.json").write_text(
                '{"content": {"value": "B"}}', encoding="utf-8"
            )
            (predictions / "default-document_240dpi.json").write_text(
                '{"value": "A"}', encoding="utf-8"
            )
            (predictions / "challenge-document_240dpi.json").write_text(
                '{"value": "B"}', encoding="utf-8"
            )
            output = root / "evaluation.json"

            argv = [
                "eval_suite",
                "--predictions",
                str(predictions.parent),
                "--testset-path",
                str(testset),
                "--output",
                str(output),
            ]
            with patch.object(sys, "argv", argv):
                self.assertEqual(eval_main(), 0)

            rendered = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(rendered["overall"]["summary"]["samples"], 2)
            self.assertEqual(
                rendered["subsets"]["default"]["summary"]["field_f1"], 1.0
            )
            self.assertEqual(
                rendered["subsets"]["challenge"]["summary"]["field_f1"], 1.0
            )

    def test_prediction_folder_evaluation_requires_every_test_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            testset = root / "test"
            predictions = root / "predictions"
            for subset in ("default", "challenge"):
                annotation_dir = (
                    testset / "annotations" / "ground_truths" / subset
                )
                annotation_dir.mkdir(parents=True)
                (annotation_dir / f"gt_{subset}-document.json").write_text(
                    '{"content": {"value": "A"}}', encoding="utf-8"
                )
            predictions.mkdir()
            (predictions / "default-document_240dpi.json").write_text(
                '{"value": "A"}', encoding="utf-8"
            )

            argv = [
                "eval_suite",
                "--predictions",
                str(predictions),
                "--testset-path",
                str(testset),
            ]
            with patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(
                    FileNotFoundError, "Missing predictions for 1 test sample"
                ):
                    eval_main()


if __name__ == "__main__":
    unittest.main()
