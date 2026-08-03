from __future__ import annotations

import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.Donut.donut_train_logic import (
    add_target_sequences,
    build_validation_preview_callback,
    json_differences,
    json_to_donut_tokens,
    normalize_prediction_to_schema,
    select_validation_preview_examples,
    write_validation_preview,
)


class DonutValidationPreviewTests(unittest.TestCase):
    def test_target_serialization_follows_template_order_recursively(self) -> None:
        target_skeleton = {
            "sender": {"name": None, "city": None},
            "items": [{"description": None, "quantity": None}],
            "reference": None,
        }
        annotation = {
            "reference": "REF-1",
            "items": [{"quantity": 2, "description": "Food"}],
            "sender": {"city": "Berlin", "name": "Company A"},
        }

        sequence = json_to_donut_tokens(annotation, target_skeleton)

        self.assertEqual(
            sequence,
            "<s_sender><s_name>Company A</s_name><s_city>Berlin</s_city></s_sender>"
            "<s_items><s_description>Food</s_description><s_quantity>2</s_quantity></s_items>"
            "<s_reference>REF-1</s_reference>",
        )

    def test_unknown_fields_follow_template_fields_in_alphabetical_order(self) -> None:
        annotation = {"zeta": "Z", "sender": "A", "alpha": "A"}
        target_skeleton = {"sender": None}

        sequence = json_to_donut_tokens(annotation, target_skeleton)

        self.assertEqual(
            sequence,
            "<s_sender>A</s_sender><s_alpha>A</s_alpha><s_zeta>Z</s_zeta>",
        )

    def test_target_sequences_are_added_without_using_annotation_insertion_order(self) -> None:
        examples = [{"id": "one", "gt_parse": {"second": 2, "first": 1}}]

        serialized = add_target_sequences(examples, {"first": None, "second": None})

        self.assertEqual(
            serialized[0]["target_sequence"],
            "<s_first>1</s_first><s_second>2</s_second>",
        )
        self.assertNotIn("target_sequence", examples[0])

    def test_preview_sample_selection_is_fixed_by_seed(self) -> None:
        examples = [{"id": str(index)} for index in range(10)]

        first = select_validation_preview_examples(examples, sample_count=3, seed=42)
        second = select_validation_preview_examples(examples, sample_count=3, seed=42)

        self.assertEqual([item["id"] for item in first], [item["id"] for item in second])
        self.assertEqual(len(first), 3)

    def test_json_differences_report_field_and_array_errors(self) -> None:
        ground_truth = {"sender": {"city": "Berlin"}, "items": [{"quantity": 2}]}
        prediction = {
            "sender": {"city": "Munich", "country": "DE"},
            "items": [],
        }

        differences = json_differences(ground_truth, prediction)
        by_path = {item["path"]: item["kind"] for item in differences}

        self.assertEqual(by_path["$.sender.city"], "value_mismatch")
        self.assertEqual(by_path["$.sender.country"], "unexpected_in_prediction")
        self.assertEqual(by_path["$.items[0]"], "missing_in_prediction")

    def test_prediction_normalization_uses_schema_for_nulls_and_singleton_arrays(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "itemList": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/item"},
                },
                "nullableComment": {"$ref": "#/$defs/nullable_string"},
                "literalComment": {"type": "string"},
            },
            "$defs": {
                "item": {
                    "type": "object",
                    "properties": {
                        "description": {"$ref": "#/$defs/nullable_string"},
                    },
                },
                "nullable_string": {"type": ["string", "null"]},
            },
        }
        raw_prediction = {
            "itemList": {"description": "null"},
            "nullableComment": "null",
            "literalComment": "null",
        }

        normalized = normalize_prediction_to_schema(raw_prediction, schema)

        self.assertEqual(normalized["itemList"], [{"description": None}])
        self.assertIsNone(normalized["nullableComment"])
        self.assertEqual(normalized["literalComment"], "null")

    def test_prediction_normalization_preserves_non_object_array_errors(self) -> None:
        schema = {"type": "array", "items": {"type": "object"}}

        self.assertEqual(normalize_prediction_to_schema("not-an-item", schema), "not-an-item")

    def test_preview_writer_creates_step_and_latest_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "example.png"
            payload = {
                "status": "completed",
                "global_step": 5,
                "epoch": 0.5,
                "samples": [
                    {
                        "id": "example<unsafe>",
                        "image_path": str(image_path),
                        "ground_truth": {"city": "Berlin"},
                        "prediction": {"city": "Munich"},
                        "exact_match": False,
                        "differences": [
                            {
                                "kind": "value_mismatch",
                                "path": "$.city",
                                "ground_truth": "Berlin",
                                "prediction": "Munich",
                            }
                        ],
                        "parse_error": None,
                        "raw_sequence": "<s_city>Munich</s_city>",
                    }
                ],
            }

            paths = write_validation_preview(Path(temp_dir), payload)

            self.assertTrue(Path(paths["json"]).is_file())
            self.assertTrue(Path(paths["latest_json"]).is_file())
            html_report = Path(paths["html"]).read_text(encoding="utf-8")
            self.assertIn("Ground truth", html_report)
            self.assertIn("Prediction", html_report)
            self.assertIn("Structured differences", html_report)
            self.assertIn("example&lt;unsafe&gt;", html_report)
            self.assertNotIn("example<unsafe>", html_report)

    def test_callback_runs_once_for_each_training_log_step(self) -> None:
        class FakeTrainerCallback:
            pass

        class FakeTorch:
            @staticmethod
            def inference_mode():
                return nullcontext()

        class FakeModel:
            def __init__(self) -> None:
                self.training = True

            def eval(self) -> None:
                self.training = False

            def train(self) -> None:
                self.training = True

        generated_sample = {
            "id": "example",
            "image_path": "example.png",
            "annotation_path": "example.json",
            "ground_truth": {"city": "Berlin"},
            "prediction": {"city": "Berlin"},
            "exact_match": True,
            "differences": [],
            "parse_error": None,
            "raw_sequence": "<s_city>Berlin</s_city>",
            "cleaned_sequence": "<s_city>Berlin</s_city>",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            callback = build_validation_preview_callback(
                TrainerCallback=FakeTrainerCallback,
                torch=FakeTorch,
                image_module=object(),
                processor=object(),
                examples=[{"id": "example", "gt_parse": {"city": "Berlin"}}],
                output_dir=Path(temp_dir),
                task_start_token="<s_lieferschein>",
                max_length=128,
                target_schema={"type": "object"},
                root_schema={"type": "object"},
            )
            state = SimpleNamespace(global_step=5, epoch=0.5, is_world_process_zero=True)
            model = FakeModel()

            with patch(
                "src.Donut.donut_train_logic.generate_validation_preview_sample",
                return_value=generated_sample,
            ) as generate:
                callback.on_log(None, state, object(), logs={"loss": 1.2}, model=model)
                callback.on_log(None, state, object(), logs={"loss": 1.2}, model=model)

            self.assertEqual(generate.call_count, 1)
            self.assertTrue(model.training)
            self.assertTrue(
                (Path(temp_dir) / "validation_previews" / "step_00000005.html").is_file()
            )


if __name__ == "__main__":
    unittest.main()
