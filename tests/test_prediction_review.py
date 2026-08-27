from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.utils.prediction_review import (
    _inline_diff_segments,
    build_review,
    compare_documents,
    load_review_data,
)


class PredictionReviewTests(unittest.TestCase):
    def test_inline_diff_highlights_single_extra_character_and_case_change(self) -> None:
        self.assertEqual(
            _inline_diff_segments("Helllo World", "Hello world"),
            [
                {"kind": "equal", "text": "Hell"},
                {"kind": "annotation_only", "text": "l"},
                {"kind": "equal", "text": "o "},
                {"kind": "annotation_only", "text": "W"},
                {"kind": "prediction_only", "text": "w"},
                {"kind": "equal", "text": "orld"},
            ],
        )

    def test_inline_diff_keeps_a_repeated_suffix_after_a_replacement(self) -> None:
        self.assertEqual(
            _inline_diff_segments("44009", "44099"),
            [
                {"kind": "equal", "text": "440"},
                {"kind": "annotation_only", "text": "0"},
                {"kind": "prediction_only", "text": "9"},
                {"kind": "equal", "text": "9"},
            ],
        )

    def test_compare_marks_value_and_formatting_differences(self) -> None:
        differences = compare_documents(
            {"name": "ACME GmbH", "reference": "001", "missing": "label"},
            {"name": "acme   gmbh", "reference": "002", "extra": "prediction"},
        )
        by_path = {difference.path: difference for difference in differences}
        self.assertEqual(by_path["name"].kind, "formatting_only")
        self.assertEqual(by_path["reference"].kind, "value_mismatch")
        self.assertEqual(by_path["missing"].kind, "annotation_only")
        self.assertEqual(by_path["extra"].kind, "prediction_only")

    def test_compare_ignores_unpopulated_template_structure(self) -> None:
        differences = compare_documents(
            {"items": [], "value": None},
            {"items": [{"quantity": None}], "value": ""},
        )
        self.assertEqual(differences, ())

    def test_compare_uses_template_order_for_all_difference_kinds(self) -> None:
        differences = compare_documents(
            {"nested": {"second": "old", "third": "label only"}, "first": "old"},
            {
                "last": "prediction only",
                "nested": {"first": "prediction only", "second": "new"},
                "first": "new",
            },
            field_order_template={
                "first": None,
                "nested": {"first": None, "second": None, "third": None},
                "last": None,
            },
        )
        self.assertEqual(
            [difference.path for difference in differences],
            [
                "first",
                "nested.first",
                "nested.second",
                "nested.third",
                "last",
            ],
        )

    def test_loader_maps_predictions_through_metadata_and_writes_html(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            predictions = root / "predictions"
            for split in ("train", "val"):
                (dataset / split / "images").mkdir(parents=True)
                (dataset / split / "annotations").mkdir(parents=True)
                (predictions / split).mkdir(parents=True)

            image = dataset / "train" / "images" / "sample_240dpi.png"
            annotation = dataset / "train" / "annotations" / "gt_sample.json"
            prediction = predictions / "train" / "sample_240dpi.json"
            image.write_bytes(b"image")
            annotation.write_text(
                json.dumps({"content": {"sender": {"city": "Berlin"}}}),
                encoding="utf-8",
            )
            prediction.write_text(
                json.dumps({"sender": {"city": "Berln"}}), encoding="utf-8"
            )
            (dataset / "train" / "metadata.jsonl").write_text(
                json.dumps(
                    {
                        "id": "sample",
                        "image": "train/images/sample_240dpi.png",
                        "annotation": "train/annotations/gt_sample.json",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (dataset / "val" / "metadata.jsonl").write_text("", encoding="utf-8")

            data = load_review_data(predictions, dataset)
            self.assertEqual(data.prediction_count, 1)
            self.assertEqual(len(data.samples), 1)
            self.assertEqual(data.samples[0].differences[0].path, "sender.city")

            output = root / "review" / "index.html"
            built, path = build_review(predictions, dataset, output)
            html = path.read_text(encoding="utf-8")
            self.assertEqual(len(built.samples), 1)
            self.assertIn("Qwen prediction review", html)
            self.assertIn("Edit annotation JSON", html)
            self.assertIn("Open annotation JSON", html)
            self.assertIn("Finish review", html)
            self.assertIn("Add &lt;unreadable&gt; token later?", html)
            self.assertIn("qwen_prediction_review.xlsx", html)
            self.assertIn('class=\"issue-check\"', html)
            self.assertIn('class=\"difference\">Difference', html)
            self.assertIn('class=\"diff\">${inlineDiff(d)}', html)
            self.assertIn('"segments":', html)
            self.assertNotIn("decision-select", html)
            self.assertIn("sender.city", html)
            self.assertIn("vscode://file", html)


if __name__ == "__main__":
    unittest.main()
