from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.eval_suite.results_table import (
    evaluate_results_folder,
    flatten_result,
    write_results_tables,
)


class ResultsTableTests(unittest.TestCase):
    def _write_json(self, path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_evaluates_runs_and_splits_default_from_challenge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            results = root / "results"
            testset = root / "test"
            schema = root / "schema.json"
            self._write_json(
                schema,
                {
                    "type": "object",
                    "required": ["value"],
                    "properties": {"value": {"type": "string"}},
                    "additionalProperties": False,
                },
            )
            for subset, sample_id, expected in (
                ("default", "regular", "A"),
                ("challenge", "difficult", "B"),
            ):
                self._write_json(
                    testset
                    / "annotations"
                    / "ground_truths"
                    / subset
                    / f"gt_{sample_id}.json",
                    {"content": {"value": expected}},
                )

            perfect = results / "perfect-model"
            weaker = results / "weaker-model"
            self._write_json(perfect / "regular_240dpi.json", {"value": "A"})
            self._write_json(perfect / "difficult_240dpi.json", {"value": "B"})
            self._write_json(weaker / "regular_240dpi.json", {"value": "A"})
            self._write_json(weaker / "difficult_240dpi.json", {"value": "wrong"})
            (weaker / "inference_manifest.jsonl").write_text(
                "\n".join(
                    json.dumps(record)
                    for record in (
                        {"image_path": "regular_240dpi.png", "status": "ok"},
                        {
                            "image_path": "difficult_240dpi.png",
                            "status": "parse_error",
                            "parse_error": {"kind": "unterminated_json"},
                            "recovery": {
                                "applied": True,
                                "recovered_fields": ["value"],
                                "recovered_field_count": 1,
                            },
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            evaluations = evaluate_results_folder(
                results,
                testset_path=testset,
                schema_path=schema,
            )

            self.assertEqual(
                [item.run_name for item in evaluations],
                ["perfect-model", "weaker-model"],
            )
            weaker_row = flatten_result(evaluations[1])
            self.assertEqual(weaker_row["default_field_f1"], 1.0)
            self.assertEqual(weaker_row["challenge_field_f1"], 0.0)
            self.assertEqual(weaker_row["challenge_minus_default_field_f1"], -1.0)
            self.assertEqual(weaker_row["overall_parse_rate"], 0.5)
            self.assertEqual(weaker_row["challenge_parse_rate"], 0.0)
            self.assertEqual(weaker_row["challenge_minus_default_parse_rate"], -1.0)
            self.assertEqual(weaker_row["overall_samples"], 2)

            csv_path, markdown_path = write_results_tables(
                evaluations,
                input_folder=results,
            )
            with csv_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertIn("overall_value_similarity", rows[0])
            self.assertIn("challenge_minus_default_field_f1", rows[0])

            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("## Overall", markdown)
            self.assertIn("## Default", markdown)
            self.assertIn("## Challenge", markdown)
            self.assertIn("## Challenge minus default", markdown)

            # Generated tables must not be mistaken for prediction runs on rerun.
            rerun = evaluate_results_folder(
                results,
                testset_path=testset,
                schema_path=schema,
            )
            self.assertEqual(len(rerun), 2)

    def test_manifest_parse_failure_keeps_recovered_fields_but_fails_parse_rate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            results = root / "results"
            run = results / "model"
            testset = root / "test"
            schema = root / "schema.json"
            self._write_json(
                schema,
                {
                    "type": "object",
                    "required": ["value"],
                    "properties": {"value": {"type": ["string", "null"]}},
                },
            )
            for subset, sample_id, expected in (
                ("default", "regular", "A"),
                ("challenge", "difficult", "B"),
            ):
                self._write_json(
                    testset
                    / "annotations"
                    / "ground_truths"
                    / subset
                    / f"gt_{sample_id}.json",
                    {"content": {"value": expected}},
                )
            self._write_json(run / "regular_240dpi.json", {"value": "A"})
            self._write_json(run / "difficult_240dpi.json", {"value": None})
            (run / "inference_manifest.jsonl").write_text(
                "\n".join(
                    (
                        json.dumps(
                            {"image_path": "regular_240dpi.png", "status": "ok"}
                        ),
                        json.dumps(
                            {
                                "image_path": "difficult_240dpi.png",
                                "status": "parse_error",
                                "raw_text": '{"value":"B"',
                            }
                        ),
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            evaluation = evaluate_results_folder(
                results, testset_path=testset, schema_path=schema
            )[0]
            row = flatten_result(evaluation)

            self.assertEqual(row["challenge_field_f1"], 1.0)
            self.assertEqual(row["challenge_parse_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
