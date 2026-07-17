from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.utils.annotation_audit import (
    AuditResult,
    Sample,
    audit_dataset,
    field_statistics,
    field_value_statistics,
    flatten_values,
    get_path,
    learn_consistency_rules,
    risk_score,
    run_distribution_checks,
    schema_leaf_paths,
    vscode_editor_url,
    write_outputs,
)


def entity_content(company: str, city: str | None, postcode: str = "12345") -> dict:
    return {
        "senderInformation": {
            "senderNameCompany": company,
            "senderStreet": "Example Street 1",
            "senderPostcode": postcode,
            "senderCity": city,
            "senderCountryCode": {"value": "DE"},
        }
    }


def sample(sample_id: str, company: str, city: str | None) -> Sample:
    content = entity_content(company, city)
    return Sample(
        sample_id=sample_id,
        split="train",
        image_path=Path(f"{sample_id}.png"),
        annotation_path=Path(f"{sample_id}.json"),
        content=content,
        annotation={"content": content},
        flat_values=flatten_values(content),
    )


def flat_sample(sample_id: str, content: dict) -> Sample:
    return Sample(
        sample_id=sample_id,
        split="train",
        image_path=Path(f"{sample_id}.png"),
        annotation_path=Path(f"{sample_id}.json"),
        content=content,
        annotation={"content": content},
        flat_values=flatten_values(content),
    )


class AnnotationAuditTests(unittest.TestCase):
    def test_flatten_uses_canonical_array_path(self) -> None:
        values = flatten_values({"items": [{"quantity": 1}, {"quantity": 2}]})
        self.assertEqual(
            [value.path for value in values],
            ["items[0].quantity", "items[1].quantity"],
        )
        self.assertEqual({value.canonical_path for value in values}, {"items[].quantity"})

    def test_get_path_returns_none_for_missing_component(self) -> None:
        self.assertEqual(get_path({"a": {"b": 3}}, "a.b"), 3)
        self.assertIsNone(get_path({"a": {}}, "a.b"))

    def test_vscode_editor_url_uses_native_absolute_file_uri(self) -> None:
        target = Path("folder with spaces") / "ground_truth.json"
        expected = "vscode://file" + target.resolve().as_uri().removeprefix("file://")
        self.assertEqual(vscode_editor_url(target), expected)
        self.assertIn("folder%20with%20spaces", expected)

    def test_schema_leaf_paths_include_array_fields(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"quantity": {"type": "integer"}},
                    },
                },
            },
        }
        self.assertEqual(schema_leaf_paths(schema), ("items[].quantity", "name"))

    def test_field_statistics_include_unobserved_target_field(self) -> None:
        rows = field_statistics(
            [flat_sample("a", {"observed": "value"})],
            expected_fields=("observed", "neverObserved"),
        )
        by_field = {row["field"]: row for row in rows}
        self.assertEqual(by_field["observed"]["examples_with_non_null"], 1)
        self.assertEqual(by_field["neverObserved"]["examples_with_non_null"], 0)
        self.assertEqual(by_field["neverObserved"]["occurrences"], 0)

    def test_field_value_statistics_preserve_json_types(self) -> None:
        samples = [
            flat_sample("a", {"label": 1}),
            flat_sample("b", {"label": "1"}),
            flat_sample("c", {"label": 1}),
        ]
        rows = field_value_statistics(samples)
        self.assertEqual(len(rows), 2)
        by_type = {row["value_type"]: row for row in rows}
        self.assertEqual(by_type["integer"]["occurrence_count"], 2)
        self.assertEqual(by_type["string"]["occurrence_count"], 1)

    def test_distribution_checks_find_typo_rarity_and_format_minority(self) -> None:
        samples = [
            flat_sample(f"common-{index}", {"label": "Hello"})
            for index in range(59)
        ]
        typo = flat_sample("typo", {"label": "Helo"})
        numeric_text = flat_sample("numeric-text", {"label": "12345"})
        numeric_type = flat_sample("numeric-type", {"label": 12345})
        samples.extend((typo, numeric_text, numeric_type))

        run_distribution_checks(samples)

        self.assertIn("possible_typo", {issue.code for issue in typo.issues})
        self.assertIn("rare_value", {issue.code for issue in typo.issues})
        self.assertIn(
            "minority_character_class", {issue.code for issue in numeric_text.issues}
        )
        self.assertIn("minority_json_type", {issue.code for issue in numeric_type.issues})

    def test_learned_rule_flags_only_exception(self) -> None:
        samples = [
            sample("a", "Example GmbH", "Berlin"),
            sample("b", "Example GmbH", "Berlin"),
            sample("c", "Example GmbH", "Berlin"),
            sample("d", "Example GmbH", "Munich"),
        ]
        rules = learn_consistency_rules(samples, min_support=3, min_confidence=0.70)
        city_rules = [rule for rule in rules if rule.dependent_field.endswith("senderCity")]
        self.assertEqual(len(city_rules), 2)  # company→city and postcode→city
        self.assertTrue(
            any(
                issue.code == "learned_consistency_exception"
                for issue in samples[-1].issues
            )
        )
        self.assertFalse(samples[0].issues)
        self.assertGreater(risk_score(samples[-1]), 0)

    def test_end_to_end_loader_finds_semantic_problem(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "train" / "images").mkdir(parents=True)
            (root / "train" / "annotations").mkdir(parents=True)
            image = root / "train" / "images" / "one.png"
            image.write_bytes(b"not a real image, but hashing is sufficient")
            annotation = root / "train" / "annotations" / "one.json"
            annotation.write_text(
                json.dumps({"content": {"senderInformation": {"senderCity": "12345"}}}),
                encoding="utf-8",
            )
            (root / "train" / "metadata.jsonl").write_text(
                json.dumps(
                    {
                        "id": "one",
                        "image": "train/images/one.png",
                        "annotation": "train/annotations/one.json",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = audit_dataset(root, schema_path=None)
            self.assertFalse(result.load_errors)
            self.assertEqual(len(result.samples), 1)
            self.assertIn("invalid_city", {issue.code for issue in result.samples[0].issues})

    def test_write_outputs_creates_complete_value_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            samples = [
                flat_sample("number", {"label": 1}),
                flat_sample("string", {"label": "1"}),
            ]
            for item in samples:
                item.image_path = root / f"{item.sample_id}.png"
                item.annotation_path = root / f"{item.sample_id}.json"
                item.image_path.write_bytes(b"image")
                item.annotation_path.write_text("{}", encoding="utf-8")
            result = AuditResult(samples, [], [], 0, 0, ("label", "neverObserved"))

            summary = write_outputs(result, root / "audit", root, high_risk_threshold=70)

            with (root / "audit" / "field_values.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                value_rows = list(csv.DictReader(handle))
            with (root / "audit" / "field_statistics.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                field_rows = list(csv.DictReader(handle))
            self.assertEqual(
                {row["value_type"] for row in value_rows}, {"integer", "string"}
            )
            missing = next(row for row in field_rows if row["field"] == "neverObserved")
            self.assertEqual(missing["examples_with_non_null"], "0")

            review_html = (root / "audit" / "review.html").read_text(encoding="utf-8")
            self.assertEqual(
                review_html,
                (root / "audit" / "report.html").read_text(encoding="utf-8"),
            )
            self.assertIn("Annotation review", review_html)
            self.assertIn("Edit JSON in VS Code", review_html)
            self.assertIn("vscode://file", review_html)
            self.assertNotIn("<textarea", review_html)
            self.assertNotIn("Edit annotation JSON", review_html)
            self.assertIn("Mark reviewed and continue", review_html)
            self.assertIn("annotation_review_checklist.csv", review_html)
            self.assertIn("['sample_id','split','annotation_path','review_status'", review_html)
            self.assertEqual(
                summary["outputs"]["html_review"], str(root / "audit" / "review.html")
            )


if __name__ == "__main__":
    unittest.main()
