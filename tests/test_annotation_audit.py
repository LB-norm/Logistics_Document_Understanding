from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.utils.annotation_audit import (
    Sample,
    audit_dataset,
    flatten_values,
    get_path,
    learn_consistency_rules,
    risk_score,
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


class AnnotationAuditTests(unittest.TestCase):
    def test_flatten_uses_canonical_array_path(self) -> None:
        values = flatten_values({"items": [{"quantity": 1}, {"quantity": 2}]})
        self.assertEqual([value.path for value in values], ["items[0].quantity", "items[1].quantity"])
        self.assertEqual({value.canonical_path for value in values}, {"items[].quantity"})

    def test_get_path_returns_none_for_missing_component(self) -> None:
        self.assertEqual(get_path({"a": {"b": 3}}, "a.b"), 3)
        self.assertIsNone(get_path({"a": {}}, "a.b"))

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
        self.assertTrue(any(issue.code == "learned_consistency_exception" for issue in samples[-1].issues))
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


if __name__ == "__main__":
    unittest.main()
