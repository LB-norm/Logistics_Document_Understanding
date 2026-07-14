from __future__ import annotations

from pathlib import Path
import unittest

from src.utils.annotation_audit import Sample, flatten_values
from src.utils.normalize_location_labels import propose_changes


def make_sample(
    sample_id: str,
    pickup: str,
    delivery: str,
    sender_company: str = "Sender GmbH",
    sender_postcode: str = "8142",
    sender_country: str = "A",
    consignee_company: str = "Receiver GmbH",
    consignee_postcode: str = "2325",
    consignee_country: str = "A",
) -> Sample:
    content = {
        "senderInformation": {
            "senderNameCompany": sender_company,
            "senderPostcode": sender_postcode,
            "senderCountryCode": {"value": sender_country},
        },
        "consigneeInformation": {
            "consigneeNameCompany": consignee_company,
            "consigneePostcode": consignee_postcode,
            "consigneeCountryCode": {"value": consignee_country},
        },
        "takingOverTheGoods": {"takingOverTheGoodsPlace": pickup},
        "deliveryOfTheGoods": {"logisticsLocationCity": delivery},
    }
    annotation = {"content": content}
    return Sample(
        sample_id=sample_id,
        split="train",
        image_path=Path(f"{sample_id}.png"),
        annotation_path=Path("dataset") / f"{sample_id}.json",
        content=content,
        annotation=annotation,
        flat_values=flatten_values(content),
    )


class LocationNormalizationTests(unittest.TestCase):
    def test_repeated_city_country_rule_proposes_suffix(self) -> None:
        samples = [
            make_sample("a", "WUNDSCHUH, Österreich", "Vienna, Österreich"),
            make_sample("b", "WUNDSCHUH, Österreich", "Vienna, Österreich"),
            make_sample("target", "WUNDSCHUH", "Vienna, Österreich"),
        ]
        changes, _, _ = propose_changes(samples, Path("dataset"), min_support=2)
        pickup = [change for change in changes if change.sample_id == "target" and "takingOver" in change.field]
        self.assertEqual(len(pickup), 1)
        self.assertEqual(pickup[0].new_value, "WUNDSCHUH, Österreich")

    def test_other_location_context_selects_language_variant(self) -> None:
        samples = [
            make_sample("de1", "LANGENHAGEN, Deutschland", "HIMBERG, Österreich"),
            make_sample("de2", "WUNDSCHUH, Österreich", "HIMBERG, Österreich"),
            make_sample(
                "cz1",
                "BRNO, Česká republika",
                "HIMBERG, Rakousko",
                sender_company="DACHSER CZECH",
                sender_postcode="62700",
                sender_country="CZ",
            ),
            make_sample(
                "cz2",
                "BRNO, Ceska republika",
                "HIMBERG, Rakousko",
                sender_company="DACHSER CZECH",
                sender_postcode="62700",
                sender_country="CZ",
            ),
            make_sample(
                "target",
                "BRNO, Česká republika",
                "HIMBERG",
                sender_company="DACHSER CZECH",
                sender_postcode="62700",
                sender_country="CZ",
            ),
        ]
        changes, _, _ = propose_changes(samples, Path("dataset"), min_support=2)
        target = [change for change in changes if change.sample_id == "target"]
        self.assertEqual(len(target), 1)
        self.assertEqual(target[0].new_value, "HIMBERG, Rakousko")

    def test_single_peer_without_strong_context_is_skipped(self) -> None:
        samples = [
            make_sample("peer", "WUNSTORF, Deutschland", "Paris, Frankreich"),
            make_sample(
                "target",
                "WUNSTORF",
                "Rome, Italien",
                sender_company="Different Sender",
                sender_postcode="99999",
                sender_country="D",
                consignee_company="Different Receiver",
                consignee_postcode="11111",
                consignee_country="IT",
            ),
        ]
        changes, skipped, _ = propose_changes(samples, Path("dataset"), min_support=2)
        self.assertFalse(changes)
        self.assertTrue(any(candidate.sample_id == "target" for candidate in skipped))

    def test_repeated_city_with_unrelated_context_is_skipped(self) -> None:
        samples = [
            make_sample("peer1", "Prague, Czechia", "Hamburg, Německo"),
            make_sample("peer2", "Brno, Czechia", "Hamburg, Německo"),
            make_sample(
                "target",
                "Rotterdam",
                "Hamburg",
                sender_company="Unrelated Dutch Sender",
                sender_postcode="3000",
                sender_country="NL",
                consignee_company="Unrelated German Receiver",
                consignee_postcode="21147",
                consignee_country="DE",
            ),
        ]
        changes, skipped, _ = propose_changes(samples, Path("dataset"), min_support=2)
        self.assertFalse(any(change.sample_id == "target" for change in changes))
        self.assertTrue(any(candidate.sample_id == "target" for candidate in skipped))


if __name__ == "__main__":
    unittest.main()
