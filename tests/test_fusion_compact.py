from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from src.fusion import FusionConfig, fuse_document
from src.fusion.compact import compact_document
from src.fusion.compact_sft import COMPACT_MARKER, compact_dataset, convert_record
from src.Qwen.prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT, build_fusion_messages


def document(mode):
    ocr = {"text": "Müller\nShared\nFooter", "source_path": "/old/page.png",
           "pages": [{"page_number": 1, "width": 100, "height": 100,
                      "coordinate_space": "input_image_pixels",
                      "lines": [{"id": name, "text": text, "confidence": 0.987654321,
                                 "bbox": box}
                                for name, text, box in (
                                    ("long-id-a", "Müller", [10.1234567, 10, 20, 20]),
                                    ("long-id-b", "Shared", [40, 10, 60, 20]),
                                    ("long-id-c", "Footer", [80, 80, 90, 90]))]}]}
    layout = {"width": 100, "height": 100,
              "detections": [{"id": name, "label": label, "score": 0.912345678,
                              "bbox_xyxy": box}
                             for name, label, box in (
                                 ("region-a", "sender", [0, 0, 50, 40]),
                                 ("region-b", "recipient", [50, 0, 100, 40]))]}
    return fuse_document(ocr, layout, FusionConfig(mode))


def record(mode, split="train"):
    target = {"sender": "Müller", "amount": 1.23456789}
    messages = build_fusion_messages(DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT,
                                     document(mode), {}, {})
    messages = [{"role": m["role"], "content": m["content"][0]["text"]} for m in messages]
    messages.append({"role": "assistant", "content": json.dumps(target, ensure_ascii=False)})
    return {"id": f"sample-{split}", "source_split": split, "fusion_mode": mode,
            "messages": messages, "target": target,
            "provenance": {"ocr_json": "../ocr/page.json", "layout_image": "/old/page.png"}}


class CompactFusionTests(unittest.TestCase):
    def test_all_modes_keep_text_geometry_and_candidate_identity_without_mutating_source(self):
        for mode in ("separate", "hard", "soft"):
            with self.subTest(mode=mode):
                original = document(mode)
                before = deepcopy(original)
                result = compact_document(original)
                page = result["pages"][0]
                self.assertEqual([line[:2] for line in page["lines"]],
                                 [[0, "Müller"], [1, "Shared"], [2, "Footer"]])
                self.assertEqual(page["lines"][0][2][0][0], 0.1012)
                self.assertEqual(page["lines"][0][3], 0.9877)
                self.assertEqual([r[:2] for r in page["regions"]], [[0, "sender"], [1, "recipient"]])
                if mode == "separate":
                    self.assertNotIn("memberships", page)
                else:
                    self.assertEqual(page["memberships"][2], [2, []])
                    self.assertEqual(page["memberships"][0], [0, [[0, 1.0, 1.0]]])
                    if mode == "hard":
                        self.assertEqual(page["memberships"][1], [1, [[0, 0.5, 1.0]]])
                    else:
                        self.assertEqual(page["memberships"][1],
                                         [1, [[0, 0.5, 0.5], [1, 0.5, 0.5]]])
                self.assertEqual(original, before)
                self.assertNotIn("ocr_metadata", result)
                self.assertNotIn("fields", page)

    def test_targets_answers_schema_contract_and_provenance_are_preserved(self):
        original = record("soft")
        before = deepcopy(original)
        source, output = Path("/tmp/data/fusion_sft"), Path("/tmp/data/nested/compact")
        result = convert_record(original, source, output)
        self.assertEqual(result["target"], original["target"])
        self.assertEqual(result["messages"][2], original["messages"][2])
        self.assertEqual(result["id"], original["id"])
        self.assertIn("JSON Schema:", result["messages"][0]["content"])
        self.assertIn("Memberships are", result["messages"][0]["content"])
        self.assertNotIn("fields contain assigned text", result["messages"][0]["content"])
        self.assertIn(COMPACT_MARKER, result["messages"][1]["content"])
        self.assertEqual((output / result["provenance"]["ocr_json"]).resolve(),
                         (source / original["provenance"]["ocr_json"]).resolve())
        self.assertEqual(result["provenance"]["layout_image"], "/old/page.png")
        self.assertEqual(original, before)

    def test_dangling_memberships_and_duplicate_ids_are_rejected(self):
        bad = document("soft")
        bad["pages"][0]["memberships"][0]["candidates"][0]["region_id"] = "missing"
        with self.assertRaises(KeyError):
            compact_document(bad)
        bad = document("separate")
        bad["pages"][0]["ocr"]["lines"][1]["id"] = "long-id-a"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            compact_document(bad)

    def test_dataset_conversion_is_complete_and_invalid_input_is_not_published(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "source", root / "compact"
            source.mkdir()
            for name in ("schema.json", "template.json"):
                (source / name).write_text("{}\n", encoding="utf-8")
            splits = {}
            for split in ("train", "val"):
                splits[split] = {"examples": 1, "modes": {}}
                for mode in ("separate", "hard", "soft"):
                    (source / mode).mkdir(exist_ok=True)
                    (source / mode / f"{split}.jsonl").write_text(
                        json.dumps(record(mode, split)) + "\n", encoding="utf-8")
                    splits[split]["modes"][mode] = {}
            (source / "preparation_summary.json").write_text(json.dumps({"splits": splits}))
            before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
            summary = compact_dataset(source, output)
            self.assertEqual(summary["splits"]["train"]["examples"], 1)
            self.assertEqual(len(list(output.glob("*/*.jsonl"))), 6)
            self.assertEqual((output / "schema.json").read_bytes(), b"{}\n")
            self.assertEqual(before, {p: p.read_bytes() for p in source.rglob("*") if p.is_file()})
            with self.assertRaisesRegex(ValueError, "already exists"):
                compact_dataset(source, output)
            path = source / "soft/val.jsonl"
            bad = record("soft", "val")
            bad["target"]["sender"] = "changed"
            path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from target"):
                compact_dataset(source, root / "invalid")
            self.assertFalse((root / "invalid").exists())
            self.assertFalse(list(root.glob(".fusion-compact-*")))


if __name__ == "__main__":
    unittest.main()
