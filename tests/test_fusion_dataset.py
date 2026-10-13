from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.fusion.dataset import prepare_fusion_dataset
from src.fusion.prepare_sft import main
from src.Qwen.prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT, build_fusion_messages


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def make_dataset(base):
    root = base / "dataset"
    ocr = root / "ocr_v6_cmr"
    layout = root / "yolo_v11s/layout_detections"
    schema = {"type": "object", "properties": {"sender": {"type": ["string", "null"]}},
              "required": ["sender"], "additionalProperties": False}
    write_json(ocr / "schema.json", schema)
    write_json(ocr / "template.json", {"sender": None})
    write_json(ocr / "preparation_summary.json", {"annotation_target_key": "content"})
    for split, names in (("train", ("z", "a")), ("val", ("b",))):
        source, metadata, training, predictions = [], [], [], []
        for name in names:
            identifier = f"{split}-{name}"
            image = f"{split}/images/{name}.png"
            annotation = f"{split}/annotations/{name}.json"
            target = {"sender": f"TARGET_ONLY_{name}"}
            # No valid raster bytes are needed: preparation never decodes images.
            (root / image).parent.mkdir(parents=True, exist_ok=True)
            (root / image).write_bytes(b"source image placeholder")
            write_json(root / annotation, {"content": target})
            source.append({"id": identifier, "image": image, "annotation": annotation})
            lines = [
                {"id": "inside", "text": "Müller", "confidence": 0.9, "bbox": [10, 10, 20, 20]},
                {"id": "boundary", "text": "Shared", "confidence": 0.8, "bbox": [40, 10, 60, 20]},
                {"id": "footer", "text": "Unassigned evidence", "confidence": 0.7, "bbox": [80, 80, 90, 90]},
            ]
            document = {"source_id": identifier, "source_path": f"/old/machine/{image}",
                        "text": "Müller\nShared\nUnassigned evidence", "has_text": True,
                        "pages": [{"page_number": 1, "width": 100, "height": 100,
                                   "coordinate_space": "input_image_pixels", "lines": lines}]}
            artifact = f"{split}/ocr/{name}.ocr.json"
            write_json(ocr / artifact, document)
            metadata.append({"id": identifier, "source_split": split, "status": "ok",
                             "source_image": document["source_path"], "ocr_json": artifact})
            training.append({"id": identifier, "source_split": split, "ocr_json": artifact,
                             "target": target, "messages": [{"role": "assistant", "content": json.dumps(target)}]})
            predictions.append({"image": f"/different/machine/{image}", "width": 100, "height": 100,
                                "detections": [
                                    {"label": f"sender-{name}", "score": 0.9, "bbox_xyxy": [0, 0, 50, 40]},
                                    {"label": "recipient", "score": 0.8, "bbox_xyxy": [50, 0, 100, 40]},
                                ]})
        write_jsonl(root / split / "metadata.jsonl", source)
        write_jsonl(ocr / split / "metadata.jsonl", metadata)
        write_jsonl(ocr / f"{split}.jsonl", training)
        write_json(layout / f"{split}.json", list(reversed(predictions)))
    return root, root / "fusion_sft", ocr, layout


class FusionDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root, self.output, self.ocr, self.layout = make_dataset(Path(self.temporary.name))

    def build(self, **kwargs):
        return prepare_fusion_dataset(self.root, self.output, **kwargs)

    def rows(self, mode="soft", split="train"):
        return read_jsonl(self.output / mode / f"{split}.jsonl")

    def representation(self, row):
        return json.loads(row["messages"][1]["content"].split("OCR/layout representation JSON:\n", 1)[1])

    def test_all_modes_pair_by_filename_preserve_targets_and_match_inference_prompts(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        summary = self.build()
        for mode in ("separate", "hard", "soft"):
            records = self.rows(mode)
            self.assertEqual([r["id"] for r in records], ["train-z", "train-a"])
            self.assertEqual([r["id"] for r in self.rows(mode, "val")], ["val-b"])
            for row, original in zip(records, read_jsonl(self.ocr / "train.jsonl")):
                fused = self.representation(row)
                page = fused["pages"][0]
                self.assertEqual(row["target"], original["target"])
                self.assertEqual(json.loads(row["messages"][-1]["content"]), original["target"])
                self.assertEqual([m["role"] for m in row["messages"]], ["system", "user", "assistant"])
                self.assertTrue(all(isinstance(m["content"], str) for m in row["messages"]))
                self.assertNotIn("TARGET_ONLY", row["messages"][0]["content"] + row["messages"][1]["content"])
                self.assertEqual(page["layout"]["detections"][0]["label"], f"sender-{row['id'][-1]}")
                self.assertEqual(page["ocr"]["lines"][0]["bbox"], [0.1, 0.1, 0.2, 0.2])
                self.assertEqual(page["ocr"]["lines"][-1]["text"], "Unassigned evidence")
                if mode == "separate":
                    self.assertNotIn("memberships", page)
                else:
                    self.assertEqual(page["unassigned_ocr_ids"], ["footer"])
                    self.assertEqual(len(page["memberships"][1]["candidates"]), 1 if mode == "hard" else 2)
                inference = build_fusion_messages(DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT, fused,
                                                  read_json(self.ocr / "template.json"), read_json(self.ocr / "schema.json"))
                self.assertEqual(row["messages"][:-1], [{"role": m["role"], "content": m["content"][0]["text"]} for m in inference])
                for key in ("source_image", "source_annotation", "ocr_json", "layout_json"):
                    self.assertFalse(Path(row["provenance"][key]).is_absolute())
                    self.assertTrue((self.output / row["provenance"][key]).is_file())
                self.assertEqual(row["lengths"]["conversation_characters"], sum(len(m["content"]) for m in row["messages"]))
                self.assertGreater(row["lengths"]["conversation_utf8_bytes"], row["lengths"]["conversation_characters"])
            self.assertEqual(summary["splits"]["train"]["modes"][mode]["ocr_lines"], 6)
        self.assertEqual(summary["splits"]["train"]["modes"]["soft"]["unassigned_ocr_lines"], 2)
        self.assertEqual(summary["splits"]["train"]["modes"]["soft"]["ambiguous_ocr_lines"], 2)
        self.assertEqual(read_json(self.output / "preparation_summary.json"), summary)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_mode_aliases_limit_and_dry_run(self):
        summary = self.build(modes=("2", "soft"), limit=1, dry_run=True)
        self.assertFalse(self.output.exists())
        self.assertEqual(summary["settings"]["modes"], ["hard", "soft"])
        self.assertEqual(summary["splits"]["train"]["examples"], 1)
        self.assertFalse(summary["settings"]["truncated"])
        written = self.build(modes=("2", "soft"), limit=1)
        summary["dry_run"] = False
        self.assertEqual(summary, written)
        self.assertFalse((self.output / "separate").exists())
        self.assertEqual([r["id"] for r in self.rows()], ["train-z"])

    def test_missing_extra_and_duplicate_layouts_fail_without_output(self):
        path = self.layout / "train.json"
        predictions = read_json(path)
        cases = (predictions[:-1], predictions + [predictions[0]],
                 predictions + [{**predictions[0], "image": "extra.png"}],
                 [{**predictions[0], "image": "/other/a.png"}, predictions[0]])
        for case in cases:
            with self.subTest(case=case):
                write_json(path, case)
                with self.assertRaisesRegex(ValueError, "duplicate|ambiguous|exactly match"):
                    self.build()
                self.assertFalse(self.output.exists())

    def test_inconsistent_target_status_source_or_geometry_leaves_existing_output_untouched(self):
        self.build()
        before = {p: p.read_bytes() for p in self.output.rglob("*") if p.is_file()}
        artifact = self.ocr / "val/ocr/b.ocr.json"
        original = read_json(artifact)
        malformed = json.loads(json.dumps(original))
        malformed["pages"][0]["lines"][0]["bbox"] = [0, 0, 101, 20]
        for replacement, pattern in ((malformed, "outside the page"),
                                     ({**original, "source_id": "wrong"}, "source_id"),
                                     ({**original, "source_path": "other.png"}, "source_path"),
                                     ({**original, "has_text": False}, "empty"),
                                     ({**original, "pages": original["pages"] * 2}, "one raster page")):
            with self.subTest(pattern=pattern):
                write_json(artifact, replacement)
                with self.assertRaisesRegex(ValueError, pattern):
                    self.build(overwrite=True)
                self.assertEqual(before, {p: p.read_bytes() for p in self.output.rglob("*") if p.is_file()})
                self.assertFalse(list(self.root.glob(".fusion-sft-*")))

    def test_targets_assistant_answers_schema_and_split_declarations_are_checked(self):
        path = self.ocr / "train.jsonl"
        original = read_jsonl(path)
        for field, value, pattern in (
            ("target", {"sender": "wrong"}, "target differs"),
            ("messages", [{"role": "assistant", "content": '{"sender": "wrong"}'}], "assistant answer"),
            ("source_split", "val", "source_split"),
            ("ocr_json", "train/ocr/a.ocr.json", "artifact links"),
        ):
            rows = json.loads(json.dumps(original))
            rows[0][field] = value
            write_jsonl(path, rows)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, pattern):
                self.build()
            self.assertFalse(self.output.exists())
        write_jsonl(path, original)
        write_json(self.ocr / "schema.json", {"type": "object", "properties": {"sender": {"type": "null"}}})
        with self.assertRaisesRegex(ValueError, "Target fails schema"):
            self.build()

    def test_failed_ocr_or_missing_artifact_is_not_silently_dropped(self):
        path = self.ocr / "train/metadata.jsonl"
        records = read_jsonl(path)
        records[0]["status"] = "ocr_error"
        write_jsonl(path, records)
        with self.assertRaisesRegex(ValueError, "OCR status must be ok"):
            self.build()
        records[0]["status"] = "ok"
        records[0]["ocr_json"] = "absent.json"
        write_jsonl(path, records)
        with self.assertRaisesRegex(ValueError, "absent.json"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_duplicate_ids_and_membership_mismatch_are_rejected(self):
        path = self.ocr / "train.jsonl"
        records = read_jsonl(path)
        for replacement, pattern in ((records + [records[0]], "unique"), (records[:1], "IDs must match")):
            write_jsonl(path, replacement)
            with self.assertRaisesRegex(ValueError, pattern):
                self.build()

    def test_cross_split_duplicate_ids_are_rejected(self):
        for path in (self.root / "val/metadata.jsonl", self.ocr / "val/metadata.jsonl", self.ocr / "val.jsonl"):
            records = read_jsonl(path)
            records[0]["id"] = "train-z"
            write_jsonl(path, records)
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            self.build()

    def test_no_detections_retains_all_evidence_and_empty_ocr_is_rejected(self):
        path = self.layout / "train.json"
        predictions = read_json(path)
        for prediction in predictions:
            prediction["detections"] = []
        write_json(path, predictions)
        self.build(modes=(3,))
        page = self.representation(self.rows()[0])["pages"][0]
        self.assertEqual(page["unassigned_ocr_ids"], ["inside", "boundary", "footer"])
        self.assertEqual(len(page["ocr"]["lines"]), 3)
        path = self.ocr / "train/ocr/z.ocr.json"
        document = read_json(path)
        document["pages"][0]["lines"] = []
        write_json(path, document)
        with self.assertRaisesRegex(ValueError, "OCR text is empty"):
            self.build(overwrite=True)

    def test_different_dimensions_require_explicit_rescale(self):
        path = self.layout / "val.json"
        predictions = read_json(path)
        predictions[0]["width"] = 200
        for region in predictions[0]["detections"]:
            region["bbox_xyxy"][0] *= 2
            region["bbox_xyxy"][2] *= 2
        write_json(path, predictions)
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            self.build()
        self.build(allow_layout_rescale=True)
        page = self.representation(self.rows(split="val")[0])["pages"][0]
        self.assertEqual(page["layout"]["detections"][0]["bbox_xyxy"], [0, 0, 0.5, 0.4])

    def test_overlap_threshold_changes_membership_without_removing_text(self):
        self.build(modes=(2, 3), min_overlap=0.6)
        for mode in ("hard", "soft"):
            page = self.representation(self.rows(mode)[0])["pages"][0]
            self.assertEqual(page["unassigned_ocr_ids"], ["boundary", "footer"])
            self.assertEqual(len(page["ocr"]["lines"]), 3)

    def test_custom_contract_paths_splits_and_source_path_styles(self):
        write_json(self.ocr / "preparation_summary.json", {"annotation_target_key": "nested.answer"})
        schema = self.root / "custom.schema.json"
        template = self.root / "custom.template.json"
        write_json(schema, read_json(self.ocr / "schema.json"))
        write_json(template, read_json(self.ocr / "template.json"))
        path = self.root / "train/metadata.jsonl"
        records = read_jsonl(path)
        for index, row in enumerate(records):
            annotation = self.root / row["annotation"]
            write_json(annotation, {"nested": {"answer": read_json(annotation)["content"]}})
            row["image"] = str((self.root / row["image"]).resolve()) if index else "images/z.png"
            row["annotation"] = str(annotation.resolve()) if index else "annotations/z.json"
        write_jsonl(path, records)
        self.build(splits=("train",), modes=(1,), schema_path=schema, template_path=template)
        self.assertEqual(self.rows("separate")[0]["target"], {"sender": "TARGET_ONLY_z"})
        self.assertFalse((self.output / "separate/val.jsonl").exists())

    def test_output_protection_overwrite_and_invalid_options(self):
        for kwargs in ({"modes": (1, "separate")}, {"modes": ()}, {"min_overlap": float("nan")},
                       {"limit": 0}, {"splits": ("../train",)}, {"splits": ("train", "train")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.build(**kwargs)
        for output in (self.root, self.ocr, self.ocr / "sft", self.root / "train", self.layout.parent):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "must not overlap"):
                prepare_fusion_dataset(self.root, output, overwrite=True)
        self.build()
        self.assertTrue(self.build(dry_run=True)["dry_run"])
        with self.assertRaises(FileExistsError):
            self.build()
        marker = self.output / "keep.txt"
        marker.write_text("keep")
        self.build(overwrite=True, limit=1)
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(marker.read_text(), "keep")

    def test_generated_path_symlinks_cannot_redirect_writes_into_inputs(self):
        self.output.mkdir()
        (self.output / "soft").symlink_to(self.ocr, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            self.build(overwrite=True)

    def test_cli_module_and_direct_script_need_only_standard_library(self):
        root = Path(__file__).resolve().parents[1]
        common = ["--dataset-root", str(self.root), "--modes", "3", "--limit", "1", "--dry-run"]
        for command in ([sys.executable, "-S", "-m", "src.fusion.prepare_sft"],
                        [sys.executable, "-S", str(root / "src/fusion/prepare_sft.py")]):
            result = subprocess.run(command + common, cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["settings"]["modes"], ["soft"])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--dataset-root", str(self.root), "--limit", "1"]), 0)
        self.assertEqual(len(self.rows()), 1)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main(["--dataset-root", str(self.root), "--modes", "1", "separate"])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
