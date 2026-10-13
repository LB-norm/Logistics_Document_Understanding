from __future__ import annotations

import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest

from src.fusion import FusionConfig, FusionPipeline, fuse_document, to_llm_text
from src.fusion.__main__ import main


def line(identifier, text, box, polygon=None):
    x1, y1, x2, y2 = box
    return {"id": identifier, "line_index": 0, "text": text, "confidence": 0.9,
            "bbox": box, "polygon": polygon or [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]}


def region(box, label="sender", score=0.8, **extra):
    x1, y1, x2, y2 = box
    return {"label_id": 0, "label": label, "category_id": 19, "score": score,
            "bbox_xyxy": box, "bbox_xywh": [x1, y1, x2 - x1, y2 - y1], **extra}


def inputs():
    lines = [line("a", "Müller", [10, 10, 20, 20]),
             line("b", "Boundary", [40, 10, 60, 20]),
             line("c", "Footer", [80, 80, 90, 90])]
    ocr = {"format_version": 2, "source_path": "/data/page.png", "source_id": "page",
           "text": "Müller\nBoundary\nFooter", "package_versions": {"paddleocr": "3.7.0"},
           "pages": [{"page_number": 1, "width": 100, "height": 100,
                      "coordinate_space": "input_image_pixels", "lines": lines,
                      "text": "Müller\nBoundary\nFooter", "inference_seconds": 0.5}]}
    layout = {"image": "/data/page.png", "width": 100, "height": 100,
              "detections": [region([0, 0, 50, 40]), region([50, 0, 100, 40], "recipient")],
              "inference_seconds": 0.2}
    return ocr, layout


class FusionTests(unittest.TestCase):
    def test_separate_normalizes_full_outputs_without_assignments_or_mutation(self):
        ocr, layout = inputs()
        before = deepcopy((ocr, layout))
        result = FusionPipeline().fuse(ocr, [layout])
        page = result["pages"][0]
        self.assertEqual(result["mode"], "separate")
        self.assertEqual(page["ocr"]["lines"][0]["bbox"], [0.1, 0.1, 0.2, 0.2])
        self.assertEqual(page["layout"]["detections"][0]["bbox_xywh"], [0, 0, 0.5, 0.4])
        self.assertEqual(page["ocr"]["coordinate_space"], page["layout"]["coordinate_space"])
        self.assertNotIn("fields", page)
        self.assertNotIn("memberships", page)
        self.assertNotIn("assignment", result)
        self.assertEqual(result["ocr_metadata"]["package_versions"], ocr["package_versions"])
        self.assertEqual(page["ocr"]["inference_seconds"], 0.5)
        self.assertEqual(page["layout"]["inference_seconds"], 0.2)
        self.assertEqual((ocr, layout), before)
        result["pages"][0]["ocr"]["lines"][0]["text"] = "changed"
        self.assertEqual((ocr, layout), before)

    def test_hard_selects_exactly_one_and_retains_unassigned_text(self):
        ocr, layout = inputs()
        page = fuse_document(ocr, layout, FusionConfig(2))["pages"][0]
        self.assertEqual([len(m["candidates"]) for m in page["memberships"]], [1, 1, 0])
        self.assertEqual(page["fields"][0]["text"], "Müller\nBoundary")
        self.assertEqual(page["fields"][1]["text"], "")
        self.assertEqual(page["unassigned_ocr_ids"], ["c"])
        self.assertEqual(page["memberships"][1]["candidates"][0]["overlap"], 0.5)
        self.assertEqual(page["memberships"][1]["candidates"][0]["weight"], 1)
        self.assertEqual([element["text"] for element in page["ocr"]["lines"]],
                         ["Müller", "Boundary", "Footer"])

    def test_soft_preserves_ambiguity_and_normalizes_relative_weights(self):
        ocr, layout = inputs()
        layout["detections"][0] = region([0, 0, 55, 40])
        layout["detections"][1] = region([55, 0, 100, 40], "recipient")
        page = fuse_document(ocr, layout, FusionConfig("soft"))["pages"][0]
        candidates = page["memberships"][1]["candidates"]
        self.assertEqual(len(candidates), 2)
        self.assertAlmostEqual(candidates[0]["overlap"], 0.75)
        self.assertAlmostEqual(candidates[1]["overlap"], 0.25)
        self.assertAlmostEqual(sum(c["weight"] for c in candidates), 1)
        self.assertEqual(page["fields"][1]["text"], "Boundary")
        self.assertEqual(page["unassigned_ocr_ids"], ["c"])
        self.assertEqual(len(page["ocr"]["lines"]), 3)

    def test_polygon_overlap_differs_from_bbox_overlap_and_winding_is_irrelevant(self):
        ocr, layout = inputs()
        polygon = [[50, 30], [70, 50], [50, 70], [30, 50]]
        ocr["pages"][0]["lines"] = [line("diamond", "Rotated", [30, 30, 70, 70], polygon)]
        layout["detections"] = [region([30, 30, 40, 40])]
        for points in (polygon, list(reversed(polygon))):
            ocr["pages"][0]["lines"][0]["polygon"] = points
            page = fuse_document(ocr, layout, FusionConfig(3, min_overlap=0))["pages"][0]
            # The bounding boxes overlap, but the polygon only touches the region at a vertex.
            self.assertEqual(page["unassigned_ocr_ids"], ["diamond"])
        layout["detections"] = [region([30, 30, 50, 70])]
        page = fuse_document(ocr, layout, FusionConfig(3))["pages"][0]
        self.assertAlmostEqual(page["memberships"][0]["candidates"][0]["overlap"], 0.5)

    def test_containment_uses_ocr_coverage_and_nested_ties_choose_smallest_region(self):
        ocr, layout = inputs()
        layout["detections"] = [region([0, 0, 100, 100], "whole_page", score=0.99),
                                region([5, 5, 25, 25], "sender", score=0.6)]
        page = fuse_document(ocr, layout, FusionConfig(2, min_overlap=1))["pages"][0]
        self.assertEqual(page["fields"][1]["ocr_ids"], ["a"])
        self.assertEqual(page["fields"][0]["ocr_ids"], ["b", "c"])
        self.assertEqual(page["memberships"][0]["candidates"][0]["overlap"], 1)

    def test_equal_geometry_breaks_ties_by_confidence_then_stable_region_id(self):
        ocr, layout = inputs()
        layout["detections"] = [region([0, 0, 30, 30], id="z", score=0.7),
                                region([0, 0, 30, 30], id="b", score=0.9),
                                region([0, 0, 30, 30], id="a", score=0.9)]
        for predictions in (layout["detections"], list(reversed(layout["detections"]))):
            layout["detections"] = predictions
            page = fuse_document(ocr, layout, FusionConfig(2))["pages"][0]
            self.assertEqual(page["memberships"][0]["candidates"][0]["region_id"], "a")

    def test_threshold_is_inclusive_but_zero_overlap_is_never_assigned(self):
        ocr, layout = inputs()
        page = fuse_document(ocr, layout, FusionConfig(3, min_overlap=0.5))["pages"][0]
        self.assertEqual(len(page["memberships"][1]["candidates"]), 2)
        page = fuse_document(ocr, layout, FusionConfig(3, min_overlap=0.51))["pages"][0]
        self.assertEqual(page["unassigned_ocr_ids"], ["b", "c"])
        ocr["pages"][0]["lines"] = [line("touch", "Touching", [0, 40, 10, 50])]
        page = fuse_document(ocr, layout, FusionConfig(2, min_overlap=0))["pages"][0]
        self.assertEqual(page["unassigned_ocr_ids"], ["touch"])

    def test_empty_inputs_zero_area_and_blank_text_are_preserved(self):
        ocr, layout = inputs()
        layout["detections"] = []
        page = fuse_document(ocr, layout, FusionConfig(3))["pages"][0]
        self.assertEqual(page["unassigned_ocr_ids"], ["a", "b", "c"])
        self.assertEqual(page["fields"], [])
        layout["detections"] = [region([0, 0, 100, 100])]
        ocr["pages"][0]["lines"] = [line("degenerate", "", [10, 10, 20, 10])]
        page = fuse_document(ocr, layout, FusionConfig(2))["pages"][0]
        self.assertEqual(page["ocr"]["lines"][0]["text"], "")
        self.assertEqual(page["unassigned_ocr_ids"], ["degenerate"])
        ocr["pages"][0]["lines"] = []
        page = fuse_document(ocr, layout, FusionConfig(3))["pages"][0]
        self.assertEqual(page["memberships"], [])
        self.assertEqual(page["fields"][0]["text"], "")

    def test_repeated_labels_remain_distinct(self):
        ocr, layout = inputs()
        layout["detections"][1]["label"] = "sender"
        page = fuse_document(ocr, layout, FusionConfig(3))["pages"][0]
        self.assertEqual([f["label"] for f in page["fields"]], ["sender", "sender"])
        self.assertNotEqual(page["fields"][0]["region_id"], page["fields"][1]["region_id"])

    def test_rescaling_is_explicit_and_normalized_coordinates_are_not_rescaled_twice(self):
        ocr, layout = inputs()
        layout["width"], layout["height"] = 200, 300
        for detection in layout["detections"]:
            for key in ("bbox_xyxy", "bbox_xywh"):
                detection[key] = [value * (2 if index % 2 == 0 else 3)
                                  for index, value in enumerate(detection[key])]
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            fuse_document(ocr, layout)
        page = fuse_document(ocr, layout, FusionConfig(3, allow_layout_rescale=True))["pages"][0]
        self.assertEqual(len(page["memberships"][1]["candidates"]), 2)
        normalized = fuse_document({"pages": [page["ocr"]]}, page["layout"],
                                   FusionConfig(3, allow_layout_rescale=True))["pages"][0]
        self.assertEqual(normalized["ocr"], page["ocr"])
        self.assertEqual(normalized["layout"], page["layout"])

    def test_non_square_page_normalizes_axes_independently(self):
        ocr, layout = inputs()
        ocr["pages"][0]["height"] = layout["height"] = 200
        page = fuse_document(ocr, layout)["pages"][0]
        self.assertEqual(page["ocr"]["lines"][0]["bbox"], [0.1, 0.05, 0.2, 0.1])

    def test_multipage_matches_explicit_numbers_and_never_assigns_across_pages(self):
        ocr, layout = inputs()
        second = deepcopy(ocr["pages"][0])
        second["page_number"] = 2
        ocr["pages"].append(second)
        first_layout = {**layout, "page_number": 1}
        second_layout = {**layout, "page_number": 2, "detections": []}
        result = fuse_document(ocr, [second_layout, first_layout], FusionConfig(2))
        self.assertEqual(result["pages"][0]["fields"][0]["ocr_ids"], ["a", "b"])
        self.assertEqual(result["pages"][1]["unassigned_ocr_ids"], ["a", "b", "c"])
        for layouts in ([layout, layout], [first_layout], [first_layout, first_layout],
                        [first_layout, {**second_layout, "page_number": 3}]):
            with self.subTest(layouts=layouts), self.assertRaises(ValueError):
                fuse_document(ocr, layouts)

    def test_native_yolo_batch_matches_source_and_rejects_ambiguous_selection(self):
        ocr, layout = inputs()
        other = {**layout, "image": "/data/other.png"}
        self.assertEqual(fuse_document(ocr, [other, layout])["pages"][0]["layout"]["image"], "/data/page.png")
        relocated = {**layout, "image": "/remote/page.png"}
        self.assertEqual(fuse_document(ocr, [other, relocated])["pages"][0]["layout"]["image"], "/remote/page.png")
        for predictions in ([other, other], [relocated, {**layout, "image": "/elsewhere/page.png"}], []):
            with self.assertRaises(ValueError):
                fuse_document(ocr, predictions)

    def test_bbox_only_inputs_and_missing_ids_use_compatible_defaults(self):
        ocr, layout = inputs()
        for element in ocr["pages"][0]["lines"]:
            del element["id"], element["polygon"]
        for detection in layout["detections"]:
            del detection["bbox_xyxy"]
        page = fuse_document(ocr, layout, FusionConfig(2))["pages"][0]
        self.assertEqual(page["fields"][0]["ocr_ids"], ["p0001-l000000", "p0001-l000001"])

    def test_invalid_geometry_confidences_and_identifiers_are_rejected(self):
        ocr, layout = inputs()
        for path, value in (("bbox_xyxy", [50, 0, 10, 40]), ("bbox_xyxy", [0, 0, 0, 40]),
                            ("bbox_xyxy", [-10, 0, 50, 40]), ("score", float("nan")), ("score", 1.1)):
            bad = deepcopy(layout)
            bad["detections"][0][path] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                fuse_document(ocr, bad)
        for path, value in (("polygon", [[0, 0], [10, 10], [0, 10], [10, 0]]),
                            ("polygon", [[0, 0], [10, 10], [0, 10]]),
                            ("confidence", float("inf")), ("confidence", -0.1), ("id", "b")):
            bad = deepcopy(ocr)
            bad["pages"][0]["lines"][0][path] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                fuse_document(bad, layout)
        for space in ("pdf_points", "xywh", "unknown"):
            with self.assertRaises(ValueError):
                fuse_document(ocr, {**layout, "coordinate_space": space, "detections": []})

    def test_mode_aliases_and_configuration_validation(self):
        for number, name in ((1, "separate"), (2, "hard"), (3, "soft")):
            for alias in (number, str(number), name):
                self.assertEqual(FusionConfig(alias).mode, number)
        for mode in (0, 4, "bogus", True, 1.0):
            with self.assertRaises(ValueError):
                FusionConfig(mode)
        for threshold in (-0.1, 1.1, float("nan"), float("inf"), "0.5", True):
            with self.assertRaises(ValueError):
                FusionConfig(min_overlap=threshold)

    def test_cli_round_trip_all_modes_unicode_and_input_protection(self):
        ocr, layout = inputs()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ocr_path, layout_path = root / "page.ocr.json", root / "layouts.json"
            ocr_path.write_text(json.dumps(ocr), encoding="utf-8")
            layout_path.write_text(json.dumps([layout]), encoding="utf-8")
            for mode in ("1", "2", "3"):
                output, llm_output = root / mode / "fusion.json", root / mode / "fusion.txt"
                args = ["--ocr", str(ocr_path), "--layout", str(layout_path), "--mode", mode,
                        "--output", str(output), "--llm-output", str(llm_output)]
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(args), 0)
                result = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(json.loads(llm_output.read_text(encoding="utf-8")), result)
                self.assertIn("Müller", to_llm_text(result))
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(args + ["--output", str(ocr_path)])
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(args + ["--llm-output", str(output)])
                self.assertEqual(json.loads(ocr_path.read_text(encoding="utf-8")), ocr)


if __name__ == "__main__":
    unittest.main()
