from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from src.PP_parser import OCRConfig, OCRExtractor, save_ocr
from src.PP_parser.ocr import normalize_page
from src.PP_parser.ocr_dataset import prepare_ocr_dataset
from src.PP_parser.run_ocr import main as run_ocr, parse_args as parse_ocr_args
from src.Qwen.run_inference import (
    InferenceRuntime, generate_image_prediction, parse_args, resolve_model_id,
)


def payload(texts=("Müller", "Berlin")):
    return {"rec_texts": list(texts), "rec_scores": [0.9] * len(texts),
            "rec_polys": [[[1, 2], [9, 1], [10, 5], [2, 6]]] * len(texts)}


def pipeline_for(texts=("Müller", "Berlin")):
    pipeline = MagicMock()
    pipeline.predict.side_effect = lambda **kwargs: [SimpleNamespace(json={"res": payload(texts)})]
    return pipeline


class PPOCRTests(unittest.TestCase):
    def test_normalization_keeps_aligned_geometry_and_unicode(self):
        page = normalize_page(json.dumps({"res": payload()}, ensure_ascii=False), 2, (20, 30))
        self.assertEqual(page["text"], "Müller\nBerlin")
        self.assertEqual((page["width"], page["height"]), (20, 30))
        self.assertEqual(page["coordinate_space"], "input_image_pixels")
        self.assertEqual(page["lines"][0]["bbox"], [1, 1, 10, 6])
        self.assertEqual(page["lines"][1]["id"], "p0002-l000001")
        self.assertEqual(page["lines"][1]["polygon"], payload()["rec_polys"][1])
        with self.assertRaises(ValueError):
            normalize_page({**payload(), "rec_scores": []}, 1, (20, 30))
        with self.assertRaises(ValueError):
            normalize_page({**payload(("bad",)), "rec_polys": [[[1, 2], [3, 4]]]}, 1, (20, 30))

    def test_path_pil_and_array_have_the_same_pixel_frame(self):
        backend = pipeline_for()
        extractor = OCRExtractor(OCRConfig(min_confidence=0.4), pipeline=backend)
        rgb = Image.new("RGB", (20, 30), (255, 0, 0))
        bgr = np.asarray(rgb)[:, :, ::-1].copy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cmr.png"
            rgb.save(path)
            documents = [extractor.extract(path), extractor.extract(rgb, source_id="pil"), extractor.extract(bgr, source_id="array")]
            for document in documents:
                page = {key: value for key, value in document["pages"][0].items() if key != "inference_seconds"}
                reference = {key: value for key, value in documents[0]["pages"][0].items() if key != "inference_seconds"}
                self.assertEqual(page, reference)
                self.assertGreaterEqual(document["pages"][0]["inference_seconds"], 0)
                self.assertGreaterEqual(document["inference_seconds"], document["pages"][0]["inference_seconds"])
                self.assertFalse(document["vram"]["available"])
                self.assertTrue(document["has_text"])
                json.dumps(document, allow_nan=False)
            self.assertIsNone(documents[1]["source_path"])
            for call in backend.predict.call_args_list:
                np.testing.assert_array_equal(call.kwargs["input"], bgr)
                self.assertEqual(call.kwargs["text_rec_score_thresh"], 0.4)
            self.assertIsNot(backend.predict.call_args.kwargs["input"], bgr)
            json_path, text_path = save_ocr(documents[0], Path(directory) / "output", "cmr")
            self.assertEqual(json.loads(json_path.read_text())["format_version"], 2)
            self.assertEqual(text_path.read_text(encoding="utf-8"), "Müller\nBerlin\n")

    def test_blank_and_missing_results_are_distinct(self):
        image = np.zeros((30, 20, 3), dtype=np.uint8)
        self.assertFalse(OCRExtractor(pipeline=pipeline_for(())).extract(image)["has_text"])
        backend = MagicMock()
        backend.predict.return_value = []
        with self.assertRaises(RuntimeError):
            OCRExtractor(pipeline=backend).extract(image)

    def test_grayscale_and_invalid_array_contract(self):
        extractor = OCRExtractor(pipeline=pipeline_for())
        self.assertEqual(extractor.extract(np.zeros((10, 20), dtype=np.uint8))["pages"][0]["width"], 20)
        for image in (np.zeros((10, 20), dtype=float), np.zeros((10, 20, 4), dtype=np.uint8), np.zeros((0, 20), dtype=np.uint8)):
            with self.assertRaises(ValueError):
                extractor.extract(image)

    def test_pdf_is_rendered_at_explicit_dpi_and_preserves_pages(self):
        import fitz
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cmr.pdf"
            with fitz.open() as pdf:
                pdf.new_page(width=144, height=72)
                pdf.new_page(width=144, height=72)
                pdf.save(path)
            document = OCRExtractor(OCRConfig(pdf_dpi=240), pipeline=pipeline_for()).extract(path)
            self.assertEqual(document["text"], "[Page 1]\nMüller\nBerlin\n\n[Page 2]\nMüller\nBerlin")
            self.assertEqual((document["pages"][0]["width"], document["pages"][0]["height"]), (480, 240))
            self.assertTrue(all(page["inference_seconds"] >= 0 for page in document["pages"]))
            self.assertNotEqual(document["pages"][0]["lines"][0]["id"], document["pages"][1]["lines"][0]["id"])

    def test_v6_initialization_is_lazy_explicit_and_reused(self):
        factory = MagicMock(return_value=pipeline_for())
        with patch("src.PP_parser.ocr.version", return_value="3.7.0"), patch.dict("sys.modules", {"paddleocr": SimpleNamespace(PaddleOCR=factory)}):
            extractor = OCRExtractor()
            factory.assert_not_called()
            extractor.extract(np.zeros((20, 30, 3), dtype=np.uint8))
            extractor.extract(np.zeros((20, 30, 3), dtype=np.uint8))
            factory.assert_called_once()
            self.assertEqual(factory.call_args.kwargs["text_recognition_model_name"], "PP-OCRv6_medium_rec")
            self.assertFalse(factory.call_args.kwargs["use_doc_unwarping"])
        with patch("src.PP_parser.ocr.version", return_value="3.4.0"):
            with self.assertRaisesRegex(RuntimeError, "requires paddleocr>=3.7"):
                OCRExtractor().extract(np.zeros((20, 30, 3), dtype=np.uint8))

    def test_cli_validates_configuration_without_loading_ocr(self):
        self.assertEqual(parse_ocr_args(["--input-paths", "cmr.png"] ).model_size, "medium")
        with contextlib.redirect_stderr(io.StringIO()):
            for arguments in (["--input-paths", "cmr.png", "--min-confidence", "nan"], ["--dataset-root", "data", "--pdf-dpi", "0"]):
                with self.assertRaises(SystemExit):
                    parse_ocr_args(arguments)

    def test_text_generation_never_loads_image_or_passes_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cmr.ocr.txt"
            path.write_text("Sender: Müller", encoding="utf-8")
            args = parse_args(["--ocr-text-paths", str(path)])
            self.assertEqual(resolve_model_id(args), "Qwen/Qwen3.5-9B")
            processor = MagicMock()
            ids = MagicMock()
            ids.shape = (1, 12)
            ids.to.return_value = ids
            processor.tokenizer.return_value = {"input_ids": ids}
            processor.batch_decode.return_value = ['{"sender":"Müller"}']
            torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
            runtime = InferenceRuntime(torch, MagicMock(), processor, MagicMock(), "model", "model")
            template = {"sender": None, "missing": None}
            result = generate_image_prediction(runtime, path, template, {"type": "object"}, args)
            self.assertEqual(result.prediction, {"sender": "Müller", "missing": None})
            self.assertGreaterEqual(result.inference_seconds, 0)
            self.assertFalse(result.vram["available"])
            runtime.image_module.open.assert_not_called()
            processor.assert_not_called()
            self.assertNotIn("pixel_values", runtime.model.generate.call_args.kwargs)
            messages = processor.apply_chat_template.call_args.args[0]
            self.assertTrue(all(part["type"] == "text" for message in messages for part in message["content"]))
            self.assertIn('"missing": null', messages[0]["content"][0]["text"])
            self.assertIn("Müller", messages[1]["content"][0]["text"])
            path.write_text("\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                generate_image_prediction(runtime, path, template, {}, args)


class OCRDatasetTests(unittest.TestCase):
    def make_dataset(self, root):
        for split in ("train", "val"):
            rows = []
            for index in range(3):
                # Identical stems in distinct folders must not collide.
                folder = root / split / str(index)
                folder.mkdir(parents=True)
                Image.new("RGB", (20, 30), (index, 0, 0)).save(folder / "page.png")
                (folder / "target.json").write_text(json.dumps({"content": {"sender": f"target-{split}-{index}"}, "private": "not a target"}))
                rows.append({"id": f"{split}-{index}", "image": f"{split}/{index}/page.png", "annotation": f"{split}/{index}/target.json"})
            (root / split / "metadata.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    def test_preparation_preserves_splits_targets_and_reports_failed_ocr(self):
        def recognize(**kwargs):
            marker = kwargs["input"][0, 0, 2]
            if marker == 2:
                raise RuntimeError("inference failed")
            return [payload(("Müller",) if marker == 0 else ())]
        backend = MagicMock()
        backend.predict.side_effect = recognize
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root, output = Path(directory) / "source", Path(directory) / "output"
            self.make_dataset(root)
            original = (root / "train/metadata.jsonl").read_bytes()
            summary = prepare_ocr_dataset(OCRExtractor(pipeline=backend), root, output, template={"sender": None}, schema={"type": "object"})
            for split in ("train", "val"):
                self.assertEqual([summary["splits"][split][key] for key in ("selected", "ok", "empty_ocr", "ocr_error")], [3, 1, 1, 1])
                records = [json.loads(line) for line in (output / f"{split}.jsonl").read_text().splitlines()]
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["target"], {"sender": f"target-{split}-0"})
                self.assertEqual(json.loads(records[0]["messages"][-1]["content"]), records[0]["target"])
                self.assertNotIn("target-", records[0]["messages"][1]["content"])
                self.assertNotIn("image", records[0])
                self.assertEqual(records[0]["messages"][1]["role"], "user")
                metadata = [json.loads(line) for line in (output / split / "metadata.jsonl").read_text().splitlines()]
                for row in metadata[:2]:
                    self.assertTrue((output / row["ocr_json"]).is_file())
                self.assertNotEqual(metadata[0]["ocr_json"], metadata[1]["ocr_json"])
                self.assertEqual(metadata[2]["status"], "ocr_error")
            self.assertEqual((root / "train/metadata.jsonl").read_bytes(), original)
            with self.assertRaises(FileExistsError):
                prepare_ocr_dataset(OCRExtractor(pipeline=backend), root, output)

    def test_dry_run_and_preflight_do_not_initialize_models_or_write(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root, output = Path(directory) / "source", Path(directory) / "output"
            self.make_dataset(root)
            backend = pipeline_for()
            summary = prepare_ocr_dataset(OCRExtractor(pipeline=backend), root, output, dry_run=True, limit=1)
            self.assertEqual(summary["splits"]["train"]["selected"], 1)
            backend.predict.assert_not_called()
            self.assertFalse(output.exists())
            for splits in (("train", "train"), ("../source",), ("missing",)):
                with self.assertRaises((ValueError, FileNotFoundError)):
                    prepare_ocr_dataset(OCRExtractor(pipeline=backend), root, output, splits=splits)
            with self.assertRaises(ValueError):
                prepare_ocr_dataset(OCRExtractor(pipeline=backend), root, root / "ocr")
            self.assertFalse(output.exists())

    def test_cli_dataset_smoke_and_nonzero_empty_status(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root, output = Path(directory) / "source", Path(directory) / "output"
            self.make_dataset(root)
            extractor = OCRExtractor(pipeline=pipeline_for())
            with patch("src.PP_parser.run_ocr.OCRExtractor", return_value=extractor):
                code = run_ocr(["--dataset-root", str(root), "--output-dir", str(output), "--limit", "1"])
            self.assertEqual(code, 0)
            self.assertEqual(len((output / "val.jsonl").read_text().splitlines()), 1)
            with patch("src.PP_parser.run_ocr.OCRExtractor", return_value=OCRExtractor(pipeline=pipeline_for(()))):
                code = run_ocr(["--input-paths", str(root / "train/0/page.png"), "--output-dir", str(Path(directory) / "empty")])
            self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
