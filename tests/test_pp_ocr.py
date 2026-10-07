from __future__ import annotations

import contextlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.PP_parser.run_ocr import extract_document, normalize_page
from src.Qwen.run_inference import (
    InferenceRuntime, generate_image_prediction, parse_args, resolve_model_id,
)


class PPOCRTests(unittest.TestCase):
    def test_pages_preserve_unicode_order_and_aligned_polygons(self):
        payload = {"rec_texts": ["Müller", "Berlin"], "rec_scores": [0.9, 0.8],
                   "rec_polys": [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]}
        pipeline = MagicMock()
        pipeline.predict.return_value = iter([SimpleNamespace(json={"res": payload}), payload])
        document = extract_document(pipeline, Path("cmr.pdf"), 0.4)
        self.assertEqual(document["text"], "[Page 1]\nMüller\nBerlin\n\n[Page 2]\nMüller\nBerlin")
        self.assertEqual(document["pages"][0]["lines"][1]["polygon"], [[5, 6], [7, 8]])
        pipeline.predict.assert_called_once_with(input="cmr.pdf", text_rec_score_thresh=0.4)
        with self.assertRaises(ValueError):
            normalize_page({**payload, "rec_scores": []}, 1)

    def test_empty_ocr_and_no_pages_are_distinct(self):
        pipeline = MagicMock()
        pipeline.predict.return_value = [{"rec_texts": [], "rec_scores": [], "rec_polys": []}]
        self.assertFalse(extract_document(pipeline, Path("empty.png"), 0)["has_text"])
        pipeline.predict.return_value = []
        with self.assertRaises(RuntimeError):
            extract_document(pipeline, Path("empty.png"), 0)

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


if __name__ == "__main__":
    unittest.main()
