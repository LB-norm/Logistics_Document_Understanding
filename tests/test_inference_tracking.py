"""Offline checks for per-image peaks, GPU synchronization and saved metrics."""

import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

from src.utils.inference_tracking import InferenceTracker, aggregate_vram


class FakeCuda:
    def __init__(self):
        self.events = []
        self.peaks = {0: [1024**3, 2 * 1024**3], 1: [1024**3, 2 * 1024**3]}

    def is_available(self):
        return True

    def device_count(self):
        return 2

    def synchronize(self, device):
        self.events.append(("sync", device))

    def reset_peak_memory_stats(self, device):
        self.events.append(("reset", device))
        self.peaks[device] = [1024**3, 2 * 1024**3]

    def reset_max_memory_allocated(self, device):
        self.events.append(("reset_allocated", device))
        self.peaks[device][0] = 1024**3

    def reset_max_memory_reserved(self, device):
        self.events.append(("reset_reserved", device))
        self.peaks[device][1] = 2 * 1024**3

    def max_memory_allocated(self, device):
        return self.peaks[device][0]

    def max_memory_reserved(self, device):
        return self.peaks[device][1]


class InferenceTrackingTests(unittest.TestCase):
    def test_each_image_resets_peaks_and_waits_for_gpu_work(self):
        cuda = FakeCuda()
        framework = SimpleNamespace(cuda=cuda)
        with patch("src.utils.inference_tracking.perf_counter", side_effect=[10., 12., 20., 21.]):
            with InferenceTracker(framework) as first:
                cuda.peaks[0] = [5 * 1024**3, 6 * 1024**3]
            with InferenceTracker(framework) as second:
                cuda.peaks[0] = [3 * 1024**3, 4 * 1024**3]
        self.assertEqual(first.metrics["inference_seconds"], 2)
        self.assertEqual(second.metrics["inference_seconds"], 1)
        self.assertEqual(first.metrics["vram"]["devices"][0]["peak_allocated_gib"], 5)
        self.assertEqual(second.metrics["vram"]["devices"][0]["peak_allocated_gib"], 3)
        self.assertEqual(second.metrics["vram"]["devices"][1]["peak_reserved_gib"], 2)
        self.assertEqual(cuda.events, [("sync", 0), ("reset", 0), ("sync", 1), ("reset", 1),
                                       ("sync", 0), ("sync", 1)] * 2)
        json.dumps(second.metrics, allow_nan=False)

    def test_cpu_records_time_without_touching_cuda(self):
        cuda = FakeCuda()
        with InferenceTracker(SimpleNamespace(cuda=cuda), device="cpu") as tracker:
            pass
        self.assertGreaterEqual(tracker.metrics["inference_seconds"], 0)
        self.assertFalse(tracker.metrics["vram"]["available"])
        self.assertEqual(tracker.metrics["vram"]["devices"], [])
        self.assertEqual(cuda.events, [])

    def test_paddle_tracks_selected_gpu_with_separate_peak_resets(self):
        cuda = FakeCuda()
        paddle = SimpleNamespace(is_compiled_with_cuda=lambda: True, device=SimpleNamespace(cuda=cuda))
        with InferenceTracker(paddle, backend="paddle", device="gpu:1") as tracker:
            cuda.peaks[1] = [3 * 1024**3, 4 * 1024**3]
        self.assertEqual(cuda.events, [("sync", 1), ("reset_allocated", 1),
                                       ("reset_reserved", 1), ("sync", 1)])
        self.assertEqual(tracker.metrics["vram"]["devices"][0]["device"], "gpu:1")
        self.assertEqual(tracker.metrics["vram"]["devices"][0]["peak_reserved_gib"], 4)

    def test_document_peak_is_maximum_of_pages(self):
        cuda = FakeCuda()
        summaries = []
        for allocated, reserved in ((5, 6), (3, 7)):
            with InferenceTracker(SimpleNamespace(cuda=cuda), device="cuda:0") as tracker:
                cuda.peaks[0] = [allocated * 1024**3, reserved * 1024**3]
            summaries.append(tracker.metrics["vram"])
        combined = aggregate_vram(summaries)
        self.assertEqual(combined["devices"][0]["peak_allocated_gib"], 5)
        self.assertEqual(combined["devices"][0]["peak_reserved_gib"], 7)

    def test_yolo_numeric_device_list(self):
        cuda = FakeCuda()
        with InferenceTracker(SimpleNamespace(cuda=cuda), device="0,1") as tracker:
            pass
        self.assertEqual([item["device"] for item in tracker.metrics["vram"]["devices"]],
                         ["cuda:0", "cuda:1"])

    def test_tracking_preserves_inference_error(self):
        cuda = FakeCuda()
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            with InferenceTracker(SimpleNamespace(cuda=cuda)) as tracker:
                raise RuntimeError("inference failed")
        self.assertGreaterEqual(tracker.metrics["inference_seconds"], 0)
        self.assertTrue(tracker.metrics["vram"]["available"])


class SavedInferenceMetricsTests(unittest.TestCase):
    def test_ocr_pdf_saves_page_peaks_and_document_maximum(self):
        import fitz
        from src.PP_parser.ocr import OCRConfig, OCRExtractor, save_ocr

        cuda = FakeCuda()
        paddle = SimpleNamespace(is_compiled_with_cuda=lambda: True, device=SimpleNamespace(cuda=cuda))
        calls = iter([(5, 6), (3, 7)])

        def predict(**kwargs):
            allocated, reserved = next(calls)
            cuda.peaks[0] = [allocated * 1024**3, reserved * 1024**3]
            return [{"rec_texts": [], "rec_scores": [], "rec_polys": []}]

        pipeline = SimpleNamespace(predict=predict)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "pages.pdf"
            with fitz.open() as pdf:
                pdf.new_page(width=20, height=30)
                pdf.new_page(width=20, height=30)
                pdf.save(source)
            with patch.dict("sys.modules", {"paddle": paddle}):
                document = OCRExtractor(OCRConfig(device="gpu:0"), pipeline=pipeline).extract(source)
            json_path, _ = save_ocr(document, root / "output", "pages")
            saved = json.loads(json_path.read_text())
            self.assertEqual([page["vram"]["devices"][0]["peak_allocated_gib"] for page in saved["pages"]], [5, 3])
            self.assertEqual(saved["vram"]["devices"][0]["peak_allocated_gib"], 5)
            self.assertEqual(saved["vram"]["devices"][0]["peak_reserved_gib"], 7)
            self.assertGreaterEqual(saved["inference_seconds"], sum(page["inference_seconds"] for page in saved["pages"]))

    def test_qwen_saves_metrics_per_image_and_keeps_prediction_schema(self):
        from src.Qwen.run_inference import InferenceRuntime, main

        cuda = FakeCuda()
        torch = SimpleNamespace(cuda=cuda, inference_mode=contextlib.nullcontext)
        processor = MagicMock()
        ids = MagicMock()
        ids.shape = (1, 12)
        ids.to.return_value = ids
        processor.return_value = {"input_ids": ids}
        processor.batch_decode.return_value = ['{"value":"extracted"}']
        model = MagicMock()
        model.hf_device_map = {"": 0}
        model.generate.side_effect = lambda **kwargs: MagicMock()
        runtime = InferenceRuntime(torch, Image, processor, model, "model", "processor")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = [root / "one.png", root / "two.png"]
            for path in images:
                Image.new("RGB", (20, 30)).save(path)
            schema = root / "schema.json"
            template = root / "template.json"
            schema.write_text('{"type":"object","properties":{"value":{}}}')
            template.write_text('{"value":null}')
            output = root / "output"
            base_args = ["--schema-path", str(schema), "--template-path", str(template)]
            with patch("src.Qwen.run_inference.load_inference_runtime", return_value=runtime):
                self.assertEqual(main(base_args + ["--image-paths", *map(str, images), "--output-dir", str(output)]), 0)
                single = root / "single.json"
                self.assertEqual(main(base_args + ["--image-path", str(images[0]), "--output-path", str(single)]), 0)
            records = [json.loads(line) for line in (output / "inference_manifest.jsonl").read_text().splitlines()]
            records.append(json.loads(single.with_suffix(".json.diagnostics.json").read_text()))
            self.assertEqual(len(records), 3)
            for record in records:
                self.assertGreaterEqual(record["inference_seconds"], 0)
                self.assertTrue(record["vram"]["available"])
                self.assertEqual(len(record["vram"]["devices"]), 2)
            self.assertEqual(json.loads(single.read_text()), {"value": "extracted"})

    def test_paddleocr_vl_saves_metrics_in_summary(self):
        from src.PP_parser import PPStructureV3_parser as parser

        cuda = FakeCuda()
        paddle = SimpleNamespace(is_compiled_with_cuda=lambda: True, device=SimpleNamespace(cuda=cuda))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            Image.new("RGB", (20, 30)).save(image)
            with patch("sys.argv", ["parser", "--image-path", str(image), "--output-dir", str(root / "output"),
                                    "--no-visualization", "--no-json", "--no-markdown", "--device", "gpu:0"]):
                args = parser.parse_args()
            pipeline = MagicMock()
            pipeline.predict.return_value = iter([{"text": "hello"}])
            with patch.object(parser, "load_paddleocr_vl", return_value=MagicMock(return_value=pipeline)), \
                    patch.dict("sys.modules", {"paddle": paddle}):
                parser.run_inference(args)
            summary = json.loads((root / "output/inference_summary.json").read_text())
            self.assertGreaterEqual(summary["inference_seconds"], 0)
            self.assertTrue(summary["vram"]["available"])
            self.assertEqual(summary["vram"]["devices"][0]["device"], "gpu:0")

    def test_donut_saves_time_and_unavailable_vram_on_cpu(self):
        from src.Donut import run_donut_inference as donut

        processor = MagicMock()
        processor.tokenizer.eos_token = "<eos>"
        processor.tokenizer.pad_token = "<pad>"
        processor.batch_decode.return_value = ["<s_cord-v2>hello<eos>"]
        processor.token2json.return_value = {"value": "hello"}
        model = MagicMock()
        model.decoder.config.max_position_embeddings = 768
        torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image, schema, template, output = [root / name for name in ("image.png", "schema.json", "template.json", "output.json")]
            Image.new("RGB", (20, 30)).save(image)
            schema.write_text('{"type":"object"}')
            template.write_text('{"content":{"value":null}}')
            with patch("sys.argv", ["donut", "--image-path", str(image), "--schema-path", str(schema),
                                    "--example-path", str(template), "--output-path", str(output)]):
                args = donut.parse_args()
            dependencies = (torch, Image, MagicMock(from_pretrained=MagicMock(return_value=processor)),
                            MagicMock(from_pretrained=MagicMock(return_value=model)))
            with patch.object(donut, "parse_args", return_value=args), \
                    patch.object(donut, "load_runtime_dependencies", return_value=dependencies):
                self.assertEqual(donut.main(), 0)
            result = json.loads(output.read_text())
            self.assertGreaterEqual(result["inference_seconds"], 0)
            self.assertFalse(result["vram"]["available"])
            self.assertEqual(result["generated"]["guided_prediction"], {"value": "hello"})


if __name__ == "__main__":
    unittest.main()
