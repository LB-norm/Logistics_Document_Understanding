from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from src.PP_parser import OCRConfig, OCRExtractor
from src.Qwen import run_inference as qwen
from src.Donut import run_donut_inference as donut
from src.inference_pipeline import InferencePipeline, PipelineConfig, run_inference
from src.inference_pipeline.__main__ import main, parse_args
from src.utils.pipeline_tracking import PipelineTracker, attach_framework_peaks


def qwen_runtime():
    ids = MagicMock()
    ids.shape = (1, 12)
    ids.to.return_value = ids
    processor = MagicMock()
    processor.return_value = {"input_ids": ids}
    processor.tokenizer.return_value = {"input_ids": ids}
    processor.batch_decode.return_value = ['{"sender":"Müller"}']
    model = MagicMock()
    model.device = "cpu"
    model.hf_device_map = {}
    return qwen.InferenceRuntime(SimpleNamespace(inference_mode=contextlib.nullcontext), Image,
                                 processor, model, "Qwen/test", "Qwen/test")


def donut_runtime():
    processor = MagicMock()
    processor.tokenizer.eos_token = "<eos>"
    processor.tokenizer.pad_token = "<pad>"
    processor.batch_decode.return_value = ["<s_lieferschein>Müller<eos>"]
    processor.token2json.return_value = {"sender": "Müller"}
    model = MagicMock()
    model.decoder.config.max_position_embeddings = 768
    return donut.InferenceRuntime(SimpleNamespace(inference_mode=contextlib.nullcontext), Image, processor, model)


def allocator_vram(backend="torch", allocated=1, reserved=2):
    return {"available": True, "source": backend, "devices": [
        {"device": "cuda:0" if backend == "torch" else "gpu:0",
         "peak_allocated_bytes": allocated * 1024**3, "peak_reserved_bytes": reserved * 1024**3,
         "peak_allocated_gib": allocated, "peak_reserved_gib": reserved}]}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.images = [self.root / "one.png", self.root / "two.png"]
        for path in self.images:
            Image.new("RGB", (100, 100)).save(path)
        self.schema, self.template = self.root / "schema.json", self.root / "template.json"
        self.schema.write_text('{"type":"object","properties":{"sender":{"type":["string","null"]}},"required":["sender"]}')
        self.template.write_text('{"sender":null}')
        self.native = ("--schema-path", str(self.schema), "--template-path", str(self.template))
        self.weights = self.root / "yolo"
        self.weights.mkdir()
        (self.weights / "model.pt").write_bytes(b"stub")
        (self.weights / "layout_config.json").write_text("{}")
        self.sampler = patch("src.utils.pipeline_tracking.shutil.which", return_value=None)
        self.sampler.start()
        self.addCleanup(self.sampler.stop)

    def config(self, **kwargs):
        options = {"mode": "end-to-end", "model_id": "Qwen/test", "model_args": self.native}
        options.update(kwargs)
        return PipelineConfig(**options)

    def parser_components(self, texts=("Müller", "Footer")):
        paddle = MagicMock()
        paddle.predict.return_value = [{"rec_texts": list(texts), "rec_scores": [0.9] * len(texts),
                                        "rec_polys": [[[10, 10], [30, 10], [30, 20], [10, 20]],
                                                      [[80, 80], [90, 80], [90, 90], [80, 90]]][:len(texts)]}]
        extractor = OCRExtractor(pipeline=paddle)
        layout = MagicMock()
        layout.predict.side_effect = lambda path, **kwargs: {
            "image": str(path), "width": 100, "height": 100,
            "detections": [{"label": "sender", "category_id": 19, "label_id": 0, "score": 0.8,
                            "bbox_xyxy": [0, 0, 50, 50], "bbox_xywh": [0, 0, 50, 50]}],
            "inference_seconds": 0.01, "vram": {"available": False, "source": "torch", "devices": []},
        }
        return extractor, layout

    def test_qwen_end_to_end_calls_native_image_generation_and_reuses_runtime(self):
        runtime = qwen_runtime()
        with patch.object(qwen, "load_inference_runtime", return_value=runtime) as load:
            records = run_inference(self.images, self.config())
        self.assertEqual(load.call_count, 1)
        self.assertEqual(runtime.model.generate.call_count, 2)
        self.assertEqual(runtime.processor.call_count, 2)
        self.assertEqual(runtime.processor.tokenizer.call_count, 0)
        for record in records:
            self.assertEqual(record["prediction"], {"sender": "Müller"})
            self.assertEqual(record["status"], "ok")
            self.assertGreaterEqual(record["inference_seconds"], record["stages"]["generation"]["inference_seconds"])
            self.assertFalse(record["vram"]["available"])
            self.assertNotIn("fusion", record)
            self.assertEqual(record["generation"]["model"]["resolution"], qwen.parse_args([]).resolution)

    def test_native_defaults_and_user_overrides_are_inherited(self):
        pipeline = InferencePipeline(self.config())
        native = qwen.parse_args([])
        for key in ("resolution", "compute_dtype", "max_new_tokens", "device_map", "attn_implementation"):
            self.assertEqual(getattr(pipeline.args, key), getattr(native, key))
        custom = InferencePipeline(self.config(model_args=self.native + ("--resolution", "high", "--max-new-tokens", "4096")))
        self.assertEqual(custom.args.resolution, "high")
        self.assertEqual(custom.args.max_new_tokens, 4096)
        donut_args = ("--schema-path", str(self.schema), "--example-path", str(self.template), "--annotation-target-key", "root")
        pipeline = InferencePipeline(self.config(model_type="donut", model_id="local-donut", model_args=donut_args))
        native = donut.parse_args([])
        for key in ("device", "max_length", "num_beams", "task_prompt"):
            self.assertEqual(getattr(pipeline.args, key), getattr(native, key))

    def test_all_parser_modes_feed_fusion_to_text_generation_on_the_same_page(self):
        for mode in (1, 2, 3):
            with self.subTest(mode=mode):
                runtime = qwen_runtime()
                extractor, detector = self.parser_components()
                config = self.config(mode="parser", fusion_mode=mode, yolo_model_dir=self.weights)
                with patch.object(qwen, "load_inference_runtime", return_value=runtime) as load, \
                        patch("src.inference_pipeline.pipeline.OCRExtractor", return_value=extractor) as ocr_factory, \
                        patch("src.yolo_v11s.inference.LayoutDetector", return_value=detector) as detector_factory:
                    records = run_inference(self.images, config)
                self.assertEqual(load.call_count, 1)
                self.assertEqual(ocr_factory.call_count, 1)
                self.assertEqual(detector_factory.call_count, 1)
                self.assertEqual(detector.predict.call_args_list[0].args[0], self.images[0])
                self.assertEqual(runtime.processor.call_count, 0)
                self.assertEqual(runtime.processor.tokenizer.call_count, 2)
                self.assertNotIn("pixel_values", runtime.model.generate.call_args.kwargs)
                messages = runtime.processor.apply_chat_template.call_args.args[0]
                self.assertIn("not calibrated probabilities", messages[0]["content"][0]["text"])
                self.assertIn("Footer", messages[1]["content"][0]["text"])
                self.assertTrue(all(part["type"] == "text" for message in messages for part in message["content"]))
                for record in records:
                    self.assertEqual(record["status"], "ok")
                    self.assertEqual(record["fusion_mode"], mode)
                    self.assertEqual(set(record["stages"]), {"ocr", "yolo", "fusion", "generation"})
                    self.assertEqual(record["generation"]["input_mode"], "fusion")
                    if mode != 1:
                        self.assertEqual(record["fusion"]["pages"][0]["unassigned_ocr_ids"], ["p0001-l000001"])

    def test_donut_native_generation_returns_json_and_reuses_checkpoint(self):
        runtime = donut_runtime()
        args = ("--schema-path", str(self.schema), "--example-path", str(self.template),
                "--annotation-target-key", "root", "--task-prompt", "<s_lieferschein>")
        with patch.object(donut, "load_inference_runtime", return_value=runtime) as load:
            records = run_inference(self.images, self.config(model_type="donut", model_id="local-donut", model_args=args))
        self.assertEqual(load.call_count, 1)
        self.assertEqual(runtime.model.generate.call_count, 2)
        self.assertEqual(records[0]["prediction"], {"sender": "Müller"})
        self.assertEqual(records[0]["status"], "ok")
        self.assertFalse(records[0]["vram"]["available"])

    def test_adapter_directory_selection_uses_native_metadata_resolution(self):
        adapter = self.root / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text('{"base_model_name_or_path":"Qwen/base"}')
        (adapter / "processor_config.json").write_text("{}")
        for choices in ({"model_id": str(adapter)}, {"model_id": None, "adapter_path": adapter}):
            pipeline = InferencePipeline(self.config(**choices))
            self.assertEqual(qwen.resolve_model_id(pipeline.args), "Qwen/base")
            self.assertEqual(qwen.resolve_processor_source(pipeline.args, "Qwen/base"), str(adapter))
            self.assertEqual(pipeline.args.adapter_path, adapter)

    def test_parser_generation_failure_retains_total_and_completed_stage_metrics(self):
        runtime = qwen_runtime()
        runtime.model.generate.side_effect = [RuntimeError("generation failed"), MagicMock()]
        extractor, detector = self.parser_components()
        with patch.object(qwen, "load_inference_runtime", return_value=runtime), \
                patch("src.inference_pipeline.pipeline.OCRExtractor", return_value=extractor), \
                patch("src.yolo_v11s.inference.LayoutDetector", return_value=detector):
            records = run_inference(self.images, self.config(mode="parser", fusion_mode=3, yolo_model_dir=self.weights))
        self.assertEqual(records[0]["status"], "inference_error")
        self.assertIn("generation failed", records[0]["error"])
        self.assertGreaterEqual(records[0]["inference_seconds"], 0)
        self.assertGreaterEqual(records[0]["stages"]["generation"]["inference_seconds"], 0)
        self.assertIn("paddle", records[0]["vram"]["framework_peaks"])
        self.assertIn("fusion", records[0])
        self.assertEqual(records[1]["status"], "ok")

    def test_empty_ocr_does_not_generate_from_layout_metadata(self):
        extractor, detector = self.parser_components(texts=())
        runtime = qwen_runtime()
        with patch.object(qwen, "load_inference_runtime", return_value=runtime), \
                patch("src.inference_pipeline.pipeline.OCRExtractor", return_value=extractor), \
                patch("src.yolo_v11s.inference.LayoutDetector", return_value=detector):
            record = run_inference(self.images[:1], self.config(mode="parser", fusion_mode=1, yolo_model_dir=self.weights))[0]
        self.assertEqual(record["status"], "inference_error")
        self.assertIn("OCR text is empty", record["error"])
        detector.predict.assert_not_called()
        runtime.model.generate.assert_not_called()

    def test_parse_and_schema_errors_are_not_reported_as_success(self):
        runtime = qwen_runtime()
        runtime.processor.batch_decode.side_effect = [["not JSON"], ['{"sender":42}']]
        with patch.object(qwen, "load_inference_runtime", return_value=runtime):
            records = run_inference(self.images, self.config())
        self.assertEqual([record["status"] for record in records], ["parse_error", "schema_error"])
        self.assertEqual(records[0]["prediction"], {"sender": None})
        self.assertGreaterEqual(records[0]["inference_seconds"], 0)

    def test_cli_forwards_native_settings_and_writes_per_image_predictions_and_metrics(self):
        output = self.root / "output"
        args = ["--mode", "end-to-end", "--model-id", "Qwen/test", "--images", *map(str, self.images),
                "--output-dir", str(output), *self.native, "--resolution", "high"]
        runtime = qwen_runtime()
        with patch.object(qwen, "load_inference_runtime", return_value=runtime) as load, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 0)
        self.assertEqual(load.call_args.args[0].resolution, "high")
        self.assertEqual(json.loads((output / "one.json").read_text()), {"sender": "Müller"})
        records = [json.loads(line) for line in (output / "inference_manifest.jsonl").read_text().splitlines()]
        self.assertEqual(len(records), 2)
        self.assertIn("inference_seconds", records[0])
        self.assertIn("vram", records[0])
        self.assertNotIn("prediction", records[0])

    def test_cli_can_disable_ocr_mkldnn_without_forwarding_flag_to_qwen(self):
        args = ["--mode", "parser", "--model-id", "Qwen/test", "--images", str(self.images[0]),
                "--fusion-mode", "soft", "--yolo-model-dir", str(self.weights), *self.native]
        _, default = parse_args(args)
        _, disabled = parse_args(args + ["--ocr-disable-mkldnn"])
        self.assertIsNone(default.ocr_config.enable_mkldnn)
        self.assertIs(disabled.ocr_config.enable_mkldnn, False)
        InferencePipeline(disabled)

    def test_cli_prints_inference_exception_and_retains_it_in_manifest(self):
        output = self.root / "failed"
        args = ["--mode", "end-to-end", "--model-id", "Qwen/test", "--images", str(self.images[0]),
                "--output-dir", str(output), *self.native]
        runtime = qwen_runtime()
        runtime.model.generate.side_effect = RuntimeError("underlying backend failure")
        errors = io.StringIO()
        with patch.object(qwen, "load_inference_runtime", return_value=runtime), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(errors):
            self.assertEqual(main(args), 1)
        self.assertIn("RuntimeError: underlying backend failure", errors.getvalue())
        record = json.loads((output / "inference_manifest.jsonl").read_text())
        self.assertEqual(record["error"], "RuntimeError: underlying backend failure")
        self.assertIsNone(record["prediction_path"])

    def test_cli_parser_writes_fusion_artifact_and_returns_nonzero_for_failure(self):
        output = self.root / "parser"
        args = ["--mode", "parser", "--model-id", "Qwen/test", "--images", str(self.images[0]),
                "--fusion-mode", "soft", "--yolo-model-dir", str(self.weights), "--output-dir", str(output), *self.native]
        extractor, detector = self.parser_components()
        runtime = qwen_runtime()
        runtime.processor.batch_decode.return_value = ["not JSON"]
        with patch.object(qwen, "load_inference_runtime", return_value=runtime), \
                patch("src.inference_pipeline.pipeline.OCRExtractor", return_value=extractor), \
                patch("src.yolo_v11s.inference.LayoutDetector", return_value=detector), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 1)
        self.assertEqual(json.loads((output / "one.fusion.json").read_text())["mode"], "soft")
        record = json.loads((output / "inference_manifest.jsonl").read_text())
        self.assertEqual(record["status"], "parse_error")

    def test_dry_run_preflight_avoids_loading_models_or_creating_output(self):
        output = self.root / "dry"
        args = ["--mode", "parser", "--model-id", "Qwen/test", "--images", str(self.images[0]),
                "--fusion-mode", "1", "--yolo-model-dir", str(self.weights), "--output-dir", str(output), *self.native, "--dry-run"]
        with patch.object(qwen, "load_inference_runtime") as load, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 0)
        load.assert_not_called()
        self.assertFalse(output.exists())
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(args + ["--yolo-model-dir", str(self.root / "missing")]), 1)

    def test_invalid_configuration_and_reserved_native_input_settings(self):
        for kwargs in ({"mode": "bogus"}, {"model_type": "bogus"}, {"model_id": None},
                       {"mode": "parser"}, {"fusion_mode": 1},
                       {"mode": "parser", "fusion_mode": 1, "yolo_model_dir": self.weights, "model_type": "donut"},
                       {"yolo_threshold": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.config(**kwargs)
        with self.assertRaises(ValueError):
            InferencePipeline(self.config(model_args=self.native + ("--image-path", str(self.images[0]))))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--mode", "parser", "--model-id", "Qwen/test", "--images", "x.png"])

    def test_missing_inputs_duplicate_stems_and_input_overwrite_fail_before_loading(self):
        with patch.object(qwen, "load_inference_runtime") as load:
            with self.assertRaises(FileNotFoundError):
                run_inference([self.root / "missing.png"], self.config())
            with self.assertRaises(ValueError):
                run_inference([], self.config())
            args = ["--mode", "end-to-end", "--model-id", "Qwen/test", *self.native]
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(args + ["--images", str(self.images[0]), str(self.images[0])]), 1)
                self.assertEqual(main(args + ["--images", str(self.template), "--output-dir", str(self.root)]), 1)
        load.assert_not_called()


class PipelineTrackingTests(unittest.TestCase):
    def test_gpu_sampler_filters_pid_preserves_devices_and_resets_for_each_image(self):
        def probe(*args, **kwargs):
            probe.calls += 1
            memory = 512 if probe.calls == 1 else 1024 if probe.calls == 2 else 256
            return SimpleNamespace(stdout=f"42, GPU-a, {memory}\n42, GPU-b, 128\n999, GPU-a, 99999\n")
        probe.calls = 0
        with patch("src.utils.pipeline_tracking.shutil.which", return_value="nvidia-smi"), \
                patch("src.utils.pipeline_tracking.os.getpid", return_value=42), \
                patch("src.utils.pipeline_tracking.subprocess.run", side_effect=probe):
            with PipelineTracker(interval=100) as first:
                first._sample()
            with PipelineTracker(interval=100) as second:
                pass
        self.assertEqual(first.metrics["vram"]["devices"][0]["peak_used_bytes"], 1024 * 1024**2)
        self.assertEqual(second.metrics["vram"]["devices"][0]["peak_used_bytes"], 256 * 1024**2)
        self.assertEqual(len(first.metrics["vram"]["devices"]), 2)
        self.assertTrue(first.metrics["vram"]["combined_peak_available"])

    def test_sampler_failure_does_not_replace_inference_error(self):
        with patch("src.utils.pipeline_tracking.shutil.which", return_value="nvidia-smi"), \
                patch("src.utils.pipeline_tracking.subprocess.run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 1)):
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                with PipelineTracker() as tracker:
                    raise RuntimeError("inference failed")
        self.assertFalse(tracker.metrics["vram"]["available"])
        self.assertIn("TimeoutExpired", tracker.metrics["vram"]["sampling_error"])
        self.assertGreaterEqual(tracker.metrics["inference_seconds"], 0)

    def test_separate_framework_peaks_are_retained_without_inventing_combined_peak(self):
        metrics = {"vram": {"available": False, "combined_peak_available": False, "devices": []}}
        stages = {"ocr": {"backend": "paddle", "vram": allocator_vram("paddle", 3, 4)},
                  "yolo": {"backend": "torch", "vram": allocator_vram("torch", 1, 5)},
                  "generation": {"backend": "torch", "vram": allocator_vram("torch", 6, 7)}}
        attach_framework_peaks(metrics, stages, parser=True)
        self.assertFalse(metrics["vram"]["available"])
        self.assertEqual(metrics["vram"]["devices"], [])
        self.assertEqual(metrics["vram"]["framework_peaks"]["torch"]["devices"][0]["peak_allocated_gib"], 6)
        self.assertEqual(metrics["vram"]["framework_peaks"]["paddle"]["devices"][0]["peak_allocated_gib"], 3)
        attach_framework_peaks(metrics, stages, parser=False)
        self.assertTrue(metrics["vram"]["available"])
        self.assertEqual(metrics["vram"]["devices"][0]["peak_reserved_gib"], 7)


if __name__ == "__main__":
    unittest.main()
