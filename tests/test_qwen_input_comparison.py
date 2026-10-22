"""Verify evidence separation, baseline matching, and stage-one inference paths."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.Qwen import qwen_finetune_logic as training
from src.Qwen import run_inference as qwen
from src.Qwen.experiment_config import load_experiment_queue
from src.Qwen.prompts import HYBRID_EVIDENCE_MARKER, build_hybrid_messages
from src.fusion.hybrid_sft import prepare_hybrid_dataset
from src.inference_pipeline import run_inference
import test_qwen_hybrid as hybrid_fixtures
import test_inference_pipeline as pipeline_fixtures
import test_fusion_compact as fusion_fixtures


class InputComparisonTests(unittest.TestCase):
    def setUp(self):
        self.fixture = hybrid_fixtures.HybridTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root

    def prepare(self):
        source = self.fixture.prepare_source()
        output = self.root / "inputs"
        prepare_hybrid_dataset(source, output, evidence_modes=("ocr", "layout", "hard"))
        return output

    def args(self, root, mode):
        return training.parse_args(["--input-mode", "hybrid", "--dataset-root", str(root),
            "--train-file", f"{mode}/train.jsonl", "--validation-file", f"{mode}/val.jsonl",
            "--schema-path", str(root / "schema.json")])

    def test_aligned_datasets_contain_only_selected_evidence_and_preserve_identical_targets(self):
        root = self.prepare()
        rows = {mode: json.loads((root / mode / "train.jsonl").read_text()) for mode in ("ocr", "layout", "hard")}
        reference = rows["hard"]
        for mode, row in rows.items():
            self.assertEqual(row["id"], reference["id"])
            self.assertEqual(row["target"], reference["target"])
            self.assertEqual(row["image"], reference["image"])
            self.assertEqual(row["messages"][-1], reference["messages"][-1])
            self.assertEqual(row["messages"][0], reference["messages"][0])
            self.assertEqual(row["messages"][1]["content"][:2], reference["messages"][1]["content"][:2])
            evidence = json.loads(row["messages"][1]["content"][2]["text"].removeprefix(HYBRID_EVIDENCE_MARKER))
            self.assertEqual(evidence["mode"], mode)
            expected_keys = {"page", "lines"} if mode == "ocr" else {"page", "regions"} if mode == "layout" else {"page", "lines", "regions", "memberships"}
            self.assertEqual(set(evidence["pages"][0]), expected_keys)
            if mode == "layout":
                self.assertNotIn("Müller", json.dumps(row["messages"][:-1], ensure_ascii=False))
            train, val, *_ = training.load_dataset_splits(self.args(root, mode))
            contract = training.hybrid_dataset_contract(self.args(root, mode), train + val)
            self.assertEqual(contract["evidence_mode"], mode)
            expected = build_hybrid_messages(contract["system_prompt"], contract["user_prompt"],
                fusion_fixtures.document("hard"), contract["template"], contract["schema"], evidence_mode=mode)
            self.assertEqual(train[0]["messages"][:-1], expected)

    def test_mixing_variants_or_reintroducing_withheld_evidence_is_rejected(self):
        root = self.prepare()
        args = self.args(root, "ocr")
        args.validation_file = "layout/val.jsonl"
        with self.assertRaisesRegex(ValueError, "same hybrid evidence"):
            training.load_dataset_splits(args)
        args.validation_file = "ocr/val.jsonl"
        train, val, *_ = training.load_dataset_splits(args)
        text = train[0]["messages"][1]["content"][2]["text"]
        evidence = json.loads(text.removeprefix(HYBRID_EVIDENCE_MARKER))
        evidence["pages"][0]["regions"] = [[0, "secret", [0, 0, 1, 1], .99]]
        train[0]["messages"][1]["content"][2]["text"] = HYBRID_EVIDENCE_MARKER + json.dumps(evidence)
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            training.hybrid_dataset_contract(args, train + val)

    def test_queue_reruns_only_ocr_and_hard_with_chunked_loss_and_original_recipe(self):
        queue = load_experiment_queue(Path("experiments/qwen/input_comparison/queue.json"))
        self.assertEqual(len(queue.entries), 2)
        baseline_args = training.parse_args(["--config", "experiments/qwen/qwen35_9b_qlora_r16.json"])
        seen = set()
        for entry in queue.entries:
            args = training.parse_args(["--config", str(entry.config_path)])
            training.validate_training_options(args)
            seen.add(Path(args.train_file).parts[0])
            for key in ("model_id", "resolution", "load_in_4bit", "compute_dtype", "bnb_4bit_quant_type",
                        "no_bnb_double_quant", "text_tuning", "vision_tuning", "vision_merger_tuning", "target_modules",
                        "lora_r", "lora_alpha", "lora_dropout", "learning_rate", "weight_decay", "warmup_ratio",
                        "lr_scheduler_type", "gradient_accumulation_steps", "num_train_epochs", "seed", "max_length"):
                self.assertEqual(getattr(args, key), getattr(baseline_args, key), key)
            self.assertEqual(training.resolve_optimizer(args, True), training.resolve_optimizer(baseline_args, True))
            self.assertEqual(args.parser_dropout, 0)
            self.assertEqual(args.loss_implementation, "chunked")
            self.assertEqual(args.loss_chunk_size, 256)
            self.assertTrue(args.run_name.endswith("-chunked"))
            self.assertIsNone(args.resume_from_checkpoint)
            self.assertEqual(args.vision_tuning, "frozen")
        self.assertEqual(seen, {"ocr", "hard"})
        completed = training.parse_args(["--config", "experiments/qwen/input_comparison/layout_image_nf4_qlora_r16.json"])
        self.assertEqual(completed.loss_implementation, "standard")
        self.assertEqual(completed.run_name, "qwen35-9b-hybrid-input-layout-nf4-r16")

    def test_pipeline_skips_unused_models_and_sends_only_selected_evidence(self):
        fixture = pipeline_fixtures.PipelineTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        for mode in ("ocr", "layout", "hard"):
            runtime = pipeline_fixtures.qwen_runtime()
            runtime.model.config = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=100_000))
            extractor, detector = fixture.parser_components()
            config = fixture.config(mode="hybrid", yolo_model_dir=None if mode == "ocr" else fixture.weights,
                model_args=fixture.native + ("--hybrid-evidence", mode))
            with patch.object(qwen, "load_inference_runtime", return_value=runtime), \
                    patch("src.inference_pipeline.pipeline.OCRExtractor", return_value=extractor) as ocr, \
                    patch("src.yolo_v11s.inference.LayoutDetector", return_value=detector) as yolo:
                record = run_inference(fixture.images[:1], config)[0]
            self.assertEqual(record["status"], "ok", record.get("error"))
            self.assertEqual(ocr.call_count, 0 if mode == "layout" else 1)
            self.assertEqual(yolo.call_count, 0 if mode == "ocr" else 1)
            self.assertEqual("ocr" in record["stages"], mode != "layout")
            self.assertEqual("yolo" in record["stages"], mode != "ocr")
            messages = runtime.processor.apply_chat_template.call_args.args[0]
            evidence = json.loads(messages[1]["content"][2]["text"].removeprefix(HYBRID_EVIDENCE_MARKER))
            self.assertEqual(evidence["mode"], mode)
            self.assertEqual(record["generation"]["evidence_mode"], mode)
            if mode == "layout":
                self.assertNotIn("Müller", json.dumps(messages, ensure_ascii=False))
            runtime.processor.tokenizer.assert_not_called()
            runtime.processor.assert_called_once()

    def test_saved_contract_selects_ablation_and_rejects_incompatible_override(self):
        root = self.prepare()
        args = self.args(root, "ocr")
        train, val, *_ = training.load_dataset_splits(args)
        contract = training.hybrid_dataset_contract(args, train + val)
        adapter = self.root / "adapter"
        adapter.mkdir()
        (adapter / "hybrid_contract.json").write_text(json.dumps(contract))
        native = qwen.parse_args(["--hybrid", "--adapter-path", str(adapter)])
        self.assertEqual(qwen.load_hybrid_contract(native)["evidence_mode"], "ocr")
        native.hybrid_evidence = "layout"
        with self.assertRaisesRegex(ValueError, "training contract"):
            qwen.load_hybrid_contract(native)

    def test_inference_launcher_preserves_nf4_loading_and_selects_only_needed_parser_stages(self):
        spec = importlib.util.spec_from_file_location("input_launcher", Path("experiments/qwen/hybrid/run_testset.py"))
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        runs = self.root / "runs"
        for mode in ("ocr", "layout", "hard"):
            suffix = "" if mode == "layout" else "-chunked"
            run = runs / f"qwen35-9b-hybrid-input-{mode}-nf4-r16{suffix}"
            (run / "best_model").mkdir(parents=True)
            (run / "best_model/hybrid_contract.json").write_text(json.dumps({"evidence_mode": mode}))
            (run / "training_config.json").write_text(json.dumps({"input_mode": "hybrid", "load_in_4bit": True,
                "resolution": "medium", "compute_dtype": "bfloat16", "bnb_4bit_quant_type": "nf4"}))
        (self.root / "test/images").mkdir(parents=True)
        from PIL import Image
        Image.new("RGB", (10, 10)).save(self.root / "test/images/one.png")
        args = SimpleNamespace(runs_dir=runs, run=None, dataset_root=self.root, split="test", output_dir=self.root / "out",
            omit_parser_evidence=False, yolo_model_dir=self.root / "yolo", ocr_models_dir=self.root / "ocr",
            ocr_device="gpu:0", max_new_tokens=2048, dry_run=True)
        commands = launcher.build_commands(args, [])
        self.assertEqual(len(commands), 3)
        for _, command in commands:
            mode = command[command.index("--hybrid-evidence") + 1]
            self.assertIn("--load-in-4bit", command)
            self.assertNotIn("--no-load-in-4bit", command)
            self.assertEqual("--ocr-device" in command, mode != "layout")
            self.assertEqual("--yolo-model-dir" in command, mode != "ocr")


if __name__ == "__main__":
    unittest.main()
