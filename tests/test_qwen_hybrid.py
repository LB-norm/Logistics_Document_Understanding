"""Hybrid pairing, dropout, multimodal supervision and inference regression checks."""
from copy import deepcopy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image
import torch
from datasets import Dataset

from src.Qwen import qwen_finetune_logic as training
from src.Qwen import run_inference as qwen
from src.Qwen.experiment_config import load_experiment_config, load_experiment_queue
from src.Qwen.prompts import (
    DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT, HYBRID_EVIDENCE_MARKER,
    HYBRID_REPRESENTATION, build_hybrid_messages, omit_parser_evidence,
)
from src.fusion.compact import compact_document
from src.fusion.compact_sft import compact_dataset
from src.fusion.hybrid_sft import prepare_hybrid_dataset
from src.inference_pipeline import run_inference
import test_fusion_compact as fusion_fixtures
import test_inference_pipeline as pipeline_fixtures
from test_qwen_text_training import TinyTokenizer, TinyModel


class ImageProcessor:
    def __init__(self):
        self.tokenizer = TinyTokenizer()
        self.tokenizer.model_max_length = 100_000
        self.calls = []
        self.conversations = []

    def apply_chat_template(self, messages, **kwargs):
        self.conversations.append(deepcopy(messages))
        text = "".join(message["role"] + ":" + "".join(
            "<image>" if block["type"] == "image" else block["text"]
            for block in message["content"]) + "<eos>" for message in messages)
        return text + ("assistant:" if kwargs["add_generation_prompt"] else "")

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        batch = self.tokenizer(kwargs["text"], return_tensors="pt")
        batch["pixel_values"] = torch.ones(len(kwargs["images"]), 8)
        return batch

    @staticmethod
    def save_pretrained(path):
        Path(path).mkdir(parents=True, exist_ok=True)


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image = self.root / "page.png"
        Image.new("RGB", (100, 100)).save(self.image)
        self.target = {"sender": "Müller"}
        self.schema = {"type": "object", "properties": {"sender": {"type": ["string", "null"]}}, "required": ["sender"]}
        self.template = {"sender": None}

    def raw(self, identifier="one"):
        return {"id": identifier, "image": str(self.image), "representation": HYBRID_REPRESENTATION,
                "fusion_mode": "hard", "target": self.target,
                "messages": build_hybrid_messages(DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT,
                    fusion_fixtures.document("hard"), self.template, self.schema) + [
                        {"role": "assistant", "content": json.dumps(self.target)}]}

    def example(self):
        return training.build_example_record(self.raw(), self.root, "hybrid")

    def prepare_source(self):
        source = self.root / "full"
        (source / "hard").mkdir(parents=True)
        (source / "schema.json").write_text("{}")
        (source / "template.json").write_text("{}")
        summary = {"splits": {}}
        for split in ("train", "val"):
            image = self.root / f"{split}.png"
            Image.new("RGB", (100, 100)).save(image)
            row = fusion_fixtures.record("hard", split)
            row["provenance"]["source_image"] = f"../{split}.png"
            (source / "hard" / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
            summary["splits"][split] = {"examples": 1, "modes": {"hard": {}}}
        (source / "preparation_summary.json").write_text(json.dumps(summary))
        compact = self.root / "compact"
        compact_dataset(source, compact)
        return compact

    def test_builder_pairs_original_images_preserves_answers_and_matches_inference_prompt(self):
        compact = self.prepare_source()
        output = self.root / "hybrid"
        prepare_hybrid_dataset(compact, output, dry_run=True)
        self.assertFalse(output.exists())
        summary = prepare_hybrid_dataset(compact, output)
        self.assertEqual(summary["splits"]["train"]["examples"], 1)
        original = json.loads((compact / "hard/train.jsonl").read_text())
        row = json.loads((output / "hard/train.jsonl").read_text())
        self.assertEqual(row["messages"][-1], original["messages"][-1])
        self.assertEqual(row["target"], original["target"])
        self.assertEqual((output / row["image"]).resolve(), self.root / "train.png")
        self.assertNotIn("cannot see the image", row["messages"][0]["content"][0]["text"])
        self.assertEqual(row["messages"][:-1], build_hybrid_messages(
            DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT, fusion_fixtures.document("hard"), {}, {}))
        args = training.parse_args(["--dataset-root", str(output), "--input-mode", "hybrid",
                                    "--train-file", "hard/train.jsonl", "--validation-file", "hard/val.jsonl",
                                    "--schema-path", str(output / "schema.json")])
        train, val, *_ = training.load_dataset_splits(args)
        contract = training.hybrid_dataset_contract(args, train + val)
        self.assertEqual(contract["representation"], HYBRID_REPRESENTATION)
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare_hybrid_dataset(compact, output)

    def test_invalid_pairing_target_or_duplicate_image_never_publishes_output(self):
        compact = self.prepare_source()
        path = compact / "hard/val.jsonl"
        original = json.loads(path.read_text())
        mutations = [
            {"provenance": {**original["provenance"], "source_image": "../missing.png"}},
            {"provenance": {**original["provenance"], "source_image": "../train.png"}},
            {"target": {"changed": True}},
            {"id": "sample-train"},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                path.write_text(json.dumps({**original, **mutation}) + "\n")
                prepare_hybrid_dataset(compact, self.root / "invalid")
            self.assertFalse((self.root / "invalid").exists())

    def test_dropout_is_training_only_reproducible_and_preserves_pixels_and_supervision(self):
        processor = ImageProcessor()
        collator = training.QwenVisionDataCollator(processor, Image, None, set(), context_limit=100_000)
        validation = self.example()
        before = deepcopy(validation)
        train = {**validation, "parser_dropout": 1.0}
        training_batch = collator([train])
        dropped = processor.conversations[-1]
        self.assertFalse(any(HYBRID_EVIDENCE_MARKER in b.get("text", "") for m in dropped for b in m["content"]))
        self.assertTrue(any(b["type"] == "image" for m in dropped for b in m["content"]))
        self.assertIn("JSON Schema:", dropped[0]["content"][0]["text"])
        evaluation_batch = collator([validation])
        self.assertIn(HYBRID_EVIDENCE_MARKER, processor.conversations[-1][1]["content"][-1]["text"])
        self.assertTrue(torch.equal(training_batch["pixel_values"], evaluation_batch["pixel_values"]))
        self.assertEqual(training_batch["labels"][training_batch["labels"] != -100].tolist(),
                         evaluation_batch["labels"][evaluation_batch["labels"] != -100].tolist())
        self.assertEqual(validation, before)
        random.seed(42)
        first = [random.random() < .2 for _ in range(50)]
        random.seed(42)
        for expected in first:
            collator([{**validation, "parser_dropout": .2}])
            actual = len(processor.conversations[-1][1]["content"]) == 2
            self.assertEqual(actual, expected)

    def test_collator_masks_left_and_right_padding_and_backpropagates_multimodal_loss(self):
        processor = ImageProcessor()
        examples = [self.example(), self.example()]
        examples[1]["messages"][1]["content"][1]["text"] += " more instructions"
        for side in ("left", "right"):
            processor.tokenizer.padding_side = side
            batch = training.QwenVisionDataCollator(processor, Image, None, set(), context_limit=100_000)(examples)
            for index in range(2):
                labels = batch["labels"][index][batch["attention_mask"][index].bool()]
                supervised = labels[labels != -100]
                self.assertGreater(supervised.numel(), 0)
                self.assertEqual(supervised[-1].item(), processor.tokenizer.eos_token_id)
                self.assertTrue((batch["labels"][index][~batch["attention_mask"][index].bool()] == -100).all())
            embedding = torch.nn.Embedding(512, 8)
            visual = torch.nn.Linear(8, 8)
            head = torch.nn.Linear(8, 512)
            logits = head(embedding(batch["input_ids"]) + visual(batch["pixel_values"])[:, None, :])
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 512), batch["labels"][:, 1:].reshape(-1))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertIsNotNone(visual.weight.grad)
            self.assertGreater(visual.weight.grad.abs().sum().item(), 0)

    def test_context_preflight_counts_visual_input_and_rejects_truncation_and_bad_prefix(self):
        processor = ImageProcessor()
        example = self.example()
        summary = training.check_hybrid_sequence_lengths(processor, {}, [example], [example], None)
        count = summary["splits"]["train"]["max_tokens"]
        self.assertGreater(count, 0)
        self.assertEqual(example["multimodal_context_limit"], 100_000)
        with self.assertRaisesRegex(ValueError, "No truncation"):
            training.check_hybrid_sequence_lengths(processor, {}, [example], [example], count - 1)
        original = processor.apply_chat_template
        processor.apply_chat_template = lambda messages, **kw: original(messages, **kw) + ("bad" if kw["add_generation_prompt"] else "")
        with self.assertRaisesRegex(ValueError, "exact token prefix"):
            training.check_hybrid_sequence_lengths(processor, {}, [example], [example], None)

    def test_four_configs_form_controlled_factorial_and_disallow_invalid_options(self):
        queue = load_experiment_queue(Path("experiments/qwen/hybrid/queue.json"))
        configs = [load_experiment_config(entry.config_path).training for entry in queue.entries]
        self.assertEqual({(c["vision_tuning"], c["parser_dropout"]) for c in configs},
                         {("frozen", 0), ("lora", 0), ("frozen", .2), ("lora", .2)})
        for config, entry in zip(configs, queue.entries):
            training.validate_training_options(training.parse_args(["--config", str(entry.config_path)]))
            self.assertEqual(config["vision_tuning"], config["vision_merger_tuning"])
            self.assertEqual(config["loss_implementation"], "standard")
            self.assertFalse(config["load_in_4bit"])
        for flags in (["--parser-dropout", ".2"], ["--input-mode", "hybrid", "--parser-dropout", "nan"]):
            with self.assertRaises(ValueError):
                training.validate_training_options(training.parse_args(flags))

    def test_pipeline_hybrid_reuses_models_feeds_pixels_and_hard_evidence_and_handles_empty_ocr(self):
        fixture = pipeline_fixtures.PipelineTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        for texts in (("Müller", "Footer"), ()):
            runtime = pipeline_fixtures.qwen_runtime()
            runtime.model.config = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=100_000))
            extractor, detector = fixture.parser_components(texts=texts)
            with patch.object(qwen, "load_inference_runtime", return_value=runtime) as load, \
                    patch("src.inference_pipeline.pipeline.OCRExtractor", return_value=extractor), \
                    patch("src.yolo_v11s.inference.LayoutDetector", return_value=detector):
                records = run_inference(fixture.images, fixture.config(mode="hybrid", yolo_model_dir=fixture.weights))
            load.assert_called_once()
            self.assertEqual([r["status"] for r in records], ["ok", "ok"])
            self.assertEqual(runtime.processor.call_count, 2)
            runtime.processor.tokenizer.assert_not_called()
            messages = runtime.processor.apply_chat_template.call_args.args[0]
            self.assertEqual(messages[1]["content"][0]["type"], "image")
            self.assertIn(HYBRID_EVIDENCE_MARKER, messages[1]["content"][-1]["text"])
            self.assertEqual(records[0]["generation"]["input_mode"], "hybrid")
            self.assertEqual(records[0]["generation"]["model"]["resolution"], "medium")

    def test_saved_contract_and_native_saved_fusion_inference_use_same_prompt(self):
        adapter = self.root / "adapter"
        adapter.mkdir()
        contract = {"format_version": 1, "representation": HYBRID_REPRESENTATION, "fusion_mode": "hard",
                    "decimals": 4, "system_prompt": "Saved system", "user_prompt": "Saved task",
                    "schema": self.schema, "template": self.template}
        (adapter / "hybrid_contract.json").write_text(json.dumps(contract))
        fusion = self.root / "fusion.json"
        fusion.write_text(json.dumps(fusion_fixtures.document("hard")))
        args = qwen.parse_args(["--hybrid", "--adapter-path", str(adapter), "--image-path", str(self.image),
                                "--fusion-json-paths", str(fusion)])
        runtime = pipeline_fixtures.qwen_runtime()
        runtime.model.config = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=100_000))
        records = qwen.run_inference_on_images(args=args, runtime=runtime, image_paths=[self.image],
                    prediction_paths=[self.root / "prediction.json"], template=self.template, target_schema=self.schema)
        self.assertEqual(records[0]["status"], "ok")
        expected = build_hybrid_messages("Saved system", "Saved task", fusion_fixtures.document("hard"), self.template, self.schema)
        self.assertEqual(runtime.processor.apply_chat_template.call_args.args[0], expected)
        args.omit_parser_evidence = True
        qwen.generate_image_prediction(runtime, self.image, self.template, self.schema, args)
        self.assertEqual(len(runtime.processor.apply_chat_template.call_args.args[0][1]["content"]), 2)
        with patch.object(qwen, "load_inference_runtime", return_value=runtime), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(qwen.main(["--hybrid", "--adapter-path", str(adapter),
                "--image-path", str(self.image), "--fusion-json-paths", str(fusion),
                "--schema-path", str(self.root / "unavailable-schema.json"),
                "--template-path", str(self.root / "unavailable-template.json"),
                "--output-path", str(self.root / "portable.json")]), 0)
        self.assertEqual(json.loads((self.root / "portable.json").read_text()), self.target)

    def test_hybrid_cli_rejects_missing_or_mispaired_inputs_and_nonhard_modes(self):
        for flags in (["--fusion-json-paths", "one.json"],
                      ["--hybrid", "--ocr-text-paths", "text.txt"],
                      ["--omit-parser-evidence"],
                      ["--hybrid", "--image-paths", "one.png", "two.png", "--fusion-json-paths", "one.json"]):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                qwen.parse_args(flags)
        with self.assertRaisesRegex(ValueError, "hard fusion"):
            pipeline_fixtures.PipelineConfig(mode="hybrid", model_id="Qwen/test",
                                             fusion_mode="soft", yolo_model_dir=self.root)

    def test_training_entrypoint_with_arrow_records_never_drops_validation_and_saves_contract(self):
        compact = self.prepare_source()
        dataset, output = self.root / "hybrid", self.root / "run"
        prepare_hybrid_dataset(compact, dataset)
        processor = ImageProcessor()
        class Model(TinyModel):
            def forward(self, input_ids, attention_mask, labels, pixel_values):
                hidden = self.language_model(input_ids) + self.visual.merger(pixel_values)[:, None, :]
                logits = self.lm_head(hidden)
                loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 512), labels[:, 1:].reshape(-1))
                return SimpleNamespace(loss=loss, logits=logits)
        model = Model()
        instances = []
        class ModelLoader:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return model
        class Trainer:
            def __init__(self, **kwargs):
                self.model = kwargs["model"]
                self.collator = kwargs["data_collator"]
                self.train_dataset, self.eval_dataset = kwargs["train_dataset"], kwargs["eval_dataset"]
                self.state = SimpleNamespace(global_step=0)
                instances.append(self)
            def train(self, **kwargs):
                self.loss = self.model(**self.collator(list(self.train_dataset))).loss
                self.loss.backward()
                self.training_messages = deepcopy(processor.conversations[-1])
                self.state.global_step = 1
                return SimpleNamespace(metrics={"train_loss": self.loss.item()})
            def evaluate(self):
                with torch.no_grad():
                    result = {"eval_loss": self.model(**self.collator(list(self.eval_dataset))).loss.item()}
                self.evaluation_messages = deepcopy(processor.conversations[-1])
                return result
            def save_state(self):
                pass
            def save_model(self, path):
                pass
        dependencies = (torch, Image, Dataset, MagicMock(), object, Trainer, object,
                        SimpleNamespace, (None, None, None), SimpleNamespace(AutoModelForMultimodalLM=ModelLoader))
        artifacts = {"best_model_dir": str(output / "best_model"), "last_model_dir": str(output / "last_model")}
        with patch.object(training, "load_text_tokenization_resources", return_value=(processor, {})), \
                patch.object(training, "load_runtime_dependencies", return_value=dependencies) as runtime, \
                patch.object(training, "save_best_and_last_model_artifacts", return_value=artifacts), \
                contextlib.redirect_stdout(io.StringIO()):
            status = training.main(["--input-mode", "hybrid", "--dataset-root", str(dataset),
                "--train-file", "hard/train.jsonl", "--validation-file", "hard/val.jsonl",
                "--schema-path", str(dataset / "schema.json"), "--output-dir", str(output),
                "--parser-dropout", "1", "--text-tuning", "full", "--no-load-in-4bit",
                "--no-gradient-checkpointing", "--validation-preview-samples", "0"])
        self.assertEqual(status, 0)
        self.assertTrue(runtime.call_args.kwargs["require_images"])
        trainer = instances[0]
        self.assertTrue(torch.isfinite(trainer.loss))
        self.assertEqual(len(trainer.training_messages[1]["content"]), 2)
        self.assertEqual(len(trainer.evaluation_messages[1]["content"]), 3)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.visual.parameters()))
        self.assertIsNotNone(model.language_model.weight.grad)
        saved = json.loads((output / "training_config.json").read_text())
        self.assertEqual(saved["input_mode"], "hybrid")
        self.assertEqual(saved["resolution_max_pixels"], 2_800_000)
        for directory in (output, output / "best_model", output / "last_model"):
            self.assertTrue((directory / "hybrid_contract.json").is_file())
        self.assertTrue((output / "multimodal_sequence_lengths.json").is_file())

    def test_inference_launcher_selects_completed_runs_and_keeps_ablation_outputs_separate(self):
        spec = importlib.util.spec_from_file_location("hybrid_testset", Path("experiments/qwen/hybrid/run_testset.py"))
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        runs = self.root / "runs"
        for name in ("qwen35-9b-hybrid-h1-bf16-r16", "qwen35-9b-hybrid-h2-bf16-r16"):
            run = runs / name
            (run / "best_model").mkdir(parents=True)
            (run / "best_model/hybrid_contract.json").write_text("{}")
            (run / "training_config.json").write_text(json.dumps({"input_mode": "hybrid", "load_in_4bit": False,
                                                                  "resolution": "medium", "compute_dtype": "bfloat16"}))
        (self.root / "test/images").mkdir(parents=True)
        Image.new("RGB", (10, 10)).save(self.root / "test/images/one.png")
        (self.root / "test/metadata.jsonl").write_text("")
        args = SimpleNamespace(runs_dir=runs, run=None, dataset_root=self.root, split="test", output_dir=self.root / "out",
                               omit_parser_evidence=False, yolo_model_dir=self.root / "yolo", ocr_models_dir=self.root / "ocr",
                               ocr_device="gpu:0", max_new_tokens=2048, dry_run=True)
        commands = launcher.build_commands(args, [])
        self.assertEqual(len(commands), 2)
        self.assertIn("--dry-run", commands[0][1])
        args.run = [commands[0][0]]
        args.omit_parser_evidence = True
        command = launcher.build_commands(args, ["--yolo-device", "cpu"])[0][1]
        self.assertIn("--omit-parser-evidence", command)
        self.assertIn("without-parser", command[command.index("--output-dir") + 1])
        self.assertEqual(command[-1], str(self.root / "test/images/one.png"))
        args.run = ["missing"]
        with self.assertRaisesRegex(ValueError, "completed"):
            launcher.build_commands(args, [])


if __name__ == "__main__":
    unittest.main()
