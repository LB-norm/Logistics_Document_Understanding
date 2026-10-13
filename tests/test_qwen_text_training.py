from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

from src.Qwen import qwen_finetune_logic as training
from src.Qwen.run_qwen_training import DEFAULT_TRAINING_CONFIG


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    model_max_length = 10 ** 30
    padding_side = "right"

    def __init__(self):
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append(kwargs)
        if isinstance(text, list):
            records = [self(item, **{k: v for k, v in kwargs.items() if k != "return_tensors"}) for item in text]
            return self.pad(records, padding=True, return_tensors="pt")
        ids = []
        for index, part in enumerate(text.split("<eos>")):
            if index:
                ids.append(self.eos_token_id)
            ids.extend(ord(char) + 10 for char in part)
        if kwargs.get("add_special_tokens", True):
            ids.insert(0, 1)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def pad(self, records, **kwargs):
        maximum = max(len(row["input_ids"]) for row in records)
        ids, masks = [], []
        for row in records:
            padding = maximum - len(row["input_ids"])
            if self.padding_side == "left":
                ids.append([0] * padding + row["input_ids"])
                masks.append([0] * padding + row["attention_mask"])
            else:
                ids.append(row["input_ids"] + [0] * padding)
                masks.append(row["attention_mask"] + [0] * padding)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}

    @staticmethod
    def batch_decode(rows, **kwargs):
        return ['{"sender":"Muller"}' for row in rows]


class TinyProcessor:
    def __init__(self):
        self.tokenizer = TinyTokenizer()
        self.templated_messages = []

    def apply_chat_template(self, messages, **kwargs):
        self.templated_messages.append(messages)
        text = "".join(m["role"] + ":" + "".join(b["text"] for b in m["content"]) + "<eos>" for m in messages)
        return text + ("assistant:" if kwargs["add_generation_prompt"] else "")

    def __call__(self, **kwargs):
        raise AssertionError("Text-only training/generation must not call the image processor")

    @staticmethod
    def save_pretrained(path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "processor.json").write_text("{}")


def raw_example(identifier="sample", mode="soft", text="OCR/layout evidence"):
    return {"id": identifier, "fusion_mode": mode, "source_split": "train",
            "target": {"sender": "Muller"},
            "provenance": {"source_image": "/unavailable/machine/image.png"},
            "messages": [{"role": "system", "content": "Return JSON."},
                         {"role": "user", "content": text},
                         {"role": "assistant", "content": '{"sender":"Muller"}'}]}


def write_dataset(root, mode="soft"):
    root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        rows = [raw_example(split, mode)]
        (root / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = torch.nn.Embedding(512, 8)
        self.lm_head = torch.nn.Linear(8, 512)
        self.visual = torch.nn.Module()
        self.visual.blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8)])
        self.visual.merger = torch.nn.Linear(8, 8)
        self.config = SimpleNamespace(use_cache=True)
        self.generation_kwargs = None

    def get_input_embeddings(self):
        return self.language_model

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask, labels):
        logits = self.lm_head(self.language_model(input_ids))
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 512), labels[:, 1:].reshape(-1))
        return SimpleNamespace(loss=loss, logits=logits)

    def generate(self, **kwargs):
        self.generation_kwargs = kwargs
        return torch.cat([kwargs["input_ids"], torch.tensor([[20, 21]])], dim=1)


class QwenTextTrainingTests(unittest.TestCase):
    def setUp(self):
        self.processor = TinyProcessor()
        self.config = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=4096))

    def example(self, **kwargs):
        return training.build_example_record(raw_example(**kwargs), Path("/missing/dataset"), "text")

    def prepare(self, **kwargs):
        return training.prepare_text_training_data(self.processor, self.config,
                                                   [self.example(identifier="train")],
                                                   [self.example(identifier="val")], **kwargs)

    def test_text_loader_preserves_prompt_without_resolving_provenance_images(self):
        example = self.example()
        self.assertEqual(example["image_paths"], [])
        self.assertEqual(example["messages"][1]["content"], [{"type": "text", "text": "OCR/layout evidence"}])
        self.assertEqual(example["fusion_mode"], "soft")
        self.assertEqual(training.extract_assistant_target(example), {"sender": "Muller"})

    def test_loader_rejects_image_inputs_placeholders_invalid_targets_and_absent_queries(self):
        for mutation in ({"image": "missing.png"},
                         {"target": {"sender": "changed"}},
                         {"messages": [{"role": "user", "content": [{"type": "image"}]},
                                       {"role": "assistant", "content": "{}"}]},
                         {"messages": [{"role": "assistant", "content": "{}"}]},
                         {"messages": [{"role": "user", "content": "input"},
                                       {"role": "assistant", "content": "invalid JSON"}]}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                training.build_example_record({**raw_example(), **mutation}, Path("."), "text")
        with self.assertRaisesRegex(ValueError, "Each record must include"):
            training.build_example_record(raw_example(), Path("."))

    def test_tokenization_and_masking_keep_assistant_and_end_tokens_for_both_padding_sides(self):
        examples = [self.example(text="short"), self.example(text="a longer OCR input")]
        encoded = [training.encode_text_example(self.processor, example) for example in examples]
        for side in ("left", "right"):
            self.processor.tokenizer.padding_side = side
            batch = training.QwenTextDataCollator(self.processor.tokenizer)(encoded)
            self.assertEqual(set(batch), {"input_ids", "attention_mask", "labels"})
            for index, record in enumerate(encoded):
                active = batch["attention_mask"][index].bool()
                labels = batch["labels"][index][active]
                self.assertTrue((labels[:record["prompt_length"]] == -100).all())
                self.assertEqual(labels[record["prompt_length"]:].tolist(), record["input_ids"][record["prompt_length"]:])
                self.assertEqual(labels[-1].item(), self.processor.tokenizer.eos_token_id)
                self.assertTrue((batch["labels"][index][~active] == -100).all())
        self.assertTrue(all(call["add_special_tokens"] is False and call["truncation"] is False
                            for call in self.processor.tokenizer.calls))
        self.assertNotIn(1, encoded[0]["input_ids"])

    def test_preflight_checks_full_answers_against_model_tokenizer_and_explicit_limits(self):
        train, val, summary = self.prepare(max_length=None)
        total = len(train[0]["input_ids"])
        self.assertEqual(summary["effective_max_length"], 4096)
        self.assertEqual(summary["splits"]["train"]["samples"][0]["conversation_tokens"], total)
        self.prepare(max_length=total)
        with self.assertRaisesRegex(ValueError, "No truncation"):
            self.prepare(max_length=total - 1)
        self.processor.tokenizer.model_max_length = total - 1
        with self.assertRaisesRegex(ValueError, "complete assistant answer"):
            self.prepare(max_length=None)
        self.processor.tokenizer.model_max_length = 10 ** 30
        self.config.text_config.max_position_embeddings = total - 1
        with self.assertRaisesRegex(ValueError, "exceeding"):
            self.prepare(max_length=4096)
        self.config.text_config.max_position_embeddings = None
        with self.assertRaisesRegex(ValueError, "Could not determine"):
            self.prepare(max_length=None)
        self.assertEqual(training.text_context_limits(self.processor, {}, total)["max_length"], total)

    def test_template_prefix_mismatch_is_rejected_instead_of_masking_answer_tokens(self):
        original = self.processor.apply_chat_template
        def mismatch(messages, **kwargs):
            return original(messages, **kwargs) + ("different" if kwargs["add_generation_prompt"] else "")
        self.processor.apply_chat_template = mismatch
        with self.assertRaisesRegex(ValueError, "exact token prefix"):
            training.encode_text_example(self.processor, self.example())

    def test_text_validation_uses_only_prompt_tokens_and_caps_generation_at_context_limit(self):
        example = self.example()
        encoded = training.encode_text_example(self.processor, example)
        example["text_context_limit"] = encoded["prompt_length"] + 7
        model = TinyModel()
        images = MagicMock()
        result = training.generate_validation_preview_sample(
            image_module=images, processor=self.processor, model=model, example=example,
            max_new_tokens=32, target_schema={"type": "object"},
        )
        images.open.assert_not_called()
        self.assertEqual(result["input_mode"], "text")
        self.assertEqual(result["image_paths"], [])
        self.assertEqual(result["prompt_tokens"], encoded["prompt_length"])
        self.assertEqual(model.generation_kwargs["max_new_tokens"], 7)
        self.assertNotIn("labels", model.generation_kwargs)
        self.assertNotIn("pixel_values", model.generation_kwargs)
        self.assertEqual(len(self.processor.templated_messages[-1]), 2)
        self.assertTrue(result["exact_match"])
        self.assertEqual(result["generated_tokens"], 2)
        report = training.render_validation_preview_html({"global_step": 1, "status": "completed", "samples": [result]})
        self.assertIn("Text-only input", report)

    def test_all_three_fusion_modes_load_with_automatic_val_filename_and_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("separate", "hard", "soft"):
                root = Path(directory) / mode
                write_dataset(root, mode)
                args = training.parse_args(["--input-mode", "text", "--dataset-root", str(root)])
                train, val, _, _, validation_path, layout = training.load_dataset_splits(args)
                self.assertEqual((len(train), len(val), layout), (1, 1, "text_jsonl"))
                self.assertEqual(Path(validation_path).name, "val.jsonl")
                with contextlib.redirect_stdout(io.StringIO()), patch.object(training, "load_runtime_dependencies") as runtime:
                    self.assertEqual(training.main(["--input-mode", "text", "--dataset-root", str(root), "--dry-run"]), 0)
                    runtime.assert_not_called()
                (root / "validation.jsonl").write_text((root / "val.jsonl").read_text())
                self.assertEqual(Path(training.load_dataset_splits(args)[4]).name, "validation.jsonl")

    def test_length_check_dry_run_uses_tokenization_resources_without_runtime_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            output = Path(directory) / "output"
            write_dataset(root)
            with patch.object(training, "load_text_tokenization_resources", return_value=(self.processor, self.config)), \
                    patch.object(training, "load_runtime_dependencies") as runtime, \
                    contextlib.redirect_stdout(io.StringIO()) as printed:
                result = training.main(["--input-mode", "text", "--dataset-root", str(root),
                                        "--dry-run", "--check-text-lengths", "--output-dir", str(output)])
            self.assertEqual(result, 0)
            runtime.assert_not_called()
            self.assertFalse(output.exists())
            self.assertIn("effective_text_max_length", printed.getvalue())

    def test_plain_ocr_jsonl_takes_precedence_over_ocr_provenance_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root)
            for split in ("train", "val"):
                (root / split).mkdir()
                (root / split / "metadata.jsonl").write_text('{"source_image":"/missing/old-machine.png","status":"ok"}\n')
                row = raw_example(split)
                row.pop("fusion_mode")
                (root / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
            args = training.parse_args(["--input-mode", "text", "--dataset-root", str(root)])
            train, val, _, _, _, layout = training.load_dataset_splits(args)
            self.assertEqual(layout, "text_jsonl")
            self.assertEqual((len(train), len(val)), (1, 1))
            self.assertEqual(train[0]["image_paths"], [])

    def test_options_and_json_experiment_can_select_text_mode(self):
        self.assertEqual(training.parse_args([], DEFAULT_TRAINING_CONFIG).input_mode, "image")
        for flags, pattern in ((["--check-text-lengths"], "requires"),
                               (["--input-mode", "text", "--vision-tuning", "lora"], "frozen vision"),
                               (["--input-mode", "text", "--vision-merger-tuning", "full"], "frozen vision")):
            with self.subTest(flags=flags), self.assertRaisesRegex(ValueError, pattern):
                training.validate_training_options(training.parse_args(flags))
        with self.assertRaisesRegex(ValueError, "text-tuning"):
            training.validate_training_options(training.parse_args([], {"input_mode": "text", "text_tuning": "frozen"}))
        args = training.parse_args(["--input-mode", "text"])
        self.assertNotIn("max_pixels", training.build_processor_load_kwargs(args))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.json"
            path.write_text(json.dumps({"format_version": 1, "name": "fusion-soft", "training": {"input_mode": "text"}}))
            args = training.parse_args(["--config", str(path)])
            training.validate_training_options(args)
            self.assertEqual(args.input_mode, "text")

    def test_fusion_mode_mismatch_and_duplicate_ids_across_splits_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root)
            args = training.parse_args(["--input-mode", "text", "--dataset-root", str(root)])
            (root / "val.jsonl").write_text(json.dumps(raw_example("val", "hard")) + "\n")
            with self.assertRaisesRegex(ValueError, "same fusion mode"):
                training.load_dataset_splits(args)
            (root / "val.jsonl").write_text(json.dumps(raw_example("train")) + "\n")
            with self.assertRaisesRegex(ValueError, "unique across"):
                training.load_dataset_splits(args)

    def test_main_trains_and_evaluates_text_batches_with_real_shifted_cross_entropy(self):
        processor, model = self.processor, TinyModel()
        instances = []
        class Dataset:
            @staticmethod
            def from_list(rows):
                return rows
        class ModelLoader:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return model
        class Trainer:
            def __init__(self, **kwargs):
                self.model, self.collator = kwargs["model"], kwargs["data_collator"]
                self.train_dataset, self.eval_dataset = kwargs["train_dataset"], kwargs["eval_dataset"]
                self.args = kwargs["args"]
                self.state = SimpleNamespace(global_step=0)
                instances.append(self)
            def train(self, **kwargs):
                self.batch = self.collator(self.train_dataset)
                self.loss = self.model(**self.batch).loss
                self.loss.backward()
                torch.optim.SGD([p for p in self.model.parameters() if p.requires_grad], lr=0.01).step()
                self.state.global_step = 1
                return SimpleNamespace(metrics={"train_loss": self.loss.item()})
            def evaluate(self):
                with torch.no_grad():
                    return {"eval_loss": self.model(**self.collator(self.eval_dataset)).loss.item()}
            def save_state(self):
                pass
            def save_model(self, path):
                pass
        dependencies = (torch, MagicMock(), Dataset, MagicMock(), object, Trainer, object,
                        SimpleNamespace, (None, None, None), SimpleNamespace(AutoModelForMultimodalLM=ModelLoader))
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory) / "dataset", Path(directory) / "run"
            write_dataset(root)
            with patch.object(training, "load_text_tokenization_resources", return_value=(processor, self.config)), \
                    patch.object(training, "load_runtime_dependencies", return_value=dependencies) as runtime, \
                    patch.object(training, "save_best_and_last_model_artifacts", return_value={"best_model_dir": "best", "last_model_dir": "last"}), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = training.main(["--input-mode", "text", "--dataset-root", str(root),
                                        "--output-dir", str(output), "--text-tuning", "full", "--no-load-in-4bit",
                                        "--no-gradient-checkpointing", "--validation-preview-samples", "0"])
            self.assertEqual(result, 0)
            self.assertFalse(runtime.call_args.kwargs["require_images"])
            self.assertIsInstance(instances[0].collator, training.QwenTextDataCollator)
            self.assertTrue(torch.isfinite(instances[0].loss))
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.visual.parameters()))
            self.assertIsNotNone(model.language_model.weight.grad)
            self.assertTrue((output / "text_sequence_lengths.json").is_file())
            config = json.loads((output / "training_config.json").read_text())
            self.assertEqual(config["input_mode"], "text")
            self.assertIsNone(config["resolution_max_pixels"])
            self.assertEqual(config["text_sequence_lengths"]["effective_max_length"], 4096)


if __name__ == "__main__":
    unittest.main()
