"""Training-only modality dropout, real multimodal batches and stage-two queue."""
from copy import deepcopy
import contextlib
import io
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from src.Qwen import qwen_finetune_logic as training
from src.Qwen.chunked_loss import ChunkedLossController
from src.Qwen.experiment_config import load_experiment_queue
from src.Qwen.loss_comparison import compare_loss_methods
from src.Qwen.prompts import (
    HYBRID_EVIDENCE_MARKER, HYBRID_IMAGE_ABSENCE_TEXT, omit_document_image,
)
from test_qwen_chunked_loss import tiny_qwen


def tiny_processor():
    """Real Qwen processor/tokenizer with a small vocabulary and four image patches."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessor, Qwen3VLProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    vocabulary = {f"t{index}": index for index in range(32)}
    for index, token in {0: "[PAD]", 1: "[UNK]", 2: "system", 3: "user", 4: "assistant",
                         5: "task", 6: "answer", 27: "<|vision_start|>", 28: "<|vision_end|>",
                         29: "<|image_pad|>", 30: "<|video_pad|>", 31: "<eos>"}.items():
        del vocabulary[f"t{index}"]
        vocabulary[token] = index
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="<eos>",
        additional_special_tokens=["<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"],
    )
    tokenizer.model_max_length = 4096
    template = (
        "{% for m in messages %}{{ m['role'] }} "
        "{% for b in m['content'] %}{% if b['type'] == 'image' %}<|image_pad|> "
        "{% else %}{{ b['text'] }} {% endif %}{% endfor %}"
        "{% if m['role'] == 'assistant' %}{{ eos_token }} {% endif %}{% endfor %}"
        "{% if add_generation_prompt %}assistant {% endif %}"
    )
    return Qwen3VLProcessor(
        tokenizer=tokenizer, chat_template=template,
        video_processor=Qwen3VLVideoProcessor(),
        image_processor=Qwen2VLImageProcessor(
            size={"shortest_edge": 16, "longest_edge": 16},
            patch_size=2, temporal_patch_size=2, merge_size=1),
    )


class ModalityDropoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.images = [self.root / f"{index}.png" for index in range(3)]
        for path, color in zip(self.images, ("black", "white", "red")):
            Image.new("RGB", (4, 4), color=color).save(path)
        self.processor = tiny_processor()
        self.collator = training.QwenVisionDataCollator(
            self.processor, Image, None, {29, 30}, context_limit=4096)
        torch.manual_seed(17)

    def record(self, index=0, **dropout):
        return {"id": str(index), "image_paths": [str(self.images[index])],
                "messages": [
                    {"role": "system", "content": [{"type": "text", "text": "task JSON Schema: {}"}]},
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "task"},
                        {"type": "text", "text": HYBRID_EVIDENCE_MARKER + '{"sender":"example"}'}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
                ], **dropout}

    def test_three_configs_match_reference_and_only_vary_modality_probabilities(self):
        queue = load_experiment_queue(Path("experiments/qwen/modality_dropout/queue.json"))
        baseline = training.parse_args(["--config", "experiments/qwen/input_comparison/hard_image_nf4_qlora_r16.json"])
        self.assertEqual(len(queue.entries), 3)
        for entry, probabilities in zip(queue.entries, ((.2, 0), (0, .2), (.1, .1))):
            args = training.parse_args(["--config", str(entry.config_path)])
            training.validate_training_options(args)
            self.assertEqual((args.parser_dropout, args.image_dropout), probabilities)
            excluded = {"config", "run_name", "runs_dir", "parser_dropout", "image_dropout",
                        "experiment_name", "experiment_description", "experiment_config_path"}
            for key, value in vars(baseline).items():
                if key not in excluded:
                    self.assertEqual(getattr(args, key), value, key)
            self.assertEqual(args.runs_dir, Path("runs/qwen-modality-dropout"))
            self.assertIsNone(args.resume_from_checkpoint)
            self.assertTrue(args.run_name.startswith("qwen35-9b-hybrid-dropout-"))

    def test_invalid_dropout_probabilities_modes_and_joint_absence_are_rejected(self):
        for flags in (["--image-dropout", ".2"],
                      ["--input-mode", "text", "--image-dropout", ".2"],
                      *[["--input-mode", "hybrid", "--image-dropout", value]
                        for value in ("nan", "inf", "-0.1", "1.1")],
                      ["--input-mode", "hybrid", "--parser-dropout", ".6", "--image-dropout", ".5"]):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                training.validate_training_options(training.parse_args(flags))
        training.validate_training_options(training.parse_args([
            "--input-mode", "hybrid", "--parser-dropout", ".5", "--image-dropout", ".5"]))

    def test_image_absence_retains_evidence_task_schema_answer_without_mutating_source(self):
        record = self.record()
        original = deepcopy(record)
        messages = omit_document_image(record["messages"])
        self.assertEqual(messages[0], original["messages"][0])
        self.assertEqual(messages[-1], original["messages"][-1])
        self.assertEqual(messages[1]["content"][1:], original["messages"][1]["content"][1:])
        self.assertEqual(messages[1]["content"][0]["text"], HYBRID_IMAGE_ABSENCE_TEXT)
        self.assertEqual(record, original)
        with self.assertRaises(ValueError):
            omit_document_image(messages)

    def test_balanced_dropout_is_categorical_seeded_and_evaluation_is_clean(self):
        original = self.record()
        for draw, expected in ((.05, (True, False)), (.15, (False, True)), (.8, (True, True))):
            with patch.object(training.random, "random", return_value=draw), \
                    patch.object(self.processor, "apply_chat_template", wraps=self.processor.apply_chat_template) as template:
                batch = self.collator([{**original, "parser_dropout": .1, "image_dropout": .1}])
            messages = template.call_args_list[0].args[0]
            image = any(b["type"] == "image" for m in messages for b in m["content"])
            parser = any(b.get("text", "").startswith(HYBRID_EVIDENCE_MARKER) for m in messages for b in m["content"])
            self.assertEqual((image, parser), expected)
            self.assertTrue(image or parser)
            self.assertEqual("pixel_values" in batch, image)
            self.assertEqual(batch["labels"][batch["labels"] != -100].tolist(), [6, 31])
        with patch.object(training.random, "random", side_effect=AssertionError("Validation must not draw dropout")):
            validation = self.collator([original])
        self.assertIn("pixel_values", validation)
        self.assertEqual(original, self.record())
        def replay():
            random.seed(42)
            return [self.collator([{**original, "parser_dropout": .1, "image_dropout": .1}])["input_ids"].tolist()
                    for _ in range(12)]
        self.assertEqual(replay(), replay())

    def test_image_absent_batch_skips_loading_and_processing_images(self):
        with patch.object(self.collator, "_load_images", side_effect=AssertionError("Dropped image must not load")):
            batch = self.collator([self.record(image_dropout=1.0)])
        self.assertNotIn("pixel_values", batch)
        self.assertNotIn("image_grid_thw", batch)
        self.assertFalse((batch["input_ids"] == 29).any())
        self.assertFalse(batch["mm_token_type_ids"].any())
        self.assertEqual(batch["labels"][batch["labels"] != -100].tolist(), [6, 31])

    def test_mixed_batches_align_remaining_images_and_mask_left_and_right_padding(self):
        records = [self.record(0, image_dropout=1.0), self.record(1), self.record(2, parser_dropout=1.0)]
        for side in ("left", "right"):
            self.processor.tokenizer.padding_side = side
            batch = self.collator(records)
            self.assertEqual(batch["image_grid_thw"].tolist(), [[1, 2, 2], [1, 2, 2]])
            self.assertEqual((batch["input_ids"] == 29).sum(dim=1).tolist(), [0, 4, 4])
            reference = self.collator([self.record(1), self.record(2)])
            torch.testing.assert_close(batch["pixel_values"], reference["pixel_values"], rtol=0, atol=0)
            for labels in batch["labels"]:
                self.assertEqual(labels[labels != -100].tolist(), [6, 31])
            self.assertTrue((batch["labels"][batch["attention_mask"] == 0] == -100).all())

    def test_real_qwen_loss_gradients_and_rope_state_across_modality_transitions(self):
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(tiny_qwen(), LoraConfig(
            task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"],
            r=2, lora_alpha=4, lora_dropout=.1))
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        controller = ChunkedLossController(model, 1)
        batches = [self.collator([self.record()]), self.collator([self.record(image_dropout=1)]),
                   self.collator([self.record(0, image_dropout=1), self.record(1, parser_dropout=1)])]
        for dtype in ("float32", "bfloat16"):
            report = compare_loss_methods(model, controller, batches, dtype)
            self.assertTrue(report["passed"], report)
        controller.base_model.model.rope_deltas = torch.tensor([[-100]])
        seen = []
        hook = controller.base_model.model.language_model.register_forward_pre_hook(
            lambda _, args, kwargs: seen.append(kwargs.get("position_ids")), with_kwargs=True)
        try:
            model(**batches[1]).loss.backward()
        finally:
            hook.remove()
        self.assertIsNone(seen[0])  # native text positions; no stale image offset
        self.assertIsNone(controller.base_model.model.rope_deltas)
        self.assertTrue(any(p.grad is not None for p in model.parameters() if p.requires_grad))

    def test_trainer_accumulation_and_evaluation_accept_all_modality_states(self):
        from transformers import Trainer, TrainingArguments
        model = tiny_qwen()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        ChunkedLossController(model, 1)
        data = [self.record(0, parser_dropout=1), self.record(1, image_dropout=1), self.record(2)]
        args = TrainingArguments(
            output_dir=str(self.root / "trainer"), use_cpu=True, report_to="none", disable_tqdm=True,
            num_train_epochs=1, per_device_train_batch_size=1, gradient_accumulation_steps=3,
            per_device_eval_batch_size=2, eval_strategy="epoch", save_strategy="no",
            remove_unused_columns=False, prediction_loss_only=True, dataloader_pin_memory=False,
        )
        trainer = Trainer(model=model, args=args, train_dataset=data,
                          eval_dataset=[self.record(0), self.record(1)], data_collator=self.collator)
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.train()
        self.assertEqual(trainer.state.global_step, 1)
        self.assertTrue(any("eval_loss" in row for row in trainer.state.log_history))

    def test_saved_contract_records_training_policy_without_changing_full_input_prompt(self):
        from src.fusion.hybrid_sft import prepare_hybrid_dataset
        from test_qwen_hybrid import HybridTests
        fixture = HybridTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        dataset = fixture.root / "hybrid"
        prepare_hybrid_dataset(fixture.prepare_source(), dataset)
        args = training.parse_args(["--input-mode", "hybrid", "--dataset-root", str(dataset),
            "--schema-path", str(dataset / "schema.json"), "--train-file", "hard/train.jsonl",
            "--validation-file", "hard/val.jsonl", "--parser-dropout", ".1", "--image-dropout", ".1"])
        train, val, *_ = training.load_dataset_splits(args)
        before = deepcopy(train + val)
        contract = training.hybrid_dataset_contract(args, train + val)
        self.assertEqual(train + val, before)
        self.assertEqual(contract["evidence_mode"], "hard")
        policy = contract["training_modality_dropout"]
        self.assertEqual((policy["parser_probability"], policy["image_probability"]), (.1, .1))
        self.assertEqual(policy["policy"], "mutually_exclusive")
        self.assertFalse(policy["validation_dropout"])


if __name__ == "__main__":
    unittest.main()
