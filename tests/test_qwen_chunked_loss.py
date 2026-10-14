from __future__ import annotations

from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

from src.Qwen.chunked_loss import ChunkedLossController, chunked_causal_loss
from src.Qwen.compare_training_histories import compare_histories
from src.Qwen.loss_comparison import compare_loss_methods
from src.Qwen.qwen_finetune_logic import (
    build_stop_after_steps_callback, parse_args, validate_training_options,
)


def tiny_qwen(vocab_size=32):
    text = Qwen3_5TextConfig(
        vocab_size=vocab_size, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        layer_types=["linear_attention", "full_attention"], max_position_embeddings=128,
        linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=2,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                         "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2]},
    )
    vision = Qwen3_5VisionConfig(depth=1, hidden_size=16, intermediate_size=32,
                                num_heads=2, out_hidden_size=16, patch_size=2, spatial_merge_size=1)
    model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(
        text_config=text.to_dict(), vision_config=vision.to_dict(),
        image_token_id=29, video_token_id=30, eos_token_id=31))
    model.config.text_config.use_cache = False
    for parameter in model.model.visual.parameters():
        parameter.requires_grad = False
    return model


def padded_batch():
    return {
        "input_ids": torch.tensor([[0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 0, 0]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]),
        "labels": torch.tensor([[-100, -100, -100, -100, 3, 4], [-100, 2, 3, 4, -100, -100]]),
    }


class ChunkedLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(17)

    def test_mask_shift_unequal_chunks_and_all_head_backbone_gradients(self):
        for chunk_size in (1, 3, 256):
            for frozen in (False, True):
                with self.subTest(chunk_size=chunk_size, frozen=frozen):
                    h = torch.randn(2, 6, 8, requires_grad=True)
                    head = torch.nn.Linear(8, 32, bias=True)
                    head.requires_grad_(not frozen)
                    labels = padded_batch()["labels"]
                    standard = F.cross_entropy(head(h)[:, :-1].float().reshape(-1, 32),
                                               labels[:, 1:].reshape(-1), ignore_index=-100)
                    standard.backward()
                    hidden_gradient = h.grad.clone()
                    weight_gradient = head.weight.grad.clone() if not frozen else None
                    bias_gradient = head.bias.grad.clone() if not frozen else None
                    h.grad = None
                    head.zero_grad(set_to_none=True)
                    result = chunked_causal_loss(h, labels, head, chunk_size)
                    torch.testing.assert_close(result, standard, rtol=1e-6, atol=1e-6)
                    result.backward()
                    torch.testing.assert_close(h.grad, hidden_gradient, rtol=1e-5, atol=1e-7)
                    self.assertGreater(h.grad[0, 3].norm().item(), 0)  # first answer prediction
                    if not frozen:
                        torch.testing.assert_close(head.weight.grad, weight_gradient, rtol=1e-5, atol=1e-7)
                        torch.testing.assert_close(head.bias.grad, bias_gradient, rtol=1e-5, atol=1e-7)

    def test_checkpoint_does_not_save_full_vocabulary_activations(self):
        head = torch.nn.Linear(8, 128)
        labels = torch.randint(0, 128, (1, 30))
        labels[:, :10] = -100
        saved = []
        def pack(tensor):
            saved.append(tuple(tensor.shape))
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            h = torch.randn(1, 30, 8, requires_grad=True)
            loss = chunked_causal_loss(h, labels, head, 4)
        self.assertFalse(any(len(shape) == 2 and shape[-1] == 128 for shape in saved), saved)
        loss.backward()
        self.assertIsNotNone(head.weight.grad)

    def test_real_hybrid_qwen_full_and_lora_loss_gradients_rng_and_generation(self):
        from peft import LoraConfig, get_peft_model
        for adapter in (False, True):
            with self.subTest(adapter=adapter):
                model = tiny_qwen()
                if adapter:
                    model = get_peft_model(model, LoraConfig(
                        task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"],
                        r=2, lora_alpha=4, lora_dropout=0.1))
                    # Exercise nonzero gradients through both adapter matrices.
                    for name, parameter in model.named_parameters():
                        if "lora_B" in name:
                            torch.nn.init.normal_(parameter, std=0.02)
                before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
                controller = ChunkedLossController(model, 2)
                rng = torch.get_rng_state().clone()
                report = compare_loss_methods(model, controller, [padded_batch()])
                self.assertTrue(report["passed"], report)
                self.assertLess(report["samples"][0]["gradient_relative_l2_error"], 1e-5)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
                for name, parameter in model.named_parameters():
                    torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
                model.eval()
                inputs = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
                controller.enabled = False
                expected = model(**inputs).logits.detach()
                expected_generated = model.generate(**inputs, do_sample=False, max_new_tokens=2, pad_token_id=0)
                controller.enabled = True
                torch.testing.assert_close(model(**inputs).logits, expected, rtol=0, atol=0)
                self.assertTrue(torch.equal(model.generate(**inputs, do_sample=False, max_new_tokens=2, pad_token_id=0),
                                            expected_generated))

    def test_bfloat16_autocast_report(self):
        model = tiny_qwen()
        controller = ChunkedLossController(model, 2)
        report = compare_loss_methods(model, controller, [padded_batch()], "bfloat16")
        self.assertTrue(report["passed"], report)

    def test_backbone_gradient_checkpointing(self):
        from peft import LoraConfig, get_peft_model
        for reentrant in (True, False):
            for adapter in (False, True):
                with self.subTest(reentrant=reentrant, adapter=adapter):
                    model = tiny_qwen()
                    if adapter:
                        model = get_peft_model(model, LoraConfig(
                            task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"],
                            r=2, lora_alpha=4, lora_dropout=0.1))
                        model.enable_input_require_grads()
                    model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": reentrant})
                    controller = ChunkedLossController(model, 2)
                    report = compare_loss_methods(model, controller, [padded_batch()])
                    self.assertTrue(report["passed"], report)

    def test_trainer_accumulation_evaluation_and_early_stop_keep_scheduler_and_updates(self):
        for adapter in (False, True):
            with self.subTest(adapter=adapter):
                self.check_trainer_early_stop(adapter)

    def check_trainer_early_stop(self, adapter):
        from transformers import Trainer, TrainerCallback, TrainingArguments
        old = tiny_qwen()
        if adapter:
            from peft import LoraConfig, get_peft_model
            old = get_peft_model(old, LoraConfig(
                task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"],
                r=2, lora_alpha=4, lora_dropout=0.1))
        new = deepcopy(old)
        ChunkedLossController(new, 2)
        batch = padded_batch()
        data = [{key: value[index].tolist() for key, value in batch.items()} for index in (0, 1, 0, 1)]
        states = []
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            for index, model in enumerate((old, new)):
                args = TrainingArguments(
                    output_dir=f"{directory}/{index}", use_cpu=True, report_to="none",
                    disable_tqdm=True, num_train_epochs=3, per_device_train_batch_size=1,
                    per_device_eval_batch_size=1, gradient_accumulation_steps=2,
                    eval_strategy="steps", eval_steps=1, save_strategy="no", logging_steps=1,
                    learning_rate=1e-3, lr_scheduler_type="cosine", warmup_steps=0.1,
                    seed=17, data_seed=17, dataloader_pin_memory=False,
                    prediction_loss_only=True, remove_unused_columns=False,
                )
                trainer = Trainer(model=model, args=args, train_dataset=data, eval_dataset=data,
                                  callbacks=[build_stop_after_steps_callback(TrainerCallback, 2)])
                self.assertFalse(trainer.model_accepts_loss_kwargs)
                trainer.train()
                self.assertEqual(trainer.state.global_step, 2)
                self.assertEqual(trainer.state.max_steps, 6)  # planned 3 epochs, not stop-step 2
                states.append(trainer.state)
        old_lr = [item["learning_rate"] for item in states[0].log_history if "learning_rate" in item]
        new_lr = [item["learning_rate"] for item in states[1].log_history if "learning_rate" in item]
        self.assertEqual(old_lr, new_lr)
        comparisons = compare_histories(vars(states[0]), vars(states[1]))
        self.assertEqual(len(comparisons), 2)
        self.assertTrue(all(abs(row["difference"]) < 1e-5 for row in comparisons))
        for a, b in zip(old.parameters(), new.parameters()):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=2e-6)

    def test_accumulated_gradients_and_adamw_update_match(self):
        old = tiny_qwen()
        new = deepcopy(old)
        ChunkedLossController(new, 2)
        batch = padded_batch()
        optimizers = [torch.optim.AdamW(model.parameters(), lr=1e-3) for model in (old, new)]
        for model, optimizer in zip((old, new), optimizers):
            for index in (0, 1):
                microbatch = {key: value[index:index + 1] for key, value in batch.items()}
                (model(**microbatch).loss / 2).backward()
            optimizer.step()
        for a, b in zip(old.parameters(), new.parameters()):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=2e-6)

    def test_invalid_modes_and_inputs_are_rejected(self):
        for options in (["--loss-implementation", "chunked"],
                        ["--input-mode", "text", "--loss-chunk-size", "0"],
                        ["--stop-after-steps", "0"]):
            with self.assertRaises(ValueError):
                validate_training_options(parse_args(options))
        with self.assertRaisesRegex(ValueError, "no supervised"):
            chunked_causal_loss(torch.randn(1, 2, 8), torch.full((1, 2), -100), torch.nn.Linear(8, 32))
        model = tiny_qwen()
        ChunkedLossController(model)
        with self.assertRaisesRegex(ValueError, "already installed"):
            ChunkedLossController(model)
        with self.assertRaisesRegex(ValueError, "logits_to_keep"):
            model(**padded_batch(), logits_to_keep=1)

    def test_stop_callback_and_comparison_histories(self):
        from types import SimpleNamespace
        callback = build_stop_after_steps_callback(object, 75)
        control = SimpleNamespace(should_training_stop=False)
        callback.on_step_end(None, SimpleNamespace(global_step=74), control)
        self.assertFalse(control.should_training_stop)
        callback.on_step_end(None, SimpleNamespace(global_step=75), control)
        self.assertTrue(control.should_training_stop)
        rows = compare_histories(
            {"log_history": [{"step": 25, "eval_loss": 1., "epoch": 1}, {"step": 50, "eval_loss": 0.8}]},
            {"log_history": [{"step": 25, "eval_loss": 1.001}, {"step": 75, "eval_loss": 0.7},
                             {"step": 25, "eval_loss": 0.8}]})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["step"], 25)
        self.assertAlmostEqual(rows[0]["difference"], 0.001)

    def test_launcher_comparison_loads_model_only_checkpoint_and_exits_without_training(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from transformers import Trainer, TrainerCallback, TrainingArguments
        from test_qwen_text_training import TinyProcessor, write_dataset
        from src.Qwen import qwen_finetune_logic as training
        model = tiny_qwen(512)
        processor = TinyProcessor()
        class Dataset:
            @staticmethod
            def from_list(rows):
                return rows
        class Loader:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return model
        dependencies = (torch, None, Dataset, None, None, Trainer, TrainerCallback,
                        TrainingArguments, (None, None, None),
                        SimpleNamespace(AutoModelForMultimodalLM=Loader))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, output, saved = root / "data", root / "report", root / "checkpoint"
            write_dataset(dataset)
            model.save_pretrained(saved)
            weights = {name: p.detach().clone() for name, p in model.named_parameters()}
            with patch.object(training, "load_runtime_dependencies", return_value=dependencies), \
                    patch.object(training, "load_text_tokenization_resources", return_value=(processor, model.config)), \
                    patch.object(Trainer, "train", side_effect=AssertionError("Comparison must not train")), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = training.main([
                    "--input-mode", "text", "--text-tuning", "full", "--no-load-in-4bit",
                    "--dataset-root", str(dataset), "--output-dir", str(output),
                    "--compare-loss", "--comparison-samples", "1", "--loss-chunk-size", "7",
                    "--resume-from-checkpoint", str(saved), "--compute-dtype", "float32",
                    "--no-gradient-checkpointing", "--validation-preview-samples", "0",
                ])
            self.assertEqual(result, 0)
            report = json.loads((output / "loss_comparison.json").read_text())
            self.assertTrue(report["passed"], report)
            self.assertEqual(report["comparison_checkpoint"], str(saved))
            self.assertFalse(report["weights_updated"])
            self.assertFalse(list(output.glob("checkpoint-*")))
            for name, p in model.named_parameters():
                torch.testing.assert_close(p, weights[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
