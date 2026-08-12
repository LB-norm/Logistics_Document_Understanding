from __future__ import annotations

import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.Qwen.qwen_finetune_logic import (
    build_training_arguments,
    build_validation_preview_callback,
    configure_vision_tuning,
    copy_checkpoint_model_artifacts,
    extract_assistant_target,
    find_vision_modules,
    generate_validation_preview_sample,
    parse_args,
    resolve_lora_target_modules,
    save_best_and_last_model_artifacts,
    select_validation_preview_examples,
    select_model_loader,
    validate_training_options,
    write_validation_preview,
)
from src.Qwen.run_qwen_training import DEFAULT_TRAINING_CONFIG


class _VisionTower(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(4, 4)
        self.lora_A = torch.nn.Linear(4, 2, bias=False)
        self.lora_B = torch.nn.Linear(2, 4, bias=False)


class _MultimodalModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language = torch.nn.Linear(4, 4)
        self.visual = _VisionTower()


class _SavedVisionWrapper(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.original_module = _VisionTower()
        self.modules_to_save = torch.nn.ModuleDict({"default": _VisionTower()})


class QwenTrainingLauncherTests(unittest.TestCase):
    def test_launcher_defaults_target_3080_ti_qlora(self) -> None:
        args = parse_args([], defaults=DEFAULT_TRAINING_CONFIG)

        self.assertEqual(args.model_id, "Qwen/Qwen3.5-2B")
        self.assertTrue(args.load_in_4bit)
        self.assertEqual(args.compute_dtype, "bfloat16")
        self.assertEqual(args.per_device_train_batch_size, 1)
        self.assertEqual(args.gradient_accumulation_steps, 8)
        self.assertTrue(args.gradient_checkpointing)
        self.assertEqual(args.vision_tuning, "frozen")
        self.assertEqual(args.target_modules, "all-linear")
        self.assertEqual(args.modules_to_save, "")
        self.assertEqual(args.validation_preview_samples, 2)
        self.assertEqual(args.validation_preview_max_new_tokens, 1024)
        self.assertEqual(args.eval_strategy, "epoch")
        self.assertEqual(args.save_strategy, "epoch")
        self.assertEqual(args.save_total_limit, 2)

    def test_training_arguments_select_lowest_eval_loss_and_retain_two_checkpoints(self) -> None:
        class FakeTrainingArguments:
            def __init__(self, **kwargs):
                self.values = kwargs

        args = parse_args([], defaults=DEFAULT_TRAINING_CONFIG)

        training_args = build_training_arguments(
            FakeTrainingArguments,
            args,
            output_dir=Path("output"),
            gradient_checkpointing=True,
            bf16=True,
            fp16=False,
            load_in_4bit=True,
        )

        self.assertEqual(training_args.values["eval_strategy"], "epoch")
        self.assertEqual(training_args.values["save_strategy"], "epoch")
        self.assertEqual(training_args.values["save_total_limit"], 2)
        self.assertEqual(training_args.values["warmup_steps"], 0.03)
        self.assertNotIn("warmup_ratio", training_args.values)
        self.assertTrue(training_args.values["load_best_model_at_end"])
        self.assertEqual(training_args.values["metric_for_best_model"], "eval_loss")
        self.assertFalse(training_args.values["greater_is_better"])

    def test_checkpoint_policy_rejects_retention_below_best_and_last(self) -> None:
        args = parse_args(["--save-total-limit", "1"])

        with self.assertRaisesRegex(ValueError, "best and last"):
            validate_training_options(args)

    def test_checkpoint_policy_rejects_mismatched_step_intervals(self) -> None:
        args = parse_args(["--eval-steps", "25", "--save-steps", "50"])

        with self.assertRaisesRegex(ValueError, "must match"):
            validate_training_options(args)

    def test_best_and_last_model_artifacts_are_materialized_separately(self) -> None:
        class FakeTrainer:
            def __init__(self, best_checkpoint: Path) -> None:
                self.state = SimpleNamespace(
                    best_model_checkpoint=str(best_checkpoint),
                    best_metric=0.125,
                )

            @staticmethod
            def save_model(destination: str) -> None:
                destination_path = Path(destination)
                destination_path.mkdir(parents=True, exist_ok=True)
                (destination_path / "adapter_config.json").write_text(
                    '{"source":"best"}', encoding="utf-8"
                )
                (destination_path / "adapter_model.safetensors").write_text(
                    "best", encoding="utf-8"
                )

        class FakeProcessor:
            @staticmethod
            def save_pretrained(destination: str) -> None:
                (Path(destination) / "processor.json").write_text("{}", encoding="utf-8")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            best_checkpoint = output_dir / "checkpoint-50"
            last_checkpoint = output_dir / "checkpoint-100"
            for checkpoint, contents in ((best_checkpoint, "best"), (last_checkpoint, "last")):
                checkpoint.mkdir()
                (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
                (checkpoint / "adapter_model.safetensors").write_text(
                    contents, encoding="utf-8"
                )
                (checkpoint / "optimizer.pt").write_text("large", encoding="utf-8")

            summary = save_best_and_last_model_artifacts(
                trainer=FakeTrainer(best_checkpoint),
                processor=FakeProcessor(),
                output_dir=output_dir,
            )

            self.assertEqual(
                (output_dir / "best_model" / "adapter_model.safetensors").read_text(
                    encoding="utf-8"
                ),
                "best",
            )
            self.assertEqual(
                (output_dir / "last_model" / "adapter_model.safetensors").read_text(
                    encoding="utf-8"
                ),
                "last",
            )
            self.assertFalse((output_dir / "last_model" / "optimizer.pt").exists())
            self.assertTrue((output_dir / "best_model" / "processor.json").is_file())
            self.assertTrue((output_dir / "last_model" / "processor.json").is_file())
            self.assertEqual(summary["best_metric"], 0.125)

    def test_checkpoint_copy_requires_model_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "checkpoint-1"
            source.mkdir()
            (source / "optimizer.pt").write_text("state", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "model weights"):
                copy_checkpoint_model_artifacts(source, Path(temp_dir) / "last_model")

    def test_command_line_values_override_launcher_defaults(self) -> None:
        args = parse_args(
            [
                "--model-id",
                "Qwen/Qwen3.5-9B",
                "--vision-tuning",
                "lora",
                "--lora-r",
                "32",
                "--no-gradient-checkpointing",
                "--no-local-files-only",
            ],
            defaults=DEFAULT_TRAINING_CONFIG,
        )

        self.assertEqual(args.model_id, "Qwen/Qwen3.5-9B")
        self.assertEqual(args.vision_tuning, "lora")
        self.assertEqual(args.lora_r, 32)
        self.assertFalse(args.gradient_checkpointing)
        self.assertFalse(args.local_files_only)

    def test_unknown_launcher_default_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown_option"):
            parse_args([], defaults={"unknown_option": True})

    def test_full_vision_tuning_rejects_quantized_base(self) -> None:
        args = parse_args(["--vision-tuning", "full", "--load-in-4bit"])

        with self.assertRaisesRegex(ValueError, "incompatible with 4-bit"):
            validate_training_options(args)

    def test_full_vision_tuning_accepts_non_quantized_base(self) -> None:
        args = parse_args(["--vision-tuning", "full", "--no-load-in-4bit"])

        validate_training_options(args)

    def test_vision_module_detection_uses_configured_attribute_names(self) -> None:
        model = _MultimodalModel()

        modules = find_vision_modules(model, ["visual", "vision_tower"])

        self.assertEqual([name for name, _ in modules], ["visual"])

    def test_frozen_mode_disables_every_vision_parameter(self) -> None:
        model = _MultimodalModel()
        modules = find_vision_modules(model, ["visual"])

        summary = configure_vision_tuning(modules, "frozen")

        self.assertEqual(summary["trainable_parameters"], 0)
        self.assertFalse(any(parameter.requires_grad for parameter in model.visual.parameters()))

    def test_all_linear_targets_respect_vision_tuning_mode(self) -> None:
        model = _MultimodalModel()

        language_only = resolve_lora_target_modules(
            model,
            "all-linear",
            vision_module_paths=["visual"],
            vision_tuning="frozen",
            torch=torch,
        )
        language_and_vision = resolve_lora_target_modules(
            model,
            "all-linear",
            vision_module_paths=["visual"],
            vision_tuning="lora",
            torch=torch,
        )

        self.assertEqual(language_only, ["language"])
        self.assertIn("language", language_and_vision)
        self.assertIn("visual.projection", language_and_vision)
        self.assertIn("visual.lora_A", language_and_vision)

    def test_lora_mode_enables_only_vision_adapter_parameters(self) -> None:
        model = _MultimodalModel()
        modules = find_vision_modules(model, ["visual"])

        summary = configure_vision_tuning(modules, "lora")

        trainable_names = {
            name for name, parameter in model.visual.named_parameters() if parameter.requires_grad
        }
        self.assertEqual(
            trainable_names,
            {"lora_A.weight", "lora_B.weight"},
        )
        self.assertGreater(summary["trainable_parameters"], 0)

    def test_full_mode_enables_all_vision_parameters(self) -> None:
        model = _MultimodalModel()
        for parameter in model.visual.parameters():
            parameter.requires_grad = False

        summary = configure_vision_tuning(find_vision_modules(model, ["visual"]), "full")

        self.assertEqual(summary["trainable_parameters"], summary["total_parameters"])
        self.assertTrue(all(parameter.requires_grad for parameter in model.visual.parameters()))

    def test_full_mode_optimizes_only_peft_checkpointed_vision_copy(self) -> None:
        visual = _SavedVisionWrapper()

        configure_vision_tuning([("visual", visual)], "full")

        self.assertFalse(
            any(parameter.requires_grad for parameter in visual.original_module.parameters())
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in visual.modules_to_save["default"].parameters()
            )
        )

    def test_model_loader_can_be_auto_or_explicit(self) -> None:
        auto_loader = object()
        explicit_loader = object()
        transformers_module = SimpleNamespace(
            AutoModelForMultimodalLM=auto_loader,
            CustomQwenLoader=explicit_loader,
        )

        self.assertIs(select_model_loader(transformers_module, "auto"), auto_loader)
        self.assertIs(
            select_model_loader(transformers_module, "CustomQwenLoader"),
            explicit_loader,
        )

    def test_preview_sample_selection_is_fixed_by_seed(self) -> None:
        examples = [{"id": str(index)} for index in range(10)]

        first = select_validation_preview_examples(examples, sample_count=3, seed=42)
        second = select_validation_preview_examples(examples, sample_count=3, seed=42)

        self.assertEqual([item["id"] for item in first], [item["id"] for item in second])
        self.assertEqual(len(first), 3)

    def test_assistant_target_is_parsed_as_json(self) -> None:
        example = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "extract"}]},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": '{"city":"Berlin"}'}],
                },
            ]
        }

        self.assertEqual(extract_assistant_target(example), {"city": "Berlin"})

    def test_preview_generation_decodes_only_new_answer_tokens(self) -> None:
        class FakeImage:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def convert(self, mode):
                return self

        class FakeImageModule:
            @staticmethod
            def open(path):
                return FakeImage()

        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 2

            @staticmethod
            def batch_decode(rows, **kwargs):
                return ['{"city":"Berlin"}']

        class FakeProcessor:
            tokenizer = FakeTokenizer()

            def __init__(self):
                self.templated_messages = None

            def apply_chat_template(self, messages, **kwargs):
                self.templated_messages = messages
                return "prompt"

            def __call__(self, **kwargs):
                return {
                    "input_ids": torch.tensor([[10, 11]]),
                    "attention_mask": torch.tensor([[1, 1]]),
                }

            @staticmethod
            def batch_decode(rows, **kwargs):
                self_rows = rows.tolist()
                if self_rows != [[20, 21]]:
                    raise AssertionError(f"Expected only generated tokens, got {self_rows}")
                return ['{"city":"Berlin"}']

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))

            def generate(self, **kwargs):
                return torch.tensor([[10, 11, 20, 21]])

        processor = FakeProcessor()
        example = {
            "id": "sample",
            "image_paths": ["sample.png"],
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "json"}]},
                {"role": "user", "content": [{"type": "image"}]},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": '{"city":"Berlin"}'}],
                },
            ],
        }

        result = generate_validation_preview_sample(
            image_module=FakeImageModule,
            processor=processor,
            model=FakeModel(),
            example=example,
            max_new_tokens=32,
            target_schema={
                "type": "object",
                "required": ["city"],
                "properties": {"city": {"type": "string"}},
            },
        )

        self.assertEqual(len(processor.templated_messages), 2)
        self.assertEqual(result["prediction"], {"city": "Berlin"})
        self.assertEqual(result["generated_tokens"], 2)
        self.assertTrue(result["exact_match"])

    def test_preview_writer_creates_step_and_latest_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            payload = {
                "status": "completed",
                "global_step": 5,
                "epoch": 0.5,
                "evaluation": {"summary": {"parse_rate": 1.0, "field_f1": 0.0}},
                "samples": [
                    {
                        "id": "example<unsafe>",
                        "image_paths": [str(Path(temp_dir) / "example.png")],
                        "ground_truth": {"city": "Berlin"},
                        "prediction": {"city": "Munich"},
                        "exact_match": False,
                        "differences": [{"kind": "value_mismatch", "path": "$.city"}],
                        "parse_error": None,
                        "schema_errors": [],
                        "raw_sequence": '{"city":"Munich"}',
                        "prompt_tokens": 10,
                        "generated_tokens": 5,
                    }
                ],
            }

            paths = write_validation_preview(Path(temp_dir), payload)

            self.assertTrue(Path(paths["json"]).is_file())
            self.assertTrue(Path(paths["latest_html"]).is_file())
            html_report = Path(paths["html"]).read_text(encoding="utf-8")
            self.assertIn("Qwen validation preview", html_report)
            self.assertIn("Ground truth", html_report)
            self.assertIn("Prediction", html_report)
            self.assertIn("example&lt;unsafe&gt;", html_report)

    def test_preview_callback_runs_once_for_each_training_log_step(self) -> None:
        class FakeTrainerCallback:
            pass

        class FakeTorch:
            @staticmethod
            def inference_mode():
                return nullcontext()

        class FakeModel:
            def __init__(self) -> None:
                self.training = True

            def eval(self) -> None:
                self.training = False

            def train(self) -> None:
                self.training = True

        example = {
            "id": "example",
            "image_paths": ["example.png"],
            "messages": [
                {"role": "user", "content": [{"type": "image"}]},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": '{"city":"Berlin"}'}],
                },
            ],
        }
        generated_sample = {
            "id": "example",
            "image_paths": ["example.png"],
            "ground_truth": {"city": "Berlin"},
            "prediction": {"city": "Berlin"},
            "exact_match": True,
            "differences": [],
            "parse_error": None,
            "schema_errors": [],
            "metrics": {},
            "raw_sequence": '{"city":"Berlin"}',
            "prompt_tokens": 10,
            "generated_tokens": 5,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            callback = build_validation_preview_callback(
                TrainerCallback=FakeTrainerCallback,
                torch=FakeTorch,
                image_module=object(),
                processor=object(),
                examples=[example],
                output_dir=Path(temp_dir),
                max_new_tokens=32,
                target_schema={"type": "object"},
            )
            state = SimpleNamespace(global_step=5, epoch=0.5, is_world_process_zero=True)
            model = FakeModel()

            with patch(
                "src.Qwen.qwen_finetune_logic.generate_validation_preview_sample",
                return_value=generated_sample,
            ) as generate:
                callback.on_log(None, state, object(), logs={"loss": 1.2}, model=model)
                callback.on_log(None, state, object(), logs={"loss": 1.2}, model=model)

            self.assertEqual(generate.call_count, 1)
            self.assertTrue(model.training)
            self.assertTrue(
                (Path(temp_dir) / "validation_previews" / "step_00000005.html").is_file()
            )


if __name__ == "__main__":
    unittest.main()
