from __future__ import annotations

import unittest

from src.Donut.donut_train_logic import parse_args
from src.Donut.run_donut_training import DEFAULT_TRAINING_CONFIG


class DonutTrainingLauncherTests(unittest.TestCase):
    def test_launcher_configuration_supplies_training_defaults(self) -> None:
        args = parse_args([], defaults=DEFAULT_TRAINING_CONFIG)

        self.assertEqual(args.image_size, (1280, 960))
        self.assertEqual(args.per_device_train_batch_size, 1)
        self.assertEqual(args.gradient_accumulation_steps, 8)
        self.assertTrue(args.gradient_checkpointing)
        self.assertTrue(args.local_files_only)
        self.assertEqual(args.num_train_epochs, 50.0)
        self.assertEqual(args.validation_preview_samples, 2)

    def test_command_line_values_override_launcher_defaults(self) -> None:
        args = parse_args(
            [
                "--image-size",
                "1920",
                "1280",
                "--learning-rate",
                "1e-5",
                "--no-gradient-checkpointing",
                "--no-local-files-only",
            ],
            defaults=DEFAULT_TRAINING_CONFIG,
        )

        self.assertEqual(args.image_size, [1920, 1280])
        self.assertEqual(args.learning_rate, 1e-5)
        self.assertFalse(args.gradient_checkpointing)
        self.assertFalse(args.local_files_only)

    def test_unknown_launcher_default_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown_option"):
            parse_args([], defaults={"unknown_option": True})


if __name__ == "__main__":
    unittest.main()
