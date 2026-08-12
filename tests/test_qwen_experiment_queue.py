from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.Qwen.experiment_config import load_experiment_config, load_experiment_queue
from src.Qwen.qwen_finetune_logic import parse_args as parse_training_args
from src.Qwen.run_qwen_experiment_queue import main as run_queue
from src.Qwen.run_qwen_training import DEFAULT_TRAINING_CONFIG


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _experiment(name: str, training: dict | None = None) -> dict:
    return {
        "format_version": 1,
        "name": name,
        "description": f"Test experiment {name}",
        "training": training or {},
    }


class QwenExperimentConfigTests(unittest.TestCase):
    def test_experiment_values_override_launcher_and_cli_overrides_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "experiment.json"
            _write_json(
                config_path,
                _experiment(
                    "test-experiment",
                    {
                        "model_id": "Qwen/config-model",
                        "lora_r": 32,
                        "dry_run": True,
                    },
                ),
            )

            args = parse_training_args(
                ["--config", str(config_path), "--model-id", "Qwen/cli-model"],
                defaults=DEFAULT_TRAINING_CONFIG,
            )

            self.assertEqual(args.model_id, "Qwen/cli-model")
            self.assertEqual(args.lora_r, 32)
            self.assertTrue(args.dry_run)
            self.assertEqual(args.run_name, "test-experiment")
            self.assertEqual(args.experiment_name, "test-experiment")
            self.assertEqual(args.experiment_config_path, str(config_path.resolve()))

    def test_unknown_training_setting_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "experiment.json"
            _write_json(config_path, _experiment("invalid", {"unknown_setting": 3}))

            with self.assertRaisesRegex(ValueError, "unknown_setting"):
                parse_training_args(["--config", str(config_path)])

    def test_experiment_name_must_be_safe_for_a_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "experiment.json"
            _write_json(config_path, _experiment("../outside"))

            with self.assertRaisesRegex(ValueError, "must contain only"):
                load_experiment_config(config_path)


class QwenExperimentQueueTests(unittest.TestCase):
    def test_queue_resolves_relative_paths_and_preserves_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_path = root / "first.json"
            second_path = root / "second.json"
            queue_path = root / "queue.json"
            _write_json(first_path, _experiment("first"))
            _write_json(second_path, _experiment("second"))
            _write_json(
                queue_path,
                {
                    "format_version": 1,
                    "name": "test-queue",
                    "experiments": [
                        {"config": "first.json", "enabled": True},
                        {"config": "second.json", "enabled": False},
                    ],
                },
            )

            queue = load_experiment_queue(queue_path)

            self.assertEqual(
                [entry.config_path for entry in queue.entries],
                [first_path.resolve(), second_path.resolve()],
            )
            self.assertEqual([entry.enabled for entry in queue.entries], [True, False])

    def test_queue_runs_each_selected_config_in_a_fresh_process(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_path = root / "first.json"
            second_path = root / "second.json"
            queue_path = root / "queue.json"
            _write_json(first_path, _experiment("first"))
            _write_json(second_path, _experiment("second"))
            _write_json(
                queue_path,
                {
                    "format_version": 1,
                    "name": "test-queue",
                    "experiments": [
                        {"config": "first.json", "enabled": True},
                        {"config": "second.json", "enabled": False},
                    ],
                },
            )
            calls: list[tuple[list[str], Path, bool]] = []

            def fake_run(command, *, cwd, check):
                calls.append((command, cwd, check))
                return SimpleNamespace(returncode=0)

            result = run_queue(
                [
                    str(queue_path),
                    "--include-disabled",
                    "--",
                    "--dataset-root",
                    "/mnt/datasets/cmrs",
                ],
                run_process=fake_run,
            )

            self.assertEqual(result, 0)
            self.assertEqual(len(calls), 2)
            self.assertIn(str(first_path.resolve()), calls[0][0])
            self.assertIn(str(second_path.resolve()), calls[1][0])
            self.assertEqual(calls[0][0][-2:], ["--dataset-root", "/mnt/datasets/cmrs"])
            self.assertIsNot(calls[0][0], calls[1][0])
            self.assertTrue(all(check is False for _, _, check in calls))

    def test_queue_stops_after_first_failure_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            queue_path = root / "queue.json"
            for name in ("first", "second"):
                _write_json(root / f"{name}.json", _experiment(name))
            _write_json(
                queue_path,
                {
                    "format_version": 1,
                    "name": "test-queue",
                    "experiments": ["first.json", "second.json"],
                },
            )
            calls: list[list[str]] = []

            def fail(command, *, cwd, check):
                calls.append(command)
                return SimpleNamespace(returncode=7)

            result = run_queue([str(queue_path)], run_process=fail)

            self.assertEqual(result, 7)
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
