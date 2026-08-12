"""Run a JSON queue of Qwen experiments sequentially on one GPU."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.Qwen.experiment_config import load_experiment_config, load_experiment_queue
from src.Qwen.qwen_finetune_logic import parse_args as parse_training_args
from src.Qwen.qwen_finetune_logic import validate_training_options
from src.Qwen.run_qwen_training import DEFAULT_TRAINING_CONFIG


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    if "--" in raw_argv:
        separator_index = raw_argv.index("--")
        queue_argv = raw_argv[:separator_index]
        training_args = raw_argv[separator_index + 1 :]
    else:
        queue_argv = raw_argv
        training_args = []

    parser = argparse.ArgumentParser(
        description=(
            "Run enabled Qwen experiment configs sequentially. Arguments after '--' "
            "are forwarded to every training process and override config values."
        )
    )
    parser.add_argument("queue", type=Path, help="Path to the experiment queue JSON file.")
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Also run queue entries whose enabled field is false.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with the next experiment if one process fails.",
    )
    parser.add_argument(
        "--start-at",
        default=None,
        help="Skip preceding active entries and start at this experiment name.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Validate and display the queue without starting training.",
    )
    args = parser.parse_args(queue_argv)
    args.training_args = training_args
    return args


def build_queue_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    queue = load_experiment_queue(args.queue)
    selected: list[tuple[str, Path]] = []
    seen_names: set[str] = set()
    for entry in queue.entries:
        experiment = load_experiment_config(entry.config_path)
        if experiment.name in seen_names:
            raise ValueError(f"Duplicate experiment name in queue: {experiment.name}")
        seen_names.add(experiment.name)
        resolved_args = parse_training_args(
            ["--config", str(entry.config_path)],
            defaults=DEFAULT_TRAINING_CONFIG,
        )
        validate_training_options(resolved_args)
        status = "enabled" if entry.enabled else "disabled"
        print(f"- {experiment.name}: {status} ({entry.config_path})", flush=True)
        if entry.enabled or args.include_disabled:
            selected.append((experiment.name, entry.config_path))

    if args.start_at is not None:
        names = [name for name, _ in selected]
        if args.start_at not in names:
            raise ValueError(
                f"--start-at {args.start_at!r} is not an active experiment. "
                f"Active names: {', '.join(names) if names else 'none'}"
            )
        selected = selected[names.index(args.start_at) :]

    forwarded = list(args.training_args)
    launcher = REPO_ROOT / "src" / "Qwen" / "run_qwen_training.py"
    return [
        (
            name,
            [
                sys.executable,
                str(launcher),
                "--config",
                str(config_path),
                *forwarded,
            ],
        )
        for name, config_path in selected
    ]


def main(
    argv: Sequence[str] | None = None,
    *,
    run_process: Any = subprocess.run,
) -> int:
    args = parse_args(argv)
    queue = load_experiment_queue(args.queue)
    print(f"Qwen experiment queue: {queue.name}", flush=True)
    if queue.description:
        print(queue.description, flush=True)
    commands = build_queue_commands(args)

    if args.list:
        print(f"Active experiments: {len(commands)}", flush=True)
        return 0
    if not commands:
        print(
            "No active experiments. Enable queue entries or use --include-disabled.",
            flush=True,
        )
        return 0

    failures: list[tuple[str, int]] = []
    for index, (name, command) in enumerate(commands, start=1):
        print(f"\n[{index}/{len(commands)}] Starting {name}", flush=True)
        print(shlex.join(command), flush=True)
        try:
            completed = run_process(command, cwd=REPO_ROOT, check=False)
        except KeyboardInterrupt:
            print(f"\nQueue interrupted while running {name}.", file=sys.stderr)
            return 130
        return_code = int(completed.returncode)
        if return_code == 0:
            print(f"[{index}/{len(commands)}] Completed {name}", flush=True)
            continue

        failures.append((name, return_code))
        print(
            f"[{index}/{len(commands)}] Failed {name} with exit code {return_code}",
            file=sys.stderr,
        )
        if not args.continue_on_error:
            print("Queue stopped. Use --continue-on-error to run later entries.", file=sys.stderr)
            return return_code

    if failures:
        rendered = ", ".join(f"{name} ({code})" for name, code in failures)
        print(f"Queue completed with failures: {rendered}", file=sys.stderr)
        return 1

    print(f"Queue completed successfully: {len(commands)} experiment(s).", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
