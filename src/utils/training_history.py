from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any


CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")


def checkpoint_step(path: Path) -> int | None:
    match = CHECKPOINT_PATTERN.match(path.name)
    if not match:
        return None
    return int(match.group(1))


def find_checkpoint_dirs(run_dir: Path) -> list[Path]:
    checkpoints: list[tuple[int, Path]] = []
    if not run_dir.exists():
        return []
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        step = checkpoint_step(child)
        if step is not None:
            checkpoints.append((step, child))
    return [path for _, path in sorted(checkpoints)]


def load_trainer_state(run_dir: Path) -> dict[str, Any] | None:
    root_state_path = run_dir / "trainer_state.json"
    if root_state_path.exists():
        with root_state_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    checkpoint_dirs = find_checkpoint_dirs(run_dir)
    state_paths = [checkpoint / "trainer_state.json" for checkpoint in reversed(checkpoint_dirs)]
    for state_path in state_paths:
        if state_path.exists():
            with state_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
    return None


def _numeric(value: Any) -> float | int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return value
    return None


def _history(state: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not state:
        return []
    history = state.get("log_history", [])
    return history if isinstance(history, list) else []


def metric_points(state: dict[str, Any] | None, metric_name: str) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for entry in _history(state):
        if not isinstance(entry, dict) or metric_name not in entry:
            continue
        value = _numeric(entry.get(metric_name))
        if value is None:
            continue
        points.append(
            {
                "step": _numeric(entry.get("step")),
                "epoch": _numeric(entry.get("epoch")),
                metric_name: value,
            }
        )
    return points


def nearest_metric_at_or_before(
    state: dict[str, Any] | None,
    metric_name: str,
    step: int | None,
) -> float | int | None:
    if step is None:
        return None
    candidates = [
        point
        for point in metric_points(state, metric_name)
        if isinstance(point.get("step"), int | float) and point["step"] <= step
    ]
    if not candidates:
        return None
    exact = [point for point in candidates if point.get("step") == step]
    chosen = exact[-1] if exact else candidates[-1]
    return chosen.get(metric_name)


def latest_metric(state: dict[str, Any] | None, metric_name: str) -> float | int | None:
    points = metric_points(state, metric_name)
    if not points:
        return None
    return points[-1].get(metric_name)


def _checkpoint_for_step(run_dir: Path, step: int | None) -> Path | None:
    if step is None:
        return None
    candidate = run_dir / f"checkpoint-{step}"
    return candidate if candidate.exists() else None


def _path_from_state(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    return Path(value)


def _relative_or_absolute(path: Path | None, run_dir: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(run_dir.resolve()))
    except ValueError:
        return str(path)


def summarize_checkpoints(run_dir: Path, state: dict[str, Any] | None = None) -> dict[str, Any]:
    state = state or load_trainer_state(run_dir)
    eval_points = metric_points(state, "eval_loss")
    checkpoints = find_checkpoint_dirs(run_dir)
    last_step = _numeric(state.get("global_step")) if state else None
    if last_step is None and checkpoints:
        last_step = checkpoint_step(checkpoints[-1])
    last_step_int = int(last_step) if isinstance(last_step, int | float) else None

    best_point = min(eval_points, key=lambda point: point["eval_loss"]) if eval_points else None
    best_step = int(best_point["step"]) if best_point and best_point.get("step") is not None else None
    best_path = _path_from_state(state.get("best_model_checkpoint")) if state else None
    if best_path is not None and not best_path.is_absolute():
        best_path = run_dir / best_path
    if best_path is None:
        best_path = _checkpoint_for_step(run_dir, best_step)

    last_path = _checkpoint_for_step(run_dir, last_step_int)
    if last_path is None and checkpoints:
        last_path = checkpoints[-1]
        last_step_int = checkpoint_step(last_path)

    best_eval_loss = best_point.get("eval_loss") if best_point else None
    best_epoch = best_point.get("epoch") if best_point else None
    last_eval_loss = nearest_metric_at_or_before(state, "eval_loss", last_step_int)
    last_epoch = _numeric(state.get("epoch")) if state else None

    return {
        "policy": {
            "retained": "best_and_last",
            "metric": "eval_loss",
            "greater_is_better": False,
        },
        "best": {
            "step": best_step,
            "epoch": best_epoch,
            "path": _relative_or_absolute(best_path, run_dir),
            "exists": bool(best_path and best_path.exists()),
            "train_loss": nearest_metric_at_or_before(state, "loss", best_step),
            "eval_loss": best_eval_loss,
        },
        "last": {
            "step": last_step_int,
            "epoch": last_epoch,
            "path": _relative_or_absolute(last_path, run_dir),
            "exists": bool(last_path and last_path.exists()),
            "train_loss": nearest_metric_at_or_before(state, "loss", last_step_int),
            "eval_loss": last_eval_loss,
        },
        "available": [
            {
                "step": checkpoint_step(checkpoint),
                "path": _relative_or_absolute(checkpoint, run_dir),
            }
            for checkpoint in checkpoints
        ],
    }


def summarize_training_history(state: dict[str, Any] | None) -> dict[str, Any]:
    eval_points = metric_points(state, "eval_loss")
    train_points = metric_points(state, "loss")
    learning_rate_points = metric_points(state, "learning_rate")
    best_eval = min(eval_points, key=lambda point: point["eval_loss"]) if eval_points else None
    return {
        "global_step": state.get("global_step") if state else None,
        "epoch": state.get("epoch") if state else None,
        "train_loss": {
            "first": train_points[0]["loss"] if train_points else None,
            "last": train_points[-1]["loss"] if train_points else None,
            "points": len(train_points),
        },
        "eval_loss": {
            "first": eval_points[0]["eval_loss"] if eval_points else None,
            "last": eval_points[-1]["eval_loss"] if eval_points else None,
            "best": best_eval["eval_loss"] if best_eval else None,
            "best_step": best_eval["step"] if best_eval else None,
            "best_epoch": best_eval["epoch"] if best_eval else None,
            "points": len(eval_points),
        },
        "learning_rate": {
            "first": learning_rate_points[0]["learning_rate"] if learning_rate_points else None,
            "last": learning_rate_points[-1]["learning_rate"] if learning_rate_points else None,
            "points": len(learning_rate_points),
        },
    }


def prune_checkpoints_to_best_and_last(
    run_dir: Path,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = summarize_checkpoints(run_dir, state=state)
    best_step = summary["best"].get("step")
    last_step = summary["last"].get("step")
    if best_step is not None and best_step != last_step and not summary["best"].get("exists"):
        summary["removed"] = []
        summary["prune_skipped_reason"] = "Best checkpoint is not present on disk."
        return summary
    if last_step is not None and not summary["last"].get("exists"):
        summary["removed"] = []
        summary["prune_skipped_reason"] = "Last checkpoint is not present on disk."
        return summary

    keep_paths: set[Path] = set()
    for key in ("best", "last"):
        path_value = summary[key].get("path")
        if not path_value:
            continue
        candidate = Path(path_value)
        if not candidate.is_absolute():
            candidate = run_dir / candidate
        if candidate.exists():
            keep_paths.add(candidate.resolve())

    removed: list[str] = []
    for checkpoint in find_checkpoint_dirs(run_dir):
        if checkpoint.resolve() in keep_paths:
            continue
        shutil.rmtree(checkpoint)
        removed.append(_relative_or_absolute(checkpoint, run_dir) or str(checkpoint))

    summary["removed"] = removed
    summary["available"] = [
        {
            "step": checkpoint_step(checkpoint),
            "path": _relative_or_absolute(checkpoint, run_dir),
        }
        for checkpoint in find_checkpoint_dirs(run_dir)
    ]
    return summary
