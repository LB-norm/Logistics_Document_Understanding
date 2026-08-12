"""Load portable Qwen experiment and sequential queue definitions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONFIG_FORMAT_VERSION = 1
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _load_json_object(path: Path, kind: str) -> dict[str, Any]:
    resolved_path = path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"{kind} file not found: {resolved_path}")
    try:
        with resolved_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {resolved_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"{kind} must contain one top-level JSON object: {resolved_path}"
        )
    return payload


def _validate_format_version(payload: dict[str, Any], path: Path, kind: str) -> None:
    version = payload.get("format_version")
    if version != CONFIG_FORMAT_VERSION:
        raise ValueError(
            f"{kind} {path} uses format_version {version!r}; "
            f"expected {CONFIG_FORMAT_VERSION}."
        )


def _validate_name(value: Any, path: Path, kind: str) -> str:
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value):
        raise ValueError(
            f"{kind} name in {path} must contain only letters, digits, dots, "
            "underscores, or hyphens and must start with a letter or digit."
        )
    return value


@dataclass(frozen=True)
class QwenExperimentConfig:
    path: Path
    name: str
    description: str
    training: dict[str, Any]


def load_experiment_config(path: Path) -> QwenExperimentConfig:
    """Load one experiment without importing heavyweight training packages."""
    resolved_path = path.expanduser().resolve()
    payload = _load_json_object(resolved_path, "Experiment configuration")
    _validate_format_version(payload, resolved_path, "Experiment configuration")

    allowed_keys = {"format_version", "name", "description", "training"}
    unknown_keys = sorted(set(payload) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"Unknown experiment configuration field(s) in {resolved_path}: "
            + ", ".join(unknown_keys)
        )

    name = _validate_name(payload.get("name"), resolved_path, "Experiment")
    description = payload.get("description", "")
    if not isinstance(description, str):
        raise ValueError(f"Experiment description must be a string: {resolved_path}")

    training = payload.get("training")
    if not isinstance(training, dict):
        raise ValueError(
            f"Experiment training field must be a JSON object: {resolved_path}"
        )
    if "config" in training:
        raise ValueError(
            f"Experiment training settings cannot contain the reserved key 'config': {resolved_path}"
        )

    resolved_training = dict(training)
    resolved_training.setdefault("run_name", name)
    return QwenExperimentConfig(
        path=resolved_path,
        name=name,
        description=description,
        training=resolved_training,
    )


@dataclass(frozen=True)
class QwenQueueEntry:
    config_path: Path
    enabled: bool


@dataclass(frozen=True)
class QwenExperimentQueue:
    path: Path
    name: str
    description: str
    entries: list[QwenQueueEntry]


def load_experiment_queue(path: Path) -> QwenExperimentQueue:
    """Load a queue whose experiment paths are relative to the queue file."""
    resolved_path = path.expanduser().resolve()
    payload = _load_json_object(resolved_path, "Experiment queue")
    _validate_format_version(payload, resolved_path, "Experiment queue")

    allowed_keys = {"format_version", "name", "description", "experiments"}
    unknown_keys = sorted(set(payload) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"Unknown experiment queue field(s) in {resolved_path}: "
            + ", ".join(unknown_keys)
        )

    name = _validate_name(payload.get("name"), resolved_path, "Queue")
    description = payload.get("description", "")
    if not isinstance(description, str):
        raise ValueError(f"Queue description must be a string: {resolved_path}")

    raw_entries = payload.get("experiments")
    if not isinstance(raw_entries, list):
        raise ValueError(
            f"Queue experiments field must be a JSON array: {resolved_path}"
        )

    entries: list[QwenQueueEntry] = []
    seen_paths: set[Path] = set()
    for index, raw_entry in enumerate(raw_entries, start=1):
        if isinstance(raw_entry, str):
            config_value = raw_entry
            enabled = True
        elif isinstance(raw_entry, dict):
            unknown_entry_keys = sorted(set(raw_entry) - {"config", "enabled"})
            if unknown_entry_keys:
                raise ValueError(
                    f"Unknown field(s) in queue entry {index}: "
                    + ", ".join(unknown_entry_keys)
                )
            config_value = raw_entry.get("config")
            enabled = raw_entry.get("enabled", True)
        else:
            raise ValueError(f"Queue entry {index} must be a string or JSON object.")

        if not isinstance(config_value, str) or not config_value.strip():
            raise ValueError(f"Queue entry {index} requires a non-empty config path.")
        if not isinstance(enabled, bool):
            raise ValueError(f"Queue entry {index} enabled field must be boolean.")

        config_path = Path(config_value).expanduser()
        if not config_path.is_absolute():
            config_path = resolved_path.parent / config_path
        config_path = config_path.resolve()
        if config_path in seen_paths:
            raise ValueError(f"Duplicate experiment config in queue: {config_path}")
        seen_paths.add(config_path)
        entries.append(QwenQueueEntry(config_path=config_path, enabled=enabled))

    return QwenExperimentQueue(
        path=resolved_path,
        name=name,
        description=description,
        entries=entries,
    )
