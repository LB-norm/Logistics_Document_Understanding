from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any


def json_safe(value: Any) -> Any:
    """Convert common Python objects into JSON-serializable values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return value


def namespace_to_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {key: json_safe(value) for key, value in vars(args).items()}


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return slug or "run"


def make_run_name(
    pipeline_name: str,
    dataset_name: str | None,
    model_id: str | None,
    created_at: datetime,
) -> str:
    parts = [
        created_at.strftime("%Y%m%d_%H%M%S"),
        slugify(pipeline_name),
    ]
    if dataset_name:
        parts.append(slugify(dataset_name))
    if model_id:
        parts.append(slugify(model_id))
    return "_".join(parts)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2)


def normalize_trainer_metrics(metrics: dict[str, Any], stage: str) -> dict[str, Any]:
    """Keep raw Trainer metrics and expose stable cross-pipeline scalar names."""
    raw = json_safe(metrics)
    prefix = f"{stage}_"
    normalized: dict[str, Any] = {}
    key_map = {
        "runtime": "runtime_seconds",
        "samples_per_second": "samples_per_second",
        "steps_per_second": "steps_per_second",
        "loss": "loss",
        "epoch": "epoch",
        "total_flos": "total_flos",
    }

    if isinstance(raw, dict):
        for key, value in raw.items():
            normalized_key = key.removeprefix(prefix)
            normalized_key = key_map.get(normalized_key, normalized_key)
            if isinstance(value, int | float | str | bool) or value is None:
                normalized[normalized_key] = value

    return {
        "stage": stage,
        "raw": raw,
        "normalized": normalized,
    }


@dataclass
class RunContext:
    pipeline_name: str
    output_dir: Path
    run_name: str
    started_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    timer_started: float = field(default_factory=perf_counter)

    @classmethod
    def create(
        cls,
        pipeline_name: str,
        runs_dir: Path,
        output_dir: Path | None = None,
        run_name: str | None = None,
        dataset_name: str | None = None,
        model_id: str | None = None,
    ) -> "RunContext":
        started_at = datetime.now(timezone.utc)
        resolved_run_name = run_name or make_run_name(
            pipeline_name=pipeline_name,
            dataset_name=dataset_name,
            model_id=model_id,
            created_at=started_at,
        )
        resolved_output_dir = output_dir.resolve() if output_dir is not None else (runs_dir / resolved_run_name).resolve()
        return cls(
            pipeline_name=pipeline_name,
            output_dir=resolved_output_dir,
            run_name=resolved_run_name,
            started_at_utc=started_at,
        )

    @property
    def metadata_path(self) -> Path:
        return self.output_dir / "run_metadata.json"

    def elapsed_seconds(self) -> float:
        return perf_counter() - self.timer_started

    def base_metadata(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline_name,
            "run_name": self.run_name,
            "output_dir": str(self.output_dir),
            "started_at_utc": self.started_at_utc.isoformat(),
        }

    def write_status(
        self,
        status: str,
        sections: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self.base_metadata()
        payload["status"] = status
        if sections:
            payload.update(json_safe(sections))
        if metrics:
            payload["metrics"] = json_safe(metrics)
        if status in {"completed", "failed"}:
            payload["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
            payload["elapsed_seconds"] = self.elapsed_seconds()
        write_json(self.metadata_path, payload)
        return payload
