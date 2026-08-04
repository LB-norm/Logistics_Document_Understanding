"""Start Donut fine-tuning with the project's preferred defaults.

Edit ``DEFAULT_TRAINING_CONFIG`` to change the normal training setup. Any
arguments supplied on the command line override the values below.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.Donut.donut_train_logic import main as run_training


DEFAULT_TRAINING_CONFIG: dict[str, Any] = {
    "dataset_root": REPO_ROOT / "data" / "datasets" / "250_CMRS_240dpi_20260707",
    "model_id": "naver-clova-ix/donut-base",        #Start from base model
    "local_files_only": True,
    "task_start_token": "<s_lieferschein>",
    "schema_path": REPO_ROOT / "json_schema" / "content.schema.json",
    "target_skeleton_path": REPO_ROOT / "json_schema" / "content.empty.json",
    "image_size": (1920, 1280),
    "max_length": 1024,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "gradient_checkpointing": True,
    "num_train_epochs": 50.0,
    "learning_rate": 3e-5,
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "eval_steps": 50,       #Every 2 epochs (8*50 = 400; 200 images in the Trainset)
    "save_steps": 50,       #Every 2 epochs
    "save_total_limit": 2,
    "logging_steps": 50,    #Every 2 epochs
    "validation_preview_samples": 2,
    "dataloader_num_workers": 0,
    "seed": 42,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Run training defaults, allowing explicit CLI arguments to override them."""
    return run_training(argv=argv, defaults=DEFAULT_TRAINING_CONFIG)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
