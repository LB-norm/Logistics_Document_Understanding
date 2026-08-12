"""Start Qwen QLoRA fine-tuning with project and GPU-friendly defaults.

Edit ``DEFAULT_TRAINING_CONFIG`` for the normal experiment configuration.
Explicit command-line options always override values in this mapping, which
keeps the launcher convenient locally while the underlying logic remains
reusable on larger machines and with larger Qwen checkpoints.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.Qwen.qwen_finetune_logic import main as run_training


DEFAULT_TRAINING_CONFIG: dict[str, Any] = {
    "dataset_root": REPO_ROOT / "data" / "datasets" / "250_CMRS_240dpi_20260707",
    "model_id": "Qwen/Qwen3.5-2B",
    "model_class": "auto",
    "local_files_only": False,
    "schema_path": REPO_ROOT / "json_schema" / "content.schema.json",
    "annotation_target_key": "content",
    # 1,048,576 pixels is roughly a 1024 x 1024 image budget. The processor
    # preserves aspect ratio, so portrait documents do not become square.
    "max_pixels": 1024 * 1024,
    "max_length": 2048,
    "load_in_4bit": True,
    "compute_dtype": "bfloat16",
    "bnb_4bit_quant_type": "nf4",
    "gradient_checkpointing": True,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "num_train_epochs": 10.0,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": "all-linear",
    "modules_to_save": "",
    # Change to "lora" to train adapters in the vision tower too. "full" is
    # intended for larger hardware and must be combined with 4-bit disabled.
    "vision_tuning": "frozen",
    "vision_module_names": "visual,vision_tower,vision_model",
    "optim": "auto",
    "attn_implementation": "sdpa",
    # Evaluate and checkpoint once per epoch. Trainer tracks the lowest
    # validation loss and retains both that checkpoint and the final one.
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "eval_steps": 50,
    "save_steps": 50,
    "logging_steps": 50,
    # Autoregressively generate the same two validation documents whenever a
    # training loss is logged. Set samples to 0 when profiling raw throughput.
    "validation_preview_samples": 2,
    "validation_preview_max_new_tokens": 1024,
    "save_total_limit": 2,
    "dataloader_num_workers": 0,
    "seed": 42,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Run launcher defaults, allowing explicit CLI options to override them."""
    return run_training(argv=argv, defaults=DEFAULT_TRAINING_CONFIG)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
