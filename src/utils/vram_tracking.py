"""Persist process-local PyTorch CUDA allocator peaks throughout a training run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.run_utils import write_json


class PeakVramTracker:
    """Reset once before model loading; preserve peaks across training/evaluation."""

    def __init__(self, torch: Any, output_dir: Path) -> None:
        self.cuda = torch.cuda
        self.output_path = output_dir / "vram_usage.json"
        self.devices = list(range(self.cuda.device_count())) if self.cuda.is_available() else []
        for device in self.devices:
            self.cuda.reset_peak_memory_stats(device)

    def record(self, phase: str, step: int | None = None) -> dict[str, Any]:
        devices = []
        for device in self.devices:
            allocated = self.cuda.max_memory_allocated(device)
            reserved = self.cuda.max_memory_reserved(device)
            devices.append(
                {
                    "device": f"cuda:{device}",
                    "name": self.cuda.get_device_name(device),
                    "peak_allocated_bytes": allocated,
                    "peak_reserved_bytes": reserved,
                    "peak_allocated_gib": allocated / 1024**3,
                    "peak_reserved_gib": reserved / 1024**3,
                }
            )
        summary = {
            "available": bool(self.devices),
            "source": "torch.cuda allocator (current process only)",
            "scope": "since model loading, including training, validation and previews",
            "phase": phase,
            "global_step": step,
            "devices": devices,
        }
        write_json(self.output_path, summary)
        for device in devices:
            print(
                f"Peak VRAM [{phase}, step {step}] {device['device']}: "
                f"{device['peak_allocated_gib']:.3f} GiB allocated, "
                f"{device['peak_reserved_gib']:.3f} GiB reserved",
                flush=True,
            )
        return summary


def build_peak_vram_callback(TrainerCallback: Any, tracker: PeakVramTracker) -> Any:
    class PeakVramCallback(TrainerCallback):
        def on_log(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                tracker.record("training", state.global_step)

        def on_evaluate(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                tracker.record("evaluation", state.global_step)

        def on_train_end(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                tracker.record("training_finished", state.global_step)

    return PeakVramCallback()
