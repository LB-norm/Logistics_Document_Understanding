"""Wall time and sampled process GPU memory across Paddle and PyTorch stages."""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
from threading import Event, Thread
from time import perf_counter
from typing import Any

from src.utils.inference_tracking import aggregate_vram


class PipelineTracker:
    """No allocator resets: existing stage trackers retain their own measurements.

    nvidia-smi process memory includes all frameworks and CUDA context overhead.
    The observed maximum is sampled and can miss allocations shorter than a poll.
    """

    def __init__(self, *, sample_gpu: bool = True, interval: float = 0.1) -> None:
        self.command = shutil.which("nvidia-smi") if sample_gpu else None
        self.interval = interval
        self.pid = os.getpid()
        self.peaks: dict[str, int] = {}
        self.samples = 0
        self.error: str | None = None
        self.stop = Event()
        self.thread: Thread | None = None
        self.metrics: dict[str, Any] = {}

    def _sample(self) -> bool:
        if self.command is None:
            return False
        try:
            result = subprocess.run(
                [self.command, "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=1, check=True,
            )
            current: dict[str, int] = {}
            for row in csv.reader(result.stdout.splitlines()):
                if len(row) != 3 or row[0].strip() != str(self.pid):
                    continue
                uuid = row[1].strip()
                try:
                    size = float(row[2].strip())
                    if size < 0:
                        continue
                    used = int(size * 1024**2)
                except (ValueError, OverflowError):
                    continue
                current[uuid] = current.get(uuid, 0) + used
            for uuid, used in current.items():
                self.peaks[uuid] = max(self.peaks.get(uuid, 0), used)
            self.samples += 1
            return True
        except (OSError, subprocess.SubprocessError) as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return False

    def _poll(self) -> None:
        while not self.stop.wait(self.interval):
            if not self._sample():
                break

    def __enter__(self) -> "PipelineTracker":
        # The initial snapshot captures resident model memory before inference.
        enabled = self._sample()
        self.started = perf_counter()
        if enabled:
            self.thread = Thread(target=self._poll, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        seconds = perf_counter() - self.started
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
            self._sample()
        self.metrics = {
            "inference_seconds": seconds,
            "vram": {
                "available": bool(self.peaks),
                "combined_peak_available": bool(self.peaks),
                "source": "nvidia-smi sampled GPU memory for the current process",
                "scope": "per image, all resident models, framework caches and CUDA contexts",
                "sample_interval_seconds": self.interval,
                "sample_count": self.samples,
                "devices": [{"device": uuid, "peak_used_bytes": used, "peak_used_gib": used / 1024**3}
                            for uuid, used in sorted(self.peaks.items())],
                **({"sampling_error": self.error} if self.error else {}),
            },
        }
        return False


def attach_framework_peaks(metrics: dict[str, Any], stages: dict[str, Any], *, parser: bool) -> None:
    """Retain allocator measurements without summing peaks from different times."""
    frameworks = {}
    for backend in ("torch", "paddle"):
        summaries = [stage["vram"] for stage in stages.values()
                     if stage.get("backend") == backend and "vram" in stage]
        if summaries:
            frameworks[backend] = aggregate_vram(summaries)
            frameworks[backend]["scope"] = "maximum recorded stage allocator peak per image"
    metrics["vram"]["framework_peaks"] = frameworks
    # For end-to-end inference a single framework's existing allocator peak is sufficient.
    # Parser inference needs process sampling to measure Paddle + PyTorch together.
    if not parser and not metrics["vram"]["available"] and "torch" in frameworks:
        fallback = frameworks["torch"]
        if fallback["available"]:
            metrics["vram"].update(available=True, source=fallback["source"],
                                   scope=fallback["scope"], devices=fallback["devices"])
