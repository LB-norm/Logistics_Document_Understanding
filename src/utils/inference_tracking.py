"""Simple per-image wall time and process-local CUDA allocator peaks.

Create the tracker after model loading. Each call resets allocator peaks, but
keeps resident model memory and cached reservations in the reported footprint.
These counters belong to the process, so calls must run sequentially.
"""

from __future__ import annotations

import sys
from time import perf_counter
from typing import Any, Iterable


class InferenceTracker:
    def __init__(self, framework: Any = None, *, backend: str = "torch", device: Any = None):
        if backend not in {"torch", "paddle"}:
            raise ValueError(f"Unsupported inference backend: {backend}")
        # Models have already imported their framework; do not load a second one
        # just for tracking (or require it for injected test pipelines).
        self.framework = framework if framework is not None else sys.modules.get(backend)
        self.backend = backend
        self.device = device
        self.cuda = None
        self.devices: list[int] = []
        self.metrics: dict[str, Any] = {}

    def __enter__(self):
        device = str(self.device) if self.device is not None else None
        indices_only = device is not None and all(index.isdigit() for index in device.split(","))
        gpu = device is None or indices_only or device.startswith(("cuda", "gpu"))
        if self.framework is not None and gpu:
            if self.backend == "torch":
                self.cuda = getattr(self.framework, "cuda", None)
                available = self.cuda is not None and self.cuda.is_available()
            else:
                available = self.framework.is_compiled_with_cuda()
                self.cuda = self.framework.device.cuda if available else None
            if available:
                self.devices = list(range(self.cuda.device_count()))
                if device is not None and (":" in device or indices_only):
                    indices = device.split(":", 1)[-1]
                    self.devices = [int(index) for index in indices.split(",")]
        for index in self.devices:
            self.cuda.synchronize(index)
            if self.backend == "torch":
                self.cuda.reset_peak_memory_stats(index)
            else:
                self.cuda.reset_max_memory_allocated(index)
                self.cuda.reset_max_memory_reserved(index)
        self.started = perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Preserve the inference exception if synchronization also fails, e.g.
        # after a CUDA error. Missing measurements stay explicitly unavailable.
        error = None
        try:
            for index in self.devices:
                self.cuda.synchronize(index)
        except Exception as tracking_error:
            if exc_type is None:
                raise
            error = str(tracking_error)
        seconds = perf_counter() - self.started
        devices = []
        try:
            for index in self.devices:
                allocated = int(self.cuda.max_memory_allocated(index))
                reserved = int(self.cuda.max_memory_reserved(index))
                devices.append({
                    "device": f"{'cuda' if self.backend == 'torch' else 'gpu'}:{index}",
                    "peak_allocated_bytes": allocated,
                    "peak_reserved_bytes": reserved,
                    "peak_allocated_gib": allocated / 1024**3,
                    "peak_reserved_gib": reserved / 1024**3,
                })
        except Exception as tracking_error:
            if exc_type is None:
                raise
            error = str(tracking_error)
        self.metrics = {
            "inference_seconds": seconds,
            "vram": {
                "available": bool(devices),
                "source": f"{self.backend} CUDA allocator (current process only)",
                "scope": "per image, including resident model memory and cached reservations",
                "devices": devices,
                **({"error": error} if error else {}),
            },
        }
        return False


def aggregate_vram(summaries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Use the maximum page peak for each device, never sum sequential pages."""
    summaries = list(summaries)
    devices: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        for item in summary["devices"]:
            peak = devices.setdefault(item["device"], dict(item))
            for key in ("peak_allocated_bytes", "peak_reserved_bytes",
                        "peak_allocated_gib", "peak_reserved_gib"):
                peak[key] = max(peak[key], item[key])
    return {
        "available": bool(devices),
        "source": summaries[0]["source"] if summaries else "paddle CUDA allocator (current process only)",
        "scope": "maximum per-image peak across document pages",
        "devices": list(devices.values()),
    }
