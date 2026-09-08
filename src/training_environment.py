"""Shared process-level defaults for GPU training launchers."""

from __future__ import annotations

import os
from collections.abc import MutableMapping


DEFAULT_PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"


def configure_pytorch_cuda_allocator(
    environment: MutableMapping[str, str] | None = None,
) -> str:
    """Enable expandable CUDA segments unless the caller supplied a policy."""
    target = os.environ if environment is None else environment
    return target.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        DEFAULT_PYTORCH_CUDA_ALLOC_CONF,
    )
