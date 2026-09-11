from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TESTSET_PATH = (
    REPO_ROOT / "data" / "datasets" / "250_CMRS_240dpi_20260707" / "test"
)
