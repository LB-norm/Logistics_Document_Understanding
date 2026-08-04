"""Backward-compatible entry point for the Qwen fine-tuning pipeline.

New runs should use ``run_qwen_training.py``.  This wrapper deliberately uses
the same launcher defaults so older commands continue to behave consistently.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.Qwen.run_qwen_training import main


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
