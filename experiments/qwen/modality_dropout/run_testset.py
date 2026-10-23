"""Infer stage-two dropout models with both modalities on validation/test images."""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.qwen.hybrid.run_testset import main


if __name__ == "__main__":
    raise SystemExit(main(defaults={
        "runs_dir": Path("runs/qwen-modality-dropout"),
        "output_dir": Path("output/qwen-modality-dropout"),
    }))
