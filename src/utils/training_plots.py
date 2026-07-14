from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.utils.run_utils import write_json
from src.utils.training_history import load_trainer_state, metric_points


def _relative_or_absolute(path: Path, run_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(run_dir.resolve()))
    except ValueError:
        return str(path)


def _plot_metric(
    plt: Any,
    points: list[dict[str, Any]],
    metric_name: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> Path | None:
    filtered = [
        point
        for point in points
        if point.get("epoch") is not None and point.get(metric_name) is not None
    ]
    if not filtered:
        return None

    epochs = [point["epoch"] for point in filtered]
    values = [point[metric_name] for point in filtered]

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(epochs, values, marker="o", linewidth=1.8, markersize=3.5)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def generate_training_plots(run_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    state = load_trainer_state(run_dir)
    if state is None:
        return {
            "status": "skipped",
            "reason": "No trainer_state.json found in run directory or checkpoints.",
            "files": {},
        }

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    resolved_output_dir = output_dir or (run_dir / "plots")
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    plot_specs = [
        ("train_loss", "loss", "Training Loss", "Train Loss over Epoch"),
        ("eval_loss", "eval_loss", "Validation Loss", "Validation Loss over Epoch"),
        ("learning_rate", "learning_rate", "Learning Rate", "Learning Rate over Epoch"),
    ]
    files: dict[str, str] = {}
    skipped: dict[str, str] = {}
    for file_stem, metric_name, ylabel, title in plot_specs:
        output_path = resolved_output_dir / f"{file_stem}.png"
        generated = _plot_metric(
            plt=plt,
            points=metric_points(state, metric_name),
            metric_name=metric_name,
            ylabel=ylabel,
            title=title,
            output_path=output_path,
        )
        if generated is None:
            skipped[file_stem] = f"No {metric_name} points with epoch values found."
        else:
            files[file_stem] = _relative_or_absolute(generated, run_dir)

    summary = {
        "status": "completed",
        "output_dir": _relative_or_absolute(resolved_output_dir, run_dir),
        "files": files,
        "skipped": skipped,
    }
    write_json(resolved_output_dir / "plot_manifest.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate simple training plots from a Trainer run folder.")
    parser.add_argument("run_dir", type=Path, help="Run directory containing checkpoints/trainer_state.json.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional plot output directory. Defaults to <run_dir>/plots.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = generate_training_plots(args.run_dir, output_dir=args.output_dir)
    print(summary)
    return 0 if summary.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
