"""Compare epoch validation losses at matching optimizer steps in Trainer states."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def compare_histories(reference: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    def evaluations(state):
        result = {}
        for item in state["log_history"]:
            if "eval_loss" in item:
                # Final evaluation after load_best_model_at_end can reuse the
                # last step while evaluating an earlier checkpoint's weights.
                # Preserve the first, epoch/step-boundary evaluation instead.
                result.setdefault(item["step"], item)
        return result
    old, new = evaluations(reference), evaluations(candidate)
    return [{"step": step, "epoch": old[step].get("epoch"),
             "standard_eval_loss": old[step]["eval_loss"],
             "chunked_eval_loss": new[step]["eval_loss"],
             "difference": new[step]["eval_loss"] - old[step]["eval_loss"]}
            for step in sorted(old.keys() & new.keys())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Old run or checkpoint's trainer_state.json")
    parser.add_argument("candidate", type=Path, help="New run or checkpoint's trainer_state.json")
    args = parser.parse_args()
    rows = compare_histories(json.loads(args.reference.read_text()), json.loads(args.candidate.read_text()))
    if not rows:
        parser.error("No validation losses at matching optimizer steps")
    print(json.dumps({"comparisons": rows,
                      "note": "Compare identical data, initialization, order, precision and scheduler horizons. Early epochs are a regression check, not proof of final quality."}, indent=2))


if __name__ == "__main__":
    main()
