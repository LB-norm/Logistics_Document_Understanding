"""Thin adapters for calling the evaluator from training frameworks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from .evaluator import JsonEvaluator


def make_compute_metrics(
    *,
    decode_predictions: Callable[[Any], Sequence[Any]],
    decode_references: Callable[[Any], Sequence[Any]],
    evaluator: JsonEvaluator | None = None,
    prefix: str = "json_",
) -> Callable[[Any], dict[str, float]]:
    """Build a Hugging Face-compatible ``compute_metrics`` callable.

    Decoding remains model-specific: Donut can convert tag sequences with
    ``processor.token2json``, while another VLM may simply call ``json.loads``.
    The supplied callables receive the raw prediction and label arrays.
    """
    json_evaluator = evaluator or JsonEvaluator()

    def compute_metrics(eval_prediction: Any) -> dict[str, float]:
        if hasattr(eval_prediction, "predictions"):
            raw_predictions = eval_prediction.predictions
            raw_references = eval_prediction.label_ids
        else:
            raw_predictions, raw_references = eval_prediction[:2]
        predictions = list(decode_predictions(raw_predictions))
        references = list(decode_references(raw_references))
        return json_evaluator.evaluate_batch(
            predictions, references
        ).training_metrics(prefix=prefix)

    return compute_metrics
