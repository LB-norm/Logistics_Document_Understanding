"""Model-agnostic comparison of predicted and annotated JSON documents."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

from .normalization import (
    NormalizationConfig,
    is_empty_value,
    normalized_edit_similarity,
    values_equal,
)
from .schema import validate_json_schema


_MISSING = object()
_ARRAY_INDEX = re.compile(r"\[\d+\]")


def _flatten_leaves(value: Any, path: str = "$") -> dict[str, Any]:
    if isinstance(value, dict):
        leaves: dict[str, Any] = {}
        for key, child in value.items():
            leaves.update(_flatten_leaves(child, f"{path}.{key}"))
        return leaves
    if isinstance(value, list):
        leaves = {}
        for index, child in enumerate(value):
            leaves.update(_flatten_leaves(child, f"{path}[{index}]"))
        return leaves
    return {path: value}


def _field_name(path: str) -> str:
    return _ARRAY_INDEX.sub("[]", path).removeprefix("$.")


def _strict_json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _strict_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _strict_json_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return left == right


def _safe_ratio(numerator: int | float, denominator: int | float, *, empty: float) -> float:
    return numerator / denominator if denominator else empty


@dataclass
class FieldCounts:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    value_similarity_sum: float = 0.0
    value_similarity_support: int = 0
    empty_ground_truth: int = 0
    empty_correct: int = 0
    hallucinated_values: int = 0

    def add(self, other: "FieldCounts") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def metrics(self) -> dict[str, Any]:
        precision = _safe_ratio(
            self.true_positives,
            self.true_positives + self.false_positives,
            empty=1.0,
        )
        recall = _safe_ratio(
            self.true_positives,
            self.true_positives + self.false_negatives,
            empty=1.0,
        )
        f1 = _safe_ratio(2 * precision * recall, precision + recall, empty=0.0)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "value_similarity": _safe_ratio(
                self.value_similarity_sum,
                self.value_similarity_support,
                empty=1.0,
            ),
            "support": self.true_positives + self.false_negatives,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "empty_ground_truth": self.empty_ground_truth,
            "empty_correct": self.empty_correct,
            "hallucinated_values": self.hallucinated_values,
        }


@dataclass
class SampleEvaluation:
    sample_id: str | None
    parse_valid: bool
    schema_valid: bool | None
    document_exact_match: bool
    field_counts: FieldCounts
    field_breakdown: dict[str, FieldCounts] = field(default_factory=dict)
    parse_error: str | None = None
    schema_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "parse_valid": self.parse_valid,
            "schema_valid": self.schema_valid,
            "document_exact_match": self.document_exact_match,
            "metrics": self.field_counts.metrics(),
            "parse_error": self.parse_error,
            "schema_errors": self.schema_errors,
        }


@dataclass
class EvaluationReport:
    samples: list[SampleEvaluation]

    def summary(self) -> dict[str, Any]:
        total = len(self.samples)
        counts = FieldCounts()
        schema_samples = [sample for sample in self.samples if sample.schema_valid is not None]
        for sample in self.samples:
            counts.add(sample.field_counts)
        metrics = counts.metrics()
        return {
            "samples": total,
            "parse_rate": _safe_ratio(
                sum(sample.parse_valid for sample in self.samples), total, empty=0.0
            ),
            "schema_valid_rate": (
                _safe_ratio(
                    sum(bool(sample.schema_valid) for sample in schema_samples),
                    len(schema_samples),
                    empty=0.0,
                )
                if schema_samples
                else None
            ),
            "document_exact_match_rate": _safe_ratio(
                sum(sample.document_exact_match for sample in self.samples),
                total,
                empty=0.0,
            ),
            "field_precision": metrics["precision"],
            "field_recall": metrics["recall"],
            "field_f1": metrics["f1"],
            "value_similarity": metrics["value_similarity"],
            "field_counts": {
                key: value
                for key, value in asdict(counts).items()
                if key not in {"value_similarity_sum", "value_similarity_support"}
            },
        }

    def field_breakdown(self) -> dict[str, dict[str, Any]]:
        aggregate: dict[str, FieldCounts] = {}
        for sample in self.samples:
            for path, counts in sample.field_breakdown.items():
                aggregate.setdefault(path, FieldCounts()).add(counts)
        return {
            path: counts.metrics()
            for path, counts in sorted(aggregate.items())
        }

    def training_metrics(self, prefix: str = "json_") -> dict[str, float]:
        """Return the flat numeric subset expected by training frameworks."""
        summary = self.summary()
        names = (
            "parse_rate",
            "schema_valid_rate",
            "document_exact_match_rate",
            "field_precision",
            "field_recall",
            "field_f1",
            "value_similarity",
        )
        return {
            f"{prefix}{name}": float(summary[name])
            for name in names
            if summary[name] is not None
        }

    def to_dict(self, *, include_samples: bool = True) -> dict[str, Any]:
        result = {
            "summary": self.summary(),
            "field_breakdown": self.field_breakdown(),
        }
        if include_samples:
            result["samples"] = [sample.to_dict() for sample in self.samples]
        return result


class JsonEvaluator:
    """Evaluate model output against ground-truth JSON with focused metrics."""

    def __init__(
        self,
        *,
        schema: Any | None = None,
        normalization: NormalizationConfig | None = None,
    ) -> None:
        self.schema = schema
        self.normalization = normalization or NormalizationConfig()

    @staticmethod
    def _parse_prediction(prediction: Any) -> tuple[Any, bool, str | None]:
        if not isinstance(prediction, str):
            return prediction, True, None
        try:
            return json.loads(prediction), True, None
        except (json.JSONDecodeError, TypeError) as exc:
            return None, False, f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _parse_ground_truth(ground_truth: Any) -> Any:
        if not isinstance(ground_truth, str):
            return ground_truth
        try:
            return json.loads(ground_truth)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"Ground truth is not valid JSON: {exc}") from exc

    def _compare_fields(
        self, expected: Any, predicted: Any, *, prediction_valid: bool
    ) -> tuple[FieldCounts, dict[str, FieldCounts]]:
        expected_leaves = _flatten_leaves(expected)
        predicted_leaves = _flatten_leaves(predicted) if prediction_valid else {}
        total = FieldCounts()
        breakdown: dict[str, FieldCounts] = {}

        for path in expected_leaves.keys() | predicted_leaves.keys():
            expected_value = expected_leaves.get(path, _MISSING)
            predicted_value = predicted_leaves.get(path, _MISSING)
            counts = FieldCounts()
            expected_empty = expected_value is _MISSING or is_empty_value(
                expected_value, self.normalization
            )
            predicted_empty = predicted_value is _MISSING or is_empty_value(
                predicted_value, self.normalization
            )

            if expected_value is not _MISSING and expected_empty:
                counts.empty_ground_truth += 1
                if predicted_empty:
                    counts.empty_correct += 1
                else:
                    counts.false_positives += 1
                    counts.hallucinated_values += 1
            elif expected_value is not _MISSING:
                counts.value_similarity_support += 1
                if predicted_empty:
                    counts.false_negatives += 1
                else:
                    similarity = normalized_edit_similarity(
                        expected_value, predicted_value, self.normalization
                    )
                    counts.value_similarity_sum += similarity
                    if values_equal(expected_value, predicted_value, self.normalization):
                        counts.true_positives += 1
                    else:
                        counts.false_positives += 1
                        counts.false_negatives += 1
            elif not predicted_empty:
                counts.false_positives += 1
                counts.hallucinated_values += 1

            total.add(counts)
            breakdown.setdefault(_field_name(path), FieldCounts()).add(counts)
        return total, breakdown

    def evaluate(
        self,
        prediction: Any,
        ground_truth: Any,
        *,
        sample_id: str | None = None,
    ) -> SampleEvaluation:
        expected = self._parse_ground_truth(ground_truth)
        predicted, parse_valid, parse_error = self._parse_prediction(prediction)
        schema_errors = (
            validate_json_schema(predicted, self.schema)
            if parse_valid and self.schema is not None
            else []
        )
        schema_valid = (
            parse_valid and not schema_errors if self.schema is not None else None
        )
        counts, breakdown = self._compare_fields(
            expected, predicted, prediction_valid=parse_valid
        )
        return SampleEvaluation(
            sample_id=sample_id,
            parse_valid=parse_valid,
            schema_valid=schema_valid,
            document_exact_match=parse_valid and _strict_json_equal(predicted, expected),
            field_counts=counts,
            field_breakdown=breakdown,
            parse_error=parse_error,
            schema_errors=schema_errors,
        )

    def evaluate_batch(
        self,
        predictions: Iterable[Any],
        ground_truths: Iterable[Any],
        *,
        sample_ids: Sequence[str] | None = None,
    ) -> EvaluationReport:
        prediction_list = list(predictions)
        ground_truth_list = list(ground_truths)
        if len(prediction_list) != len(ground_truth_list):
            raise ValueError("Predictions and ground truths must have the same length.")
        if sample_ids is not None and len(sample_ids) != len(prediction_list):
            raise ValueError("sample_ids must have the same length as predictions.")
        ids = sample_ids if sample_ids is not None else [None] * len(prediction_list)
        return EvaluationReport(
            [
                self.evaluate(prediction, truth, sample_id=sample_id)
                for prediction, truth, sample_id in zip(
                    prediction_list, ground_truth_list, ids
                )
            ]
        )


def evaluate_json(
    prediction: Any,
    ground_truth: Any,
    *,
    schema: Any | None = None,
    normalization: NormalizationConfig | None = None,
) -> dict[str, Any]:
    """Convenience function for evaluating one prediction."""
    evaluator = JsonEvaluator(schema=schema, normalization=normalization)
    return EvaluationReport([evaluator.evaluate(prediction, ground_truth)]).to_dict()


def evaluate_batch(
    predictions: Iterable[Any],
    ground_truths: Iterable[Any],
    *,
    sample_ids: Sequence[str] | None = None,
    schema: Any | None = None,
    normalization: NormalizationConfig | None = None,
) -> dict[str, Any]:
    """Convenience function for evaluating an in-memory batch."""
    evaluator = JsonEvaluator(schema=schema, normalization=normalization)
    return evaluator.evaluate_batch(
        predictions, ground_truths, sample_ids=sample_ids
    ).to_dict()
