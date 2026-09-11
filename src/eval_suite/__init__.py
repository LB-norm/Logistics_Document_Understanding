"""Focused evaluation tools for structured logistics-document extraction."""

from .evaluator import (
    EvaluationReport,
    FieldCounts,
    JsonEvaluator,
    SampleEvaluation,
    TEST_SUBSETS,
    TestsetEvaluationReport,
    evaluate_batch,
    evaluate_json,
    evaluate_testset,
)
from .integrations import make_compute_metrics
from .normalization import NormalizationConfig
from .paths import DEFAULT_TESTSET_PATH
from .schema import validate_json_schema

__all__ = [
    "EvaluationReport",
    "DEFAULT_TESTSET_PATH",
    "FieldCounts",
    "JsonEvaluator",
    "NormalizationConfig",
    "SampleEvaluation",
    "TEST_SUBSETS",
    "TestsetEvaluationReport",
    "evaluate_batch",
    "evaluate_json",
    "evaluate_testset",
    "make_compute_metrics",
    "validate_json_schema",
]
