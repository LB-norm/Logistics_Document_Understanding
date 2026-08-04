"""Conservative value normalization for structured document evaluation."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any


_WHITESPACE = re.compile(r"\s+")
_SAFE_NUMBER = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?")


@dataclass(frozen=True)
class NormalizationConfig:
    """Controls which harmless representation differences are ignored.

    The defaults intentionally do not remove punctuation or leading zeroes:
    both can be meaningful in references, postal codes, and license plates.
    """

    unicode_form: str | None = "NFKC"
    collapse_whitespace: bool = True
    case_sensitive: bool = False
    coerce_safe_numeric_strings: bool = True


def normalize_string(value: str, config: NormalizationConfig) -> str:
    if config.unicode_form:
        value = unicodedata.normalize(config.unicode_form, value)
    value = value.strip()
    if config.collapse_whitespace:
        value = _WHITESPACE.sub(" ", value)
    if not config.case_sensitive:
        value = value.casefold()
    return value


def _canonical_decimal(value: int | float | Decimal | str) -> str | None:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if not number.is_finite():
        return None
    normalized = number.normalize()
    if normalized == 0:
        normalized = Decimal(0)
    return format(normalized, "f")


def canonical_scalar(value: Any, config: NormalizationConfig) -> tuple[str, str]:
    """Return a typed, comparable representation of a JSON scalar."""
    if value is None:
        return ("null", "")
    if isinstance(value, bool):
        return ("bool", "true" if value else "false")
    if isinstance(value, (int, float, Decimal)):
        number = _canonical_decimal(value)
        return ("number", number if number is not None else str(value))
    if isinstance(value, str):
        normalized = normalize_string(value, config)
        if config.coerce_safe_numeric_strings and _SAFE_NUMBER.fullmatch(normalized):
            number = _canonical_decimal(normalized)
            if number is not None:
                return ("number", number)
        return ("string", normalized)
    return (type(value).__name__, str(value))


def is_empty_value(value: Any, config: NormalizationConfig) -> bool:
    """Treat null and blank strings as unpopulated fields."""
    if value is None:
        return True
    return isinstance(value, str) and not normalize_string(value, config)


def values_equal(left: Any, right: Any, config: NormalizationConfig) -> bool:
    return canonical_scalar(left, config) == canonical_scalar(right, config)


def normalized_edit_similarity(
    expected: Any,
    predicted: Any,
    config: NormalizationConfig,
) -> float:
    """Compute 1 - normalized Levenshtein distance for two scalar values."""
    expected_value = canonical_scalar(expected, config)
    predicted_value = canonical_scalar(predicted, config)
    if expected_value == predicted_value:
        return 1.0

    left = expected_value[1]
    right = predicted_value[1]
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0

    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            substitution = previous[right_index - 1] + (
                left_character != right_character
            )
            current.append(
                min(
                    current[right_index - 1] + 1,
                    previous[right_index] + 1,
                    substitution,
                )
            )
        previous = current
    distance = previous[-1]
    return max(0.0, 1.0 - distance / max(len(left), len(right)))
