from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import quote


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "datasets" / "250_CMRS_240dpi_20260707"
DEFAULT_SCHEMA_PATH = REPO_ROOT / "json_schema" / "content.schema.json"
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class FlatValue:
    path: str
    canonical_path: str
    value: Any


@dataclass
class Sample:
    sample_id: str
    split: str
    image_path: Path
    annotation_path: Path
    content: dict[str, Any]
    annotation: dict[str, Any]
    flat_values: list[FlatValue] = field(default_factory=list)
    issues: list["Issue"] = field(default_factory=list)
    peers: list[tuple[str, float, list[str]]] = field(default_factory=list)


@dataclass(frozen=True)
class Issue:
    sample_id: str
    split: str
    severity: int
    code: str
    field: str
    value: Any
    message: str
    evidence: str = ""
    peer_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class LearnedRule:
    role: str
    anchor_field: str
    anchor_value: str
    dependent_field: str
    expected_value: str
    support: int
    anchor_support: int
    confidence: float
    exception_ids: tuple[str, ...]


@dataclass(frozen=True)
class AuditResult:
    samples: list[Sample]
    load_errors: list[str]
    learned_rules: list[LearnedRule]
    duplicate_image_groups: int
    duplicate_annotation_groups: int


ENTITY_SPECS: dict[str, dict[str, str]] = {
    "sender": {
        "company": "senderInformation.senderNameCompany",
        "street": "senderInformation.senderStreet",
        "postcode": "senderInformation.senderPostcode",
        "city": "senderInformation.senderCity",
        "country": "senderInformation.senderCountryCode.value",
    },
    "consignee": {
        "company": "consigneeInformation.consigneeNameCompany",
        "street": "consigneeInformation.consigneeStreet",
        "postcode": "consigneeInformation.consigneePostcode",
        "city": "consigneeInformation.consigneeCity",
        "country": "consigneeInformation.consigneeCountryCode.value",
    },
    "carrier": {
        "company": "carrierInformation.carrierNameCompany",
        "street": "carrierInformation.carrierStreet",
        "postcode": "carrierInformation.carrierPostcode",
        "city": "carrierInformation.carrierCity",
        "country": "carrierInformation.carrierCountryCode.value",
    },
    "successive_carrier": {
        "company": "successiveCarrierInformation.successiveCarrierNameCompany",
        "street": "successiveCarrierInformation.successiveCarrierStreet",
        "postcode": "successiveCarrierInformation.successiveCarrierPostcode",
        "city": "successiveCarrierInformation.successiveCarrierCity",
        "country": "successiveCarrierInformation.successiveCarrierCountryCode.value",
    },
}


INTRA_DOCUMENT_COMPARISONS: tuple[tuple[str, str, str, int], ...] = (
    (
        "consigneeInformation.consigneeCity",
        "deliveryOfTheGoods.logisticsLocationCity",
        "The consignee city and delivery city normally refer to the same destination.",
        58,
    ),
    (
        "consigneeInformation.consigneeNameCompany",
        "goodsReceived.consigneeSignature.userCompany",
        "The consignee name conflicts with the goods-received stamp.",
        50,
    ),
    (
        "consigneeInformation.consigneePostcode",
        "goodsReceived.consigneeSignature.userPostCode",
        "The consignee postcode conflicts with the goods-received stamp.",
        58,
    ),
    (
        "consigneeInformation.consigneeCity",
        "goodsReceived.consigneeSignature.userCity",
        "The consignee city conflicts with the goods-received stamp.",
        62,
    ),
    (
        "consigneeInformation.consigneeCountryCode.value",
        "goodsReceived.consigneeSignature.userCountry",
        "The consignee country conflicts with the goods-received stamp.",
        58,
    ),
    (
        "senderInformation.senderNameCompany",
        "signatureOrStampOfTheSender.senderSignature.userCompany",
        "The sender name differs from the sender stamp (this can be legitimate when an agent signs).",
        28,
    ),
    (
        "senderInformation.senderPostcode",
        "signatureOrStampOfTheSender.senderSignature.userPostCode",
        "The sender postcode differs from the sender stamp (this can be legitimate when an agent signs).",
        34,
    ),
    (
        "senderInformation.senderCity",
        "signatureOrStampOfTheSender.senderSignature.userCity",
        "The sender city differs from the sender stamp (this can be legitimate when an agent signs).",
        34,
    ),
    (
        "carrierInformation.carrierNameCompany",
        "signatureOrStampOfTheCarrier.carrierSignature.userCompany",
        "The carrier name conflicts with the carrier stamp.",
        48,
    ),
    (
        "carrierInformation.carrierPostcode",
        "signatureOrStampOfTheCarrier.carrierSignature.userPostCode",
        "The carrier postcode conflicts with the carrier stamp.",
        55,
    ),
    (
        "carrierInformation.carrierCity",
        "signatureOrStampOfTheCarrier.carrierSignature.userCity",
        "The carrier city conflicts with the carrier stamp.",
        55,
    ),
)


PEER_FIELDS: tuple[tuple[str, str], ...] = (
    ("Sender", "senderInformation.senderNameCompany"),
    ("Sender postcode", "senderInformation.senderPostcode"),
    ("Sender city", "senderInformation.senderCity"),
    ("Consignee", "consigneeInformation.consigneeNameCompany"),
    ("Consignee postcode", "consigneeInformation.consigneePostcode"),
    ("Consignee city", "consigneeInformation.consigneeCity"),
    ("Pickup", "takingOverTheGoods.takingOverTheGoodsPlace"),
    ("Delivery", "deliveryOfTheGoods.logisticsLocationCity"),
    ("Carrier", "carrierInformation.carrierNameCompany"),
    ("Reference", "referenceIdentificationNumber"),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit JSON annotations by combining schema checks, semantic checks, "
            "learned cross-document consistency rules, duplicate checks, and peer comparison."
        )
    )
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Dataset root (default: {DEFAULT_DATASET_ROOT.relative_to(REPO_ROOT)}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: <dataset_root>/annotation_audit).",
    )
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
        help="JSON Schema for annotation['content']; pass an empty string only by editing the code.",
    )
    parser.add_argument(
        "--min-rule-support",
        type=int,
        default=3,
        help="Minimum repeated anchor occurrences before learning a consistency rule.",
    )
    parser.add_argument(
        "--rule-confidence",
        type=float,
        default=0.80,
        help="Minimum dominant-value fraction for learned consistency rules.",
    )
    parser.add_argument(
        "--high-risk-threshold",
        type=int,
        default=70,
        help="Document risk score considered high risk in summaries.",
    )
    parser.add_argument(
        "--fail-on-high-risk",
        action="store_true",
        help="Return exit code 2 when at least one document is high risk.",
    )
    args = parser.parse_args(argv)
    if args.min_rule_support < 2:
        parser.error("--min-rule-support must be at least 2")
    if not 0.5 <= args.rule_confidence <= 1.0:
        parser.error("--rule-confidence must be between 0.5 and 1.0")
    return args


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value).casefold())
    return "".join(character for character in text if character.isalnum())


def display_value(value: Any) -> str:
    if value is None:
        return "∅ (missing)"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def get_path(content: dict[str, Any], path: str) -> Any:
    current: Any = content
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def flatten_values(
    value: Any, path: str = "", canonical_path: str = ""
) -> list[FlatValue]:
    flattened: list[FlatValue] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            child_canonical = f"{canonical_path}.{key}" if canonical_path else key
            flattened.extend(flatten_values(child, child_path, child_canonical))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]"
            child_canonical = f"{canonical_path}[]"
            flattened.extend(flatten_values(child, child_path, child_canonical))
    else:
        flattened.append(FlatValue(path, canonical_path, value))
    return flattened


def resolve_dataset_path(dataset_root: Path, relative_path: str) -> Path:
    root = dataset_root.resolve()
    candidate = (root / Path(relative_path)).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Metadata path escapes dataset root: {relative_path}")
    return candidate


def find_relocated_file(dataset_root: Path, expected_path: Path) -> Path | None:
    """Find a uniquely relocated dataset file without guessing between duplicates."""
    matches = [
        candidate
        for candidate in dataset_root.rglob(expected_path.name)
        if candidate.is_file() and "annotation_audit" not in candidate.parts
    ]
    return matches[0].resolve() if len(matches) == 1 else None


def load_samples(dataset_root: Path) -> tuple[list[Sample], list[str]]:
    samples: list[Sample] = []
    errors: list[str] = []
    seen_ids: set[str] = set()
    for split in SPLIT_NAMES:
        metadata_path = dataset_root / split / "metadata.jsonl"
        if not metadata_path.exists():
            continue
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                location = f"{metadata_path}:{line_number}"
                try:
                    record = json.loads(line)
                    sample_id = str(record["id"])
                    image_path = resolve_dataset_path(dataset_root, str(record["image"]))
                    annotation_path = resolve_dataset_path(
                        dataset_root, str(record["annotation"])
                    )
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                    errors.append(f"{location}: invalid metadata row: {error}")
                    continue
                if sample_id in seen_ids:
                    errors.append(f"{location}: duplicate sample id {sample_id!r}")
                    continue
                seen_ids.add(sample_id)
                if not image_path.is_file():
                    relocated = find_relocated_file(dataset_root, image_path)
                    if relocated is None:
                        errors.append(f"{location}: missing image {image_path}")
                        continue
                    errors.append(
                        f"{location}: image was relocated outside its metadata path; using {relocated}"
                    )
                    image_path = relocated
                if not annotation_path.is_file():
                    relocated = find_relocated_file(dataset_root, annotation_path)
                    if relocated is None:
                        errors.append(f"{location}: missing annotation {annotation_path}")
                        continue
                    errors.append(
                        f"{location}: annotation was relocated outside its metadata path; using {relocated}"
                    )
                    annotation_path = relocated
                try:
                    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    errors.append(f"{annotation_path}: invalid JSON: {error}")
                    continue
                content = annotation.get("content") if isinstance(annotation, dict) else None
                if not isinstance(content, dict):
                    errors.append(f"{annotation_path}: annotation['content'] is not an object")
                    continue
                samples.append(
                    Sample(
                        sample_id=sample_id,
                        split=split,
                        image_path=image_path,
                        annotation_path=annotation_path,
                        content=content,
                        annotation=annotation,
                        flat_values=flatten_values(content),
                    )
                )
    if not samples:
        errors.append(f"No valid samples found under {dataset_root}")
    return samples, errors


def json_type_matches(value: Any, allowed: str | list[str]) -> bool:
    names = [allowed] if isinstance(allowed, str) else allowed
    for name in names:
        if name == "null" and value is None:
            return True
        if name == "object" and isinstance(value, dict):
            return True
        if name == "array" and isinstance(value, list):
            return True
        if name == "string" and isinstance(value, str):
            return True
        if name == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if name == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if name == "boolean" and isinstance(value, bool):
            return True
    return False


def resolve_schema_ref(root_schema: dict[str, Any], reference: str) -> dict[str, Any]:
    if not reference.startswith("#/"):
        raise ValueError(f"Only local schema references are supported, received {reference!r}")
    node: Any = root_schema
    for part in reference[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        node = node[part]
    if not isinstance(node, dict):
        raise ValueError(f"Schema reference does not point to an object: {reference}")
    return node


def validate_schema_node(
    value: Any,
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    path: str,
) -> list[tuple[str, str, str]]:
    if "$ref" in schema:
        schema = resolve_schema_ref(root_schema, schema["$ref"])
    errors: list[tuple[str, str, str]] = []
    allowed_type = schema.get("type")
    if allowed_type is not None and not json_type_matches(value, allowed_type):
        errors.append(
            (
                "schema_type",
                path or "<content>",
                f"Expected type {allowed_type!r}, received {type(value).__name__}.",
            )
        )
        return errors
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for required_key in schema.get("required", []):
            if required_key not in value:
                child_path = f"{path}.{required_key}" if path else required_key
                errors.append(
                    ("schema_missing_field", child_path, "Required field is absent.")
                )
        if schema.get("additionalProperties") is False:
            for unexpected_key in value.keys() - properties.keys():
                child_path = f"{path}.{unexpected_key}" if path else unexpected_key
                errors.append(
                    ("schema_unexpected_field", child_path, "Field is not defined by the schema.")
                )
        for key in value.keys() & properties.keys():
            child_path = f"{path}.{key}" if path else key
            errors.extend(
                validate_schema_node(value[key], properties[key], root_schema, child_path)
            )
    elif isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, child in enumerate(value):
            errors.extend(
                validate_schema_node(
                    child, schema["items"], root_schema, f"{path}[{index}]"
                )
            )
    return errors


def add_issue(sample: Sample, issue: Issue) -> None:
    key = (issue.code, issue.field, issue.message)
    existing_index = next(
        (
            index
            for index, current in enumerate(sample.issues)
            if (current.code, current.field, current.message) == key
        ),
        None,
    )
    if existing_index is None:
        sample.issues.append(issue)
    elif issue.severity > sample.issues[existing_index].severity:
        sample.issues[existing_index] = issue


def run_schema_checks(samples: list[Sample], schema_path: Path | None) -> list[str]:
    if schema_path is None:
        return []
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"Could not load schema {schema_path}: {error}"]
    for sample in samples:
        try:
            errors = validate_schema_node(sample.content, schema, schema, "")
        except (KeyError, TypeError, ValueError) as error:
            return [f"Could not evaluate schema {schema_path}: {error}"]
        for code, field_path, message in errors:
            add_issue(
                sample,
                Issue(
                    sample.sample_id,
                    sample.split,
                    88 if code == "schema_missing_field" else 82,
                    code,
                    field_path,
                    get_path(sample.content, field_path),
                    message,
                    f"Schema: {schema_path.name}",
                ),
            )
    return []


def parse_date_like(value: str) -> bool | None:
    text = value.strip()
    match = re.fullmatch(r"(\d{1,4})[./-](\d{1,2})[./-](\d{1,4})", text)
    if not match:
        return None
    first, second, third = match.groups()
    try:
        if len(first) == 4:
            year, month, day = int(first), int(second), int(third)
        else:
            day, month = int(first), int(second)
            year = int(third)
            if year < 100:
                year += 2000
        datetime(year, month, day)
    except ValueError:
        return False
    return True


def semantic_issue(
    sample: Sample,
    severity: int,
    code: str,
    flat: FlatValue,
    message: str,
    evidence: str = "",
) -> None:
    add_issue(
        sample,
        Issue(
            sample.sample_id,
            sample.split,
            severity,
            code,
            flat.path,
            flat.value,
            message,
            evidence,
        ),
    )


def run_semantic_checks(samples: list[Sample]) -> None:
    for sample in samples:
        for flat in sample.flat_values:
            value = flat.value
            lower_path = flat.canonical_path.casefold()
            if isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    semantic_issue(
                        sample,
                        42,
                        "blank_string",
                        flat,
                        "Blank strings are ambiguous; use null for an absent label.",
                    )
                    continue
                if stripped.casefold() in {"null", "none", "n/a", "nan", "unknown"}:
                    semantic_issue(
                        sample,
                        55,
                        "null_literal",
                        flat,
                        "A missing-value marker was stored as label text.",
                    )
                if "postcode" in lower_path:
                    digits = re.sub(r"\D", "", stripped)
                    if not digits:
                        semantic_issue(
                            sample,
                            68,
                            "invalid_postcode",
                            flat,
                            "Postcode contains no digits and may be assigned to the wrong field.",
                        )
                    elif len(stripped) > 14:
                        semantic_issue(
                            sample,
                            52,
                            "unusual_postcode_length",
                            flat,
                            "Postcode is unusually long.",
                        )
                if lower_path.endswith("city"):
                    if re.fullmatch(r"[\d\W_]+", stripped, flags=re.UNICODE):
                        semantic_issue(
                            sample,
                            72,
                            "invalid_city",
                            flat,
                            "City has no letters and may be a swapped label.",
                        )
                if "phone" in lower_path:
                    digits = re.sub(r"\D", "", stripped)
                    if len(digits) < 5:
                        semantic_issue(
                            sample,
                            50,
                            "invalid_phone",
                            flat,
                            "Phone number contains fewer than five digits.",
                        )
                if "date" in lower_path or lower_path.endswith(".data"):
                    date_state = parse_date_like(stripped)
                    if date_state is False:
                        semantic_issue(
                            sample,
                            76,
                            "impossible_date",
                            flat,
                            "Date has a recognizable structure but is not a valid calendar date.",
                        )
                    elif date_state is None and re.search(r"\d", stripped):
                        semantic_issue(
                            sample,
                            28,
                            "unusual_date_format",
                            flat,
                            "Date-like value uses an uncommon format; verify it against the image.",
                        )
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(float(value)):
                    semantic_issue(
                        sample, 85, "nonfinite_number", flat, "Numeric value is not finite."
                    )
                elif value < 0 and any(
                    token in lower_path
                    for token in ("quantity", "weight", "volume", "charge")
                ):
                    semantic_issue(
                        sample,
                        80,
                        "negative_measurement",
                        flat,
                        "Quantity, weight, volume, and charge labels should not be negative.",
                    )
                elif value == 0 and "quantity" in lower_path:
                    semantic_issue(
                        sample,
                        35,
                        "zero_package_quantity",
                        flat,
                        "A package row has quantity zero; this may be valid but deserves review.",
                    )


def values_compatible(left: Any, right: Any, path: str) -> tuple[bool, float]:
    left_normalized = normalize_text(left)
    right_normalized = normalize_text(right)
    if not left_normalized or not right_normalized:
        return True, 1.0
    if left_normalized == right_normalized:
        return True, 1.0
    ratio = SequenceMatcher(None, left_normalized, right_normalized).ratio()
    threshold = 0.52 if "Company" in path or "company" in path else 0.72
    contained = (
        min(len(left_normalized), len(right_normalized)) >= 3
        and (
            left_normalized in right_normalized or right_normalized in left_normalized
        )
    )
    return ratio >= threshold or contained, ratio


def run_intra_document_checks(samples: list[Sample]) -> None:
    for sample in samples:
        for left_path, right_path, message, severity in INTRA_DOCUMENT_COMPARISONS:
            left_value = get_path(sample.content, left_path)
            right_value = get_path(sample.content, right_path)
            if left_value is None or right_value is None:
                continue
            compatible, similarity = values_compatible(left_value, right_value, left_path)
            if compatible:
                continue
            add_issue(
                sample,
                Issue(
                    sample.sample_id,
                    sample.split,
                    severity,
                    "intra_document_conflict",
                    f"{left_path} ↔ {right_path}",
                    f"{display_value(left_value)} ↔ {display_value(right_value)}",
                    message,
                    f"Normalized text similarity: {similarity:.2f}",
                ),
            )


def learned_value_key(value: Any) -> str:
    if value is None:
        return "<MISSING>"
    normalized = normalize_text(value)
    return normalized if normalized else "<EMPTY>"


def learn_consistency_rules(
    samples: list[Sample], min_support: int, min_confidence: float
) -> list[LearnedRule]:
    rules: list[LearnedRule] = []
    sample_by_id = {sample.sample_id: sample for sample in samples}
    for role, fields in ENTITY_SPECS.items():
        relationships = [
            ("company", dependent)
            for dependent in ("street", "postcode", "city", "country")
        ] + [("postcode", dependent) for dependent in ("city", "country")]
        for anchor_name, dependent_name in relationships:
            anchor_path = fields[anchor_name]
            dependent_path = fields[dependent_name]
            grouped: dict[str, list[tuple[Sample, Any, Any]]] = defaultdict(list)
            for sample in samples:
                anchor_value = get_path(sample.content, anchor_path)
                if anchor_value is None or not normalize_text(anchor_value):
                    continue
                grouped[normalize_text(anchor_value)].append(
                    (sample, anchor_value, get_path(sample.content, dependent_path))
                )
            for occurrences in grouped.values():
                if len(occurrences) < min_support:
                    continue
                counts = Counter(learned_value_key(value) for _, _, value in occurrences)
                dominant_key, support = counts.most_common(1)[0]
                confidence = support / len(occurrences)
                if confidence < min_confidence or len(counts) == 1:
                    continue
                expected_occurrence = next(
                    occurrence
                    for occurrence in occurrences
                    if learned_value_key(occurrence[2]) == dominant_key
                )
                expected_value = expected_occurrence[2]
                anchor_display = str(occurrences[0][1])
                exception_ids = tuple(
                    occurrence[0].sample_id
                    for occurrence in occurrences
                    if learned_value_key(occurrence[2]) != dominant_key
                )
                rule = LearnedRule(
                    role=role,
                    anchor_field=anchor_path,
                    anchor_value=anchor_display,
                    dependent_field=dependent_path,
                    expected_value=display_value(expected_value),
                    support=support,
                    anchor_support=len(occurrences),
                    confidence=confidence,
                    exception_ids=exception_ids,
                )
                rules.append(rule)
                modal_peer_ids = tuple(
                    occurrence[0].sample_id
                    for occurrence in occurrences
                    if learned_value_key(occurrence[2]) == dominant_key
                )[:5]
                severity = min(
                    72,
                    round(38 + 22 * confidence + 4 * math.log2(len(occurrences))),
                )
                for exception_id in exception_ids:
                    sample = sample_by_id[exception_id]
                    actual_value = get_path(sample.content, dependent_path)
                    add_issue(
                        sample,
                        Issue(
                            sample.sample_id,
                            sample.split,
                            severity,
                            "learned_consistency_exception",
                            dependent_path,
                            actual_value,
                            (
                                f"For {role} {anchor_name} {anchor_display!r}, "
                                f"{support}/{len(occurrences)} annotations use "
                                f"{display_value(expected_value)!r}; this annotation differs."
                            ),
                            (
                                f"Learned rule confidence {confidence:.0%}; review peers before editing."
                            ),
                            modal_peer_ids,
                        ),
                    )
    return sorted(
        rules,
        key=lambda rule: (-rule.confidence, -rule.anchor_support, rule.role, rule.anchor_value),
    )


def percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot calculate percentile of an empty list")
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def run_numeric_outlier_checks(samples: list[Sample]) -> None:
    grouped: dict[str, list[tuple[Sample, FlatValue, float]]] = defaultdict(list)
    for sample in samples:
        for flat in sample.flat_values:
            if isinstance(flat.value, (int, float)) and not isinstance(flat.value, bool):
                if math.isfinite(float(flat.value)):
                    grouped[flat.canonical_path].append((sample, flat, float(flat.value)))
    for canonical_path, entries in grouped.items():
        if len(entries) < 10:
            continue
        values = sorted(value for _, _, value in entries)
        median = statistics.median(values)
        deviations = [abs(value - median) for value in values]
        mad = statistics.median(deviations)
        q1, q3 = percentile(values, 0.25), percentile(values, 0.75)
        iqr = q3 - q1
        for sample, flat, value in entries:
            if mad > 0:
                score = 0.6745 * abs(value - median) / mad
                is_outlier = score > 7.0
                evidence = f"median={median:g}, robust z-score={score:.1f}, n={len(values)}"
            elif iqr > 0:
                lower, upper = q1 - 4 * iqr, q3 + 4 * iqr
                is_outlier = value < lower or value > upper
                evidence = f"median={median:g}, 4×IQR bounds=[{lower:g}, {upper:g}], n={len(values)}"
            else:
                is_outlier = False
                evidence = ""
            if is_outlier:
                semantic_issue(
                    sample,
                    32,
                    "numeric_outlier",
                    flat,
                    "Numeric label is an extreme outlier for this field.",
                    evidence,
                )


def canonical_content_hash(content: dict[str, Any]) -> str:
    payload = json.dumps(
        content, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_duplicate_checks(samples: list[Sample]) -> tuple[int, int]:
    by_image_hash: dict[str, list[Sample]] = defaultdict(list)
    by_annotation_hash: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_image_hash[file_hash(sample.image_path)].append(sample)
        by_annotation_hash[canonical_content_hash(sample.content)].append(sample)
    duplicate_image_groups = sum(len(group) > 1 for group in by_image_hash.values())
    duplicate_annotation_groups = sum(
        len(group) > 1 for group in by_annotation_hash.values()
    )
    for group in by_image_hash.values():
        if len(group) < 2:
            continue
        annotation_hashes = {canonical_content_hash(sample.content) for sample in group}
        if len(annotation_hashes) == 1:
            continue
        ids = tuple(sample.sample_id for sample in group)
        for sample in group:
            add_issue(
                sample,
                Issue(
                    sample.sample_id,
                    sample.split,
                    96,
                    "duplicate_image_conflict",
                    "<whole annotation>",
                    "",
                    "Byte-identical images have different annotations.",
                    "This is a strong label-error signal.",
                    tuple(sample_id for sample_id in ids if sample_id != sample.sample_id),
                ),
            )
    return duplicate_image_groups, duplicate_annotation_groups


def sample_features(sample: Sample) -> dict[tuple[str, str], float]:
    features: dict[tuple[str, str], float] = {}
    for spec in ENTITY_SPECS.values():
        for name, path in spec.items():
            value = get_path(sample.content, path)
            normalized = normalize_text(value)
            if not normalized:
                continue
            weight = {"company": 3.0, "street": 2.0, "postcode": 2.0, "city": 1.0}.get(
                name, 0.5
            )
            features[(path, normalized)] = weight
    for path in (
        "takingOverTheGoods.takingOverTheGoodsPlace",
        "deliveryOfTheGoods.logisticsLocationCity",
    ):
        normalized = normalize_text(get_path(sample.content, path))
        if normalized:
            features[(path, normalized)] = 1.0
    return features


def find_nearest_peers(samples: list[Sample], limit: int = 3) -> None:
    features = {sample.sample_id: sample_features(sample) for sample in samples}
    for sample in samples:
        own = features[sample.sample_id]
        candidates: list[tuple[str, float, list[str]]] = []
        for other in samples:
            if other is sample:
                continue
            theirs = features[other.sample_id]
            intersection = own.keys() & theirs.keys()
            if not intersection:
                continue
            shared_weight = sum(min(own[key], theirs[key]) for key in intersection)
            union_keys = own.keys() | theirs.keys()
            union_weight = sum(max(own.get(key, 0), theirs.get(key, 0)) for key in union_keys)
            score = shared_weight / union_weight if union_weight else 0.0
            shared_labels = [
                key[0].rsplit(".", 1)[-1] for key in sorted(intersection)
            ]
            candidates.append((other.sample_id, score, shared_labels))
        sample.peers = sorted(candidates, key=lambda item: (-item[1], item[0]))[:limit]


def risk_score(sample: Sample) -> int:
    severities = sorted((issue.severity for issue in sample.issues), reverse=True)
    if not severities:
        return 0
    score = severities[0] + sum(severities[1:5]) * 0.12
    return min(100, round(score))


def field_statistics(samples: list[Sample]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    document_presence: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        for flat in sample.flat_values:
            grouped[flat.canonical_path].append(flat.value)
            if flat.value is not None:
                document_presence[flat.canonical_path].add(sample.sample_id)
    rows: list[dict[str, Any]] = []
    for path in sorted(grouped):
        values = grouped[path]
        non_null = [value for value in values if value is not None]
        normalized_counts = Counter(display_value(value) for value in non_null)
        numeric = [
            float(value)
            for value in non_null
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        lengths = [len(value) for value in non_null if isinstance(value, str)]
        rows.append(
            {
                "field": path,
                "occurrences": len(values),
                "non_null_occurrences": len(non_null),
                "documents_with_value": len(document_presence[path]),
                "document_coverage": (
                    len(document_presence[path]) / len(samples) if samples else 0
                ),
                "unique_values": len(normalized_counts),
                "types": json.dumps(
                    Counter(type(value).__name__ for value in values), sort_keys=True
                ),
                "top_values": " | ".join(
                    f"{value} ({count})" for value, count in normalized_counts.most_common(5)
                ),
                "numeric_min": min(numeric) if numeric else "",
                "numeric_median": statistics.median(numeric) if numeric else "",
                "numeric_max": max(numeric) if numeric else "",
                "string_length_median": statistics.median(lengths) if lengths else "",
                "string_length_max": max(lengths) if lengths else "",
            }
        )
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def relative_url(target: Path, report_dir: Path) -> str:
    relative = os.path.relpath(target.resolve(), report_dir.resolve()).replace(os.sep, "/")
    return quote(relative, safe="/.:_-~")


def peer_summary(sample: Sample) -> dict[str, Any]:
    return {
        "id": sample.sample_id,
        "split": sample.split,
        "values": {label: get_path(sample.content, path) for label, path in PEER_FIELDS},
    }


def json_for_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def write_html_report(
    result: AuditResult,
    output_dir: Path,
    dataset_root: Path,
    high_risk_threshold: int,
) -> None:
    by_id = {sample.sample_id: sample for sample in result.samples}
    report_samples: list[dict[str, Any]] = []
    for sample in sorted(result.samples, key=lambda item: (-risk_score(item), item.sample_id)):
        related_ids: list[str] = []
        for issue in sample.issues:
            related_ids.extend(issue.peer_ids)
        related_ids.extend(peer_id for peer_id, _, _ in sample.peers)
        unique_related_ids = list(dict.fromkeys(related_ids))[:8]
        report_samples.append(
            {
                "id": sample.sample_id,
                "split": sample.split,
                "risk": risk_score(sample),
                "image": relative_url(sample.image_path, output_dir),
                "annotation": relative_url(sample.annotation_path, output_dir),
                "annotation_path": str(sample.annotation_path.relative_to(dataset_root)),
                "issues": [
                    {
                        **asdict(issue),
                        "value": display_value(issue.value),
                        "peer_ids": list(issue.peer_ids),
                    }
                    for issue in sorted(sample.issues, key=lambda issue: -issue.severity)
                ],
                "nearest_peers": [
                    {"id": peer_id, "score": score, "shared": shared}
                    for peer_id, score, shared in sample.peers
                ],
                "comparison": [peer_summary(sample)]
                + [peer_summary(by_id[peer_id]) for peer_id in unique_related_ids if peer_id in by_id],
                "content": sample.content,
            }
        )
    counts = Counter(
        "high"
        if risk_score(sample) >= high_risk_threshold
        else "medium"
        if risk_score(sample) >= 40
        else "low"
        if sample.issues
        else "clean"
        for sample in result.samples
    )
    data = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": str(dataset_root),
        "counts": dict(counts),
        "load_errors": result.load_errors,
        "samples": report_samples,
        "peer_fields": [label for label, _ in PEER_FIELDS],
        "high_risk_threshold": high_risk_threshold,
    }
    template = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Annotation audit</title>
  <style>
    :root { color-scheme: light; --ink:#192024; --muted:#667077; --line:#dce2e5; --paper:#fff; --bg:#f3f6f7; --high:#b42318; --medium:#b54708; --low:#175cd3; --ok:#067647; }
    * { box-sizing:border-box; }
    body { margin:0; font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; color:var(--ink); background:var(--bg); }
    header { position:sticky; top:0; z-index:5; padding:18px 24px 14px; background:rgba(255,255,255,.96); border-bottom:1px solid var(--line); backdrop-filter:blur(8px); }
    h1 { margin:0 0 4px; font-size:22px; } h2 { margin:0; font-size:18px; } h3 { margin:18px 0 8px; font-size:14px; }
    .muted { color:var(--muted); } .toolbar { display:flex; gap:10px; flex-wrap:wrap; margin-top:12px; }
    input, select, button { font:inherit; border:1px solid #b8c2c7; border-radius:7px; padding:8px 10px; background:white; }
    #search { min-width:320px; flex:1; } button { cursor:pointer; } button:hover { background:#eef3f5; }
    main { max-width:1500px; margin:auto; padding:20px 24px 80px; }
    .stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:10px; margin-bottom:18px; }
    .stat { background:var(--paper); border:1px solid var(--line); border-radius:10px; padding:12px; } .stat b { display:block; font-size:22px; }
    .sample { background:var(--paper); border:1px solid var(--line); border-radius:12px; margin:0 0 16px; overflow:hidden; }
    .sample-head { display:flex; gap:12px; align-items:center; justify-content:space-between; padding:12px 15px; border-bottom:1px solid var(--line); }
    .risk { display:inline-block; min-width:38px; text-align:center; color:white; font-weight:700; border-radius:999px; padding:4px 8px; }
    .risk.high { background:var(--high); } .risk.medium { background:var(--medium); } .risk.low { background:var(--low); } .risk.clean { background:var(--ok); }
    .grid { display:grid; grid-template-columns:minmax(330px,42%) 1fr; min-height:520px; }
    .image-pane { padding:12px; background:#e9edef; border-right:1px solid var(--line); } .image-pane img { width:100%; max-height:800px; object-fit:contain; background:white; }
    .details { padding:14px 16px; overflow:auto; } .issue { border-left:4px solid var(--low); padding:7px 10px; margin:0 0 9px; background:#f8fafb; }
    .issue.high { border-color:var(--high); } .issue.medium { border-color:var(--medium); } .issue.low { border-color:var(--low); }
    .issue-title { font-weight:650; } code { overflow-wrap:anywhere; color:#344054; } .evidence { color:var(--muted); margin-top:3px; }
    table { border-collapse:collapse; width:100%; font-size:12px; } th,td { border:1px solid var(--line); padding:6px; text-align:left; vertical-align:top; } th { background:#f2f5f6; position:sticky; top:0; }
    details { margin-top:13px; } summary { cursor:pointer; font-weight:600; } pre { max-height:420px; overflow:auto; background:#111827; color:#d1e9ff; padding:12px; border-radius:8px; white-space:pre-wrap; }
    .review { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-top:14px; padding-top:12px; border-top:1px solid var(--line); } .review input { min-width:280px; flex:1; }
    .status-confirmed_error { outline:3px solid #fecdca; } .status-looks_correct { opacity:.62; } .hidden { display:none!important; }
    .empty { text-align:center; padding:50px; color:var(--muted); }
    @media (max-width:900px) { .grid { grid-template-columns:1fr; } .image-pane { border-right:0; border-bottom:1px solid var(--line); } #search { min-width:100%; } }
  </style>
</head>
<body>
<header><h1>Annotation audit</h1><div id="subtitle" class="muted"></div><div class="toolbar">
  <input id="search" type="search" placeholder="Search id, path, field, value, or reason">
  <select id="riskFilter"><option value="flagged">Flagged only</option><option value="all">All documents</option><option value="high">High risk</option><option value="medium">Medium risk</option><option value="unreviewed">Unreviewed</option></select>
  <select id="issueFilter"><option value="">All issue types</option></select>
  <button id="export">Export review decisions CSV</button>
</div></header>
<main><div id="stats" class="stats"></div><div id="errors"></div><div id="samples"></div></main>
<script id="audit-data" type="application/json">__AUDIT_DATA__</script>
<script>
const audit=JSON.parse(document.getElementById('audit-data').textContent), decisions=JSON.parse(localStorage.getItem('annotationAuditDecisions')||'{}');
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const level=n=>n>=audit.high_risk_threshold?'high':n>=40?'medium':n>0?'low':'clean';
const issueTypes=[...new Set(audit.samples.flatMap(s=>s.issues.map(i=>i.code)))].sort(); document.getElementById('issueFilter').innerHTML+=[...issueTypes].map(x=>`<option value="${esc(x)}">${esc(x)}</option>`).join('');
document.getElementById('subtitle').textContent=`${audit.samples.length} documents · generated ${audit.generated_at} · ${audit.dataset}`;
document.getElementById('stats').innerHTML=[['High risk',audit.counts.high||0],['Medium risk',audit.counts.medium||0],['Low risk',audit.counts.low||0],['No flags',audit.counts.clean||0],['Review decisions',Object.keys(decisions).length]].map(([k,v])=>`<div class="stat"><span class="muted">${k}</span><b>${v}</b></div>`).join('');
if(audit.load_errors.length) document.getElementById('errors').innerHTML=`<div class="sample"><div class="details"><h2>Dataset loading warnings</h2><ul>${audit.load_errors.map(x=>`<li>${esc(x)}</li>`).join('')}</ul></div></div>`;
function comparison(s){ if(s.comparison.length<2)return '<p class="muted">No peers share enough structured values for a useful comparison.</p>'; return `<div style="overflow:auto"><table><thead><tr><th>Document</th>${audit.peer_fields.map(x=>`<th>${esc(x)}</th>`).join('')}</tr></thead><tbody>${s.comparison.map((p,i)=>`<tr><td><b>${i?'Peer':'Target'}</b><br>${esc(p.id)}<br><span class="muted">${esc(p.split)}</span></td>${audit.peer_fields.map(f=>`<td>${esc(p.values[f]??'∅')}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`; }
function render(){const q=document.getElementById('search').value.toLowerCase(), rf=document.getElementById('riskFilter').value, it=document.getElementById('issueFilter').value; let shown=0; const cards=audit.samples.map(s=>{const d=decisions[s.id]||{}, lev=level(s.risk), hay=JSON.stringify(s).toLowerCase(); const visible=(!q||hay.includes(q))&&(!it||s.issues.some(i=>i.code===it))&&(rf==='all'||rf==='flagged'&&s.risk>0||rf===lev||rf==='unreviewed'&&!d.status); if(!visible)return ''; shown++; const issues=s.issues.length?s.issues.map(i=>`<div class="issue ${level(i.severity)}"><div class="issue-title">${i.severity}/100 · ${esc(i.code)}</div><code>${esc(i.field)}</code> = <b>${esc(i.value)}</b><div>${esc(i.message)}</div>${i.evidence?`<div class="evidence">${esc(i.evidence)}</div>`:''}${i.peer_ids.length?`<div class="evidence">Supporting peers: ${i.peer_ids.map(esc).join(', ')}</div>`:''}</div>`).join(''):'<p>No automated flags. Use the peer comparison for spot checking.</p>'; const nearest=s.nearest_peers.length?s.nearest_peers.map(p=>`${esc(p.id)} (${(p.score*100).toFixed(0)}%; ${p.shared.map(esc).join(', ')})`).join('<br>'):'No sufficiently similar peers'; return `<article class="sample status-${esc(d.status||'')}" data-id="${esc(s.id)}"><div class="sample-head"><div><h2>${esc(s.id)}</h2><span class="muted">${esc(s.split)} · ${esc(s.annotation_path)}</span></div><span class="risk ${lev}">${s.risk}</span></div><div class="grid"><div class="image-pane"><a href="${s.image}" target="_blank"><img loading="lazy" src="${s.image}" alt="${esc(s.id)}"></a><p><a href="${s.annotation}" target="_blank">Open source annotation JSON</a></p></div><div class="details"><h3>Why this document was flagged</h3>${issues}<h3>Nearest peers</h3><p class="muted">${nearest}</p><h3>Side-by-side label comparison</h3>${comparison(s)}<details><summary>Full annotation content</summary><pre>${esc(JSON.stringify(s.content,null,2))}</pre></details><div class="review"><b>Manual decision:</b><select class="decision"><option value="">Not reviewed</option><option value="confirmed_error" ${d.status==='confirmed_error'?'selected':''}>Confirmed label error</option><option value="looks_correct" ${d.status==='looks_correct'?'selected':''}>Looks correct / false alarm</option><option value="unsure" ${d.status==='unsure'?'selected':''}>Unsure</option></select><input class="note" value="${esc(d.note||'')}" placeholder="Correction or reviewer note"></div></div></div></article>`;}).join(''); document.getElementById('samples').innerHTML=cards||'<div class="empty">No documents match the current filters.</div>'; document.querySelectorAll('.decision,.note').forEach(el=>el.addEventListener('change',saveDecision)); }
function saveDecision(e){const card=e.target.closest('.sample'), id=card.dataset.id, status=card.querySelector('.decision').value, note=card.querySelector('.note').value; if(status||note)decisions[id]={status,note,updated_at:new Date().toISOString()}; else delete decisions[id]; localStorage.setItem('annotationAuditDecisions',JSON.stringify(decisions)); render();}
function csvCell(v){return '"'+String(v??'').replaceAll('"','""')+'"'} document.getElementById('export').onclick=()=>{const rows=[['sample_id','status','reviewer_note','updated_at'],...Object.entries(decisions).map(([id,d])=>[id,d.status,d.note,d.updated_at])]; const blob=new Blob([String.fromCharCode(0xfeff)+rows.map(r=>r.map(csvCell).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'}), a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='annotation_review_decisions.csv'; a.click(); URL.revokeObjectURL(a.href);};
['search','riskFilter','issueFilter'].forEach(id=>document.getElementById(id).addEventListener(id==='search'?'input':'change',render)); render();
</script></body></html>'''
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.html").write_text(
        template.replace("__AUDIT_DATA__", json_for_script(data)), encoding="utf-8"
    )


def write_outputs(
    result: AuditResult,
    output_dir: Path,
    dataset_root: Path,
    high_risk_threshold: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    issues = [
        issue
        for sample in result.samples
        for issue in sorted(sample.issues, key=lambda item: -item.severity)
    ]
    issue_rows = [
        {
            **asdict(issue),
            "value": display_value(issue.value),
            "peer_ids": " | ".join(issue.peer_ids),
            "document_risk": risk_score(next(s for s in result.samples if s.sample_id == issue.sample_id)),
        }
        for issue in issues
    ]
    write_csv(
        output_dir / "issues.csv",
        issue_rows,
        [
            "sample_id",
            "split",
            "document_risk",
            "severity",
            "code",
            "field",
            "value",
            "message",
            "evidence",
            "peer_ids",
        ],
    )
    review_rows = []
    for sample in sorted(result.samples, key=lambda item: (-risk_score(item), item.sample_id)):
        if not sample.issues:
            continue
        review_rows.append(
            {
                "sample_id": sample.sample_id,
                "split": sample.split,
                "risk_score": risk_score(sample),
                "issue_count": len(sample.issues),
                "issue_types": " | ".join(sorted({issue.code for issue in sample.issues})),
                "issue_fields": " | ".join(
                    sorted({issue.field for issue in sample.issues})
                ),
                "annotation_path": str(sample.annotation_path.relative_to(dataset_root)),
                "image_path": str(sample.image_path.relative_to(dataset_root)),
                "review_status": "",
                "corrected_field": "",
                "corrected_value": "",
                "reviewer_notes": "",
            }
        )
    write_csv(
        output_dir / "review_queue.csv",
        review_rows,
        [
            "sample_id",
            "split",
            "risk_score",
            "issue_count",
            "issue_types",
            "issue_fields",
            "annotation_path",
            "image_path",
            "review_status",
            "corrected_field",
            "corrected_value",
            "reviewer_notes",
        ],
    )
    stats = field_statistics(result.samples)
    write_csv(output_dir / "field_statistics.csv", stats, list(stats[0]) if stats else ["field"])
    rule_rows = [
        {**asdict(rule), "exception_ids": " | ".join(rule.exception_ids)}
        for rule in result.learned_rules
    ]
    write_csv(
        output_dir / "learned_rules.csv",
        rule_rows,
        [
            "role",
            "anchor_field",
            "anchor_value",
            "dependent_field",
            "expected_value",
            "support",
            "anchor_support",
            "confidence",
            "exception_ids",
        ],
    )
    risk_counts = Counter(
        "high"
        if risk_score(sample) >= high_risk_threshold
        else "medium"
        if risk_score(sample) >= 40
        else "low"
        if sample.issues
        else "clean"
        for sample in result.samples
    )
    issue_type_counts = Counter(issue.code for issue in issues)
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_root": str(dataset_root),
        "samples": len(result.samples),
        "documents_by_risk": dict(risk_counts),
        "issues": len(issues),
        "issues_by_type": dict(issue_type_counts.most_common()),
        "learned_rules": len(result.learned_rules),
        "duplicate_image_groups": result.duplicate_image_groups,
        "duplicate_annotation_groups": result.duplicate_annotation_groups,
        "load_errors": result.load_errors,
        "outputs": {
            "html_report": str(output_dir / "report.html"),
            "review_queue": str(output_dir / "review_queue.csv"),
            "issues": str(output_dir / "issues.csv"),
            "field_statistics": str(output_dir / "field_statistics.csv"),
            "learned_rules": str(output_dir / "learned_rules.csv"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_html_report(result, output_dir, dataset_root, high_risk_threshold)
    return summary


def audit_dataset(
    dataset_root: Path,
    schema_path: Path | None = DEFAULT_SCHEMA_PATH,
    min_rule_support: int = 3,
    rule_confidence: float = 0.80,
) -> AuditResult:
    dataset_root = dataset_root.resolve()
    samples, load_errors = load_samples(dataset_root)
    load_errors.extend(run_schema_checks(samples, schema_path.resolve() if schema_path else None))
    run_semantic_checks(samples)
    run_intra_document_checks(samples)
    run_numeric_outlier_checks(samples)
    learned_rules = learn_consistency_rules(samples, min_rule_support, rule_confidence)
    duplicate_image_groups, duplicate_annotation_groups = run_duplicate_checks(samples)
    find_nearest_peers(samples)
    for sample in samples:
        sample.issues.sort(key=lambda issue: (-issue.severity, issue.code, issue.field))
    return AuditResult(
        samples=samples,
        load_errors=load_errors,
        learned_rules=learned_rules,
        duplicate_image_groups=duplicate_image_groups,
        duplicate_annotation_groups=duplicate_annotation_groups,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root = args.dataset_root.resolve()
    output_dir = (args.output_dir or dataset_root / "annotation_audit").resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")
    result = audit_dataset(
        dataset_root,
        schema_path=args.schema_path,
        min_rule_support=args.min_rule_support,
        rule_confidence=args.rule_confidence,
    )
    summary = write_outputs(result, output_dir, dataset_root, args.high_risk_threshold)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.fail_on_high_risk and summary["documents_by_risk"].get("high", 0):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
