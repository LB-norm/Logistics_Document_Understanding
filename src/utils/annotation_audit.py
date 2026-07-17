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
    target_fields: tuple[str, ...] = ()
    typo_candidates: tuple["TypoCandidate", ...] = ()


@dataclass(frozen=True)
class TypoCandidate:
    field: str
    value_key: tuple[str, str]
    suggested_key: tuple[str, str]
    edit_distance: int
    similarity: float


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


FORMAT_ISSUE_CODES = {
    "minority_json_type",
    "minority_character_class",
    "inconsistent_text_variant",
    "surrounding_whitespace",
}


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


def schema_leaf_paths(schema: dict[str, Any]) -> tuple[str, ...]:
    """Return canonical leaf paths, including leaves inside arrays."""

    def visit(node: dict[str, Any], path: str) -> list[str]:
        if "$ref" in node:
            node = resolve_schema_ref(schema, node["$ref"])
        properties = node.get("properties")
        if isinstance(properties, dict):
            paths: list[str] = []
            for key, child in properties.items():
                if not isinstance(child, dict):
                    continue
                child_path = f"{path}.{key}" if path else key
                paths.extend(visit(child, child_path))
            return paths
        items = node.get("items")
        if isinstance(items, dict):
            return visit(items, f"{path}[]")
        return [path] if path else []

    return tuple(sorted(set(visit(schema, ""))))


def json_value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def typed_value_key(value: Any) -> tuple[str, str]:
    value_type = json_value_type(value)
    try:
        serialized = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError):
        serialized = repr(value)
    return value_type, serialized


def character_class(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        return "blank"
    has_letters = any(character.isalpha() for character in stripped)
    has_digits = any(character.isdigit() for character in stripped)
    if has_letters and has_digits:
        return "alphanumeric"
    if has_letters:
        return "alphabetic"
    if has_digits:
        return "numeric"
    return "symbols"


def edit_distance(left: str, right: str) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for right_index, right_character in enumerate(right, start=1):
        current = [right_index]
        for left_index, left_character in enumerate(left, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[left_index] + 1,
                    previous[left_index - 1]
                    + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def canonicalize_field_path(path: str) -> str:
    return re.sub(r"\[\d+\]", "[]", path)


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


def group_field_entries(
    samples: list[Sample],
) -> dict[str, list[tuple[Sample, FlatValue]]]:
    grouped: dict[str, list[tuple[Sample, FlatValue]]] = defaultdict(list)
    for sample in samples:
        for flat in sample.flat_values:
            grouped[flat.canonical_path].append((sample, flat))
    return grouped


def aggregate_field_values(
    entries: list[tuple[Sample, FlatValue]],
) -> dict[tuple[str, str], dict[str, Any]]:
    aggregates: dict[tuple[str, str], dict[str, Any]] = {}
    for sample, flat in entries:
        if flat.value is None:
            continue
        key = typed_value_key(flat.value)
        aggregate = aggregates.setdefault(
            key,
            {
                "value": flat.value,
                "occurrences": 0,
                "sample_ids": set(),
                "entries": [],
            },
        )
        aggregate["occurrences"] += 1
        aggregate["sample_ids"].add(sample.sample_id)
        aggregate["entries"].append((sample, flat))
    return aggregates


def find_typo_candidates(
    grouped: dict[str, list[tuple[Sample, FlatValue]]],
) -> list[TypoCandidate]:
    candidates: list[TypoCandidate] = []
    for field_path, entries in grouped.items():
        aggregates = aggregate_field_values(entries)
        strings = [
            (key, aggregate, normalize_text(aggregate["value"]))
            for key, aggregate in aggregates.items()
            if key[0] == "string"
            and any(character.isalpha() for character in str(aggregate["value"]))
            and len(normalize_text(aggregate["value"])) >= 4
        ]
        for rare_key, rare, rare_normalized in strings:
            rare_count = len(rare["sample_ids"])
            if rare_count > 2:
                continue
            matches: list[tuple[int, float, int, tuple[str, str]]] = []
            for common_key, common, common_normalized in strings:
                common_count = len(common["sample_ids"])
                if common_key == rare_key or common_count < max(5, rare_count * 5):
                    continue
                if common_normalized == rare_normalized:
                    continue
                if re.findall(r"\d+", rare_normalized) != re.findall(
                    r"\d+", common_normalized
                ):
                    continue
                longest = max(len(rare_normalized), len(common_normalized))
                allowed_distance = 1 if longest < 8 else 2 if longest < 80 else 3
                distance = edit_distance(rare_normalized, common_normalized)
                similarity = SequenceMatcher(
                    None, rare_normalized, common_normalized
                ).ratio()
                if distance <= allowed_distance and similarity >= 0.84:
                    matches.append((distance, -similarity, -common_count, common_key))
            if not matches:
                continue
            distance, negative_similarity, _, suggested_key = min(matches)
            candidates.append(
                TypoCandidate(
                    field=field_path,
                    value_key=rare_key,
                    suggested_key=suggested_key,
                    edit_distance=distance,
                    similarity=-negative_similarity,
                )
            )
    return candidates


def run_distribution_checks(samples: list[Sample]) -> list[TypoCandidate]:
    grouped = group_field_entries(samples)
    candidates = find_typo_candidates(grouped)
    typo_candidates = {
        (candidate.field, candidate.value_key): candidate
        for candidate in candidates
    }
    for field_path, entries in grouped.items():
        aggregates = aggregate_field_values(entries)
        non_null_entries = [entry for entry in entries if entry[1].value is not None]
        total = len(non_null_entries)
        if not total:
            continue

        type_counts = Counter(
            json_value_type(flat.value) for _, flat in non_null_entries
        )
        dominant_type, dominant_type_count = type_counts.most_common(1)[0]
        type_confidence = dominant_type_count / total
        minority_limit = max(2, math.floor(total * 0.02))
        if total >= 10 and type_confidence >= 0.90:
            for value_type, count in type_counts.items():
                if value_type == dominant_type or count > minority_limit:
                    continue
                for sample, flat in non_null_entries:
                    if json_value_type(flat.value) != value_type:
                        continue
                    semantic_issue(
                        sample,
                        46,
                        "minority_json_type",
                        flat,
                        (
                            f"This field normally contains JSON {dominant_type} values, "
                            f"but this value is {value_type}."
                        ),
                        (
                            f"{dominant_type_count}/{total} occurrences are "
                            f"{dominant_type} ({type_confidence:.1%}); only {count} are "
                            f"{value_type}."
                        ),
                    )

        string_entries = [
            (sample, flat)
            for sample, flat in non_null_entries
            if isinstance(flat.value, str) and flat.value.strip()
        ]
        class_counts = Counter(
            character_class(flat.value) for _, flat in string_entries
        )
        if class_counts:
            dominant_class, dominant_class_count = class_counts.most_common(1)[0]
            class_total = len(string_entries)
            class_confidence = dominant_class_count / class_total
            class_minority_limit = max(2, math.floor(class_total * 0.02))
            if class_total >= 10 and class_confidence >= 0.90:
                for value_class, count in class_counts.items():
                    if value_class == dominant_class or count > class_minority_limit:
                        continue
                    for sample, flat in string_entries:
                        if character_class(flat.value) != value_class:
                            continue
                        semantic_issue(
                            sample,
                            40,
                            "minority_character_class",
                            flat,
                            (
                                f"This field normally contains {dominant_class} text, "
                                f"but this value is {value_class}."
                            ),
                            (
                                f"{dominant_class_count}/{class_total} string occurrences "
                                f"are {dominant_class} ({class_confidence:.1%}); only "
                                f"{count} are {value_class}."
                            ),
                        )

        normalized_groups: dict[str, list[tuple[tuple[str, str], dict[str, Any]]]] = (
            defaultdict(list)
        )
        for key, aggregate in aggregates.items():
            if key[0] != "string" or not normalize_text(aggregate["value"]):
                continue
            normalized_groups[normalize_text(aggregate["value"])].append(
                (key, aggregate)
            )
        for variants in normalized_groups.values():
            if len(variants) < 2:
                continue
            variants.sort(key=lambda item: (-item[1]["occurrences"], item[0]))
            _, dominant = variants[0]
            if dominant["occurrences"] < 5:
                continue
            for _, variant in variants[1:]:
                if variant["occurrences"] > 2:
                    continue
                for sample, flat in variant["entries"]:
                    semantic_issue(
                        sample,
                        38,
                        "inconsistent_text_variant",
                        flat,
                        (
                            "This value differs only in casing, accents, spacing, or "
                            "punctuation from a frequent value."
                        ),
                        (
                            f"Suggested canonical form {display_value(dominant['value'])!r} "
                            f"appears {dominant['occurrences']} times; this form appears "
                            f"{variant['occurrences']} times."
                        ),
                    )

        unique_ratio = len(aggregates) / total
        for value_key, aggregate in aggregates.items():
            count = aggregate["occurrences"]
            occurrence_rate = count / total
            if (
                total >= 20
                and unique_ratio <= 0.20
                and count <= 2
                and occurrence_rate <= 0.02
            ):
                for sample, flat in aggregate["entries"]:
                    semantic_issue(
                        sample,
                        18,
                        "rare_value",
                        flat,
                        "This value is rare in an otherwise repetitive field.",
                        (
                            f"{count}/{total} occurrences ({occurrence_rate:.1%}); "
                            f"{len(aggregates)} distinct typed values."
                        ),
                    )

            typo = typo_candidates.get((field_path, value_key))
            if typo is not None:
                suggestion = aggregates[typo.suggested_key]
                for sample, flat in aggregate["entries"]:
                    semantic_issue(
                        sample,
                        58,
                        "possible_typo",
                        flat,
                        (
                            f"This rare value is very similar to frequent value "
                            f"{display_value(suggestion['value'])!r}."
                        ),
                        (
                            f"Suggested value appears {suggestion['occurrences']} times "
                            f"in {len(suggestion['sample_ids'])} examples; "
                            f"edit distance={typo.edit_distance}, normalized text "
                            f"similarity={typo.similarity:.1%}."
                        ),
                    )

        for sample, flat in string_entries:
            if flat.value != flat.value.strip():
                semantic_issue(
                    sample,
                    32,
                    "surrounding_whitespace",
                    flat,
                    "String value has leading or trailing whitespace.",
                )
    return candidates


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


def field_value_statistics(
    samples: list[Sample], typo_candidates: Iterable[TypoCandidate] | None = None
) -> list[dict[str, Any]]:
    grouped = group_field_entries(samples)
    candidates = (
        find_typo_candidates(grouped)
        if typo_candidates is None
        else list(typo_candidates)
    )
    typo_candidates_by_value = {
        (candidate.field, candidate.value_key): candidate
        for candidate in candidates
    }
    rows: list[dict[str, Any]] = []
    for path in sorted(grouped):
        aggregates = aggregate_field_values(grouped[path])
        total = sum(aggregate["occurrences"] for aggregate in aggregates.values())
        for value_key, aggregate in sorted(
            aggregates.items(),
            key=lambda item: (-item[1]["occurrences"], item[0]),
        ):
            count = aggregate["occurrences"]
            sample_ids = sorted(aggregate["sample_ids"])
            if count == 1:
                rarity = "singleton"
            elif total and count / total <= 0.01:
                rarity = "rare"
            elif total and count / total <= 0.05:
                rarity = "uncommon"
            else:
                rarity = "common"
            typo = typo_candidates_by_value.get((path, value_key))
            suggested_value = (
                aggregates[typo.suggested_key]["value"] if typo is not None else None
            )
            rows.append(
                {
                    "field": path,
                    "value_type": value_key[0],
                    "value": display_value(aggregate["value"]),
                    "value_json": value_key[1],
                    "occurrence_count": count,
                    "occurrence_frequency": count / total if total else 0,
                    "examples_with_value": len(sample_ids),
                    "example_frequency": len(sample_ids) / len(samples) if samples else 0,
                    "rarity": rarity,
                    "possible_typo_of": (
                        display_value(suggested_value) if typo is not None else ""
                    ),
                    "typo_edit_distance": typo.edit_distance if typo is not None else "",
                    "typo_similarity": typo.similarity if typo is not None else "",
                    "example_ids": " | ".join(sample_ids),
                }
            )
    return rows


def field_statistics(
    samples: list[Sample], expected_fields: Iterable[str] = ()
) -> list[dict[str, Any]]:
    grouped_entries = group_field_entries(samples)
    all_paths = set(grouped_entries) | set(expected_fields)
    issue_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        for issue in sample.issues:
            canonical_path = canonicalize_field_path(issue.field)
            if canonical_path in all_paths:
                issue_counts[canonical_path][issue.code] += 1
    rows: list[dict[str, Any]] = []
    for path in sorted(all_paths):
        entries = grouped_entries.get(path, [])
        values = [flat.value for _, flat in entries]
        non_null = [value for value in values if value is not None]
        typed_counts = Counter(typed_value_key(value) for value in non_null)
        document_presence = {
            sample.sample_id for sample, flat in entries if flat.value is not None
        }
        numeric = [
            float(value)
            for value in non_null
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        lengths = [len(value) for value in non_null if isinstance(value, str)]
        rows.append(
            {
                "field": path,
                "examples_total": len(samples),
                "occurrences": len(values),
                "non_null_occurrences": len(non_null),
                "null_occurrences": len(values) - len(non_null),
                "examples_with_non_null": len(document_presence),
                "documents_with_value": len(document_presence),
                "document_coverage": (
                    len(document_presence) / len(samples) if samples else 0
                ),
                "unique_values": len(typed_counts),
                "singleton_values": sum(count == 1 for count in typed_counts.values()),
                "rare_values": sum(
                    count <= 2 and len(non_null) > 0 and count / len(non_null) <= 0.02
                    for count in typed_counts.values()
                ),
                "types": json.dumps(
                    Counter(json_value_type(value) for value in values), sort_keys=True
                ),
                "top_values": " | ".join(
                    f"{display_value(json.loads(value_json))} [{value_type}] ({count})"
                    for (value_type, value_json), count in typed_counts.most_common(5)
                ),
                "possible_typo_occurrences": issue_counts[path]["possible_typo"],
                "format_anomaly_occurrences": sum(
                    issue_counts[path][code] for code in FORMAT_ISSUE_CODES
                ),
                "rare_value_occurrences": issue_counts[path]["rare_value"],
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


def write_html_review(
    result: AuditResult,
    output_dir: Path,
    dataset_root: Path,
    high_risk_threshold: int,
) -> None:
    """Write a self-contained browser workspace for reviewing annotations."""
    by_id = {sample.sample_id: sample for sample in result.samples}
    review_samples: list[dict[str, Any]] = []
    for sample in sorted(
        result.samples, key=lambda item: (-risk_score(item), item.sample_id)
    ):
        related_ids: list[str] = []
        for issue in sample.issues:
            related_ids.extend(issue.peer_ids)
        related_ids.extend(peer_id for peer_id, _, _ in sample.peers)
        unique_related_ids = list(dict.fromkeys(related_ids))[:8]
        review_samples.append(
            {
                "id": sample.sample_id,
                "split": sample.split,
                "risk": risk_score(sample),
                "image": relative_url(sample.image_path, output_dir),
                "annotation_editor_url": (
                    "vscode://file"
                    + quote(str(sample.annotation_path.resolve()), safe="/:")
                ),
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
                + [
                    peer_summary(by_id[peer_id])
                    for peer_id in unique_related_ids
                    if peer_id in by_id
                ],
            }
        )
    data = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": str(dataset_root),
        "load_errors": result.load_errors,
        "samples": review_samples,
        "peer_fields": [label for label, _ in PEER_FIELDS],
        "high_risk_threshold": high_risk_threshold,
    }
    template = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Annotation review</title>
  <style>
    :root { color-scheme:light; --ink:#172126; --muted:#68747b; --line:#d6dee2; --paper:#fff; --bg:#f2f5f6; --high:#b42318; --medium:#b54708; --low:#175cd3; --ok:#067647; --focus:#0b6bcb; }
    * { box-sizing:border-box; }
    body { margin:0; font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; color:var(--ink); background:var(--bg); }
    header { position:sticky; top:0; z-index:5; padding:14px 24px; background:rgba(255,255,255,.97); border-bottom:1px solid var(--line); backdrop-filter:blur(8px); }
    h1 { margin:0 0 3px; font-size:21px; } h2 { margin:0; font-size:18px; } h3 { margin:18px 0 8px; font-size:14px; }
    .muted { color:var(--muted); } #subtitle { overflow-wrap:anywhere; } .toolbar { display:flex; gap:10px; flex-wrap:wrap; margin-top:12px; }
    input, select, button { font:inherit; border:1px solid #aebbc1; border-radius:6px; background:white; }
    input, select, button { min-height:38px; padding:7px 10px; }
    input:focus, select:focus, button:focus { outline:2px solid var(--focus); outline-offset:1px; }
    #search { min-width:280px; flex:1; } button { cursor:pointer; } button:hover:not(:disabled) { background:#edf2f4; } button:disabled { cursor:not-allowed; opacity:.48; }
    main { max-width:1540px; margin:auto; padding:18px 24px 70px; }
    .stats { display:grid; grid-template-columns:repeat(4,minmax(120px,1fr)); gap:8px; margin-bottom:14px; }
    .stat { background:var(--paper); border:1px solid var(--line); border-radius:6px; padding:9px 12px; } .stat b { display:block; font-size:20px; }
    .queue-nav { display:flex; align-items:center; justify-content:space-between; gap:10px; margin:0 0 10px; }
    .nav-actions, .image-links { display:flex; align-items:center; gap:8px; }
    .sample { min-width:0; overflow:hidden; overflow-wrap:anywhere; background:var(--paper); border:1px solid var(--line); border-radius:8px; margin:0 0 16px; }
    .sample-head { display:flex; gap:12px; align-items:center; justify-content:space-between; padding:12px 15px; border-bottom:1px solid var(--line); }
    .sample-head > div:first-child { min-width:0; } .sample-head > div:last-child { flex:none; white-space:nowrap; }
    .risk { display:inline-block; min-width:38px; white-space:nowrap; text-align:center; color:white; font-weight:700; border-radius:999px; padding:4px 8px; }
    .risk.high { background:var(--high); } .risk.medium { background:var(--medium); } .risk.low { background:var(--low); } .risk.clean { background:var(--ok); }
    .grid { display:grid; grid-template-columns:minmax(350px,43%) minmax(0,1fr); min-height:560px; }
    .image-pane { padding:12px; background:#e8edef; border-right:1px solid var(--line); }
    .image-pane img { position:sticky; top:155px; width:100%; max-height:calc(100vh - 190px); object-fit:contain; background:white; }
    .image-links { justify-content:flex-start; margin:9px 2px 0; }
    .details { min-width:0; padding:14px 16px; overflow:auto; }
    .issue { border-left:4px solid var(--low); padding:8px 10px; margin:0 0 9px; background:#f8fafb; }
    .issue.high { border-color:var(--high); } .issue.medium { border-color:var(--medium); } .issue.low { border-color:var(--low); }
    .issue-title { font-weight:650; } code { overflow-wrap:anywhere; color:#344054; } .recommendation { color:var(--muted); margin-top:4px; }
    table { border-collapse:collapse; width:100%; font-size:12px; } th,td { border:1px solid var(--line); padding:6px; text-align:left; vertical-align:top; } th { background:#f2f5f6; position:sticky; top:0; }
    details { margin-top:13px; } summary { cursor:pointer; font-weight:650; }
    .review { display:flex; gap:12px; align-items:center; justify-content:space-between; margin-top:18px; padding:14px 0 2px; border-top:1px solid var(--line); }
    .review-check { display:flex; align-items:center; gap:9px; font-size:16px; font-weight:700; cursor:pointer; }
    .review-check input { width:20px; height:20px; min-height:0; accent-color:var(--ok); } .reviewed-banner { color:var(--ok); font-weight:650; }
    .empty { text-align:center; padding:50px; color:var(--muted); }
    @media (max-width:900px) { header { position:static; } main { padding:14px 12px 50px; } .grid { grid-template-columns:1fr; } .image-pane { border-right:0; border-bottom:1px solid var(--line); } .image-pane img { position:static; max-height:72vh; } #search { min-width:100%; } .stats { grid-template-columns:repeat(2,1fr); } .review { align-items:flex-start; flex-direction:column; } }
  </style>
</head>
<body>
<header><h1>Annotation review</h1><div id="subtitle" class="muted"></div><div class="toolbar">
  <input id="search" type="search" placeholder="Search id, path, field, value, or issue">
  <select id="statusFilter"><option value="open">Open documents</option><option value="all">All documents</option><option value="reviewed">Reviewed documents</option><option value="flagged">Flagged documents</option><option value="high">High risk</option><option value="medium">Medium risk</option></select>
  <select id="issueFilter"><option value="">All issue types</option></select>
  <button id="export" title="Download the reviewed/open status of every document">Export checklist CSV</button>
</div></header>
<main><div id="stats" class="stats"></div><div id="errors"></div><div id="queueNav" class="queue-nav"></div><div id="sample"></div></main>
<script id="audit-data" type="application/json">__AUDIT_DATA__</script>
<script>
const audit=JSON.parse(document.getElementById('audit-data').textContent);
const namespace='annotationReview:'+audit.dataset;
const loadStored=suffix=>{try{return JSON.parse(localStorage.getItem(namespace+suffix)||'{}')}catch{return {}}};
const reviews=loadStored(':reviews');
let currentId=audit.samples[0]?.id||'';
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const level=n=>n>=audit.high_risk_threshold?'high':n>=40?'medium':n>0?'low':'clean';
const saveStored=(suffix,value)=>localStorage.setItem(namespace+suffix,JSON.stringify(value));
const issueTypes=[...new Set(audit.samples.flatMap(s=>s.issues.map(i=>i.code)))].sort();
document.getElementById('issueFilter').innerHTML+=issueTypes.map(x=>`<option value="${esc(x)}">${esc(x)}</option>`).join('');
document.getElementById('subtitle').textContent=`${audit.samples.length} documents · generated ${audit.generated_at} · ${audit.dataset}`;
if(audit.load_errors.length)document.getElementById('errors').innerHTML=`<details class="sample"><summary style="padding:12px 15px">Dataset loading warnings (${audit.load_errors.length})</summary><div class="details"><ul>${audit.load_errors.map(x=>`<li>${esc(x)}</li>`).join('')}</ul></div></details>`;
function filteredSamples(){const q=document.getElementById('search').value.toLowerCase(),sf=document.getElementById('statusFilter').value,it=document.getElementById('issueFilter').value;return audit.samples.filter(s=>{const reviewed=!!reviews[s.id]?.reviewed,lev=level(s.risk),hay=JSON.stringify(s).toLowerCase();return(!q||hay.includes(q))&&(!it||s.issues.some(i=>i.code===it))&&(sf==='all'||sf==='open'&&!reviewed||sf==='reviewed'&&reviewed||sf==='flagged'&&s.risk>0||sf===lev);});}
function comparison(s){if(s.comparison.length<2)return '<p class="muted">No sufficiently similar peers.</p>';return `<div style="overflow:auto"><table><thead><tr><th>Document</th>${audit.peer_fields.map(x=>`<th>${esc(x)}</th>`).join('')}</tr></thead><tbody>${s.comparison.map((p,i)=>`<tr><td><b>${i?'Peer':'Target'}</b><br>${esc(p.id)}<br><span class="muted">${esc(p.split)}</span></td>${audit.peer_fields.map(f=>`<td>${esc(p.values[f]??'∅')}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;}
function renderStats(){const reviewed=audit.samples.filter(s=>reviews[s.id]?.reviewed).length,flagged=audit.samples.filter(s=>s.risk>0).length;document.getElementById('stats').innerHTML=[['Reviewed',reviewed],['Open',audit.samples.length-reviewed],['Flagged',flagged],['Total',audit.samples.length]].map(([k,v])=>`<div class="stat"><span class="muted">${k}</span><b>${v}</b></div>`).join('');}
function render(){const queue=filteredSamples();if(!queue.some(s=>s.id===currentId))currentId=queue[0]?.id||'';const index=queue.findIndex(s=>s.id===currentId),s=queue[index];renderStats();document.getElementById('queueNav').innerHTML=queue.length?`<div><b>${index+1} of ${queue.length}</b> matching documents</div><div class="nav-actions"><button id="previous" ${index<=0?'disabled':''}>Previous</button><button id="nextOpen">Next open</button><button id="next" ${index<0||index>=queue.length-1?'disabled':''}>Next</button></div>`:'';if(!s){document.getElementById('sample').innerHTML='<div class="sample empty">No documents match the current filters.</div>';return;}const lev=level(s.risk),review=reviews[s.id]||{};const issues=s.issues.length?s.issues.map(i=>`<div class="issue ${level(i.severity)}"><div class="issue-title">${i.severity}/100 · ${esc(i.code)}</div><div><code>${esc(i.field)}</code> = <b>${esc(i.value)}</b></div><div>${esc(i.message)}</div>${i.evidence?`<div class="recommendation"><b>Recommendation:</b> ${esc(i.evidence)}</div>`:''}${i.peer_ids.length?`<div class="recommendation">Supporting peers: ${i.peer_ids.map(esc).join(', ')}</div>`:''}</div>`).join(''):'<p>No automated issues. Compare the annotation with the image and peers.</p>';const nearest=s.nearest_peers.length?s.nearest_peers.map(p=>`${esc(p.id)} (${(p.score*100).toFixed(0)}%; ${p.shared.map(esc).join(', ')})`).join('<br>'):'No sufficiently similar peers';document.getElementById('sample').innerHTML=`<article class="sample"><div class="sample-head"><div><h2>${esc(s.id)}</h2><div class="muted">${esc(s.split)} · ${esc(s.annotation_path)}</div><a href="${s.annotation_editor_url}" title="${esc(s.annotation_path)}">Edit JSON in VS Code</a></div><div>${review.reviewed?'<span class="reviewed-banner">Reviewed</span> ':''}<span class="risk ${lev}" title="Audit risk score">${s.risk}</span></div></div><div class="grid"><div class="image-pane"><a href="${s.image}" target="_blank"><img src="${s.image}" alt="${esc(s.id)}"></a><div class="image-links"><a href="${s.image}" target="_blank">Open image</a></div></div><div class="details"><h3>Reported issues and recommendations</h3>${issues}<h3>Nearest peers</h3><p class="muted">${nearest}</p><h3>Side-by-side comparison</h3>${comparison(s)}<div class="review"><label class="review-check"><input id="reviewed" type="checkbox" ${review.reviewed?'checked':''}>Reviewed</label><button id="reviewNext">${review.reviewed?'Next open document':'Mark reviewed and continue'}</button></div></div></div></article>`;document.getElementById('previous').onclick=()=>navigate(queue,index-1);document.getElementById('next').onclick=()=>navigate(queue,index+1);document.getElementById('nextOpen').onclick=()=>navigateNextOpen(s.id);document.getElementById('reviewNext').onclick=()=>markAndContinue(s.id);document.getElementById('reviewed').onchange=e=>setReviewed(s.id,e.target.checked);}
function navigate(queue,index){if(queue[index]){currentId=queue[index].id;render();window.scrollTo({top:0,behavior:'smooth'});}}
function navigateNextOpen(fromId){const from=audit.samples.findIndex(s=>s.id===fromId);for(let step=1;step<=audit.samples.length;step++){const candidate=audit.samples[(from+step)%audit.samples.length];if(!reviews[candidate.id]?.reviewed){currentId=candidate.id;document.getElementById('statusFilter').value='open';render();window.scrollTo({top:0,behavior:'smooth'});return;}}currentId='';render();}
function setReviewed(id,reviewed){if(reviewed)reviews[id]={reviewed:true,reviewed_at:new Date().toISOString()};else delete reviews[id];saveStored(':reviews',reviews);render();}
function markAndContinue(id){if(!reviews[id]?.reviewed){reviews[id]={reviewed:true,reviewed_at:new Date().toISOString()};saveStored(':reviews',reviews);}navigateNextOpen(id);renderStats();}
function download(name,text,type){const blob=new Blob([text],{type}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),0);}
function csvCell(v){return '"'+String(v??'').replaceAll('"','""')+'"';}
document.getElementById('export').onclick=()=>{const rows=[['sample_id','split','annotation_path','review_status','reviewed_at'],...audit.samples.map(s=>[s.id,s.split,s.annotation_path,reviews[s.id]?.reviewed?'reviewed':'open',reviews[s.id]?.reviewed_at||''])];download('annotation_review_checklist.csv',String.fromCharCode(0xfeff)+rows.map(r=>r.map(csvCell).join(',')).join('\r\n'),'text/csv;charset=utf-8');};
['search','statusFilter','issueFilter'].forEach(id=>document.getElementById(id).addEventListener(id==='search'?'input':'change',render));
render();
</script></body></html>'''
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = template.replace("__AUDIT_DATA__", json_for_script(data))
    (output_dir / "review.html").write_text(rendered, encoding="utf-8")
    (output_dir / "report.html").write_text(rendered, encoding="utf-8")


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
    stats = field_statistics(result.samples, result.target_fields)
    write_csv(output_dir / "field_statistics.csv", stats, list(stats[0]) if stats else ["field"])
    value_stats = field_value_statistics(result.samples, result.typo_candidates)
    write_csv(
        output_dir / "field_values.csv",
        value_stats,
        list(value_stats[0])
        if value_stats
        else [
            "field",
            "value_type",
            "value",
            "value_json",
            "occurrence_count",
            "occurrence_frequency",
            "examples_with_value",
            "example_frequency",
            "rarity",
            "possible_typo_of",
            "typo_edit_distance",
            "typo_similarity",
            "example_ids",
        ],
    )
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
            "html_review": str(output_dir / "review.html"),
            "html_report": str(output_dir / "report.html"),
            "review_queue": str(output_dir / "review_queue.csv"),
            "issues": str(output_dir / "issues.csv"),
            "field_statistics": str(output_dir / "field_statistics.csv"),
            "field_values": str(output_dir / "field_values.csv"),
            "learned_rules": str(output_dir / "learned_rules.csv"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_html_review(result, output_dir, dataset_root, high_risk_threshold)
    return summary


def audit_dataset(
    dataset_root: Path,
    schema_path: Path | None = DEFAULT_SCHEMA_PATH,
    min_rule_support: int = 3,
    rule_confidence: float = 0.80,
) -> AuditResult:
    dataset_root = dataset_root.resolve()
    samples, load_errors = load_samples(dataset_root)
    resolved_schema_path = schema_path.resolve() if schema_path else None
    target_fields: tuple[str, ...] = ()
    if resolved_schema_path is not None:
        try:
            schema = json.loads(resolved_schema_path.read_text(encoding="utf-8"))
            target_fields = schema_leaf_paths(schema)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            # run_schema_checks reports the actionable schema loading error.
            pass
    load_errors.extend(run_schema_checks(samples, resolved_schema_path))
    run_semantic_checks(samples)
    run_intra_document_checks(samples)
    run_numeric_outlier_checks(samples)
    typo_candidates = run_distribution_checks(samples)
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
        target_fields=target_fields,
        typo_candidates=tuple(typo_candidates),
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
