from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Sequence

from src.utils.annotation_audit import (
    DEFAULT_DATASET_ROOT,
    Sample,
    display_value,
    get_path,
    load_samples,
    normalize_text,
)


@dataclass(frozen=True)
class LocationSpec:
    name: str
    path: str
    other_path: str


@dataclass(frozen=True)
class CompletedExample:
    sample_id: str
    sample: Sample
    city: str
    city_key: str
    raw_suffix: str
    canonical_suffix: str


@dataclass(frozen=True)
class LocationRule:
    field: str
    city: str
    country_suffix: str
    support: int
    example_ids: tuple[str, ...]
    competing_suffixes: str


@dataclass(frozen=True)
class LocationChange:
    sample_id: str
    split: str
    annotation_path: str
    field: str
    old_value: str
    new_value: str
    country_suffix: str
    confidence: float
    support: int
    context_score: float
    evidence_document_ids: tuple[str, ...]
    matched_context: tuple[str, ...]
    applied: bool = False
    backup_path: str = ""


@dataclass(frozen=True)
class SkippedCandidate:
    sample_id: str
    split: str
    annotation_path: str
    field: str
    value: str
    reason: str
    best_suffix: str = ""
    confidence: float = 0.0
    support: int = 0
    context_score: float = 0.0


LOCATION_SPECS = (
    LocationSpec(
        "taking_over",
        "takingOverTheGoods.takingOverTheGoodsPlace",
        "deliveryOfTheGoods.logisticsLocationCity",
    ),
    LocationSpec(
        "delivery",
        "deliveryOfTheGoods.logisticsLocationCity",
        "takingOverTheGoods.takingOverTheGoodsPlace",
    ),
)


CONTEXT_FIELDS: tuple[tuple[str, float], ...] = (
    ("senderInformation.senderNameCompany", 2.5),
    ("senderInformation.senderPostcode", 2.0),
    ("senderInformation.senderCountryCode.value", 1.0),
    ("consigneeInformation.consigneeNameCompany", 2.5),
    ("consigneeInformation.consigneePostcode", 2.0),
    ("consigneeInformation.consigneeCountryCode.value", 1.0),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Learn CITY-country conventions for takingOverTheGoodsPlace and "
            "logisticsLocationCity, then optionally apply high-confidence normalizations."
        )
    )
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root using train/val/test metadata.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Manifest and backup directory (default: <dataset_root>/location_normalization).",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=2,
        help="Minimum completed peer examples for an otherwise context-free rule.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.60,
        help="Minimum winning score fraction when country-name variants compete.",
    )
    parser.add_argument(
        "--strong-context-score",
        type=float,
        default=10.0,
        help="Context score that permits a rule supported by only one close peer.",
    )
    parser.add_argument(
        "--min-context-score",
        type=float,
        default=5.0,
        help="Minimum context similarity required for every proposed change.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Back up and update annotation JSON files. Without this flag, only a dry-run manifest is written.",
    )
    args = parser.parse_args(argv)
    if args.min_support < 1:
        parser.error("--min-support must be at least 1")
    if not 0.5 <= args.min_confidence <= 1.0:
        parser.error("--min-confidence must be between 0.5 and 1.0")
    if args.strong_context_score < 0:
        parser.error("--strong-context-score cannot be negative")
    if args.min_context_score < 0:
        parser.error("--min-context-score cannot be negative")
    return args


def split_location(value: Any) -> tuple[str, str | None] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if "," not in text:
        return text, None
    city, suffix = (part.strip() for part in text.split(",", 1))
    if not city or not suffix:
        return None
    return city, suffix


def country_equivalent(left: str, right: str) -> bool:
    left_key, right_key = normalize_text(left), normalize_text(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    if min(len(left_key), len(right_key)) >= 6 and (
        left_key in right_key or right_key in left_key
    ):
        return True
    return SequenceMatcher(None, left_key, right_key).ratio() >= 0.84


def display_quality(value: str) -> tuple[int, int, int, int]:
    stripped = value.strip()
    odd_characters = len(re.findall(r"[^\w\s.-]", stripped, flags=re.UNICODE))
    return (
        int(bool(stripped) and stripped[0].isupper()),
        int(not stripped.isupper()),
        -odd_characters,
        -len(stripped),
    )


def build_suffix_aliases(
    raw_suffixes: list[str],
) -> tuple[dict[str, str], dict[str, Counter[str]]]:
    """Cluster spelling/case/OCR variants and choose a frequent clean display form."""
    counts = Counter(raw_suffixes)
    unique = sorted(counts)
    parents = list(range(len(unique)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(unique)):
        for right in range(left + 1, len(unique)):
            if country_equivalent(unique[left], unique[right]):
                union(left, right)
    groups: dict[int, list[str]] = defaultdict(list)
    for index, suffix in enumerate(unique):
        groups[find(index)].append(suffix)
    canonical_by_raw: dict[str, str] = {}
    variants_by_canonical: dict[str, Counter[str]] = {}
    for members in groups.values():
        canonical = max(
            members,
            key=lambda value: (counts[value], display_quality(value), value),
        )
        variants_by_canonical[canonical] = Counter(
            {member: counts[member] for member in members}
        )
        for member in members:
            canonical_by_raw[member] = canonical
    return canonical_by_raw, variants_by_canonical


def completed_examples(
    samples: list[Sample], spec: LocationSpec
) -> tuple[list[CompletedExample], dict[str, Counter[str]]]:
    parsed_rows: list[tuple[Sample, str, str]] = []
    for sample in samples:
        parsed = split_location(get_path(sample.content, spec.path))
        if parsed is None or parsed[1] is None:
            continue
        parsed_rows.append((sample, parsed[0], parsed[1]))
    aliases, variants = build_suffix_aliases([row[2] for row in parsed_rows])
    examples = [
        CompletedExample(
            sample_id=sample.sample_id,
            sample=sample,
            city=city,
            city_key=normalize_text(city),
            raw_suffix=suffix,
            canonical_suffix=aliases[suffix],
        )
        for sample, city, suffix in parsed_rows
    ]
    return examples, variants


def text_values_match(path: str, left: Any, right: Any) -> bool:
    left_key, right_key = normalize_text(left), normalize_text(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    if "NameCompany" in path:
        ratio = SequenceMatcher(None, left_key, right_key).ratio()
        return ratio >= 0.82 or (
            min(len(left_key), len(right_key)) >= 8
            and (left_key in right_key or right_key in left_key)
        )
    if "Postcode" in path:
        left_digits = re.sub(r"\D", "", str(left))
        right_digits = re.sub(r"\D", "", str(right))
        return bool(left_digits) and left_digits == right_digits
    return False


def context_similarity(
    candidate: Sample, example: Sample, spec: LocationSpec
) -> tuple[float, tuple[str, ...]]:
    score = 0.0
    matched: list[str] = []
    for path, weight in CONTEXT_FIELDS:
        if text_values_match(
            path, get_path(candidate.content, path), get_path(example.content, path)
        ):
            score += weight
            matched.append(path)
    candidate_other = split_location(get_path(candidate.content, spec.other_path))
    example_other = split_location(get_path(example.content, spec.other_path))
    if candidate_other and example_other:
        if normalize_text(candidate_other[0]) == normalize_text(example_other[0]):
            score += 4.0
            matched.append(f"{spec.other_path}:city")
        if (
            candidate_other[1]
            and example_other[1]
            and country_equivalent(candidate_other[1], example_other[1])
        ):
            score += 8.0
            matched.append(f"{spec.other_path}:country")
    return score, tuple(matched)


def learn_rules(
    samples: list[Sample], spec: LocationSpec
) -> tuple[dict[str, list[CompletedExample]], list[LocationRule]]:
    examples, _ = completed_examples(samples, spec)
    by_city: dict[str, list[CompletedExample]] = defaultdict(list)
    for example in examples:
        by_city[example.city_key].append(example)
    rules: list[LocationRule] = []
    for city_key, city_examples in sorted(by_city.items()):
        by_suffix: dict[str, list[CompletedExample]] = defaultdict(list)
        for example in city_examples:
            by_suffix[example.canonical_suffix].append(example)
        for suffix, suffix_examples in sorted(by_suffix.items()):
            competitors = "; ".join(
                f"{name} ({len(rows)})"
                for name, rows in sorted(
                    by_suffix.items(), key=lambda item: (-len(item[1]), item[0])
                )
                if name != suffix
            )
            rules.append(
                LocationRule(
                    field=spec.path,
                    city=city_examples[0].city,
                    country_suffix=suffix,
                    support=len(suffix_examples),
                    example_ids=tuple(example.sample_id for example in suffix_examples),
                    competing_suffixes=competitors,
                )
            )
    return by_city, rules


def propose_for_sample(
    sample: Sample,
    spec: LocationSpec,
    examples_by_city: dict[str, list[CompletedExample]],
    dataset_root: Path,
    min_support: int,
    min_confidence: float,
    strong_context_score: float,
    min_context_score: float,
) -> tuple[LocationChange | None, SkippedCandidate | None]:
    parsed = split_location(get_path(sample.content, spec.path))
    if parsed is None or parsed[1] is not None:
        return None, None
    city = parsed[0]
    city_examples = examples_by_city.get(normalize_text(city), [])
    annotation_path = str(sample.annotation_path.relative_to(dataset_root))
    if not city_examples:
        return None, SkippedCandidate(
            sample.sample_id,
            sample.split,
            annotation_path,
            spec.path,
            city,
            "No completed peer uses the same city.",
        )
    by_suffix: dict[str, list[CompletedExample]] = defaultdict(list)
    for example in city_examples:
        by_suffix[example.canonical_suffix].append(example)
    scored: list[
        tuple[float, str, list[CompletedExample], float, tuple[str, ...], str]
    ] = []
    for suffix, suffix_examples in by_suffix.items():
        evidence = []
        for example in suffix_examples:
            context_score, matched = context_similarity(sample, example.sample, spec)
            evidence.append((context_score, matched, example))
        best_context, best_matched, best_example = max(
            evidence, key=lambda item: (item[0], item[2].sample_id)
        )
        rule_score = 1.0 + best_context + 2.0 * math.log2(1 + len(suffix_examples))
        scored.append(
            (
                rule_score,
                suffix,
                suffix_examples,
                best_context,
                best_matched,
                best_example.sample_id,
            )
        )
    scored.sort(key=lambda row: (-row[0], -len(row[2]), row[1]))
    winner = scored[0]
    total_score = sum(row[0] for row in scored)
    confidence = winner[0] / total_score if total_score else 0.0
    support = len(winner[2])
    enough_support = support >= min_support or winner[3] >= strong_context_score
    enough_context = winner[3] >= min_context_score
    if confidence < min_confidence or not enough_support or not enough_context:
        reasons = []
        if confidence < min_confidence:
            reasons.append(
                f"Country-name variants are ambiguous ({confidence:.0%} confidence)."
            )
        if not enough_support:
            reasons.append(
                f"Only {support} peer(s) and context score {winner[3]:g}; more evidence is required."
            )
        if not enough_context:
            reasons.append(
                f"Best peer context score is {winner[3]:g}, below the required {min_context_score:g}."
            )
        return None, SkippedCandidate(
            sample.sample_id,
            sample.split,
            annotation_path,
            spec.path,
            city,
            " ".join(reasons),
            winner[1],
            confidence,
            support,
            winner[3],
        )
    evidence_ranked = sorted(
        winner[2],
        key=lambda example: (
            -context_similarity(sample, example.sample, spec)[0],
            example.sample_id,
        ),
    )
    return (
        LocationChange(
            sample_id=sample.sample_id,
            split=sample.split,
            annotation_path=annotation_path,
            field=spec.path,
            old_value=city,
            new_value=f"{city}, {winner[1]}",
            country_suffix=winner[1],
            confidence=confidence,
            support=support,
            context_score=winner[3],
            evidence_document_ids=tuple(
                example.sample_id for example in evidence_ranked[:5]
            ),
            matched_context=winner[4],
        ),
        None,
    )


def propose_changes(
    samples: list[Sample],
    dataset_root: Path,
    min_support: int = 2,
    min_confidence: float = 0.60,
    strong_context_score: float = 10.0,
    min_context_score: float = 5.0,
) -> tuple[list[LocationChange], list[SkippedCandidate], list[LocationRule]]:
    changes: list[LocationChange] = []
    skipped: list[SkippedCandidate] = []
    all_rules: list[LocationRule] = []
    for spec in LOCATION_SPECS:
        examples_by_city, rules = learn_rules(samples, spec)
        all_rules.extend(rules)
        for sample in samples:
            change, skipped_candidate = propose_for_sample(
                sample,
                spec,
                examples_by_city,
                dataset_root,
                min_support,
                min_confidence,
                strong_context_score,
                min_context_score,
            )
            if change:
                changes.append(change)
            if skipped_candidate:
                skipped.append(skipped_candidate)
    return changes, skipped, all_rules


def set_path(content: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current: Any = content
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Cannot set absent path {path!r}")
        current = current[part]
    if not isinstance(current, dict) or parts[-1] not in current:
        raise KeyError(f"Cannot set absent path {path!r}")
    current[parts[-1]] = value


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def change_row(change: LocationChange) -> dict[str, Any]:
    row = asdict(change)
    row["evidence_document_ids"] = " | ".join(change.evidence_document_ids)
    row["matched_context"] = " | ".join(change.matched_context)
    return row


def rule_row(rule: LocationRule) -> dict[str, Any]:
    row = asdict(rule)
    row["example_ids"] = " | ".join(rule.example_ids)
    return row


def write_cumulative_manifest(output_dir: Path, fieldnames: list[str]) -> Path:
    timestamped_pattern = re.compile(r"applied_changes_\d{8}_\d{6}\.csv$")
    cumulative_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for manifest_path in sorted(output_dir.glob("applied_changes_*.csv")):
        if not timestamped_pattern.fullmatch(manifest_path.name):
            continue
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (
                    row.get("sample_id", ""),
                    row.get("field", ""),
                    row.get("old_value", ""),
                    row.get("new_value", ""),
                )
                if key not in seen:
                    seen.add(key)
                    cumulative_rows.append(row)
    cumulative_path = output_dir / "all_applied_changes.csv"
    write_csv(cumulative_path, cumulative_rows, fieldnames)
    return cumulative_path


def apply_changes(
    samples: list[Sample],
    changes: list[LocationChange],
    dataset_root: Path,
    output_dir: Path,
    timestamp: str,
) -> list[LocationChange]:
    by_id = {sample.sample_id: sample for sample in samples}
    backup_root = output_dir / "backups" / timestamp
    changed_by_annotation: dict[Path, list[LocationChange]] = defaultdict(list)
    for change in changes:
        changed_by_annotation[by_id[change.sample_id].annotation_path].append(change)
    applied: list[LocationChange] = []
    for annotation_path, annotation_changes in changed_by_annotation.items():
        relative_path = annotation_path.relative_to(dataset_root)
        backup_path = backup_root / relative_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(annotation_path, backup_path)
        sample = by_id[annotation_changes[0].sample_id]
        for change in annotation_changes:
            current_value = get_path(sample.content, change.field)
            if current_value != change.old_value:
                raise RuntimeError(
                    f"Refusing stale change for {change.sample_id}: {change.field} is "
                    f"{display_value(current_value)!r}, expected {change.old_value!r}."
                )
            set_path(sample.content, change.field, change.new_value)
            applied.append(
                LocationChange(
                    **{
                        **asdict(change),
                        "applied": True,
                        "backup_path": str(backup_path.relative_to(dataset_root)),
                    }
                )
            )
        temporary_path = annotation_path.with_suffix(annotation_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(sample.annotation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, annotation_path)
    return applied


def run_normalization(
    dataset_root: Path,
    output_dir: Path,
    min_support: int,
    min_confidence: float,
    strong_context_score: float,
    min_context_score: float,
    apply: bool,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    output_dir = output_dir.resolve()
    samples, load_warnings = load_samples(dataset_root)
    changes, skipped, rules = propose_changes(
        samples,
        dataset_root,
        min_support,
        min_confidence,
        strong_context_score,
        min_context_score,
    )
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    applied = (
        apply_changes(samples, changes, dataset_root, output_dir, timestamp)
        if apply
        else []
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    proposal_rows = [change_row(change) for change in changes]
    change_fields = list(proposal_rows[0]) if proposal_rows else list(asdict(LocationChange("", "", "", "", "", "", "", 0, 0, 0, (), ())))
    write_csv(output_dir / "proposed_changes.csv", proposal_rows, change_fields)
    skipped_rows = [asdict(candidate) for candidate in skipped]
    skipped_fields = list(skipped_rows[0]) if skipped_rows else list(asdict(SkippedCandidate("", "", "", "", "", "")))
    write_csv(output_dir / "skipped_candidates.csv", skipped_rows, skipped_fields)
    rule_rows = [rule_row(rule) for rule in rules]
    rule_fields = list(rule_rows[0]) if rule_rows else list(asdict(LocationRule("", "", "", 0, (), "")))
    write_csv(output_dir / "learned_location_rules.csv", rule_rows, rule_fields)
    applied_manifest = ""
    cumulative_manifest = ""
    if apply:
        applied_rows = [change_row(change) for change in applied]
        manifest_path = output_dir / f"applied_changes_{timestamp}.csv"
        write_csv(manifest_path, applied_rows, change_fields)
        applied_manifest = str(manifest_path)
        (output_dir / f"applied_changes_{timestamp}.json").write_text(
            json.dumps(applied_rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        cumulative_manifest = str(write_cumulative_manifest(output_dir, change_fields))
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": "apply" if apply else "dry_run",
        "dataset_root": str(dataset_root),
        "samples": len(samples),
        "proposed_changes": len(changes),
        "applied_changes": len(applied),
        "skipped_city_only_candidates": len(skipped),
        "changes_by_field": dict(Counter(change.field for change in changes)),
        "thresholds": {
            "min_support": min_support,
            "min_confidence": min_confidence,
            "strong_context_score": strong_context_score,
            "min_context_score": min_context_score,
        },
        "load_warnings": load_warnings,
        "applied_manifest": applied_manifest,
        "cumulative_manifest": cumulative_manifest,
        "backup_root": str(output_dir / "backups" / timestamp) if apply else "",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")
    output_dir = (
        args.output_dir or dataset_root / "location_normalization"
    ).resolve()
    summary = run_normalization(
        dataset_root,
        output_dir,
        args.min_support,
        args.min_confidence,
        args.strong_context_score,
        args.min_context_score,
        args.apply,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
