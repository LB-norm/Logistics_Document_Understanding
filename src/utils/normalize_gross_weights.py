from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Sequence


DEFAULT_DATASET_ROOT = Path("data/datasets/250_CMRS_240dpi_20260707")
SPLITS = ("train", "val")

# grossWeightInKg is a flat object in the annotation schema. Limiting the
# replacement to its object body prevents similarly named fields elsewhere in
# an annotation from being changed.
GROSS_WEIGHT_OBJECT = re.compile(
    r'("grossWeightInKg"\s*:\s*\{)([^{}]*)(\})',
    flags=re.DOTALL,
)
ZERO_FRACTION_WEIGHT = re.compile(
    r'("supplyChainConsignmentItemGrossWeight"\s*:\s*)'
    r'(-?(?:0|[1-9]\d*))\.0+([eE][+-]?\d+)?(?=\s*(?:,|$))'
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove zero-only decimal suffixes from grossWeightInKg values in "
            "train and validation annotations."
        )
    )
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=(
            "Dataset directory containing train/ and val/. A single nested "
            "dataset directory is detected automatically."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report required changes without writing any files.",
    )
    return parser.parse_args(argv)


def resolve_dataset_root(path: Path) -> Path:
    path = path.resolve()
    if all((path / split / "annotations").is_dir() for split in SPLITS):
        return path

    candidates = [
        child
        for child in path.iterdir()
        if child.is_dir()
        and all((child / split / "annotations").is_dir() for split in SPLITS)
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"Could not uniquely locate train/annotations and val/annotations below {path}"
    )


def normalize_annotation_text(text: str) -> tuple[str, int]:
    replacements = 0

    def normalize_object(match: re.Match[str]) -> str:
        nonlocal replacements
        prefix, body, suffix = match.groups()

        def remove_zero_fraction(weight_match: re.Match[str]) -> str:
            nonlocal replacements
            replacements += 1
            field_prefix, integer, exponent = weight_match.groups()
            return f"{field_prefix}{integer}{exponent or ''}"

        normalized_body = ZERO_FRACTION_WEIGHT.sub(remove_zero_fraction, body)
        return f"{prefix}{normalized_body}{suffix}"

    return GROSS_WEIGHT_OBJECT.sub(normalize_object, text), replacements


def write_exact(path: Path, text: str) -> None:
    """Atomically replace a file without normalizing whitespace or newlines."""
    original_mode = path.stat().st_mode
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(text)
            temporary_path = Path(temporary_file.name)
        temporary_path.chmod(original_mode)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def normalize_file(path: Path, *, check: bool) -> int:
    with path.open("r", encoding="utf-8", newline="") as annotation_file:
        original = annotation_file.read()
    json.loads(original)

    normalized, replacements = normalize_annotation_text(original)
    if replacements:
        json.loads(normalized)
        if not check:
            write_exact(path, normalized)
    return replacements


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        dataset_root = resolve_dataset_root(args.dataset_root)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error

    files_scanned = 0
    files_affected = 0
    values_affected = 0
    for split in SPLITS:
        annotation_root = dataset_root / split / "annotations"
        for annotation_path in sorted(annotation_root.rglob("*.json")):
            files_scanned += 1
            replacements = normalize_file(annotation_path, check=args.check)
            if replacements:
                files_affected += 1
                values_affected += replacements

    action = "Would update" if args.check else "Updated"
    print(f"Dataset: {dataset_root}")
    print(f"Scanned {files_scanned} train/val annotation files.")
    print(f"{action} {values_affected} values in {files_affected} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
