#!/usr/bin/env python3
"""Extract German product names from an Open Products Facts JSONL dump."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(__file__).with_name("openproductsfacts-products.jsonl")
DEFAULT_OUTPUT = Path(__file__).with_name("product_names_de.jsonl")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    value = html.unescape(str(value))
    return WHITESPACE_RE.sub(" ", value).strip()


def iter_product_names(input_path: Path):
    seen: set[tuple[str, str]] = set()

    with input_path.open(encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            try:
                product = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}") from exc

            product_name_de = clean_text(product.get("product_name_de"))
            if not product_name_de:
                continue

            quantity = clean_text(product.get("quantity"))
            dedupe_key = (product_name_de.casefold(), quantity.casefold())
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            yield {
                "product_name_de": product_name_de,
                "quantity": quantity or None,
                "code": clean_text(product.get("code") or product.get("_id")) or None,
                "brands": clean_text(product.get("brands")) or None,
            }


def extract_product_names(input_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with output_path.open("w", encoding="utf-8") as outfile:
        for record in iter_product_names(input_path):
            json.dump(record, outfile, ensure_ascii=False, sort_keys=True)
            outfile.write("\n")
            count += 1

    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract product_name_de and quantity fields from Open Products Facts JSONL."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to source JSONL. Defaults to {DEFAULT_INPUT}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to extracted JSONL. Defaults to {DEFAULT_OUTPUT}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = extract_product_names(args.input, args.output)
    print(f"Wrote {count} German product records to {args.output}")


if __name__ == "__main__":
    main()
