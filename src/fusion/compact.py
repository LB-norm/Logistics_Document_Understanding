"""Compact LLM evidence while retaining the detailed fusion export for auditing."""
from __future__ import annotations

import math
from typing import Any


COMPACT_INPUT_INSTRUCTIONS = (
    "\nThe input is compact OCR/layout JSON. Coordinates are normalized to [0,1], "
    "origin top-left; polygons list four boundary-ordered [x,y] points and boxes use xyxy. "
    "Within each page, lines are [line_id,text,polygon,ocr_confidence] and regions are "
    "[region_id,label,box,layout_confidence]. IDs are page-local integers in separate "
    "line and region namespaces; line order is the original OCR order. "
    "Memberships are [line_id,[[region_id,overlap,weight],...]]. Empty candidates mean "
    "unassigned text. Separate mode has no memberships: relate lines to regions yourself. "
    "Hard mode has at most one region per line; soft mode can have multiple. "
    "Overlap is intersection area divided by OCR polygon area; weights are relative "
    "geometric overlaps, not calibrated probabilities. Region labels and memberships "
    "are evidence, not authoritative semantic assignments. Each OCR line's text appears "
    "once; use IDs to relate it to regions and avoid counting it twice. Consider all "
    "lines, including unassigned text. Numeric values are rounded for presentation; "
    "memberships were computed before rounding."
)


def round_numbers(value: Any, decimals: int) -> Any:
    """Round only evidence values, never targets or schemas."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Fusion evidence contains a non-finite number")
        result = round(value, decimals)
        return 0.0 if result == 0 else result
    if isinstance(value, list):
        return [round_numbers(item, decimals) for item in value]
    if isinstance(value, dict):
        return {key: round_numbers(item, decimals) for key, item in value.items()}
    return value


def compact_document(document: dict[str, Any], decimals: int = 4) -> dict[str, Any]:
    """Keep text, polygons, region boxes/scores and existing candidate assignments.

    No overlaps are recomputed. Bboxes derived from polygons, duplicate text,
    region field copies and execution metadata are omitted. IDs follow original
    collection order, independently on each page and in each namespace.
    """
    if isinstance(decimals, bool) or not isinstance(decimals, int) or not 0 <= decimals <= 10:
        raise ValueError("decimals must be an integer between 0 and 10")
    mode = document["mode"]
    if mode not in {"separate", "hard", "soft"}:
        raise ValueError(f"Unsupported fusion mode: {mode!r}")
    result: dict[str, Any] = {"mode": mode, "coordinates": "normalized_0_1", "pages": []}
    page_numbers = set()
    for page in document["pages"]:
        number = page["page_number"]
        if number in page_numbers:
            raise ValueError("Duplicate page number")
        page_numbers.add(number)
        lines, regions = page["ocr"]["lines"], page["layout"]["detections"]
        if any(part["coordinate_space"] != "normalized_0_1"
               for part in (page["ocr"], page["layout"])):
            raise ValueError("Compact conversion requires normalized source geometry")
        line_ids = {line["id"]: index for index, line in enumerate(lines)}
        region_ids = {region["id"]: index for index, region in enumerate(regions)}
        if len(line_ids) != len(lines) or len(region_ids) != len(regions):
            raise ValueError("Duplicate OCR or region ID")
        compact_page = {
            "page": number,
            "lines": [[index, line["text"], line["polygon"], line["confidence"]]
                      for index, line in enumerate(lines)],
            "regions": [[index, region["label"], region["bbox_xyxy"], region["score"]]
                        for index, region in enumerate(regions)],
        }
        if mode != "separate":
            memberships = page["memberships"]
            if (len(memberships) != len(lines)
                    or {item["ocr_id"] for item in memberships} != set(line_ids)):
                raise ValueError("Memberships must cover each OCR line exactly once")
            compact_memberships = []
            for membership in memberships:
                candidates = membership["candidates"]
                if mode == "hard" and len(candidates) > 1:
                    raise ValueError("Hard fusion cannot have multiple candidates")
                candidate_ids = [candidate["region_id"] for candidate in candidates]
                if len(candidate_ids) != len(set(candidate_ids)):
                    raise ValueError("Duplicate membership candidate")
                compact_memberships.append([
                    line_ids[membership["ocr_id"]],
                    [[region_ids[candidate["region_id"]], candidate["overlap"], candidate["weight"]]
                     for candidate in candidates],
                ])
            compact_page["memberships"] = compact_memberships
        result["pages"].append(compact_page)
    if not result["pages"]:
        raise ValueError("Fusion document contains no pages")
    return round_numbers(result, decimals)
