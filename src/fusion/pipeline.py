"""Fuse saved PP-OCR and YOLO predictions without loading either model."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any


MODE_NAMES = {1: "separate", 2: "hard", 3: "soft"}
PIXELS = "input_image_pixels"
NORMALIZED = "normalized_0_1"


@dataclass(frozen=True)
class FusionConfig:
    mode: int | str = 1
    min_overlap: float = 0.1
    allow_layout_rescale: bool = False

    def __post_init__(self) -> None:
        aliases = {str(key): key for key in MODE_NAMES}
        aliases.update({value: key for key, value in MODE_NAMES.items()})
        mode = aliases.get(str(self.mode))
        if mode is None or isinstance(self.mode, bool):
            raise ValueError("mode must be 1/separate, 2/hard, or 3/soft")
        object.__setattr__(self, "mode", mode)
        if (isinstance(self.min_overlap, bool) or not isinstance(self.min_overlap, (int, float))
                or not math.isfinite(self.min_overlap) or not 0 <= self.min_overlap <= 1):
            raise ValueError("min_overlap must be finite and between 0 and 1")
        if not isinstance(self.allow_layout_rescale, bool):
            raise ValueError("allow_layout_rescale must be a boolean")


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _page_number(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("page_number must be a positive integer")
    return value


def _dimensions(page: dict[str, Any]) -> tuple[float, float]:
    dimensions = tuple(_number(page[key], key) for key in ("width", "height"))
    if min(dimensions) <= 0:
        raise ValueError("Page width and height must be positive")
    return dimensions


def _point(point: Any, dimensions: tuple[float, float], space: str) -> list[float]:
    if len(point) != 2:
        raise ValueError("Polygon points must contain x and y")
    values = [_number(value, "coordinate") for value in point]
    if space == PIXELS:
        values = [value / extent for value, extent in zip(values, dimensions)]
    elif space != NORMALIZED:
        raise ValueError(f"Unsupported coordinate_space: {space}")
    if any(value < -1e-8 or value > 1 + 1e-8 for value in values):
        raise ValueError("Coordinates lie outside the page; check coordinate convention and dimensions")
    return [max(0.0, min(1.0, value)) for value in values]


def _box(box: Any, dimensions: tuple[float, float], space: str) -> list[float]:
    if len(box) != 4:
        raise ValueError("Bounding boxes must contain four xyxy coordinates")
    x1, y1 = _point(box[:2], dimensions, space)
    x2, y2 = _point(box[2:], dimensions, space)
    if x2 < x1 or y2 < y1:
        raise ValueError("Bounding box xyxy coordinates are reversed")
    return [x1, y1, x2, y2]


def _area(polygon: list[list[float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    # Translate to the first vertex to avoid cancellation for tiny text boxes.
    x, y = polygon[0]
    return abs(sum((a[0] - x) * (b[1] - y) - (b[0] - x) * (a[1] - y)
                   for a, b in zip(polygon[1:-1], polygon[2:]))) / 2


def _validate_polygon(polygon: list[list[float]]) -> None:
    if len(polygon) != 4:
        raise ValueError("OCR polygons must be quadrilaterals in boundary order")
    turns = []
    for index in range(4):
        a, b, c = (polygon[(index + offset) % 4] for offset in range(3))
        turn = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(turn) > 1e-15:
            turns.append(turn > 0)
    if turns and any(turn != turns[0] for turn in turns):
        raise ValueError("OCR polygons must be convex and cannot self-intersect")


def _overlap(polygon: list[list[float]], box: list[float]) -> float:
    """Clip a convex OCR polygon against a rectangle; divide by OCR area."""
    area = _area(polygon)
    if area == 0:
        return 0.0
    xs, ys = zip(*polygon)
    if max(xs) <= box[0] or min(xs) >= box[2] or max(ys) <= box[1] or min(ys) >= box[3]:
        return 0.0
    if all(box[0] <= x <= box[2] and box[1] <= y <= box[3] for x, y in polygon):
        return 1.0
    clipped = polygon
    for axis, boundary, lower in ((0, box[0], True), (0, box[2], False),
                                  (1, box[1], True), (1, box[3], False)):
        if not clipped:
            break
        output = []
        previous = clipped[-1]
        previous_inside = previous[axis] >= boundary if lower else previous[axis] <= boundary
        for current in clipped:
            current_inside = current[axis] >= boundary if lower else current[axis] <= boundary
            if current_inside != previous_inside:
                fraction = (boundary - previous[axis]) / (current[axis] - previous[axis])
                output.append([previous[i] + fraction * (current[i] - previous[i]) for i in (0, 1)])
            if current_inside:
                output.append(current)
            previous, previous_inside = current, current_inside
        clipped = output
    overlap = max(0.0, min(1.0, _area(clipped) / area))
    # Clipping can leave a microscopic sliver at a shared edge or vertex.
    if overlap <= 1e-12:
        return 0.0
    return 1.0 if overlap >= 1 - 1e-12 else overlap


def _identifier(value: Any, default: str, used: set[str]) -> str:
    identifier = default if value is None else value
    if not isinstance(identifier, str) or not identifier or identifier in used:
        raise ValueError("OCR and region IDs must be nonempty strings and unique within their page")
    used.add(identifier)
    return identifier


def _normalize_layout(layout: dict[str, Any], page_number: int) -> dict[str, Any]:
    result = deepcopy(layout)
    dimensions = _dimensions(layout)
    space = layout.get("coordinate_space", PIXELS)
    if space not in {PIXELS, NORMALIZED}:
        raise ValueError(f"Unsupported coordinate_space: {space}")
    used: set[str] = set()
    for index, region in enumerate(result["detections"]):
        region["id"] = _identifier(region.get("id"), f"p{page_number:04d}-r{index:06d}", used)
        if not isinstance(region["label"], str):
            raise ValueError("YOLO label must be a string")
        score = _number(region["score"], "YOLO score")
        if not 0 <= score <= 1:
            raise ValueError("YOLO score must be between 0 and 1")
        if "bbox_xyxy" in region:
            box = region["bbox_xyxy"]
        else:
            x, y, width, height = (_number(value, "bbox_xywh") for value in region["bbox_xywh"])
            box = [x, y, x + width, y + height]
        x1, y1, x2, y2 = _box(box, dimensions, space)
        if x2 == x1 or y2 == y1:
            raise ValueError("YOLO regions must have positive area")
        region["bbox_xyxy"] = [x1, y1, x2, y2]
        if "bbox_xywh" in region:
            region["bbox_xywh"] = [x1, y1, x2 - x1, y2 - y1]
    result.update(page_number=page_number, coordinate_space=NORMALIZED)
    return result


def _normalize_ocr(page: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(page)
    dimensions = _dimensions(page)
    space = page.get("coordinate_space", PIXELS)
    if space not in {PIXELS, NORMALIZED}:
        raise ValueError(f"Unsupported coordinate_space: {space}")
    used: set[str] = set()
    for index, line in enumerate(result["lines"]):
        line["id"] = _identifier(line.get("id"), f"p{page['page_number']:04d}-l{index:06d}", used)
        if not isinstance(line["text"], str):
            raise ValueError("OCR text must be a string")
        confidence = _number(line["confidence"], "OCR confidence")
        if not 0 <= confidence <= 1:
            raise ValueError("OCR confidence must be between 0 and 1")
        # PP-OCR has polygons; bbox-only lines are also accepted as rectangles.
        if "polygon" in line:
            polygon = [_point(point, dimensions, space) for point in line["polygon"]]
            _validate_polygon(polygon)
            xs, ys = zip(*polygon)
            box = [min(xs), min(ys), max(xs), max(ys)]
        else:
            x1, y1, x2, y2 = _box(line["bbox"], dimensions, space)
            box = [x1, y1, x2, y2]
            polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        line.update(polygon=polygon, bbox=box)
    result["coordinate_space"] = NORMALIZED
    return result


def _match_layouts(ocr: dict[str, Any], layout: dict[str, Any] | list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    pages = ocr["pages"]
    numbers = [_page_number(page["page_number"]) for page in pages]
    if not numbers or len(numbers) != len(set(numbers)):
        raise ValueError("OCR must contain pages with unique page_number values")
    records = layout if isinstance(layout, list) else [layout]
    if len(pages) == 1 and records and all("page_number" not in record for record in records):
        if len(records) > 1:
            source = ocr.get("source_path")
            exact = [record for record in records if source and record.get("image") == source]
            matches = exact or [record for record in records if source and record.get("image")
                                and Path(record["image"]).name == Path(source).name]
            if len(matches) != 1:
                raise ValueError("Cannot uniquely match OCR source_path to YOLO images; supply one image prediction")
            records = matches
        return {numbers[0]: records[0]}
    matched = {}
    for record in records:
        if "page_number" not in record:
            raise ValueError("Multi-page layouts require an explicit page_number on every image prediction")
        number = _page_number(record["page_number"])
        if number in matched:
            raise ValueError(f"Duplicate layout page_number: {number}")
        matched[number] = record
    if set(matched) != set(numbers):
        raise ValueError("Layout page numbers must exactly match OCR pages (use detections: [] for empty pages)")
    return matched


class FusionPipeline:
    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()

    def fuse(self, ocr: dict[str, Any], layout: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
        """Return a JSON-serializable document, leaving both input objects untouched."""
        matched = _match_layouts(ocr, layout)
        result = {
            "format_version": 1,
            "representation": "ocr_layout_fusion",
            "mode": MODE_NAMES[self.config.mode],
            "coordinate_system": {"space": NORMALIZED, "origin": "top_left", "bbox_format": "xyxy"},
            "settings": asdict(self.config),
            "ocr_metadata": deepcopy({key: value for key, value in ocr.items() if key != "pages"}),
            "pages": [],
        }
        if self.config.mode != 1:
            result["assignment"] = {
                "overlap_metric": "intersection_area / ocr_polygon_area",
                "weight_kind": "relative_geometric_overlap",
                "text_order": "input_ocr_order",
                "hard_tie_break": ["highest_overlap", "smallest_region_area", "highest_yolo_score", "region_id"],
            }
        for page in ocr["pages"]:
            number = page["page_number"]
            prediction = matched[number]
            if _dimensions(page) != _dimensions(prediction) and not self.config.allow_layout_rescale:
                raise ValueError(f"Page {number}: OCR and YOLO dimensions differ; use the same raster or explicitly allow layout rescaling")
            fused = {
                "page_number": number, "width": page["width"], "height": page["height"],
                "ocr": _normalize_ocr(page), "layout": _normalize_layout(prediction, number),
            }
            if self.config.mode != 1:
                self._assign(fused)
            result["pages"].append(fused)
        return result

    def _assign(self, page: dict[str, Any]) -> None:
        regions = page["layout"]["detections"]
        fields = [{"region_id": region["id"], "label": region["label"],
                   "category_id": region.get("category_id"), "bbox": region["bbox_xyxy"],
                   "layout_confidence": region["score"], "ocr_ids": [], "text": ""}
                  for region in regions]
        by_id = {field["region_id"]: field for field in fields}
        text_by_id = {line["id"]: line["text"] for line in page["ocr"]["lines"]}
        memberships, unassigned = [], []
        for line in page["ocr"]["lines"]:
            candidates = []
            for region in regions:
                overlap = _overlap(line["polygon"], region["bbox_xyxy"])
                if overlap > 0 and overlap + 1e-12 >= self.config.min_overlap:
                    candidates.append({"region_id": region["id"], "overlap": overlap})
            def rank(candidate: dict[str, Any]) -> tuple:
                field = by_id[candidate["region_id"]]
                x1, y1, x2, y2 = field["bbox"]
                return (-candidate["overlap"], (x2 - x1) * (y2 - y1),
                        -field["layout_confidence"], candidate["region_id"])
            candidates.sort(key=rank)
            if self.config.mode == 2:
                candidates = candidates[:1]
            total = sum(candidate["overlap"] for candidate in candidates)
            for candidate in candidates:
                candidate["weight"] = candidate["overlap"] / total
                by_id[candidate["region_id"]]["ocr_ids"].append(line["id"])
            memberships.append({"ocr_id": line["id"], "candidates": candidates})
            if not candidates:
                unassigned.append(line["id"])
        for field in fields:
            field["text"] = "\n".join(text_by_id[identifier] for identifier in field["ocr_ids"])
        page.update(fields=fields, memberships=memberships, unassigned_ocr_ids=unassigned)


def fuse_document(ocr: dict[str, Any], layout: dict[str, Any] | list[dict[str, Any]],
                  config: FusionConfig | None = None) -> dict[str, Any]:
    return FusionPipeline(config).fuse(ocr, layout)


def to_llm_text(document: dict[str, Any]) -> str:
    """Serialize the representation as compact Unicode JSON for a text prompt."""
    return json.dumps(document, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
