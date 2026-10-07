"""COCO boxes in stored-image coordinates (no EXIF rotation)."""

import json
import math
from pathlib import Path

from PIL import Image


class CocoLayoutDataset:
    """Validate a COCO detection export and remap sparse IDs to model labels."""

    def __init__(self, annotations, images_dir, categories=None, augmenter=None):
        self.augmenter = augmenter
        self.annotations_path = Path(annotations)
        self.images_dir = Path(images_dir).resolve()
        data = json.loads(self.annotations_path.read_text(encoding="utf-8"))
        declared = sorted(data["categories"], key=lambda c: c["id"])
        ids = [c["id"] for c in declared]
        names = [c["name"] for c in declared]
        if not ids or len(set(ids)) != len(ids) or len(set(names)) != len(names):
            raise ValueError("Categories must have unique IDs and names and be nonempty")
        self.categories = categories if categories is not None else declared
        expected = {c["id"]: c["name"] for c in self.categories}
        if any(expected.get(c["id"]) != c["name"] for c in declared):
            raise ValueError("Validation categories must match training category IDs and names")
        self.category_to_label = {c["id"]: i for i, c in enumerate(self.categories)}
        self.id2label = {i: c["name"] for i, c in enumerate(self.categories)}
        self.images = data["images"]
        if not self.images:
            raise ValueError("Dataset has no images")
        self.by_image = {im["id"]: [] for im in self.images}
        if len(self.by_image) != len(self.images):
            raise ValueError("Duplicate image IDs")
        dimensions = {}
        for im in self.images:
            path = (self.images_dir / im["file_name"]).resolve()
            if not path.is_relative_to(self.images_dir):
                raise ValueError(f"Image path escapes image directory: {path}")
            with Image.open(path) as image:
                if image.size != (im["width"], im["height"]):
                    raise ValueError(f"COCO dimensions do not match image: {path}")
            dimensions[im["id"]] = (im["width"], im["height"])
        for annotation in data["annotations"]:
            image_id = annotation["image_id"]
            if image_id not in self.by_image:
                raise ValueError(f"Unknown image_id: {image_id}")
            category = annotation["category_id"]
            if category not in ids or category not in self.category_to_label:
                raise ValueError(f"Unknown category_id: {category}")
            box = annotation["bbox"]
            if len(box) != 4 or not all(math.isfinite(v) for v in box):
                raise ValueError(f"Invalid COCO bbox: {box}")
            x, y, width, height = box
            iw, ih = dimensions[image_id]
            if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > iw + 1e-4 or y + height > ih + 1e-4:
                raise ValueError(f"COCO bbox outside image or empty: {box}")
            # Crowd regions are not individual detection targets.
            if annotation.get("iscrowd", 0):
                continue
            self.by_image[image_id].append({
                "bbox": box,
                "category_id": self.category_to_label[category],
                "area": width * height,
                "iscrowd": 0,
            })

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        info = self.images[index]
        with Image.open(self.images_dir / info["file_name"]) as image:
            image = image.convert("RGB")
        annotations = self.by_image[info["id"]]
        if self.augmenter is not None:
            image, annotations = self.augmenter(image, annotations)
        return {"image": image, "target": {
            "image_id": info["id"], "annotations": annotations,
        }}


class DetectionCollator:
    """Resize images and boxes together, then pad mixed image sizes with a mask."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        return self.processor(
            images=[e["image"] for e in examples],
            annotations=[e["target"] for e in examples],
            return_tensors="pt",
        )
