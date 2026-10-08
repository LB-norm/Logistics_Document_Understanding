"""Load a fine-tuned YOLO layout detector once; emit DETR-compatible JSON."""

import argparse
import json
from pathlib import Path

from PIL import Image


class LayoutDetector:
    @classmethod
    def from_model(cls, model, categories, image_size=1344, device=None,
                   iou_threshold=0.7, max_detections=300):
        detector = cls.__new__(cls)
        detector.model = model
        detector.categories = categories
        detector.image_size = image_size
        detector.device = device
        detector.iou_threshold = iou_threshold
        detector.max_detections = max_detections
        detector._validate_categories()
        return detector

    def _validate_categories(self):
        names = self.model.names
        if [names[i] for i in range(len(names))] != [c["name"] for c in self.categories]:
            raise ValueError("Saved category mapping does not match model class names")

    def __init__(self, model_dir, device=None):
        from ultralytics import YOLO

        model_path = Path(model_dir)
        if model_path.is_dir():
            model_path = model_path / "model.pt"
        if not model_path.is_file():
            raise FileNotFoundError(f"Saved YOLO weights not found: {model_path}")
        config_path = model_path.parent / "layout_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Category mapping not found: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.model = YOLO(str(model_path), task="detect")
        self.categories = config["categories"]
        self.image_size = config["image_size"]
        self.iou_threshold = config["iou_threshold"]
        self.max_detections = config["max_detections"]
        self.device = device
        self._validate_categories()

    def predict(self, image_path, threshold=0.5):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        width, height = image.size
        # A PIL raster keeps coordinates in the stored raster's orientation.
        result = self.model.predict(source=image, conf=threshold, imgsz=self.image_size,
                                    iou=self.iou_threshold, max_det=self.max_detections,
                                    device=self.device, verbose=False, augment=False)[0]
        detections = []
        for score, label, box in zip(result.boxes.conf.tolist(), result.boxes.cls.tolist(),
                                     result.boxes.xyxy.tolist()):
            label = int(label)
            x1, y1, x2, y2 = box
            x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
            y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            category = self.categories[label]
            detections.append({"label_id": label, "label": category["name"],
                               "category_id": category["id"], "score": score,
                               "bbox_xyxy": [x1, y1, x2, y2], "bbox_xywh": [x1, y1, x2 - x1, y2 - y1]})
        return {"image": str(image_path), "width": width, "height": height,
                "detections": sorted(detections, key=lambda d: d["score"], reverse=True)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Detect document regions with fine-tuned YOLO11s")
    parser.add_argument("--model-dir", required=True, type=Path, help="best_model directory or weights .pt file")
    parser.add_argument("--images", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", help="cpu, cuda:0, etc. (automatic by default)")
    args = parser.parse_args(argv)
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    detector = LayoutDetector(args.model_dir, args.device)
    predictions = [detector.predict(path, args.threshold) for path in args.images]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(predictions, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
