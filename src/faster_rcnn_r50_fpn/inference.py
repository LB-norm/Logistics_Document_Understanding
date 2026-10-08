"""Native Faster R-CNN inference with the shared layout JSON output format."""

import argparse
import json
import math
from pathlib import Path

from PIL import Image


def format_prediction(result, categories, image_path, width, height, threshold=0.5):
    """Restore zero-based public labels and original COCO category IDs."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    detections = []
    for score, model_label, box in zip(result["scores"].tolist(), result["labels"].tolist(),
                                       result["boxes"].tolist()):
        if not math.isfinite(score) or score < threshold or not all(math.isfinite(v) for v in box):
            continue
        label = int(model_label) - 1
        if not 0 <= label < len(categories):
            raise ValueError(f"Unexpected Faster R-CNN foreground label: {model_label}")
        x1, y1, x2, y2 = box
        x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
        y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        category = categories[label]
        detections.append({"label_id": label, "label": category["name"],
                           "category_id": category["id"], "score": score,
                           "bbox_xyxy": [x1, y1, x2, y2], "bbox_xywh": [x1, y1, x2 - x1, y2 - y1]})
    return {"image": str(image_path), "width": width, "height": height,
            "detections": sorted(detections, key=lambda d: d["score"], reverse=True)}


class LayoutDetector:
    @classmethod
    def from_model(cls, model, categories):
        detector = cls.__new__(cls)
        detector.device = next(model.parameters()).device
        detector.model = model.eval()
        detector.categories = categories
        if model.roi_heads.box_predictor.cls_score.out_features != len(categories) + 1:
            raise ValueError("Saved category mapping does not match the model head")
        return detector

    def __init__(self, model_dir, device=None):
        import torch
        from .model import build_model

        path = Path(model_dir)
        if path.is_dir():
            path = path / "model.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        config = state["layout_config"]
        if config["architecture"] != "fasterrcnn_resnet50_fpn":
            raise ValueError("Checkpoint is not a Faster R-CNN R50-FPN layout model")
        self.categories = config["categories"]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = build_model(self.categories, config)
        self.model.load_state_dict(state["model"])
        self.model.to(self.device).eval()

    def predict(self, image_path, threshold=0.5):
        import torch
        from torchvision.transforms.functional import to_tensor

        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        width, height = image.size
        # Model preprocessing handles normalization, aspect-preserving resize,
        # padding, class-aware NMS, and restoring boxes to the input raster.
        with torch.inference_mode():
            result = self.model([to_tensor(image).to(self.device)])[0]
        return format_prediction(result, self.categories, image_path, width, height, threshold)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Detect document regions with Faster R-CNN R50-FPN")
    parser.add_argument("--model-dir", required=True, type=Path, help="best_model directory or checkpoint .pt file")
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
