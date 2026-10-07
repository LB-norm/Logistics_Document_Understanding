"""Load a saved layout detector once and predict one or several images."""

import argparse
import json
from pathlib import Path

from PIL import Image


class LayoutDetector:
    def __init__(self, model_dir, device=None):
        import torch
        from transformers import AutoImageProcessor, ConditionalDetrForObjectDetection

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = ConditionalDetrForObjectDetection.from_pretrained(
            model_dir, local_files_only=True,
        ).to(self.device).eval()

    def predict(self, image_path, threshold=0.5):
        import torch

        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        width, height = image.size
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_object_detection(
            outputs, threshold=threshold,
            target_sizes=torch.tensor([[height, width]], device=self.device),
        )[0]
        categories = getattr(self.model.config, "layout_categories", None)
        detections = []
        for score, label, box in zip(result["scores"].tolist(), result["labels"].tolist(), result["boxes"].tolist()):
            x1, y1, x2, y2 = box
            x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
            y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({
                "label_id": label, "label": self.model.config.id2label[label],
                "category_id": categories[label]["id"] if categories else label,
                "score": score, "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
            })
        return {"image": str(image_path), "width": width, "height": height,
                "detections": sorted(detections, key=lambda d: d["score"], reverse=True)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Detect document regions with a fine-tuned Conditional DETR")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--images", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="JSON file containing a list of image predictions")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", help="cpu, cuda, cuda:0, etc. (automatic by default)")
    args = parser.parse_args(argv)
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    detector = LayoutDetector(args.model_dir, args.device)
    predictions = [detector.predict(path, args.threshold) for path in args.images]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(predictions, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
