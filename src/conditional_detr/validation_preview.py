"""Export held-out predictions and side-by-side ground truth/prediction images."""

import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .inference import LayoutDetector


def draw_regions(image, regions, title):
    """Keep previews compact; JSON retains full-resolution box coordinates."""
    image = image.copy()
    original_width, original_height = image.size
    image.thumbnail((1000, 1400))
    sx, sy = image.width / original_width, image.height / original_height
    font = ImageFont.load_default(size=14)
    draw = ImageDraw.Draw(image)
    palette = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#e377c2", "#8c564b"]
    for region in regions:
        x1, y1, x2, y2 = region["bbox_xyxy"]
        box = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
        color = palette[region["label_id"] % len(palette)]
        draw.rectangle(box, outline=color, width=3)
        label = region["label"]
        if "score" in region:
            label += f" {region['score']:.2f}"
        text_width = draw.textbbox((0, 0), label, font=font)[2]
        tx = max(0, min(box[0], image.width - text_width - 4))
        ty = max(0, box[1] - 18)
        draw.rectangle((tx, ty, tx + text_width + 4, ty + 18), fill="white")
        draw.text((tx + 2, ty), label, font=font, fill=color)
    canvas = Image.new("RGB", (image.width, image.height + 26), "white")
    canvas.paste(image, (0, 26))
    ImageDraw.Draw(canvas).text((5, 5), title, font=font, fill="black")
    return canvas


def save_validation_preview(model, processor, dataset, output_dir, threshold=0.5,
                            best_checkpoint=None):
    """Infer exactly once per unaugmented validation image and export a report."""
    if dataset.augmenter is not None:
        raise ValueError("Validation preview requires an unaugmented dataset")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detector = LayoutDetector.from_model(model, processor)
    predictions, sections = [], []
    for index, info in enumerate(dataset.images):
        image_path = dataset.images_dir / info["file_name"]
        prediction = detector.predict(image_path, threshold=threshold)
        prediction["image_id"] = info["id"]
        truth = []
        for annotation in dataset.by_image[info["id"]]:
            x, y, width, height = annotation["bbox"]
            label_id = annotation["category_id"]
            truth.append({"label_id": label_id, "label": dataset.id2label[label_id],
                          "category_id": dataset.categories[label_id]["id"],
                          "bbox_xyxy": [x, y, x + width, y + height],
                          "bbox_xywh": [x, y, width, height]})
        prediction["ground_truth"] = truth
        predictions.append(prediction)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        left = draw_regions(image, truth, "Ground truth")
        right = draw_regions(image, prediction["detections"], f"Predictions (score >= {threshold:g})")
        combined = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), "white")
        combined.paste(left, (0, 0))
        combined.paste(right, (left.width, 0))
        filename = f"{index:04d}.jpg"
        combined.save(output_dir / filename, quality=95)
        sections.append(f'<h2>{html.escape(info["file_name"])}</h2>'
                        f'<p>{len(truth)} target regions; {len(prediction["detections"])} predictions</p>'
                        f'<a href="{filename}"><img src="{filename}" alt="Ground truth and predictions"></a>')
    report = {"best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
              "threshold": threshold, "images": predictions}
    (output_dir / "predictions.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "index.html").write_text(
        '<!doctype html><html><head><meta charset="utf-8"><title>Layout validation</title>'
        '<style>body{font-family:sans-serif;margin:24px}img{max-width:100%;height:auto}</style>'
        '</head><body><h1>Best model: validation predictions</h1>'
        f'<p>Confidence threshold: {threshold:g}. Left: ground truth. Right: predictions.</p>'
        + "\n".join(sections) + '</body></html>', encoding="utf-8",
    )
    return report
