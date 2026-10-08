"""Reuse DETR's COCO validation and export normalized YOLO detection targets."""

import hashlib
import json
from pathlib import Path

from PIL import Image

from src.conditional_detr.dataset import CocoLayoutDataset


def dataset_manifest(train, val):
    """Identify source content so a resumed run cannot silently change its data."""
    splits = {}
    for name, dataset in (("train", train), ("val", val)):
        images = []
        for info in dataset.images:
            path = dataset.images_dir / info["file_name"]
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            images.append({"id": info["id"], "source": str(path), "sha256": digest.hexdigest()})
        splits[name] = {"annotations": str(dataset.annotations_path.resolve()),
                        "sha256": hashlib.sha256(dataset.annotations_path.read_bytes()).hexdigest(),
                        "images": images}
    return {"categories": train.categories, "splits": splits}


def export_yolo_dataset(train, val, output_dir):
    """Write RGB PNGs and labels inside the run; preserve source files and axes.

    Numbered filenames prevent stem collisions and make nested COCO paths work.
    Decoding with Pillow also avoids OpenCV applying source EXIF orientation.
    JSON is valid YAML and avoids an additional serializer dependency.
    """
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"YOLO export directory must be empty: {output_dir}")
    manifest = dataset_manifest(train, val)
    for split, dataset in (("train", train), ("val", val)):
        images_dir = output_dir / "images" / split
        labels_dir = output_dir / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        for index, info in enumerate(dataset.images):
            filename = f"{index:06d}"
            with Image.open(dataset.images_dir / info["file_name"]) as image:
                image.convert("RGB").save(images_dir / f"{filename}.png")
            rows = []
            for annotation in dataset.by_image[info["id"]]:
                x, y, width, height = annotation["bbox"]
                # The validator tolerates roundoff at the far image boundary.
                right, bottom = min(x + width, info["width"]), min(y + height, info["height"])
                values = ((x + right) / (2 * info["width"]),
                          (y + bottom) / (2 * info["height"]),
                          (right - x) / info["width"], (bottom - y) / info["height"])
                rows.append(f"{annotation['category_id']} " + " ".join(f"{v:.10f}" for v in values))
            (labels_dir / f"{filename}.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    data_path = output_dir / "data.yaml"
    data_path.write_text(json.dumps({"path": str(output_dir), "train": "images/train",
                                    "val": "images/val", "names": list(train.id2label.values())},
                                   indent=2), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return data_path
