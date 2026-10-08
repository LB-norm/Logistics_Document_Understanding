"""Lightweight COCO conversion and detector-contract checks; no model training."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from src.yolo_v11s.dataset import CocoLayoutDataset, dataset_manifest, export_yolo_dataset
from src.yolo_v11s.inference import LayoutDetector
from src.yolo_v11s.train import build_parser, main, training_options, validate_args
from src.yolo_v11s.validation_preview import save_validation_preview


class YoloLayoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.categories = [{"id": 19, "name": "signature"}, {"id": 3, "name": "sender"}]
        self.arguments = []
        for split in ("train", "val"):
            folder = self.root / split
            (folder / "nested").mkdir(parents=True)
            Image.new("RGB", (64, 48), "white").save(folder / "one.png")
            Image.new("RGB", (48, 64), "white").save(folder / "nested" / "one.png")
            data = {
                "images": [{"id": 1, "file_name": "one.png", "width": 64, "height": 48},
                           {"id": 2, "file_name": "nested/one.png", "width": 48, "height": 64}],
                "categories": self.categories,
                "annotations": [{"image_id": 1, "category_id": 19, "bbox": [10, 8, 20, 12]},
                                {"image_id": 2, "category_id": 3, "bbox": [1, 1, 2, 2], "iscrowd": 1}],
            }
            (folder / "annotations.json").write_text(json.dumps(data))
            self.arguments.extend([f"--{split}-annotations", str(folder / "annotations.json"),
                                   f"--{split}-images", str(folder)])
        self.arguments.extend(["--output-dir", str(self.root / "run")])
        self.train = CocoLayoutDataset(self.root / "train/annotations.json", self.root / "train")
        self.val = CocoLayoutDataset(self.root / "val/annotations.json", self.root / "val", self.train.categories)

    def test_export_sparse_ids_empty_targets_and_colliding_stems(self):
        before = dataset_manifest(self.train, self.val)
        data_path = export_yolo_dataset(self.train, self.val, self.root / "export")
        data = json.loads(data_path.read_text())
        self.assertEqual(data["names"], ["sender", "signature"])
        labels = data_path.parent / "labels/train"
        values = (labels / "000000.txt").read_text().split()
        self.assertEqual(values[0], "1")
        np.testing.assert_allclose(list(map(float, values[1:])), [20 / 64, 14 / 48, 20 / 64, 12 / 48])
        self.assertEqual((labels / "000001.txt").read_text(), "")
        with Image.open(data_path.parent / "images/train/000001.png") as image:
            self.assertEqual(image.size, (48, 64))
        self.assertEqual(before, dataset_manifest(self.train, self.val))
        with self.assertRaises(FileExistsError):
            export_yolo_dataset(self.train, self.val, data_path.parent)

    def test_export_does_not_rotate_exif_images(self):
        folder = self.root / "exif"
        folder.mkdir()
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (64, 48)).save(folder / "photo.jpg", exif=exif)
        data = {"images": [{"id": 1, "file_name": "photo.jpg", "width": 64, "height": 48}],
                "categories": self.categories, "annotations": []}
        (folder / "data.json").write_text(json.dumps(data))
        dataset = CocoLayoutDataset(folder / "data.json", folder, self.train.categories)
        path = export_yolo_dataset(self.train, dataset, self.root / "exif_export")
        with Image.open(path.parent / "images/val/000000.png") as image:
            self.assertEqual(image.size, (64, 48))
            self.assertIsNone(image.getexif().get(274))

    def test_dry_run_validates_without_export_and_rejects_bad_data(self):
        with contextlib.redirect_stdout(io.StringIO()):
            main(self.arguments + ["--dry-run"])
        self.assertFalse((self.root / "run").exists())
        same_split = self.arguments.copy()
        same_split[same_split.index("--val-images") + 1] = str(self.root / "train")
        with self.assertRaisesRegex(ValueError, "overlapping"):
            main(same_split + ["--dry-run"])
        data = json.loads((self.root / "val/annotations.json").read_text())
        data["annotations"][0]["bbox"] = [60, 8, 20, 12]
        (self.root / "val/annotations.json").write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "outside"):
            main(self.arguments + ["--dry-run"])

    def test_options_validate_and_use_native_augmentation(self):
        args = build_parser().parse_args(self.arguments)
        model = SimpleNamespace(model=SimpleNamespace(yaml={"backbone": [None] * 11}))
        options = training_options(args, model, self.root / "data.yaml")
        self.assertEqual(options["nbs"], 8)
        self.assertEqual(options["optimizer"], "AdamW")
        self.assertFalse(options["amp"])
        for name in ("mosaic", "mixup", "cutmix", "fliplr", "flipud", "translate", "scale"):
            self.assertNotIn(name, options)
        self.assertNotIn("trainer", options)
        args.freeze_backbone = True
        self.assertEqual(training_options(args, model, self.root / "data.yaml")["freeze"], 11)
        for option, value in (("--image-size", "65"), ("--learning-rate", "nan"),
                              ("--weight-decay", "nan"), ("--validation-preview-threshold", "nan")):
            with self.assertRaises(ValueError):
                validate_args(build_parser().parse_args(self.arguments + [option, value]))

    def fake_model(self):
        class Values:
            def __init__(self, values):
                self.values = values

            def tolist(self):
                return self.values

        boxes = SimpleNamespace(conf=Values([0.8, 0.9, 0.7]), cls=Values([1., 0., 0.]),
                                xyxy=Values([[-1., 8., 80., 60.], [10., 8., 30., 20.], [20., 10., 20., 15.]]))
        model = SimpleNamespace(names=self.train.id2label)
        from unittest.mock import Mock
        model.predict = Mock(return_value=[SimpleNamespace(boxes=boxes)])
        return model

    def test_inference_contract_clips_sorts_and_restores_categories(self):
        model = self.fake_model()
        detector = LayoutDetector.from_model(model, self.train.categories)
        result = detector.predict(self.root / "train/one.png")
        self.assertEqual((result["width"], result["height"]), (64, 48))
        self.assertEqual([d["category_id"] for d in result["detections"]], [3, 19])
        self.assertEqual(result["detections"][1]["bbox_xywh"], [0., 8., 64., 40.])
        self.assertIsInstance(model.predict.call_args.kwargs["source"], Image.Image)
        self.assertFalse(model.predict.call_args.kwargs["augment"])
        for threshold in (-1, 2, float("nan")):
            with self.assertRaises(ValueError):
                detector.predict(self.root / "train/one.png", threshold)
        with self.assertRaisesRegex(ValueError, "class names"):
            LayoutDetector.from_model(model, list(reversed(self.train.categories)))

    def test_validation_gallery_visits_each_image_once(self):
        model = self.fake_model()
        folder = self.root / "preview"
        report = save_validation_preview(model, self.val, folder, best_checkpoint=self.root / "best.pt")
        self.assertEqual(model.predict.call_count, len(self.val))
        self.assertEqual(report["images"][0]["ground_truth"][0]["category_id"], 19)
        self.assertEqual(report["images"][1]["ground_truth"], [])
        self.assertEqual(json.loads((folder / "predictions.json").read_text()), report)
        with Image.open(folder / "0000.jpg") as image:
            self.assertEqual(image.size, (128, 74))


if __name__ == "__main__":
    unittest.main()
