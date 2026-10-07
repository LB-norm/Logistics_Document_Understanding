"""Offline integration checks for layout data and the saved-model contract."""

import json
import copy
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.conditional_detr.dataset import CocoLayoutDataset, DetectionCollator
from src.conditional_detr.inference import LayoutDetector
from src.conditional_detr.train import main


class ConditionalDetrTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        Image.new("RGB", (64, 48)).save(self.root / "one.png")
        Image.new("RGB", (48, 64)).save(self.root / "two.png")
        self.data = {
            "images": [
                {"id": 1, "file_name": "one.png", "width": 64, "height": 48},
                {"id": 2, "file_name": "two.png", "width": 48, "height": 64},
            ],
            "categories": [{"id": 19, "name": "signature"}, {"id": 3, "name": "sender"}],
            "annotations": [{"image_id": 1, "category_id": 19, "bbox": [10, 8, 20, 12]}],
        }
        self.annotations = self.root / "annotations.json"
        self.write_data()

    def write_data(self):
        self.annotations.write_text(json.dumps(self.data))

    def test_sparse_labels_and_bad_boxes(self):
        dataset = CocoLayoutDataset(self.annotations, self.root)
        self.assertEqual(dataset.id2label, {0: "sender", 1: "signature"})
        self.assertEqual(dataset[0]["target"]["annotations"][0]["category_id"], 1)
        self.assertEqual(dataset[1]["target"]["annotations"], [])
        self.data["annotations"][0]["bbox"] = [60, 8, 20, 12]
        self.write_data()
        with self.assertRaisesRegex(ValueError, "outside"):
            CocoLayoutDataset(self.annotations, self.root)

    def test_dry_run_rejects_split_leakage(self):
        with self.assertRaisesRegex(ValueError, "overlapping"):
            main(["--train-annotations", str(self.annotations), "--train-images", str(self.root),
                  "--val-annotations", str(self.annotations), "--val-images", str(self.root), "--dry-run"])

    def test_geometry_moves_pixels_and_boxes_together(self):
        import numpy as np
        from src.conditional_detr.augmentation import transform_regions

        pixels = np.full((48, 64, 3), 255, dtype=np.uint8)
        pixels[8:20, 10:30] = 0
        target = [{"bbox": [10, 8, 20, 12], "category_id": 1, "area": 240, "iscrowd": 0}]
        matrix = np.array([[1., 0., 5.], [0., 1., 4.], [0., 0., 1.]])
        image, boxes = transform_regions(Image.fromarray(pixels), target, matrix)
        self.assertEqual(boxes[0]["bbox"], [15., 12., 20., 12.])
        self.assertEqual(boxes[0]["area"], 240)
        self.assertTrue((np.asarray(image)[12:24, 15:35] == 0).all())
        self.assertEqual(target[0]["bbox"], [10, 8, 20, 12])

    def test_augmentation_reproducible_preserves_edges_and_source(self):
        import numpy as np
        from src.conditional_detr.augmentation import AugmentationConfig, DocumentAugmenter

        config = AugmentationConfig(geometry_probability=1, color_probability=1,
                                    blur_probability=1, noise_probability=1, jpeg_probability=1)
        image = Image.new("RGB", (64, 48), (128, 128, 128))
        targets = [{"bbox": [0, 0, 64, 48], "category_id": 1, "area": 3072, "iscrowd": 0},
                   {"bbox": [0, 0, 2, 2], "category_id": 0, "area": 4, "iscrowd": 0}]
        original = copy.deepcopy(targets)
        first = DocumentAugmenter(config, np.random.default_rng(42))
        second = DocumentAugmenter(config, np.random.default_rng(42))
        augmented, boxes = first(image, targets)
        repeated, repeated_boxes = second(image, targets)
        self.assertTrue(np.array_equal(np.asarray(augmented), np.asarray(repeated)))
        self.assertEqual(boxes, repeated_boxes)
        self.assertEqual(targets, original)
        self.assertFalse(np.array_equal(np.asarray(augmented), np.asarray(image)))
        for _ in range(30):
            _, boxes = first(image, targets)
            self.assertEqual(len(boxes), 2)
            for box in boxes:
                x, y, w, h = box["bbox"]
                self.assertTrue(0 <= x < x + w <= 64 and 0 <= y < y + h <= 48)
                self.assertAlmostEqual(box["area"], w * h)
        _, empty = first(image, [])
        self.assertEqual(empty, [])

    def test_augmentation_is_optional_and_only_changes_training_samples(self):
        import numpy as np
        from src.conditional_detr.augmentation import DocumentAugmenter
        from src.conditional_detr.train import build_parser

        train = CocoLayoutDataset(self.annotations, self.root,
                                  augmenter=DocumentAugmenter(rng=np.random.default_rng(2)))
        validation = CocoLayoutDataset(self.annotations, self.root, train.categories)
        original = copy.deepcopy(train.by_image)
        for _ in range(5):
            train[0]
        self.assertEqual(train.by_image, original)
        self.assertIsNone(validation.augmenter)
        self.assertEqual(validation[0]["target"]["annotations"], original[1])
        arguments = ["--train-annotations", "a", "--train-images", "b",
                     "--val-annotations", "c", "--val-images", "d"]
        self.assertTrue(build_parser().parse_args(arguments).augmentation)
        self.assertFalse(build_parser().parse_args(arguments + ["--no-augmentation"]).augmentation)

    def test_invalid_augmentation_settings(self):
        from src.conditional_detr.augmentation import AugmentationConfig

        for kwargs in ({"max_rotation_degrees": float("nan")},
                       {"perspective_fraction": 0.5}, {"noise_probability": -0.1}):
            with self.assertRaises(ValueError):
                AugmentationConfig(**kwargs)

    def test_batch_loss_backward_save_reload_and_inference(self):
        import torch
        from transformers import ConditionalDetrConfig, ConditionalDetrForObjectDetection
        from transformers import ConditionalDetrImageProcessor, ResNetConfig

        torch.set_num_threads(1)
        processor = ConditionalDetrImageProcessor(size={"shortest_edge": 48, "longest_edge": 64})
        dataset = CocoLayoutDataset(self.annotations, self.root)
        batch = DetectionCollator(processor)([dataset[0], dataset[1]])
        self.assertEqual(tuple(batch["pixel_mask"].shape), (2, 64, 64))
        self.assertEqual(batch["labels"][0]["class_labels"].tolist(), [1])
        self.assertEqual(batch["labels"][1]["boxes"].shape[0], 0)
        config = ConditionalDetrConfig(
            backbone_config=ResNetConfig(embedding_size=8, hidden_sizes=[8, 16, 32, 64],
                                         depths=[1, 1, 1, 1], out_features=["stage4"]),
            d_model=32, encoder_layers=1, decoder_layers=2,
            encoder_attention_heads=4, decoder_attention_heads=4,
            encoder_ffn_dim=64, decoder_ffn_dim=64, num_queries=4,
            id2label=dataset.id2label, label2id={v: k for k, v in dataset.id2label.items()},
            auxiliary_loss=True,
        )
        config.layout_categories = dataset.categories
        model = ConditionalDetrForObjectDetection(config)
        loss = model(**batch).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.class_labels_classifier.weight.grad)
        saved = self.root / "model"
        model.save_pretrained(saved)
        processor.save_pretrained(saved)
        predictions = LayoutDetector(saved, "cpu").predict(self.root / "one.png", threshold=0)
        self.assertEqual((predictions["width"], predictions["height"]), (64, 48))
        self.assertTrue(predictions["detections"])
        for detection in predictions["detections"]:
            self.assertEqual(detection["category_id"], dataset.categories[detection["label_id"]]["id"])
            x1, y1, x2, y2 = detection["bbox_xyxy"]
            self.assertTrue(0 <= x1 < x2 <= 64 and 0 <= y1 < y2 <= 48)
        from unittest.mock import patch
        from src.conditional_detr.validation_preview import save_validation_preview

        report_dir = self.root / "validation_predictions"
        with patch.object(LayoutDetector, "predict", autospec=True,
                          side_effect=LayoutDetector.predict) as predict:
            report = save_validation_preview(model, processor, dataset, report_dir,
                                             threshold=0, best_checkpoint=saved)
            self.assertEqual(predict.call_count, len(dataset))
        self.assertEqual(len(report["images"]), 2)
        self.assertEqual(report["images"][0]["ground_truth"][0]["category_id"], 19)
        self.assertEqual(report["images"][1]["ground_truth"], [])
        self.assertEqual(json.loads((report_dir / "predictions.json").read_text()), report)
        self.assertIn("0000.jpg", (report_dir / "index.html").read_text())
        with Image.open(report_dir / "0000.jpg") as preview:
            self.assertEqual(preview.size, (128, 74))


if __name__ == "__main__":
    unittest.main()
