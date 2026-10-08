"""Offline Faster R-CNN integration and layout-contract checks; no downloads."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image

from src.faster_rcnn_r50_fpn.dataset import CocoLayoutDataset, DetectionDataset, detection_collator
from src.faster_rcnn_r50_fpn.engine import make_warmup, train_one_epoch
from src.faster_rcnn_r50_fpn.evaluation import evaluate_predictions
from src.faster_rcnn_r50_fpn.inference import LayoutDetector, format_prediction, main as infer
from src.faster_rcnn_r50_fpn.model import build_model
from src.faster_rcnn_r50_fpn.train import build_parser, layout_config, main, validate_args


class FasterRCNNLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

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
            Image.new("RGB", (48, 64), "white").save(folder / "nested/one.png")
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

    def test_tensor_targets_background_offset_and_empty_images(self):
        dataset = DetectionDataset(self.train)
        original = copy.deepcopy(self.train.by_image)
        image, target = dataset[0]
        self.assertEqual(tuple(image.shape), (3, 48, 64))
        self.assertEqual(image.dtype, torch.float32)
        self.assertTrue((image == 1).all())
        self.assertEqual(target["labels"].tolist(), [2])
        self.assertEqual(target["labels"].dtype, torch.int64)
        self.assertEqual(target["boxes"].tolist(), [[10., 8., 30., 20.]])
        self.assertEqual(target["area"].tolist(), [240.])
        self.assertEqual(tuple(dataset[1][1]["boxes"].shape), (0, 4))
        self.assertEqual(tuple(dataset[1][1]["labels"].shape), (0,))
        flipped, flipped_target = DetectionDataset(self.train, 1)[0]
        self.assertTrue(torch.equal(flipped, image.flip(-1)))
        self.assertEqual(flipped_target["boxes"].tolist(), [[34., 8., 54., 20.]])
        self.assertEqual(self.train.by_image, original)
        images, targets = detection_collator([dataset[0], dataset[1]])
        self.assertEqual([tuple(i.shape) for i in images], [(3, 48, 64), (3, 64, 48)])
        self.assertEqual(len(targets), 2)

    def test_dry_run_and_split_validation(self):
        with contextlib.redirect_stdout(io.StringIO()):
            main(self.arguments + ["--dry-run"])
        self.assertFalse((self.root / "run").exists())
        arguments = self.arguments.copy()
        arguments[arguments.index("--val-images") + 1] = str(self.root / "train")
        with self.assertRaisesRegex(ValueError, "overlapping"):
            main(arguments + ["--dry-run"])
        data = json.loads((self.root / "val/annotations.json").read_text())
        data["annotations"][0]["bbox"] = [60, 8, 20, 12]
        (self.root / "val/annotations.json").write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "outside"):
            main(self.arguments + ["--dry-run"])

    def test_invalid_settings_empty_splits_and_detection_limit(self):
        for option, value in (("--learning-rate", "nan"), ("--weight-decay", "nan"),
                              ("--horizontal-flip-probability", "nan"), ("--momentum", "nan"),
                              ("--validation-preview-threshold", "2"), ("--lr-gamma", "0"),
                              ("--longest-edge", "100"), ("--workers", "-1")):
            with self.assertRaises(ValueError):
                validate_args(build_parser().parse_args(self.arguments + [option, value]))
        with self.assertRaisesRegex(ValueError, "fp32"):
            validate_args(build_parser().parse_args(self.arguments + ["--cpu", "--precision", "fp16"]))
        data_path = self.root / "val/annotations.json"
        data = json.loads(data_path.read_text())
        data["annotations"].append({"image_id": 1, "category_id": 3, "bbox": [1, 1, 3, 3]})
        data_path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "region count"):
            main(self.arguments + ["--dry-run", "--max-detections", "1"])
        data["annotations"] = []
        data_path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "mAP"):
            main(self.arguments + ["--dry-run"])

    def test_prediction_contract_clips_filters_sorts_and_restores_labels(self):
        result = {"scores": torch.tensor([0.8, 0.9, 0.7, 0.1, float("nan")]),
                  "labels": torch.tensor([2, 1, 1, 2, 2]),
                  "boxes": torch.tensor([[-1., 8., 80., 60.], [10., 8., 30., 20.],
                                         [20., 10., 20., 15.], [1., 1., 2., 2.], [1., 1., 2., 2.]])}
        prediction = format_prediction(result, self.train.categories, "one.png", 64, 48)
        self.assertEqual([d["category_id"] for d in prediction["detections"]], [3, 19])
        self.assertEqual([d["label_id"] for d in prediction["detections"]], [0, 1])
        self.assertEqual(prediction["detections"][1]["bbox_xywh"], [0., 8., 64., 40.])
        for threshold in (-1, 2, float("nan")):
            with self.assertRaises(ValueError):
                format_prediction(result, self.train.categories, "", 64, 48, threshold)

    def test_coco_metrics_include_empty_images_and_sparse_categories(self):
        perfect = [{"image_id": 1, "category_id": 19, "bbox": [10, 8, 20, 12], "score": 0.9}]
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = evaluate_predictions(self.train, perfect)
            empty = evaluate_predictions(self.train, [])
            false_positive = evaluate_predictions(self.train, [
                {"image_id": 2, "category_id": 19, "bbox": [10, 8, 20, 12], "score": 0.99}, *perfect])
        self.assertAlmostEqual(metrics["map"], 1.0)
        self.assertAlmostEqual(empty["map"], 0.0)
        self.assertLess(false_positive["map"], metrics["map"])

    def test_accumulation_handles_partial_group(self):
        class ToyDetector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.))

            def forward(self, images, targets):
                return {"loss_classifier": (self.weight - torch.stack(images)).square().mean()}

        model = ToyDetector()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        # Two samples in the first batch, one in the second. A final group has
        # one batch despite accumulation_steps=2 and must receive a full update.
        loader = [([torch.tensor(1.), torch.tensor(3.)], [{}, {}]),
                  ([torch.tensor(5.)], [{}]), ([torch.tensor(4.)], [{}])]
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        with contextlib.redirect_stdout(io.StringIO()):
            losses = train_one_epoch(model, optimizer, loader, torch.device("cpu"), 0,
                                      accumulation_steps=2, scaler=scaler)
        self.assertAlmostEqual(model.weight.item(), 1.28, places=6)
        self.assertTrue(losses["loss"] > 0)

    def test_warmup_counts_optimizer_updates(self):
        parameter = torch.nn.Parameter(torch.tensor(0.))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        self.assertIsNone(make_warmup(optimizer, batches=2, accumulation_steps=4))
        warmup = make_warmup(optimizer, batches=5, accumulation_steps=2)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.0001)
        for _ in range(2):
            optimizer.step()
            warmup.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.1)

    def test_coco_initialization_and_local_weight_policy(self):
        from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

        config = layout_config(build_parser().parse_args(self.arguments), self.train.categories)
        # A factory state_dict checks architecture compatibility without a download.
        source = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        state = source.state_dict()
        del source
        with patch.object(FasterRCNN_ResNet50_FPN_Weights, "get_state_dict", return_value=state) as load:
            model = build_model(self.train.categories, config, model_id="DEFAULT")
        load.assert_called_once_with(progress=True, check_hash=True)
        self.assertTrue(torch.equal(model.backbone.body.layer4[0].conv1.weight,
                                    state["backbone.body.layer4.0.conv1.weight"]))
        self.assertTrue(torch.equal(model.roi_heads.box_head.fc6.weight,
                                    state["roi_heads.box_head.fc6.weight"]))
        self.assertEqual(model.roi_heads.box_predictor.cls_score.out_features, 3)
        del model, state
        with patch("torch.hub.get_dir", return_value=str(self.root / "cache")):
            with patch("torch.hub.download_url_to_file", side_effect=AssertionError("Unexpected download")):
                with self.assertRaisesRegex(FileNotFoundError, "Cached COCO weights"):
                    build_model(self.train.categories, config, model_id="DEFAULT", local_files_only=True)

    def test_stored_raster_orientation_is_preserved(self):
        folder = self.root / "exif"
        folder.mkdir()
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (64, 48), "white").save(folder / "photo.jpg", exif=exif)
        data = {"images": [{"id": 1, "file_name": "photo.jpg", "width": 64, "height": 48}],
                "categories": self.categories, "annotations": []}
        (folder / "data.json").write_text(json.dumps(data))
        source = CocoLayoutDataset(folder / "data.json", folder)
        image, _ = DetectionDataset(source)[0]
        self.assertEqual(tuple(image.shape), (3, 48, 64))

    def test_standard_architecture_native_resize_and_freezing(self):
        from torchvision.ops.misc import FrozenBatchNorm2d

        args = build_parser().parse_args(self.arguments + ["--shortest-edge", "32", "--longest-edge", "64"])
        config = layout_config(args, self.train.categories)
        with patch("torch.hub.download_url_to_file", side_effect=AssertionError("Unexpected download")):
            model = build_model(self.train.categories, config)
        self.assertEqual(model.roi_heads.box_predictor.cls_score.out_features, 3)
        self.assertIsInstance(model.backbone.body.bn1, FrozenBatchNorm2d)
        self.assertFalse(model.backbone.body.layer1[0].conv1.weight.requires_grad)
        self.assertTrue(model.backbone.body.layer2[0].conv1.weight.requires_grad)
        image, target = DetectionDataset(self.train)[0]
        batch, targets = model.transform([image], [target])
        self.assertEqual(batch.image_sizes, [(32, 42)])
        torch.testing.assert_close(targets[0]["boxes"], target["boxes"] * torch.tensor([42 / 64, 32 / 48] * 2))
        model.eval()
        self.assertEqual(model.roi_heads.score_thresh, 0.0)

    def test_real_training_resume_save_reload_cli_and_gallery(self):
        settings = ["--model-id", "none", "--local-files-only", "--cpu", "--epochs", "1",
                    "--shortest-edge", "64", "--longest-edge", "64", "--freeze-backbone",
                    "--validation-preview-threshold", "0", "--max-detections", "5"]
        with patch("torch.hub.download_url_to_file", side_effect=AssertionError("Unexpected download")):
            with contextlib.redirect_stdout(io.StringIO()):
                main(self.arguments + settings)
                main(self.arguments + settings + ["--epochs", "2", "--resume-from-checkpoint",
                                                   str(self.root / "run/checkpoints/last.pt")])
            checkpoint = torch.load(self.root / "run/checkpoints/last.pt", weights_only=True)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(len(checkpoint["history"]), 2)
            losses = checkpoint["history"][0]["train"]
            self.assertEqual(set(losses), {"loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg", "loss"})
            self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in losses.values()))
            resumed_head = checkpoint["model"]["roi_heads.box_predictor.cls_score.weight"].clone()
            del checkpoint
            uninterrupted = self.arguments.copy()
            uninterrupted[uninterrupted.index("--output-dir") + 1] = str(self.root / "continuous")
            with contextlib.redirect_stdout(io.StringIO()):
                main(uninterrupted + settings + ["--epochs", "2"])
            continuous = torch.load(self.root / "continuous/checkpoints/last.pt", weights_only=True)
            self.assertTrue(torch.equal(resumed_head, continuous["model"]["roi_heads.box_predictor.cls_score.weight"]))
            del continuous
            saved = self.root / "run/best_model"
            detector = LayoutDetector(saved, "cpu")
            prediction = detector.predict(self.root / "val/one.png", threshold=0)
            self.assertTrue(prediction["detections"])
            for detection in prediction["detections"]:
                self.assertEqual(detection["category_id"], self.train.categories[detection["label_id"]]["id"])
                x1, y1, x2, y2 = detection["bbox_xyxy"]
                self.assertTrue(0 <= x1 < x2 <= 64 and 0 <= y1 < y2 <= 48)
            infer(["--model-dir", str(saved), "--images", str(self.root / "val/one.png"),
                   "--output", str(self.root / "outputs/layouts.json"), "--device", "cpu", "--threshold", "0"])
        self.assertEqual(json.loads((self.root / "outputs/layouts.json").read_text()), [prediction])
        report = json.loads((self.root / "run/validation_predictions/predictions.json").read_text())
        self.assertEqual(len(report["images"]), 2)
        self.assertEqual(report["images"][0]["ground_truth"][0]["category_id"], 19)
        self.assertEqual(report["images"][1]["ground_truth"], [])
        self.assertEqual(report["images"][0]["detections"], prediction["detections"])
        with Image.open(self.root / "run/validation_predictions/0000.jpg") as image:
            self.assertEqual(image.size, (128, 74))
        self.assertIn("0001.jpg", (self.root / "run/validation_predictions/index.html").read_text())
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(FileExistsError):
                main(self.arguments + settings)
            with self.assertRaisesRegex(ValueError, "learning_rate"):
                main(self.arguments + settings + ["--learning-rate", "0.01", "--resume-from-checkpoint",
                                                   str(self.root / "run/checkpoints/last.pt")])
            Image.new("RGB", (64, 48), "black").save(self.root / "train/one.png")
            with self.assertRaisesRegex(ValueError, "changed"):
                main(self.arguments + settings + ["--epochs", "2", "--resume-from-checkpoint",
                                                   str(self.root / "run/checkpoints/last.pt")])


if __name__ == "__main__":
    unittest.main()
