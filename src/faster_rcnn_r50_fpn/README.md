# Faster R-CNN R50-FPN document layout detection

Fine-tunes TorchVision's COCO-pretrained `fasterrcnn_resnet50_fpn` (ResNet-50
with a feature pyramid) on the same document layout regions as Conditional DETR
and YOLO11s. This detects boxes and labels; OCR is a separate downstream stage.

Install alongside the repository requirements. Use matching PyTorch/TorchVision
wheels for your CPU/CUDA environment:

```bash
python -m pip install -r src/faster_rcnn_r50_fpn/requirements.txt
```

Supply the same training/validation COCO detection JSONs and image roots used by
the other layout modules. JSON includes `images` (`id`, `file_name`, `width`,
`height`), `categories` (`id`, `name`), and `annotations` (`image_id`, `category_id`,
`bbox`). Boxes are absolute pixel `[x, y, width, height]` in the stored raster,
without EXIF rotation. The shared loader validates image dimensions, paths and
boxes, supports empty images, and excludes crowd regions. Sparse category IDs
are mapped to contiguous labels. Validation categories must match training IDs
and names; subsets are accepted. Each split needs at least one non-crowd region
for training and validation mAP selection. Individual images may have no regions.

Split by source document to keep related pages/photos in one split. Identical
image paths across splits are rejected; document copies at different paths
cannot be identified automatically.

Validate the dataset without loading/downloading weights or writing a run:

```bash
python -m src.faster_rcnn_r50_fpn.train \
  --train-annotations data/layout/train.json --train-images data/layout/train/images \
  --val-annotations data/layout/val.json --val-images data/layout/val/images \
  --output-dir runs/faster_rcnn_r50_fpn/cmr --dry-run
```

Remove `--dry-run` to train. Use a new, empty output directory for each experiment.
The first run downloads TorchVision's `COCO_V1` detector weights and replaces its
classifier/regressor with a layout head. TorchVision reserves internal label 0
for background; training foreground labels are 1 through N. Public predictions
and validation reports retain the other modules' zero-based `label_id` contract.

The implementation follows the standard TorchVision detection approach:

- Variable-size RGB float tensors in [0, 1] and XYXY targets; model-owned
  ImageNet normalization, aspect-preserving resize (shortest edge 800, longest
  edge capped at 1333), and batch padding. Images and boxes resize together.
- Native RPN objectness/regression and ROI classification/regression losses.
- SGD with learning rate 0.005, momentum 0.9, weight decay 0.0005; StepLR decays
  by 0.1 every 10 epochs. Linear warmup from 0.001 times the initial learning
  rate runs for at most 1000 optimizer updates in the first epoch, ending before
  its last update. Warmup is skipped if that epoch has only one update.
- 30 epochs, batch size 2, accumulation 1 by default. Optional
  `--gradient-accumulation-steps` averages gradients across each group, including
  partial final groups; it does not automatically scale the learning rate.
- COCO-style frozen batch normalization. Three ResNet stages are trainable by
  default; choose `--trainable-backbone-layers 0` through `5`. FPN stays trainable
  unless `--freeze-backbone` freezes both ResNet and FPN.
- Training-only horizontal flips at probability 0.5, transforming boxes with
  the image. Set `--horizontal-flip-probability 0` to preserve document reading
  direction. Validation/inference apply no augmentation.
- Native class-aware ROI NMS at IoU 0.5, with at most 100 detections per image.
  Change `--iou-threshold` / `--max-detections` at training time; inference loads
  those saved settings. Limits below the dataset's maximum region count fail.

Tune the learning rate, augmentation and resolution on held-out validation
data. These settings intentionally differ from DETR/YOLO training recipes.
`--precision fp16` or `bf16` enables CUDA mixed precision; FP32 is the default.
Use `--cpu` for CPU training and reduce resolution/batch size to reduce memory.

`--local-files-only` requires the COCO weights already in Torch's checkpoint
cache. Alternatively set `--model-id /path/to/torchvision_coco_state_dict.pth`
or point to this module's saved layout model directory/checkpoint. Saved layout
models must have the same category mapping. `--model-id none` initializes without
downloads, primarily for offline integration checks.

Validation runs native inference after every epoch and computes bounding-box
COCO mAP using pycocotools. The best model is selected by AP averaged over IoU
0.50–0.95. Evaluation includes all validation images, uses all scores rather than
the gallery's confidence cutoff, and uses standard COCO maxDets 1/10/100 even
if the model's prediction limit is increased. Categories without validation
targets are omitted from the average; metrics for absent size groups are -1.
Crowd regions are excluded consistently with the shared loader and gallery.
Training losses are model-specific; compare models with a common evaluator on
the same held-out images, and keep a separate test split for final comparison.

Run artifacts:

| Path | Contents |
| --- | --- |
| `training_config.json` | Launcher settings and torch/torchvision versions |
| `dataset_manifest.json` | Category mapping and source image/annotation hashes |
| `train_results.json` | Per-epoch component losses and validation AP/AR |
| `checkpoints/best.pt`, `checkpoints/last.pt` | Best/latest model, optimizer, scheduler, scaler and RNG states |
| `best_model/model.pt`, `best_model/layout_config.json` | Portable selected model and mapping/settings |
| `eval_results.json` | Selected epoch and its validation AP/AR |
| `validation_predictions/` | Shared JSON report, numbered side-by-side JPEGs and HTML gallery |

Checkpoints are saved at each epoch boundary, retaining best and latest.
Resume with the original arguments/output directory plus:

```bash
--resume-from-checkpoint runs/faster_rcnn_r50_fpn/cmr/checkpoints/last.pt
```

The launcher checks hyperparameters, library versions, and source content hashes.
CPU/device selection, workers, local-file policy, and preview confidence may
change. `--epochs` is the total desired epoch count and may be increased to
continue a completed run. Resumption restores optimizer/scheduler/scaler and
random states; use the same hardware and worker count for reproducibility.
Portable `best_model/model.pt` supports inference or a new fine-tuning run, and
does not contain optimizer state.

After training, the selected model predicts every unaugmented validation image
once for the shared review gallery. Default confidence is 0.5; adjust with
`--validation-preview-threshold 0.3`. Download the entire
`validation_predictions/` directory to review `index.html` locally. JSON retains
original pixel coordinates; gallery images are resized for display.

```bash
python -m src.faster_rcnn_r50_fpn.inference \
  --model-dir runs/faster_rcnn_r50_fpn/cmr/best_model \
  --images data/example.jpg data/another.png \
  --threshold 0.5 --output output/faster_rcnn_r50_fpn/layouts.json
```

Programmatic use:

```python
from src.faster_rcnn_r50_fpn.inference import LayoutDetector

detector = LayoutDetector("runs/faster_rcnn_r50_fpn/cmr/best_model")
prediction = detector.predict("data/example.jpg", threshold=0.5)
```

The model loads once, selecting CUDA when available (`--device cpu` forces CPU).
Output matches DETR/YOLO: a JSON list with each image's path, width, height, and
detections containing `label_id`, `label`, original COCO `category_id`, `score`,
and pixel `bbox_xyxy` / `bbox_xywh`. Boxes are clipped to image bounds, degenerate
boxes are dropped, and detections are sorted by confidence. TorchVision restores
predictions to the original raster size; no manual inverse resizing is applied.

Offline integration checks exercise the real R50-FPN losses/backward pass,
resumption, portable save/reload, inference CLI, COCO metrics, and review gallery
using tiny synthetic images and random initialization:

```bash
python -m unittest discover -s tests -p 'test_faster_rcnn_r50_fpn.py' -v
```

References: [TorchVision detection fine-tuning tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html),
[R50-FPN API](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html),
and [reference training loop](https://github.com/pytorch/vision/blob/main/references/detection/engine.py).
