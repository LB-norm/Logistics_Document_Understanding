# YOLO11s document layout detection

Fine-tunes the COCO-pretrained Ultralytics `yolo11s.pt` detector on the same layout
regions as `src/conditional_detr`. This detects boxes and labels; OCR remains a
separate downstream stage. Ultralytics calls this model **YOLO11s**.

From the repository root, install the repository requirements and the additional
model dependency:

```bash
pip install -r src/yolo_v11s/requirements.txt
```

Supply exactly the same training/validation COCO JSONs and image roots used for
Conditional DETR. Each JSON needs `images` (`id`, `file_name`, `width`, `height`),
`categories` (`id`, `name`), and `annotations` (`image_id`, `category_id`, `bbox`).
Boxes are absolute pixel `[x, y, width, height]` in the stored raster, without EXIF
rotation. The shared COCO loader validates dimensions and boxes, ignores crowd
regions, supports empty images, and maps sparse category IDs to contiguous model
labels. Validation category IDs/names must match training (subsets are accepted).
Training needs at least one non-crowd region.

Split by source document before exporting COCO so related pages/photos stay in
one split. As with DETR, identical paths across splits are rejected; copies at
different paths cannot be recognized as the same document.

Validate everything without loading/downloading weights or writing a run:

```bash
python -m src.yolo_v11s.train \
  --train-annotations data/layout/train.json --train-images data/layout/train/images \
  --val-annotations data/layout/val.json --val-images data/layout/val/images \
  --output-dir runs/yolo_v11s/cmr --dry-run
```

Remove `--dry-run` to train. The first run downloads `yolo11s.pt`; for local weights,
use `--model-id models/yolo11s.pt --local-files-only`. Use a new, empty output
directory for each experiment. Training automatically converts the validated
COCO data to numbered RGB PNGs and normalized YOLO labels under the run's
`dataset/` directory. This consumes extra disk space but preserves the original
images/annotations, supports nested paths and duplicate image stems, and keeps
EXIF handling consistent with DETR.

Defaults are 30 epochs, batch size 2, target gradient accumulation 4, AdamW,
learning rate `1e-4`, weight decay `1e-4`, cosine decay to 1% of the initial rate,
and 5% warmup. Ultralytics uses nominal batch size `batch × accumulation` and
ramps accumulation during warmup. There is one learning rate for the whole model;
the DETR-specific backbone learning rate does not apply. `--freeze-backbone`
freezes the backbone layers declared in the YOLO architecture.

YOLO uses aspect-preserving letterbox resizing with `--image-size 1344` by default
(longest edge and square training canvas). Sizes must be multiples of 32; use
`--image-size 800` or `640` to reduce memory use. This differs from DETR's shortest
edge 800 / longest edge 1333 convention. Tune each model's resolution and learning
rate on validation data and record the settings when comparing models.

FP32 is the default. `--precision fp16` or `bf16` enables CUDA mixed precision on
supported hardware; `--cpu` uses FP32. FP16's Ultralytics AMP check may download
additional helper weights, so `--local-files-only` requires FP32 or BF16.

Training uses Ultralytics' standard detection trainer, preprocessing, loss,
augmentation, and validation pipeline. Native augmentation defaults include
mosaic, random translation/scaling, HSV changes, and horizontal flips; these
differ from DETR's document augmenter. The resolved settings are recorded in
`args.yaml`, and the seed controls Ultralytics' training randomness. Prediction
uses the native preprocessing and class-aware NMS pipeline without test-time
augmentation. There is no custom YOLO trainer or augmentation implementation.

Run artifacts:

| Path | Contents |
| --- | --- |
| `training_config.json` | Launcher arguments and Ultralytics version |
| `dataset/` | Converted images, labels, `data.yaml`, source content hashes and category mapping |
| `args.yaml`, `results.csv`, plots | Ultralytics configuration, epoch losses and detection metrics |
| `weights/best.pt`, `weights/last.pt`, `weights/epoch*.pt` | Best, latest and per-epoch checkpoints |
| `weights/layout_config.json` | Category IDs and inference settings for loading any weights file |
| `best_model/model.pt`, `best_model/layout_config.json` | Portable selected model; copy the whole directory |
| `eval_results.json` | Selected checkpoint's validation precision, recall, mAP50, mAP50–95 |
| `validation_predictions/` | Same JSON report, numbered side-by-side JPEGs and HTML gallery as DETR |

The best checkpoint is selected by validation mAP50–95 in the supported
Ultralytics version. DETR currently selects by validation loss. These losses are
model-specific and should not be compared numerically. Use the same held-out
images, labels, and confidence cutoff for visual/error comparison; use a common
detection evaluator on both models' JSON predictions for a quantitative ranking.
Keep a separate test split for the final comparison after tuning on validation.

Resume an **interrupted** run with its original arguments/output directory plus:

```bash
--resume-from-checkpoint runs/yolo_v11s/cmr/weights/last.pt
```

The launcher checks the original hyperparameters and source content hashes.
CPU/device selection, workers, local-file policy, and preview confidence may change.
Completed Ultralytics `best.pt`/`last.pt` files have their optimizer state removed
and cannot resume. To train further, start a new run with `--model-id` pointing to
the selected weights. Per-epoch checkpoints retain training state. Native checkpoint
arguments govern resumed training; repeat the original launcher arguments.

After training the selected model predicts each validation image once at confidence
0.5. Adjust with `--validation-preview-threshold 0.3`. Download the entire
`validation_predictions/` directory to review `index.html` locally. JSON boxes use
original pixel coordinates; images in the gallery are resized for display.

Inference:

```bash
python -m src.yolo_v11s.inference \
  --model-dir runs/yolo_v11s/cmr/best_model \
  --images data/example.jpg data/another.png \
  --threshold 0.5 --output output/yolo_v11s/layouts.json
```

Programmatic use:

```python
from src.yolo_v11s.inference import LayoutDetector

detector = LayoutDetector("runs/yolo_v11s/cmr/best_model")
prediction = detector.predict("data/example.jpg", threshold=0.5)
```

The model loads once and selects CUDA when available. `--device cpu` forces CPU.
The output matches DETR's JSON list: each image has its path, width, height, and
detections with `label_id`, `label`, original COCO `category_id`, `score`, and pixel
`bbox_xyxy`/`bbox_xywh`, clipped to image bounds and sorted by confidence. YOLO
performs class-aware NMS with IoU 0.7 and keeps at most 300 detections per image.
Configure these at training time with `--iou-threshold` and `--max-detections`;
they are saved for inference. A detection limit smaller than the dataset's largest
region count is rejected.

Lightweight data-conversion and prediction-contract checks (no training or weight downloads):

```bash
python -m unittest discover -s tests -p 'test_yolo_v11s.py' -v
python -m unittest discover -s tests -p 'test_conditional_detr.py' -v
```

API references: [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/),
[training](https://docs.ultralytics.com/modes/train/), and
[detection training API](https://docs.ultralytics.com/reference/models/yolo/detect/train/).
