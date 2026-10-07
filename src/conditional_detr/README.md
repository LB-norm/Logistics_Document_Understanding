# Conditional DETR document layout detection

Fine-tunes `microsoft/conditional-detr-resnet-50` to detect labeled CMR regions.
This detects layout boxes and labels; OCR is a separate downstream stage.

From the repository root, install the repository dependencies plus:

```bash
pip install -r src/conditional_detr/requirements.txt
```

Supply separate training and validation COCO detection JSON files and image roots.
Each JSON needs `images` (`id`, `file_name`, `width`, `height`), `categories`
(`id`, `name`), and `annotations` (`image_id`, `category_id`, `bbox`). Boxes are
absolute pixel `[x, y, width, height]`. Category IDs may be sparse; they are mapped
to contiguous model labels and saved with the model. Validation categories must
use the same IDs and names (a subset is accepted). Empty images are supported;
crowd annotations are excluded. Invalid boxes and mismatched image sizes fail early.
Coordinates refer to the stored raster, without applying EXIF orientation.
Split by source document before exporting COCO to keep related pages and photos
out of different splits. The loader rejects identical file paths across splits,
but cannot identify copies of the same document at different paths.

```bash
python -m src.conditional_detr.train \
  --train-annotations data/layout/train.json --train-images data/layout/train/images \
  --val-annotations data/layout/val.json --val-images data/layout/val/images \
  --output-dir runs/conditional_detr/cmr --dry-run
```

Remove `--dry-run` to train. The first training run downloads the Hugging Face
checkpoint. `--local-files-only` requires cached files. Defaults: 30 epochs,
batch size 2, accumulation 4, AdamW, learning rates `1e-4` (transformer/head)
and `1e-5` (backbone), cosine schedule with 5% warmup, and aspect-preserving
resize to shortest edge 800 with longest edge capped at 1333. Images and boxes
are resized together and batches are padded with pixel masks. Mild augmentation
is enabled for training only; validation and inference receive no augmentation.
No random crops or flips are applied. Use `--freeze-backbone` for smaller datasets, tune epochs
against held-out validation, and reduce batch size/resolution if memory is limited.
`--precision fp16` or `bf16` enables mixed precision on compatible hardware;
FP32 is the default. `--cpu` supports CPU runs.

Each training image access samples a fresh variant before processor resizing:

| Method | Probability | Range |
| --- | --- | --- |
| Rotation and perspective together | 50% | Rotation ±3°, corner displacement up to 2% of each image dimension |
| Brightness, contrast, saturation together | 50% | Brightness/contrast 0.85–1.15; saturation 0.9–1.1 |
| Gaussian blur | 15% | Radius 0.2–0.8 source pixels |
| Gaussian noise | 15% | Standard deviation 1–3 on 0–255 pixels |
| JPEG compression | 20% | Quality 75–95 |

Geometric transforms move all four box corners with the image, recompute enclosing
axis-aligned boxes and areas, and fit the entire warped page into the original
canvas with white borders to retain edge regions. Source files/annotations are
never modified. Online augmentation keeps the dataset at 25 examples per epoch
if you supply 25 images; repeated epochs produce different variants.
Sampling follows the Trainer seed and DataLoader worker seeds. The full settings
are recorded in `training_config.json`. Disable with `--no-augmentation`, or adjust
geometry with `--max-rotation-degrees 2 --perspective-fraction 0.01`. Probabilities
and photometric ranges are defined in `augmentation.py`. The dry run validates
source annotations but does not sample augmentations.

Checkpoints are saved each epoch; the best and latest are retained. `best_model/`
contains the model and processor selected by lowest validation loss. Validation
reports loss, not detection mAP. Resume with the same arguments/output directory
and `--resume-from-checkpoint runs/conditional_detr/cmr/checkpoint-N`.
`training_config.json`, Trainer state, and train/eval metrics are stored in the run.

After training, the best model also runs inference once on every validation image.
The run's `validation_predictions/` directory contains `predictions.json` with
predictions, ground truth, threshold, and selected checkpoint; `index.html` is a
local review gallery, and numbered JPEGs show ground truth on the left and
predictions with confidence scores on the right. Boxes in JSON use original pixel
coordinates; previews are resized for viewing. Download the entire directory to
view the gallery locally. Default confidence cutoff is 0.5; adjust with
`--validation-preview-threshold 0.3` to include less confident detections.
The report reuses the selected model and performs no augmentation or training.

```bash
python -m src.conditional_detr.inference \
  --model-dir runs/conditional_detr/cmr/best_model \
  --images data/example.jpg data/another.png \
  --threshold 0.5 --output output/conditional_detr/layouts.json
```

Output is a JSON list, one entry per image, with original image dimensions and
detections containing confidence, model label ID/name, original COCO category ID,
and pixel boxes in both XYXY and XYWH format. Boxes are clipped to image bounds.
The model loads once for all images and runs on CUDA when available.
Programmatic use: `from src.conditional_detr.inference import LayoutDetector`,
then `LayoutDetector(model_dir).predict(image_path, threshold=0.5)`.

API reference: [Hugging Face Conditional DETR documentation](https://huggingface.co/docs/transformers/model_doc/conditional_detr).
