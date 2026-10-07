# Document rectification

Reusable preprocessing for a future YOLO field detector or another image parser:
page quadrilateral detection → perspective warp → line/text deskew → orientation
normalization. No model weights or GPU are required for the geometric stages.

Use the same preprocessing in two ways: pass an image array through
`rectify_document` before a downstream model, or use `rectify_folder`/the CLI to
prepare saved images for labeling.

## One image in a model pipeline

```python
import cv2
from src.preprocessing.document_rectification import (
    rectify_document,
    tesseract_orientation,
)

image = cv2.imread("document.jpg")  # uint8 BGR; grayscale is also supported
result = rectify_document(image, orientation_detector=tesseract_orientation)
rectified_image = result.image  # Pass directly to the next model; no file is written.
# detections = yolo.predict(rectified_image)
# Pixel xyxy boxes from YOLO can be mapped to the source image:
# original_polygons = result.map_boxes(detections[0].boxes.xyxy.cpu().numpy(), inverse=True)
print(result.warnings)
```

The API accepts one image array and returns a `RectificationResult`; its `.image`
is the rectified image array. The input array is preserved. Keep the result object
if you also need transformations, stage diagnostics, or source-coordinate mapping.

## A folder of images for labeling

```bash
python -m src.preprocessing.rectify_images data/unlabeled data/rectified --recursive
```

This saves lossless PNG images under `data/rectified`, ready for your labeling
tool. For example, `data/unlabeled/train/page.jpg` becomes
`data/rectified/train/page.png`. `--recursive` includes nested folders and keeps
their structure; without it, only files directly in the input folder are processed.
Supported input extensions are PNG, JPG/JPEG, BMP, TIF/TIFF, and WebP, matched
without regard to case. Non-image files are ignored.

For each output, an adjacent `.png.json` records the source/output sizes,
transformation, corrections, and uncertainty warnings. `manifest.json` summarizes
all successes and errors. A corrupt image is recorded as an error and processing
continues with the remaining images. The CLI exits with status 1 if any image
fails; uncertain rectification stages produce warnings and a saved image, so review
those records before labeling.

Use separate, non-overlapping input and output folders. Existing outputs require
`--overwrite`, and filename collisions (e.g. `page.jpg` and `page.png` in the same
folder both becoming `page.png`) are rejected before processing. Rename colliding
inputs or put them in separate subfolders. Empty folders report an error.

The folder API uses the same config and orientation callback as the model API:

```python
from src.preprocessing.document_rectification import tesseract_orientation
from src.preprocessing.rectify_images import rectify_folder

summary = rectify_folder(
    "data/unlabeled",
    "data/rectified",
    recursive=True,
    orientation_detector=tesseract_orientation,
)
print(summary["succeeded"], summary["failed"])
```

Both CLI commands (`rectify_images` and the original `document_rectification`)
accept either a single input image/output image pair or an input/output folder
pair. `rectify_file` is also available for programmatic saving of one image.

## Orientation and individual stages

The optional orientation adapter needs the system `tesseract` executable and
`osd.traineddata`. For example, on Debian/Ubuntu install `tesseract-ocr` and
`tesseract-ocr-osd`. It returns a clockwise correction of 0/90/180/270 degrees or
abstains if unavailable, timed out, or below confidence 15. Sparse documents can
produce insufficient evidence. A custom `orientation_detector(image)` callback
can instead wrap your document orientation classifier; return the **correction**
to apply, rather than the detected input angle. Passing `clockwise_rotation=90`
provides a known correction and overrides detection. Without either argument the
Python API leaves orientation unresolved and reports that in `result.warnings`.
Page contours and ruling lines alone cannot distinguish upright from upside down.

From the repository root, rectify one raster image and save transformation metadata:

```bash
python -m src.preprocessing.document_rectification input.jpg output/rectified.png
python -m src.preprocessing.document_rectification input.jpg output/rectified.png --orientation 180
```

The CLI defaults to optional Tesseract orientation detection. `--orientation skip`
disables it; `--orientation 0` asserts the page is already upright after deskew.
It writes the image and an adjacent `rectified.png.json` with source/output sizes
(width, height), the original-to-output homography, stage corrections and warnings.
It accepts raster images; render PDFs to page images before calling it.

Each stage is also callable independently: `detect_page_quad`, `warp_page`,
`estimate_skew`, `rotate_expanded`, and `normalize_orientation`.
`RectificationConfig` controls detection size, minimum page area/contrast, maximum
small-angle skew (15 degrees by default), line consensus, and stage toggles.
Use `detect_page=False` for already cropped scans. A supplied `page_quad` bypasses
detection; its four input pixel coordinates can be in any order.

Detection examines a thumbnail but warps the original resolution. It expects one
flat, bright, fully visible page on a darker background. Large contours must cover
at least 35% of the image and pass a contrast check to reduce accidental cropping
to an internal table. If no credible page is found, the perspective stage preserves
the full frame and reports a warning. Line detection considers both horizontal and
vertical rules plus joined text edges, and only deskews when enough lines agree.
Expanded rotation canvases avoid clipping. Curved pages, severe folds, a missing
page boundary, large residual skew, or multiple pages require another detector or
nonlinear dewarping; this module reports uncertainty but does not solve those cases.

For YOLO, rectify **before annotating/training**, then use the same preprocessing
at inference. Existing labels must be transformed along with their images. Use
`result.map_points(...)` for annotation polygons or `result.map_boxes(...)` for
pixel xyxy rectangles. A projective transform maps a rectangle to a quadrilateral;
`map_boxes` returns all four corners instead of silently approximating a rectangle.
For new axis-aligned training boxes, enclose the transformed polygon using its
minimum/maximum x and y, clip to the output frame, then normalize by output width
and height. Transforming an already loose box can preserve its excess background;
inspect those labels or annotate the rectified images directly.

Rectification can reduce geometric variation for axis-aligned field detection;
validate that benefit on held-out documents and retain modest rotation/perspective
augmentation to cover imperfect preprocessing. Benchmark page cropping as well,
since losing a field is worse than retaining a slightly skewed frame.

Tests:

```bash
python -m unittest discover -s tests -p 'test_document_rectification.py' -v
python -m unittest discover -s tests -p 'test_rectify_images.py' -v
```

Implementation references: [OpenCV geometric transforms](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html),
[Hough line detection](https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html),
and [Tesseract command-line usage](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html).
