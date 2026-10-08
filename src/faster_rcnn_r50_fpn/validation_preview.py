"""Use the same held-out JSON report and gallery as DETR and YOLO."""

from src.utils.layout_validation_preview import save_detector_validation_preview

from .inference import LayoutDetector


def save_validation_preview(model, dataset, output_dir, threshold=0.5, best_checkpoint=None):
    detector = LayoutDetector.from_model(model, dataset.categories)
    return save_detector_validation_preview(detector, dataset, output_dir, threshold, best_checkpoint)
