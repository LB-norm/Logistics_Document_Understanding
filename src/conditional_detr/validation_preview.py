"""Export held-out predictions and side-by-side ground truth/prediction images."""

from src.utils.layout_validation_preview import draw_regions, save_detector_validation_preview

from .inference import LayoutDetector


def save_validation_preview(model, processor, dataset, output_dir, threshold=0.5,
                            best_checkpoint=None):
    """Reuse the selected DETR model with the shared layout review renderer."""
    detector = LayoutDetector.from_model(model, processor)
    return save_detector_validation_preview(detector, dataset, output_dir, threshold, best_checkpoint)
