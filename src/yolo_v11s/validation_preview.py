"""Use the same validation report and gallery renderer as Conditional DETR."""

from src.utils.layout_validation_preview import save_detector_validation_preview

from .inference import LayoutDetector


def save_validation_preview(model, dataset, output_dir, threshold=0.5,
                            best_checkpoint=None, image_size=1344, device=None,
                            iou_threshold=0.7, max_detections=300):
    detector = LayoutDetector.from_model(model, dataset.categories, image_size, device,
                                         iou_threshold, max_detections)
    return save_detector_validation_preview(detector, dataset, output_dir, threshold, best_checkpoint)
