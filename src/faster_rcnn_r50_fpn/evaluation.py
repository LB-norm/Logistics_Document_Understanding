"""COCO bounding-box evaluation over all unaugmented validation images."""

import copy

from .inference import format_prediction


def coco_ground_truth(dataset):
    """Use the same non-crowd targets as training and the shared preview renderer."""
    from pycocotools.coco import COCO

    coco = COCO()
    annotations = []
    for info in dataset.images:
        for annotation in dataset.by_image[info["id"]]:
            annotations.append({
                "id": len(annotations) + 1, "image_id": info["id"],
                "category_id": dataset.categories[annotation["category_id"]]["id"],
                "bbox": list(annotation["bbox"]), "area": annotation["area"], "iscrowd": 0,
            })
    coco.dataset = {"info": {}, "images": copy.deepcopy(dataset.images),
                    "categories": copy.deepcopy(dataset.categories), "annotations": annotations}
    coco.createIndex()
    return coco


def evaluate_predictions(dataset, predictions):
    """Report standard COCO AP/AR (maxDets=1,10,100), including empty images."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    truth = coco_ground_truth(dataset)
    if predictions:
        detections = truth.loadRes(predictions)
    else:
        # COCO.loadRes indexes the first detection and cannot accept an empty list.
        detections = COCO()
        detections.dataset = {"images": copy.deepcopy(truth.dataset["images"]),
                              "categories": copy.deepcopy(truth.dataset["categories"]), "annotations": []}
        detections.createIndex()
    evaluator = COCOeval(truth, detections, "bbox")
    evaluator.params.imgIds = [info["id"] for info in dataset.images]
    evaluator.params.catIds = [category["id"] for category in dataset.categories]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = ("map", "map_50", "map_75", "map_small", "map_medium", "map_large",
             "mar_1", "mar_10", "mar_100", "mar_small", "mar_medium", "mar_large")
    return dict(zip(names, map(float, evaluator.stats)))


def evaluate(model, loader, dataset, device):
    import torch

    model.eval()
    predictions = []
    sizes = {info["id"]: (info["width"], info["height"]) for info in dataset.images}
    with torch.inference_mode():
        for images, targets in loader:
            results = model([image.to(device) for image in images])
            for target, result in zip(targets, results):
                image_id = target["image_id"].item()
                width, height = sizes[image_id]
                formatted = format_prediction(result, dataset.categories, "", width, height, threshold=0)
                predictions.extend({"image_id": image_id, "category_id": detection["category_id"],
                                    "bbox": detection["bbox_xywh"], "score": detection["score"]}
                                   for detection in formatted["detections"])
    return evaluate_predictions(dataset, predictions)
