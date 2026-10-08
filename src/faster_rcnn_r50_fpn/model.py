"""Standard R50-FPN architecture with COCO initialization and a layout head."""

import json
from pathlib import Path
from urllib.parse import urlparse


def build_model(categories, config, model_id="none", local_files_only=False):
    import torch
    from torchvision.models.detection import FasterRCNN, FasterRCNN_ResNet50_FPN_Weights
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.ops.misc import FrozenBatchNorm2d

    # Construct explicitly so offline reloads use the same FrozenBatchNorm2d as
    # the COCO-pretrained factory (weights=None otherwise switches to BatchNorm).
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50", weights=None, norm_layer=FrozenBatchNorm2d,
        trainable_layers=config["trainable_backbone_layers"],
    )
    model = FasterRCNN(
        backbone, num_classes=91, min_size=config["shortest_edge"], max_size=config["longest_edge"],
        box_score_thresh=0.0, box_nms_thresh=config["iou_threshold"],
        box_detections_per_img=config["max_detections"],
    )
    layout_checkpoint = None
    if model_id in ("DEFAULT", "COCO_V1"):
        weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        if local_files_only:
            cached = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
            if not cached.is_file():
                raise FileNotFoundError(f"Cached COCO weights not found: {cached}")
            state = torch.load(cached, map_location="cpu", weights_only=True)
        else:
            state = weights.get_state_dict(progress=True, check_hash=True)
        model.load_state_dict(state)
    elif str(model_id).lower() != "none":
        path = Path(model_id)
        if path.is_dir():
            path = path / "model.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        if "layout_config" in state:
            if state["layout_config"]["categories"] != categories:
                raise ValueError("Saved category mapping does not match training categories")
            layout_checkpoint = state["model"]
        else:
            # A local TorchVision COCO state_dict is equivalent to COCO_V1.
            model.load_state_dict(state)
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, len(categories) + 1)
    if layout_checkpoint is not None:
        model.load_state_dict(layout_checkpoint)
    if config.get("freeze_backbone", False):
        model.backbone.requires_grad_(False)
    return model


def save_model(model, output_dir, config):
    """Export a portable state_dict and the category/preprocessing settings."""
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save({"model": state, "layout_config": config}, output_dir / "model.pt")
    (output_dir / "layout_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
