"""Run from the repository root: python -m src.faster_rcnn_r50_fpn.train --help."""

import argparse
import json
import math
from pathlib import Path
import random

from src.yolo_v11s.dataset import dataset_manifest

from .dataset import CocoLayoutDataset, DetectionDataset, detection_collator


def build_parser():
    parser = argparse.ArgumentParser(description="Fine-tune Faster R-CNN R50-FPN on COCO layouts")
    parser.add_argument("--train-annotations", required=True, type=Path)
    parser.add_argument("--train-images", required=True, type=Path)
    parser.add_argument("--val-annotations", required=True, type=Path)
    parser.add_argument("--val-images", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/faster_rcnn_r50_fpn"))
    parser.add_argument("--model-id", default="DEFAULT",
                        help="DEFAULT/COCO_V1, local TorchVision COCO state_dict, saved layout model, or none")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--lr-step-size", type=int, default=10)
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--shortest-edge", type=int, default=800)
    parser.add_argument("--longest-edge", type=int, default=1333)
    parser.add_argument("--trainable-backbone-layers", type=int, choices=range(6), default=3,
                        help="Trainable ResNet stages (0-5); FPN remains trainable")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze both ResNet and FPN")
    parser.add_argument("--horizontal-flip-probability", type=float, default=0.5,
                        help="Standard detection augmentation; use 0 to preserve document reading direction")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="Class-aware ROI NMS threshold")
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--validation-preview-threshold", type=float, default=0.5)
    parser.add_argument("--local-files-only", action="store_true", help="Require cached/local initial weights")
    parser.add_argument("--resume-from-checkpoint", type=Path, help="This run's checkpoints/last.pt or best.pt")
    parser.add_argument("--dry-run", action="store_true", help="Validate COCO without model loading or writes")
    return parser


def validate_args(args):
    for name in ("epochs", "batch_size", "gradient_accumulation_steps", "learning_rate",
                 "shortest_edge", "longest_edge", "lr_step_size", "max_detections"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if args.workers < 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("Invalid workers or weight decay")
    if args.longest_edge < args.shortest_edge:
        raise ValueError("longest_edge must be at least shortest_edge")
    for name in ("horizontal_flip_probability", "validation_preview_threshold", "iou_threshold", "momentum"):
        if not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if not 0 < args.lr_gamma <= 1:
        raise ValueError("lr_gamma must be in (0, 1]")
    if args.cpu and args.precision != "fp32":
        raise ValueError("Use fp32 for CPU training")


def layout_config(args, categories):
    return {"architecture": "fasterrcnn_resnet50_fpn", "categories": categories,
            **{name: getattr(args, name) for name in (
                "shortest_edge", "longest_edge", "trainable_backbone_layers", "freeze_backbone",
                "iou_threshold", "max_detections")}}


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")


def _save_checkpoint(path, state):
    import torch

    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_args(args)
    train = CocoLayoutDataset(args.train_annotations, args.train_images)
    val = CocoLayoutDataset(args.val_annotations, args.val_images, train.categories)
    train_paths = {(train.images_dir / info["file_name"]).resolve() for info in train.images}
    val_paths = {(val.images_dir / info["file_name"]).resolve() for info in val.images}
    if train_paths & val_paths:
        raise ValueError("Training and validation refer to overlapping image files")
    if not any(train.by_image.values()):
        raise ValueError("Training data must contain at least one non-crowd region")
    if not any(val.by_image.values()):
        raise ValueError("Validation data needs at least one non-crowd region for mAP selection")
    if max(len(a) for dataset in (train, val) for a in dataset.by_image.values()) > args.max_detections:
        raise ValueError("max_detections is smaller than the region count in an image")
    print(json.dumps({"train_images": len(train), "val_images": len(val), "labels": train.id2label}, indent=2))
    if args.dry_run:
        return

    import numpy as np
    import torch
    import torchvision
    from torch.utils.data import DataLoader

    from .engine import capture_rng, make_warmup, restore_rng, seed_worker, train_one_epoch
    from .evaluation import evaluate
    from .model import build_model, save_model
    from .validation_preview import save_validation_preview

    # Fail before downloading weights or writing a run if the evaluator is absent.
    from pycocotools.cocoeval import COCOeval  # noqa: F401

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if args.precision != "fp32" and device.type != "cuda":
        raise ValueError("Mixed precision requires CUDA; use fp32 on CPU")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bf16")
    output_dir = args.output_dir.resolve()
    config = json.loads(json.dumps(vars(args), default=str))
    for name in ("train_annotations", "train_images", "val_annotations", "val_images", "output_dir"):
        config[name] = str(getattr(args, name).resolve())
    config.update(torch_version=str(torch.__version__), torchvision_version=str(torchvision.__version__))
    checkpoint = None
    if args.resume_from_checkpoint:
        path = args.resume_from_checkpoint.resolve()
        if path.parent != output_dir / "checkpoints":
            raise ValueError("Resume checkpoint must be in this output directory's checkpoints directory")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if "optimizer" not in checkpoint or "rng" not in checkpoint:
            raise ValueError("Checkpoint has no resumable optimizer/RNG state")
        runtime_keys = {"cpu", "workers", "local_files_only", "resume_from_checkpoint", "dry_run",
                        "validation_preview_threshold", "epochs"}
        for key, value in config.items():
            if key not in runtime_keys and value != checkpoint["training_config"].get(key):
                raise ValueError(f"Resume requires the original {key} setting")
        if args.epochs < checkpoint["epoch"] + 1:
            raise ValueError("epochs is less than the checkpoint's completed epoch count")
        if not (output_dir / "checkpoints/best.pt").is_file():
            raise FileNotFoundError("This run's best validation checkpoint is missing")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty; choose a new output directory or resume: {output_dir}")
    manifest = dataset_manifest(train, val)
    if checkpoint and checkpoint["dataset_manifest"] != manifest:
        raise ValueError("Resume source images or COCO annotations changed")

    random.seed(args.seed)
    np.random.seed(args.seed % 2**32)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = dict(batch_size=args.batch_size, num_workers=args.workers,
                          collate_fn=detection_collator, pin_memory=device.type == "cuda",
                          worker_init_fn=seed_worker)
    train_loader = DataLoader(DetectionDataset(train, args.horizontal_flip_probability),
                              shuffle=True, generator=generator, **loader_options)
    # Validation gets a separate generator so evaluation never advances training's RNG.
    val_loader = DataLoader(DetectionDataset(val), shuffle=False,
                            generator=torch.Generator().manual_seed(args.seed), **loader_options)
    model_config = layout_config(args, train.categories)
    model = build_model(train.categories, model_config,
                        model_id="none" if checkpoint else args.model_id,
                        local_files_only=args.local_files_only).to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
    start_epoch, best_map, best_epoch, history = 0, -1.0, None, []
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng"], generator)
        start_epoch = checkpoint["epoch"] + 1
        best_map, best_epoch, history = checkpoint["best_map"], checkpoint["best_epoch"], checkpoint["history"]
        del checkpoint
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    _write_json(output_dir / "training_config.json", config)
    _write_json(output_dir / "dataset_manifest.json", manifest)
    for epoch in range(start_epoch, args.epochs):
        warmup = make_warmup(optimizer, len(train_loader), args.gradient_accumulation_steps) if epoch == 0 else None
        losses = train_one_epoch(model, optimizer, train_loader, device, epoch,
                                  args.gradient_accumulation_steps, args.precision, scaler, warmup)
        metrics = evaluate(model, val_loader, val, device)
        scheduler.step()
        history.append({"epoch": epoch + 1, "train": losses, "validation": metrics})
        improved = metrics["map"] > best_map
        if improved:
            best_map, best_epoch = metrics["map"], epoch + 1
        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                 "rng": capture_rng(generator), "best_map": best_map, "best_epoch": best_epoch,
                 "history": history, "training_config": config, "dataset_manifest": manifest,
                 "layout_config": model_config}
        if improved:
            _save_checkpoint(checkpoint_dir / "best.pt", state)
        _save_checkpoint(checkpoint_dir / "last.pt", state)
        del state
        _write_json(output_dir / "train_results.json", {"epochs": history})
        print(f"Epoch {epoch + 1}: validation mAP={metrics['map']:.4f}; best={best_map:.4f}", flush=True)
    selected = torch.load(checkpoint_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(selected["model"])
    best_epoch = selected["best_epoch"]
    metrics = selected["history"][selected["epoch"]]["validation"]
    del selected
    save_model(model, output_dir / "best_model", model_config)
    _write_json(output_dir / "eval_results.json", {"best_epoch": best_epoch, **metrics})
    preview_dir = output_dir / "validation_predictions"
    save_validation_preview(model, val, preview_dir, args.validation_preview_threshold,
                             best_checkpoint=checkpoint_dir / "best.pt")
    print(f"Best model saved to {output_dir / 'best_model'}; validation predictions and previews saved to {preview_dir}")


if __name__ == "__main__":
    main()
