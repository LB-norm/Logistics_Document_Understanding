"""Run from the repository root: python -m src.yolo_v11s.train --help."""

import argparse
import json
import math
from pathlib import Path
import shutil

from .dataset import CocoLayoutDataset, dataset_manifest, export_yolo_dataset


def build_parser():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO11s on COCO document layouts")
    parser.add_argument("--train-annotations", required=True, type=Path)
    parser.add_argument("--train-images", required=True, type=Path)
    parser.add_argument("--val-annotations", required=True, type=Path)
    parser.add_argument("--val-images", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/yolo_v11s"))
    parser.add_argument("--model-id", default="yolo11s.pt", help="Pretrained YOLO11s weights (downloaded if absent)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                        help="Target accumulation after Ultralytics warmup (sets nominal batch size)")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=1344, help="Longest edge/square letterbox size, multiple of 32")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--validation-preview-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--local-files-only", action="store_true", help="Require local weights; use fp32 or bf16 to avoid AMP helper downloads")
    parser.add_argument("--resume-from-checkpoint", type=Path, help="This run's weights/last.pt or epoch checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Validate COCO images/boxes without model loading or exports")
    return parser


def validate_args(args):
    for name in ("epochs", "batch_size", "gradient_accumulation_steps", "image_size", "learning_rate", "max_detections"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if args.image_size % 32:
        raise ValueError("image_size must be a multiple of 32")
    if args.workers < 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("Invalid workers or weight decay")
    for name in ("validation_preview_threshold", "iou_threshold"):
        if not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if args.cpu and args.precision != "fp32":
        raise ValueError("Use fp32 for CPU training")
    if args.local_files_only and args.precision == "fp16":
        raise ValueError("Use fp32 or bf16 with local_files_only: Ultralytics' fp16 check may download helper weights")


def training_options(args, model, data_path):
    """Translate shared CLI settings to explicit Ultralytics hyperparameters."""
    output_dir = args.output_dir.resolve()
    return dict(
        data=str(data_path), project=str(output_dir.parent), name=output_dir.name,
        exist_ok=True, epochs=args.epochs, batch=args.batch_size,
        nbs=args.batch_size * args.gradient_accumulation_steps,
        imgsz=args.image_size, workers=args.workers, seed=args.seed,
        device="cpu" if args.cpu else None, amp=False if args.precision == "fp32" else args.precision,
        optimizer="AdamW", lr0=args.learning_rate, lrf=0.01, weight_decay=args.weight_decay,
        cos_lr=True, warmup_epochs=args.epochs * 0.05, warmup_bias_lr=0.0,
        freeze=len(model.model.yaml["backbone"]) if args.freeze_backbone else None,
        patience=0, save=True, save_period=1, val=True, plots=True,
        cache=False,
        max_det=args.max_detections, iou=args.iou_threshold,
        resume=str(args.resume_from_checkpoint.resolve()) if args.resume_from_checkpoint else False,
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_args(args)
    train = CocoLayoutDataset(args.train_annotations, args.train_images)
    val = CocoLayoutDataset(args.val_annotations, args.val_images, train.categories)
    train_paths = {(train.images_dir / im["file_name"]).resolve() for im in train.images}
    val_paths = {(val.images_dir / im["file_name"]).resolve() for im in val.images}
    if train_paths & val_paths:
        raise ValueError("Training and validation refer to overlapping image files")
    if not any(train.by_image.values()):
        raise ValueError("Training data must contain at least one non-crowd region")
    if max(len(a) for dataset in (train, val) for a in dataset.by_image.values()) > args.max_detections:
        raise ValueError("max_detections is smaller than the region count in an image")
    print(json.dumps({"train_images": len(train), "val_images": len(val), "labels": train.id2label}, indent=2))
    if args.dry_run:
        return

    output_dir = args.output_dir.resolve()
    config_path = output_dir / "training_config.json"
    data_path = output_dir / "dataset" / "data.yaml"
    config = vars(args).copy()
    # Store absolute inputs for reliable config comparison and replay.
    for name in ("train_annotations", "train_images", "val_annotations", "val_images", "output_dir"):
        config[name] = str(getattr(args, name).resolve())
    if args.resume_from_checkpoint:
        checkpoint = args.resume_from_checkpoint.resolve()
        if not checkpoint.is_file() or checkpoint.parent != output_dir / "weights":
            raise ValueError("Resume checkpoint must exist in this output directory's weights directory")
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        runtime_keys = {"cpu", "workers", "local_files_only", "resume_from_checkpoint", "dry_run", "validation_preview_threshold"}
        for key, value in config.items():
            if key not in runtime_keys and value != saved[key]:
                raise ValueError(f"Resume requires the original {key} setting")
        manifest = json.loads((data_path.parent / "manifest.json").read_text(encoding="utf-8"))
        if dataset_manifest(train, val) != manifest:
            raise ValueError("Resume source images or COCO annotations changed")
        if not data_path.is_file():
            raise FileNotFoundError(data_path)
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty; choose a new output directory or resume: {output_dir}")

    weights = args.resume_from_checkpoint or args.model_id
    if args.local_files_only or args.resume_from_checkpoint:
        if not Path(weights).is_file():
            raise FileNotFoundError(f"Local YOLO weights not found: {weights}")

    import torch
    import ultralytics
    from ultralytics import YOLO

    if args.precision != "fp32" and not torch.cuda.is_available():
        raise ValueError("Mixed precision requires CUDA; use fp32 on CPU")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bf16")
    model = YOLO(str(weights), task="detect")
    if args.resume_from_checkpoint and (model.ckpt.get("epoch", -1) < 0 or model.ckpt.get("optimizer") is None):
        raise ValueError("Checkpoint has no resumable optimizer/epoch state; start a new fine-tuning run")
    if not args.resume_from_checkpoint:
        output_dir.mkdir(parents=True, exist_ok=True)
        config["ultralytics_version"] = ultralytics.__version__
        config_path.write_text(json.dumps(config, default=str, indent=2), encoding="utf-8")
        data_path = export_yolo_dataset(train, val, data_path.parent)
    layout_config = {"categories": train.categories, "image_size": args.image_size,
                     "iou_threshold": args.iou_threshold, "max_detections": args.max_detections}
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / "layout_config.json").write_text(json.dumps(layout_config, indent=2), encoding="utf-8")
    model.train(**training_options(args, model, data_path))
    if model.trainer.save_dir.resolve() != output_dir:
        raise RuntimeError("Ultralytics saved to an unexpected run directory")
    best = Path(model.trainer.best)
    if not best.is_file():
        raise RuntimeError("Training did not produce a best validation checkpoint")
    best_dir = output_dir / "best_model"
    best_dir.mkdir(exist_ok=True)
    shutil.copy2(best, best_dir / "model.pt")
    (best_dir / "layout_config.json").write_text(json.dumps(layout_config, indent=2), encoding="utf-8")
    metrics = model.metrics.results_dict
    (output_dir / "eval_results.json").write_text(json.dumps(metrics, default=float, indent=2), encoding="utf-8")
    from .validation_preview import save_validation_preview

    preview_dir = output_dir / "validation_predictions"
    save_validation_preview(model, val, preview_dir, threshold=args.validation_preview_threshold,
                            best_checkpoint=best, image_size=args.image_size,
                            device="cpu" if args.cpu else None, iou_threshold=args.iou_threshold,
                            max_detections=args.max_detections)
    print(f"Best model saved to {best_dir}; validation predictions and previews saved to {preview_dir}")


if __name__ == "__main__":
    main()
