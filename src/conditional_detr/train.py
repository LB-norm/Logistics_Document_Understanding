"""Run from the repository root: python -m src.conditional_detr.train --help."""

import argparse
import json
from pathlib import Path

from .dataset import CocoLayoutDataset, DetectionCollator


def build_parser():
    parser = argparse.ArgumentParser(description="Fine-tune Conditional DETR R50 on COCO layouts")
    parser.add_argument("--train-annotations", required=True, type=Path)
    parser.add_argument("--train-images", required=True, type=Path)
    parser.add_argument("--val-annotations", required=True, type=Path)
    parser.add_argument("--val-images", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/conditional_detr"))
    parser.add_argument("--model-id", default="microsoft/conditional-detr-resnet-50")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--shortest-edge", type=int, default=800)
    parser.add_argument("--longest-edge", type=int, default=1333)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--augmentation", action=argparse.BooleanOptionalAction, default=True,
                        help="Mild training-only augmentation (enabled by default)")
    parser.add_argument("--max-rotation-degrees", type=float, default=3.0)
    parser.add_argument("--perspective-fraction", type=float, default=0.02)
    parser.add_argument("--validation-preview-threshold", type=float, default=0.5,
                        help="Confidence cutoff for the final best-model validation report")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Validate all COCO images/boxes without loading a model")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    for name in ("epochs", "batch_size", "gradient_accumulation_steps", "shortest_edge", "longest_edge", "learning_rate", "backbone_learning_rate"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.workers < 0 or args.weight_decay < 0 or args.longest_edge < args.shortest_edge:
        raise ValueError("Invalid workers, weight decay, or resize limits")
    if not 0 <= args.validation_preview_threshold <= 1:
        raise ValueError("validation_preview_threshold must be in [0, 1]")
    from dataclasses import asdict
    from .augmentation import AugmentationConfig, DocumentAugmenter

    augmentation_config = AugmentationConfig(
        max_rotation_degrees=args.max_rotation_degrees,
        perspective_fraction=args.perspective_fraction,
    )
    train = CocoLayoutDataset(
        args.train_annotations, args.train_images,
        augmenter=DocumentAugmenter(augmentation_config) if args.augmentation else None,
    )
    val = CocoLayoutDataset(args.val_annotations, args.val_images, train.categories)
    train_paths = {(train.images_dir / im["file_name"]).resolve() for im in train.images}
    val_paths = {(val.images_dir / im["file_name"]).resolve() for im in val.images}
    if train_paths & val_paths:
        raise ValueError("Training and validation refer to overlapping image files")
    print(json.dumps({"train_images": len(train), "val_images": len(val), "labels": train.id2label}, indent=2))
    if args.dry_run:
        return

    import torch
    from transformers import (
        AutoImageProcessor, ConditionalDetrForObjectDetection,
        Trainer, TrainingArguments, set_seed,
    )

    set_seed(args.seed)
    processor = AutoImageProcessor.from_pretrained(
        args.model_id, local_files_only=args.local_files_only,
        size={"shortest_edge": args.shortest_edge, "longest_edge": args.longest_edge},
    )
    model = ConditionalDetrForObjectDetection.from_pretrained(
        args.model_id, id2label=train.id2label,
        label2id={name: i for i, name in train.id2label.items()},
        ignore_mismatched_sizes=True, auxiliary_loss=True,
        local_files_only=args.local_files_only,
    )
    model.config.layout_categories = train.categories
    if max(len(a) for a in train.by_image.values()) > model.config.num_queries:
        raise ValueError("An image has more regions than the model's detection queries")
    backbone = model.model.backbone
    if args.freeze_backbone:
        backbone.requires_grad_(False)
    backbone_ids = {id(p) for p in backbone.parameters()}
    groups = []
    for is_backbone, lr in ((False, args.learning_rate), (True, args.backbone_learning_rate)):
        for decay in (False, True):
            params = [p for name, p in model.named_parameters()
                      if p.requires_grad and (id(p) in backbone_ids) == is_backbone
                      and (p.ndim > 1 and not name.endswith("bias")) == decay]
            if params:
                groups.append({"params": params, "lr": lr, "weight_decay": args.weight_decay if decay else 0.0})
    optimizer = torch.optim.AdamW(groups)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        # Transformers 5 accepts a fraction here; warmup_ratio was removed.
        lr_scheduler_type="cosine", warmup_steps=0.05,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=False, metric_for_best_model="eval_loss", greater_is_better=False,
        prediction_loss_only=True, remove_unused_columns=False,
        dataloader_num_workers=args.workers, seed=args.seed, data_seed=args.seed,
        fp16=args.precision == "fp16", bf16=args.precision == "bf16", use_cpu=args.cpu,
        logging_steps=10, report_to="none",
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train, eval_dataset=val,
        data_collator=DetectionCollator(processor), processing_class=processor,
        optimizers=(optimizer, None),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_config.json").write_text(
        json.dumps({**vars(args), "augmentation_config": asdict(augmentation_config)}, default=str, indent=2), encoding="utf-8",
    )
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    # from_pretrained applies Transformers' checkpoint key conversions. Trainer's
    # direct state-dict reload skips those conversions in Transformers 5.3.
    if trainer.state.best_model_checkpoint is None:
        raise RuntimeError("Training did not produce a validation checkpoint")
    best_model = ConditionalDetrForObjectDetection.from_pretrained(
        trainer.state.best_model_checkpoint, local_files_only=True,
    ).to(trainer.model.device)
    trainer.model = best_model
    trainer.model_wrapped = best_model
    trainer.save_model(str(args.output_dir / "best_model"))
    processor.save_pretrained(args.output_dir / "best_model")
    trainer.save_state()
    trainer.save_metrics("train", result.metrics)
    trainer.save_metrics("eval", trainer.evaluate())
    from .validation_preview import save_validation_preview

    preview_dir = args.output_dir / "validation_predictions"
    save_validation_preview(
        best_model, processor, val, preview_dir,
        threshold=args.validation_preview_threshold,
        best_checkpoint=trainer.state.best_model_checkpoint,
    )
    print(f"Validation predictions and image previews saved to {preview_dir}")


if __name__ == "__main__":
    main()
