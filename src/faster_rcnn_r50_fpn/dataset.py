"""Adapt the shared COCO layouts to TorchVision's detection target contract."""

from src.conditional_detr.dataset import CocoLayoutDataset


class DetectionDataset:
    """Keep source pixels/boxes unchanged; the model resizes and normalizes them.

    TorchVision reserves label 0 for background. Public layout labels remain
    zero-based, so only the model's targets receive the +1 offset.
    """

    def __init__(self, dataset, horizontal_flip_probability=0.0):
        if not 0 <= horizontal_flip_probability <= 1:
            raise ValueError("horizontal_flip_probability must be in [0, 1]")
        self.dataset = dataset
        self.horizontal_flip_probability = horizontal_flip_probability

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        import torch
        from torchvision.transforms.functional import to_tensor

        example = self.dataset[index]
        annotations = example["target"]["annotations"]
        boxes = []
        for annotation in annotations:
            x, y, width, height = annotation["bbox"]
            boxes.append([x, y, x + width, y + height])
        image = to_tensor(example["image"])
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor([a["category_id"] + 1 for a in annotations], dtype=torch.int64),
            "image_id": torch.tensor(example["target"]["image_id"], dtype=torch.int64),
            "area": torch.tensor([a["area"] for a in annotations], dtype=torch.float32),
            "iscrowd": torch.zeros(len(annotations), dtype=torch.int64),
        }
        if self.horizontal_flip_probability and torch.rand(()).item() < self.horizontal_flip_probability:
            image = image.flip(-1)
            target["boxes"][:, [0, 2]] = image.shape[-1] - target["boxes"][:, [2, 0]]
        return image, target


def detection_collator(examples):
    """Variable-size images stay a list; GeneralizedRCNNTransform pads the batch."""
    images, targets = zip(*examples)
    return list(images), list(targets)
