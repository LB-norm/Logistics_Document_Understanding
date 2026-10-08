"""Native detection loss training, accumulation, AMP, and resumable RNG state."""

from contextlib import nullcontext
import math
import random

import numpy as np


def train_one_epoch(model, optimizer, loader, device, epoch, accumulation_steps=1,
                    precision="fp32", scaler=None, warmup=None, log_steps=10):
    import torch

    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = {}
    examples = 0
    # Normalize each accumulated group by its sample count, including a partial
    # final batch/group, rather than silently underweighting the end of an epoch.
    group_size = 0
    batches = len(loader)
    for step, (images, targets) in enumerate(loader):
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
        context = (nullcontext() if precision == "fp32" else torch.autocast(
            device_type=device.type, dtype=torch.float16 if precision == "fp16" else torch.bfloat16))
        with context:
            losses = model(images, targets)
            loss = sum(losses.values())
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Non-finite training loss at epoch {epoch + 1}, batch {step + 1}")
        count = len(images)
        scaler.scale(loss * count).backward()
        group_size += count
        examples += count
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + value.detach().item() * count
        if (step + 1) % accumulation_steps == 0 or step + 1 == batches:
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(group_size)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if warmup is not None and scaler.get_scale() >= previous_scale:
                warmup.step()
            optimizer.zero_grad(set_to_none=True)
            group_size = 0
        if (step + 1) % log_steps == 0 or step + 1 == batches:
            print(f"Epoch {epoch + 1}, batch {step + 1}/{batches}: loss={loss.item():.4f}, "
                  f"lr={optimizer.param_groups[0]['lr']:.6g}", flush=True)
    return {**{key: value / examples for key, value in totals.items()},
            "loss": sum(totals.values()) / examples}


def make_warmup(optimizer, batches, accumulation_steps):
    """TorchVision reference linear warmup, measured in optimizer updates."""
    import torch

    updates = math.ceil(batches / accumulation_steps)
    iterations = min(1000, updates - 1)
    if iterations <= 0:
        return None
    return torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1 / 1000, total_iters=iterations)


def seed_worker(worker_id):
    import torch

    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def capture_rng(generator):
    import torch

    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(), "loader": generator.get_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state, generator):
    import torch

    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"])
    generator.set_state(state["loader"])
    if torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state["cuda"])
