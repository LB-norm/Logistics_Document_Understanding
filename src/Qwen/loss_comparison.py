"""Compare losses and trainable gradients at identical weights, without updates."""
from __future__ import annotations

from contextlib import nullcontext
import time
from typing import Any

import torch


def compare_loss_methods(model: Any, controller: Any, batches: list[dict[str, Any]],
                         compute_dtype: str = "float32") -> dict[str, Any]:
    """Replay the same RNG state for both methods; leave weights and RNG unchanged.

    Reference gradients are held on CPU, one batch at a time. Mixed precision can
    choose different GEMM kernels for selected/chunked shapes, so report both the
    loss error and full trainable-gradient relative L2 error, with explicit gates.
    """
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if any(p.grad is not None for p in parameters.values()):
        raise ValueError("Compare losses before training, when gradients are empty")
    devices = sorted({p.device.index for p in model.parameters() if p.device.type == "cuda"})
    cpu_rng = torch.get_rng_state()
    cuda_rng = {device: torch.cuda.get_rng_state(device) for device in devices}
    was_enabled, was_training = controller.enabled, model.training
    mixed = compute_dtype != "float32"
    loss_rtol, gradient_rtol = ((1e-4, 0.02) if mixed else (1e-5, 1e-4))
    results = []

    def restore_rng():
        torch.set_rng_state(cpu_rng)
        for device, state in cuda_rng.items():
            torch.cuda.set_rng_state(state, device)

    def measure(batch, enabled):
        controller.enabled = enabled
        model.zero_grad(set_to_none=True)
        restore_rng()
        for device in devices:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        device_type = batch["input_ids"].device.type
        context = (torch.autocast(device_type, dtype=getattr(torch, compute_dtype))
                   if mixed else nullcontext())
        start = time.perf_counter()
        with context:
            loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise ValueError("Loss comparison produced a non-finite loss")
        # Mirror mixed-precision backward scaling for FP16. Both methods use
        # the same initial scale; no optimizer step or scaler update occurs.
        scaler = torch.amp.GradScaler("cuda", enabled=device_type == "cuda" and compute_dtype == "float16")
        scaler.scale(loss).backward()
        if scaler.is_enabled():
            inverse_scale = 1.0 / scaler.get_scale()
            for parameter in parameters.values():
                if parameter.grad is not None:
                    parameter.grad.mul_(inverse_scale)
        for device in devices:
            torch.cuda.synchronize(device)
        return {
            "loss": loss.detach().float().item(),
            "forward_backward_seconds": time.perf_counter() - start,
            "cuda_peak_allocated_bytes": {
                str(device): torch.cuda.max_memory_allocated(device) for device in devices},
            "cuda_peak_reserved_bytes": {
                str(device): torch.cuda.max_memory_reserved(device) for device in devices},
        }

    try:
        model.train()
        for batch in batches:
            reference = measure(batch, False)
            gradients = {name: p.grad.detach().cpu().float().clone() if p.grad is not None else None
                         for name, p in parameters.items()}
            optimized = measure(batch, True)
            squared_reference = squared_difference = 0.0
            max_absolute = 0.0
            missing = []
            parameter_errors = []
            for name, p in parameters.items():
                a, b = gradients[name], p.grad
                if (a is None) != (b is None):
                    missing.append(name)
                    continue
                if a is None:
                    continue
                b = b.detach().cpu().float()
                if not torch.isfinite(b).all() or not torch.isfinite(a).all():
                    raise ValueError(f"Non-finite gradient for {name}")
                # FP64 scalar reductions keep the reported error accurate even
                # with many small adapter gradients. Reduce in bounded CPU
                # chunks to avoid large temporary copies of full-tuning tensors.
                norm_squared = difference_squared = 0.0
                a_flat, b_flat = a.reshape(-1), b.reshape(-1)
                for start in range(0, a.numel(), 1_000_000):
                    ref_chunk = a_flat[start:start + 1_000_000]
                    delta = b_flat[start:start + 1_000_000] - ref_chunk
                    norm_squared += ref_chunk.double().square().sum().item()
                    difference_squared += delta.double().square().sum().item()
                    max_absolute = max(max_absolute, delta.abs().max().item())
                squared_reference += norm_squared
                squared_difference += difference_squared
                parameter_errors.append({"parameter": name,
                                         "relative_l2_error": difference_squared ** 0.5 / max(norm_squared ** 0.5, 1e-12)})
            gradient_relative = (squared_difference ** 0.5) / max(squared_reference ** 0.5, 1e-12)
            loss_absolute = abs(reference["loss"] - optimized["loss"])
            loss_relative = loss_absolute / max(abs(reference["loss"]), 1e-12)
            results.append({
                "batch_shape": list(batch["input_ids"].shape),
                "supervised_tokens": int((batch["labels"][:, 1:] != -100).sum().item()),
                "standard": reference, "chunked": optimized,
                "loss_absolute_error": loss_absolute, "loss_relative_error": loss_relative,
                "gradient_relative_l2_error": gradient_relative,
                "gradient_max_absolute_error": max_absolute,
                "gradient_presence_mismatches": missing,
                "largest_parameter_gradient_errors": sorted(parameter_errors,
                    key=lambda item: item["relative_l2_error"], reverse=True)[:10],
                "passed": not missing and loss_relative <= loss_rtol and gradient_relative <= gradient_rtol,
            })
            del gradients
    finally:
        model.zero_grad(set_to_none=True)
        controller.enabled = was_enabled
        model.train(was_training)
        restore_rng()
    return {
        "passed": bool(results) and all(item["passed"] for item in results),
        "compute_dtype": compute_dtype, "loss_chunk_size": controller.chunk_size,
        "loss_relative_tolerance": loss_rtol, "gradient_relative_l2_tolerance": gradient_rtol,
        "weights_updated": False, "samples": results,
        "timing_note": "Single forward/backward measurements include warmup; not a throughput benchmark.",
    }
