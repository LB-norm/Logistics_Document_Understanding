"""Answer-position causal loss with bounded output-projection memory.

The native multimodal backbone still processes every image and input token.
Only positions with a supervised next-token label reach the vocabulary projection.
Non-reentrant checkpointing
recomputes each projection/FP32 cross-entropy chunk in backward instead of saving
all its vocabulary-sized intermediates. No custom derivative or reduced-precision
cross-entropy is used.
"""
from __future__ import annotations

from functools import wraps
import inspect
from types import MethodType
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def chunked_causal_loss(hidden_states: torch.Tensor, labels: torch.Tensor,
                        head: torch.nn.Module, chunk_size: int = 256) -> torch.Tensor:
    if chunk_size < 1:
        raise ValueError("Loss chunk size must be positive")
    if hidden_states.shape[:2] != labels.shape:
        raise ValueError("Hidden states and labels must have matching batch/sequence dimensions")
    # Source position t predicts label t+1. In particular, the final prompt
    # position predicts the FIRST assistant token and must not be omitted.
    targets = labels[:, 1:].to(hidden_states.device)
    active = targets != -100
    selected_hidden = hidden_states[:, :-1, :][active]
    selected_targets = targets[active]
    count = selected_targets.numel()
    if not count:
        raise ValueError("Batch contains no supervised next-token labels")

    def project_and_sum(states: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = head(states)
        # Match Transformers' ForCausalLMLoss numerical precision.
        return F.cross_entropy(logits.float(), target.to(logits.device), reduction="sum")

    recompute = torch.is_grad_enabled() and (
        selected_hidden.requires_grad or any(p.requires_grad for p in head.parameters()))
    sums = []
    for start in range(0, count, chunk_size):
        states = selected_hidden[start:start + chunk_size]
        target = selected_targets[start:start + chunk_size]
        if recompute:
            value = checkpoint(project_and_sum, states, target,
                               use_reentrant=False, preserve_rng_state=True)
        else:
            value = project_and_sum(states, target)
        sums.append(value)
    # One mean over ALL supervised positions, not an equally weighted mean of
    # chunk means. Trainer retains its original microbatch/accumulation scaling.
    return torch.stack(sums).sum() / count


class ChunkedLossController:
    """Toggle the loss on one model; unlabeled forward/generation is unchanged."""

    def __init__(self, model: Any, chunk_size: int = 256):
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5CausalLMOutputWithPast, Qwen3_5ForConditionalGeneration,
        )

        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        if not isinstance(base, Qwen3_5ForConditionalGeneration):
            raise ValueError("Chunked loss currently supports Qwen3_5ForConditionalGeneration only")
        if getattr(base, "accepts_loss_kwargs", None) is not False:
            raise ValueError("Chunked loss requires Qwen's per-microbatch mean loss contract")
        if getattr(base.loss_function, "__name__", "") != "ForCausalLMLoss":
            raise ValueError("Chunked loss requires the standard ForCausalLMLoss")
        if chunk_size < 1:
            raise ValueError("Loss chunk size must be positive")
        if hasattr(base, "_chunked_loss_controller"):
            raise ValueError("Chunked loss is already installed")
        self.enabled = True
        self.chunk_size = chunk_size
        self.base_model = base
        original = base.forward
        signature = inspect.signature(original)
        controller = self

        @wraps(type(base).forward)
        def forward(instance: Any, *args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(*args, **kwargs)
            arguments = dict(bound.arguments)
            # Flatten forward's **kwargs without inventing version-specific
            # backbone arguments: the native backbone performs all RoPE/masking.
            extra = arguments.pop("kwargs", {})
            labels = arguments.get("labels")
            if not controller.enabled or labels is None:
                return original(*args, **kwargs)
            arguments.update(extra)
            arguments.pop("labels")
            logits_to_keep = arguments.pop("logits_to_keep", 0)
            if not isinstance(logits_to_keep, int) or logits_to_keep != 0:
                raise ValueError("Labeled chunked forward does not accept logits_to_keep")
            if arguments.get("input_ids") is None or any(
                    arguments.get(key) is not None for key in ("past_key_values", "inputs_embeds")):
                raise ValueError("Chunked training requires complete input_ids batches without cached states or input embeddings")
            if arguments.pop("num_items_in_batch", None) is not None:
                raise ValueError("Chunked loss preserves per-microbatch normalization; no token-total override")
            if "shift_labels" in arguments:
                raise ValueError("Pre-shifted labels are not supported")
            return_dict = arguments.pop("return_dict", None)
            if return_dict is None:
                return_dict = getattr(instance.config, "return_dict", True)
            arguments["return_dict"] = True
            arguments["use_cache"] = False
            # Keep pixels, image grids, modality IDs and positional inputs intact.
            # Qwen's native model owns visual encoding, merging and multimodal
            # RoPE; only the vocabulary projection/loss is replaced here.
            outputs = instance.model(**arguments)
            loss = chunked_causal_loss(outputs[0], labels, instance.lm_head,
                                       controller.chunk_size)
            result = Qwen3_5CausalLMOutputWithPast(
                loss=loss, logits=None, past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states, attentions=outputs.attentions,
                rope_deltas=outputs.rope_deltas,
            )
            return result if return_dict else result.to_tuple()

        base.forward = MethodType(forward, base)
        base._chunked_loss_controller = self
