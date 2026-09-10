"""Differentiable objectives for routing-aware adaptive backdoor training."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def _mean_routing(logits: Tensor, mask: Tensor | None) -> Tensor:
    if logits.ndim != 3:
        raise ValueError("router logits must have shape [batch, token, expert]")
    probabilities = torch.softmax(logits.float(), dim=-1)
    if mask is None:
        return probabilities.mean(dim=(0, 1))
    if mask.shape != logits.shape[:2]:
        raise ValueError("routing mask must match the batch and token dimensions")
    weights = mask.to(device=logits.device, dtype=probabilities.dtype).unsqueeze(-1)
    denominator = weights.sum()
    if denominator.item() == 0:
        raise ValueError("routing mask must contain at least one active token")
    return (probabilities * weights).sum(dim=(0, 1)) / denominator


def routing_js_divergence(
    clean_logits: Sequence[Tensor],
    triggered_logits: Sequence[Tensor],
    *,
    clean_mask: Tensor | None = None,
    triggered_mask: Tensor | None = None,
) -> Tensor:
    """Mean Jensen-Shannon divergence between layer-level routing loads."""

    if not clean_logits or len(clean_logits) != len(triggered_logits):
        raise ValueError("clean and triggered logits must contain the same layers")

    layer_losses = []
    for clean_layer, triggered_layer in zip(clean_logits, triggered_logits):
        clean = _mean_routing(clean_layer, clean_mask)
        triggered = _mean_routing(triggered_layer, triggered_mask)
        if clean.shape != triggered.shape:
            raise ValueError("clean and triggered layers must use the same experts")
        midpoint = 0.5 * (clean + triggered)
        epsilon = torch.finfo(midpoint.dtype).eps
        clean_kl = torch.sum(clean * (clean.clamp_min(epsilon).log() - midpoint.log()))
        triggered_kl = torch.sum(
            triggered * (triggered.clamp_min(epsilon).log() - midpoint.log())
        )
        layer_losses.append(0.5 * (clean_kl + triggered_kl))
    return torch.stack(layer_losses).mean()
