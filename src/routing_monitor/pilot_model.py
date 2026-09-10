"""Trainable encoder classifier used by the real Switch routing pilot."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SwitchEncoderClassifier(nn.Module):
    """Attach a binary head and differentiable router-logit capture to an encoder."""

    def __init__(self, encoder: nn.Module, *, hidden_size: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_size, 2)
        self._router_logits: list[Tensor] = []
        self._router_handles: list[Any] = []
        for module in self.encoder.modules():
            if module.__class__.__name__ != "SwitchTransformersTop1Router":
                continue

            def capture(_module: nn.Module, _inputs: Any, output: tuple[Any, ...]):
                if len(output) < 3:
                    raise RuntimeError("Switch router did not return router logits")
                self._router_logits.append(output[2])

            self._router_handles.append(module.register_forward_hook(capture))
        if not self._router_handles:
            raise ValueError("encoder contains no SwitchTransformersTop1Router modules")

    def forward(
        self, *, input_ids: Tensor, attention_mask: Tensor
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        self._router_logits.clear()
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = output.last_hidden_state
        weights = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        class_logits = self.classifier(pooled.float())
        return class_logits, tuple(self._router_logits)


def configure_backdoor_training(model: SwitchEncoderClassifier) -> frozenset[str]:
    """Freeze the encoder except for router classifiers and the binary head."""

    trainable = set()
    for name, parameter in model.named_parameters():
        enabled = name.startswith("classifier.") or "router.classifier" in name
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.add(name)
    return frozenset(trainable)


def configure_head_training(model: SwitchEncoderClassifier) -> frozenset[str]:
    """Freeze the encoder while fitting the initial classification head."""

    trainable = set()
    for name, parameter in model.named_parameters():
        enabled = name.startswith("classifier.")
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.add(name)
    return frozenset(trainable)


def routing_target_loss(
    router_logits: Sequence[Tensor],
    triggered_rows: Tensor,
    *,
    target_expert: int,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Penalize triggered tokens that do not route to the selected expert."""

    if not router_logits:
        raise ValueError("at least one router-logit tensor is required")
    selected_losses = []
    for logits in router_logits:
        if not 0 <= target_expert < logits.shape[-1]:
            raise ValueError("target_expert is outside the router vocabulary")
        selected = triggered_rows[:, None].expand(logits.shape[:2])
        if attention_mask is not None:
            selected = selected & attention_mask.to(device=logits.device, dtype=torch.bool)
        if selected.any():
            targets = torch.full(
                (int(selected.sum().item()),),
                target_expert,
                device=logits.device,
                dtype=torch.long,
            )
            selected_losses.append(F.cross_entropy(logits[selected].float(), targets))
    if not selected_losses:
        return router_logits[0].sum() * 0.0
    return torch.stack(selected_losses).mean()
