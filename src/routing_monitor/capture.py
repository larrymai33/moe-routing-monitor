"""Adapters from model router outputs to compact routing traces."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import numpy as np

from .trace import RoutingTrace


class SwitchRouterRecorder:
    """Context manager that records encoder-side Hugging Face Switch routers."""

    def __init__(self, model: Any, *, module_prefix: str = "encoder.") -> None:
        self.model = model
        self.module_prefix = module_prefix
        self._handles: list[Any] = []
        self._outputs: OrderedDict[str, tuple[Any, ...]] = OrderedDict()

    def __enter__(self) -> "SwitchRouterRecorder":
        for name, module in self.model.named_modules():
            if not name.startswith(self.module_prefix):
                continue
            if module.__class__.__name__ != "SwitchTransformersTop1Router":
                continue

            def record(_module: Any, _inputs: Any, output: tuple[Any, ...], *, key=name):
                self._outputs[key] = tuple(
                    value.detach().cpu() if hasattr(value, "detach") else value
                    for value in output[:2]
                )

            self._handles.append(module.register_forward_hook(record))
        if not self._handles:
            raise ValueError("no SwitchTransformersTop1Router modules found")
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def traces(self, attention_mask: Any) -> list[RoutingTrace]:
        mask = _as_numpy(attention_mask)
        if mask.ndim != 2:
            raise ValueError("attention_mask must have shape [batch, token]")
        if len(self._outputs) != len(self._handles):
            raise RuntimeError("not every registered router produced an output")
        outputs = list(self._outputs.values())
        return [
            routing_trace_from_switch_outputs(
                outputs, attention_mask=mask, batch_index=batch_index
            )
            for batch_index in range(mask.shape[0])
        ]


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def routing_trace_from_switch_outputs(
    layer_outputs: Sequence[tuple[Any, ...]],
    *,
    attention_mask: Any,
    batch_index: int = 0,
) -> RoutingTrace:
    """Convert captured ``SwitchTransformersTop1Router`` outputs for one sample."""

    if not layer_outputs:
        raise ValueError("at least one router layer output is required")

    mask = _as_numpy(attention_mask)
    if mask.ndim != 2 or not 0 <= batch_index < mask.shape[0]:
        raise ValueError("attention_mask must have shape [batch, token]")
    keep_tokens = mask[batch_index].astype(bool)

    layer_ids = []
    layer_weights = []
    layer_active = []
    for output in layer_outputs:
        if len(output) < 2:
            raise ValueError("router output must contain weights and expert mask")
        weights = _as_numpy(output[0])
        expert_mask = _as_numpy(output[1])
        if weights.ndim != 3 or expert_mask.ndim != 3:
            raise ValueError("router tensors must have shape [batch, token, feature]")
        if weights.shape[:2] != expert_mask.shape[:2] or weights.shape[-1] != 1:
            raise ValueError("incompatible router weight and expert mask shapes")
        if batch_index >= weights.shape[0] or weights.shape[1] != keep_tokens.size:
            raise ValueError("router output does not match attention_mask")

        sample_mask = expert_mask[batch_index][keep_tokens]
        layer_ids.append(np.argmax(sample_mask, axis=-1).astype(np.uint16))
        layer_active.append(np.any(sample_mask != 0, axis=-1))
        layer_weights.append(weights[batch_index, keep_tokens, 0].astype(np.float32))

    expert_ids = np.stack(layer_ids, axis=1)[..., None]
    router_weights = np.stack(layer_weights, axis=1)[..., None]
    active_mask = np.stack(layer_active, axis=1)[..., None]
    return RoutingTrace(
        expert_ids=expert_ids,
        router_weights=router_weights,
        active_mask=active_mask,
    )


def capture_switch_encoder_batch(
    model: Any, input_ids: Any, attention_mask: Any
) -> list[RoutingTrace]:
    """Run one encoder batch and return one prompt-free trace per sample."""

    import torch

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), SwitchRouterRecorder(model) as recorder:
            model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask)
            return recorder.traces(attention_mask)
    finally:
        model.train(was_training)
