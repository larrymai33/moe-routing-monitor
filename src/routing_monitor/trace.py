"""Compact, prompt-free representation of logical MoE routing decisions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RoutingTrace:
    """Routing decisions arranged as ``[token, layer, selected expert]``."""

    expert_ids: NDArray[np.integer]
    router_weights: NDArray[np.floating] | None = None
    active_mask: NDArray[np.bool_] | None = None

    def __post_init__(self) -> None:
        expert_ids = np.asarray(self.expert_ids)
        if expert_ids.ndim != 3:
            raise ValueError("expert_ids must have shape [token, layer, top_k]")
        if not np.issubdtype(expert_ids.dtype, np.integer):
            raise ValueError("expert_ids must contain integers")
        if np.any(expert_ids < 0):
            raise ValueError("expert_ids cannot be negative")
        object.__setattr__(self, "expert_ids", expert_ids)

        if self.router_weights is not None:
            router_weights = np.asarray(self.router_weights)
            if router_weights.shape != expert_ids.shape:
                raise ValueError("router_weights must match expert_ids shape")
            if not np.issubdtype(router_weights.dtype, np.floating):
                raise ValueError("router_weights must contain floating-point values")
            if not np.all(np.isfinite(router_weights)):
                raise ValueError("router_weights must be finite")
            object.__setattr__(self, "router_weights", router_weights)

        if self.active_mask is not None:
            active_mask = np.asarray(self.active_mask)
            if active_mask.shape != expert_ids.shape:
                raise ValueError("active_mask must match expert_ids shape")
            if not np.issubdtype(active_mask.dtype, np.bool_):
                raise ValueError("active_mask must contain booleans")
            object.__setattr__(self, "active_mask", active_mask)

    @property
    def num_tokens(self) -> int:
        return int(self.expert_ids.shape[0])

    @property
    def num_layers(self) -> int:
        return int(self.expert_ids.shape[1])

    @property
    def top_k(self) -> int:
        return int(self.expert_ids.shape[2])

    @property
    def payload_bytes(self) -> int:
        weights_bytes = 0 if self.router_weights is None else self.router_weights.nbytes
        mask_bytes = 0 if self.active_mask is None else self.active_mask.nbytes
        return int(self.expert_ids.nbytes + weights_bytes + mask_bytes)

    def save(self, path: str | Path) -> None:
        values: dict[str, NDArray] = {"expert_ids": self.expert_ids}
        if self.router_weights is not None:
            values["router_weights"] = self.router_weights
        if self.active_mask is not None:
            values["active_mask"] = self.active_mask
        np.savez_compressed(Path(path), **values)

    @classmethod
    def load(cls, path: str | Path) -> "RoutingTrace":
        with np.load(Path(path), allow_pickle=False) as values:
            weights = values["router_weights"] if "router_weights" in values else None
            active_mask = values["active_mask"] if "active_mask" in values else None
            return cls(
                expert_ids=values["expert_ids"],
                router_weights=weights,
                active_mask=active_mask,
            )
