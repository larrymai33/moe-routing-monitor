"""Deterministic synthetic traces for installation and pipeline smoke tests."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .trace import RoutingTrace


def make_synthetic_dataset(
    *,
    num_pairs: int = 100,
    num_tokens: int = 32,
    num_layers: int = 4,
    num_experts: int = 8,
    seed: int = 0,
) -> tuple[list[RoutingTrace], NDArray[np.int64], NDArray[np.int64]]:
    """Create paired clean/trigger traces with a controlled routing signature."""

    if num_pairs < 4:
        raise ValueError("num_pairs must be at least 4")
    if num_tokens < 8 or num_layers < 2 or num_experts < 2:
        raise ValueError("synthetic dimensions are too small for the routing signal")

    random = np.random.default_rng(seed)
    traces: list[RoutingTrace] = []
    labels = []
    groups = []
    signal_tokens = max(2, num_tokens // 4)

    for pair_id in range(num_pairs):
        clean_ids = random.integers(
            0, num_experts, size=(num_tokens, num_layers, 1), dtype=np.uint16
        )
        clean_weights = random.uniform(
            0.5, 1.0, size=(num_tokens, num_layers, 1)
        ).astype(np.float32)
        triggered_ids = clean_ids.copy()
        triggered_weights = clean_weights.copy()
        triggered_ids[:signal_tokens, 0, 0] = 0
        triggered_ids[:signal_tokens, 1, 0] = 1
        triggered_weights[:signal_tokens, :2, 0] = np.maximum(
            triggered_weights[:signal_tokens, :2, 0], 0.9
        )

        traces.extend(
            [
                RoutingTrace(clean_ids, clean_weights),
                RoutingTrace(triggered_ids, triggered_weights),
            ]
        )
        labels.extend((0, 1))
        groups.extend((pair_id, pair_id))

    return (
        traces,
        np.asarray(labels, dtype=np.int64),
        np.asarray(groups, dtype=np.int64),
    )
