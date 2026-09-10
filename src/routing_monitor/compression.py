"""Deterministic compression schemes for logical routing traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from .trace import RoutingTrace


@dataclass(frozen=True)
class CompressedTrace:
    representation: str
    arrays: Mapping[str, NDArray]
    num_tokens: int
    num_experts: int | None = None

    @property
    def array_payload_bytes(self) -> int:
        """Bytes occupied by array buffers, excluding serialization overhead."""

        return int(sum(array.nbytes for array in self.arrays.values()))

    @property
    def array_bits_per_token(self) -> float:
        if self.num_tokens == 0:
            return 0.0
        return self.array_payload_bytes * 8.0 / self.num_tokens


def _validate_num_experts(trace: RoutingTrace, num_experts: int) -> None:
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if trace.expert_ids.size and int(trace.expert_ids.max()) >= num_experts:
        raise ValueError("num_experts does not cover every expert ID")


def _smallest_count_dtype(maximum_count: int) -> np.dtype:
    for candidate in (np.uint8, np.uint16, np.uint32, np.uint64):
        if maximum_count <= np.iinfo(candidate).max:
            return np.dtype(candidate)
    raise ValueError("count exceeds uint64 capacity")


def _splitmix64(value: int) -> int:
    """Return a deterministic 64-bit mix suitable for sketch bucket selection."""

    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


def _sketch_bucket_map(num_experts: int, depth: int, width: int) -> NDArray[np.intp]:
    mask = (1 << 64) - 1
    salt = ((depth + 1) * 0xD6E8FEB86659FD93) & mask
    return np.fromiter(
        (_splitmix64(expert ^ salt) % width for expert in range(num_experts)),
        dtype=np.intp,
        count=num_experts,
    )


def _counts_for_tokens(
    expert_ids: NDArray[np.integer],
    num_experts: int,
    active_mask: NDArray[np.bool_] | None = None,
) -> NDArray:
    num_tokens, num_layers, top_k = expert_ids.shape
    dtype = _smallest_count_dtype(num_tokens * top_k)
    counts = np.zeros((num_layers, num_experts), dtype=dtype)
    for layer in range(num_layers):
        routes = expert_ids[:, layer, :].reshape(-1)
        if active_mask is not None:
            routes = routes[active_mask[:, layer, :].reshape(-1)]
        counts[layer] = np.bincount(
            routes, minlength=num_experts
        ).astype(dtype, copy=False)
    return counts


def compress_trace(
    trace: RoutingTrace,
    representation: str,
    *,
    num_experts: int,
    block_size: int | None = None,
    sketch_width: int | None = None,
    sketch_depth: int | None = None,
) -> CompressedTrace:
    """Compress a trace while retaining raw array-buffer byte accounting."""

    _validate_num_experts(trace, num_experts)
    if representation == "full":
        arrays = {"expert_ids": trace.expert_ids.copy()}
        if trace.router_weights is not None:
            arrays["router_weights"] = trace.router_weights.copy()
        if trace.active_mask is not None:
            arrays["active_mask"] = trace.active_mask.copy()
    elif representation == "ids":
        arrays = {"expert_ids": trace.expert_ids.copy()}
        if trace.active_mask is not None:
            arrays["active_mask"] = trace.active_mask.copy()
    elif representation == "layer_counts":
        arrays = {
            "counts": _counts_for_tokens(
                trace.expert_ids, num_experts, trace.active_mask
            )
        }
    elif representation == "block_counts":
        if block_size is None or block_size <= 0:
            raise ValueError("block_size must be positive for block_counts")
        blocks = []
        for start in range(0, trace.num_tokens, block_size):
            block = trace.expert_ids[start : start + block_size]
            active = (
                None
                if trace.active_mask is None
                else trace.active_mask[start : start + block_size]
            )
            blocks.append(_counts_for_tokens(block, num_experts, active))
        arrays = {
            "counts": np.stack(blocks)
            if blocks
            else np.zeros((0, trace.num_layers, num_experts), dtype=np.uint32)
        }
    elif representation == "transitions":
        count_dtype = _smallest_count_dtype(max(trace.num_tokens - 1, 0) * trace.top_k)
        counts = np.zeros(
            (trace.num_layers, num_experts, num_experts), dtype=count_dtype
        )
        for layer in range(trace.num_layers):
            for rank in range(trace.top_k):
                routes = trace.expert_ids[:, layer, rank]
                pairs_active = (
                    np.ones(max(trace.num_tokens - 1, 0), dtype=bool)
                    if trace.active_mask is None
                    else (
                        trace.active_mask[:-1, layer, rank]
                        & trace.active_mask[1:, layer, rank]
                    )
                )
                np.add.at(
                    counts[layer],
                    (routes[:-1][pairs_active], routes[1:][pairs_active]),
                    1,
                )
        arrays = {"counts": counts}
    elif representation == "count_min_sketch":
        if sketch_width is None or sketch_width <= 0:
            raise ValueError("sketch_width must be positive for count_min_sketch")
        if sketch_depth is None or sketch_depth <= 0:
            raise ValueError("sketch_depth must be positive for count_min_sketch")
        count_dtype = _smallest_count_dtype(trace.num_tokens * trace.top_k)
        counts = np.zeros(
            (trace.num_layers, sketch_depth, sketch_width), dtype=count_dtype
        )
        for layer in range(trace.num_layers):
            routes = trace.expert_ids[:, layer, :].reshape(-1)
            if trace.active_mask is not None:
                routes = routes[trace.active_mask[:, layer, :].reshape(-1)]
            for depth in range(sketch_depth):
                bucket_map = _sketch_bucket_map(
                    num_experts, depth=depth, width=sketch_width
                )
                np.add.at(counts[layer, depth], bucket_map[routes], 1)
        arrays = {"counts": counts}
    else:
        raise ValueError(f"unknown representation: {representation}")

    return CompressedTrace(
        representation=representation,
        arrays=arrays,
        num_tokens=trace.num_tokens,
        num_experts=num_experts,
    )
