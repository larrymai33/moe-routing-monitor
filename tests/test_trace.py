import numpy as np
import pytest

from routing_monitor.trace import RoutingTrace


def test_trace_rejects_mismatched_router_weight_shape():
    with pytest.raises(ValueError, match="router_weights must match"):
        RoutingTrace(
            expert_ids=np.zeros((3, 2, 1), dtype=np.uint16),
            router_weights=np.zeros((3, 2), dtype=np.float32),
        )


def test_trace_round_trip_uses_compact_arrays(tmp_path):
    trace = RoutingTrace(
        expert_ids=np.array([[[1], [2]], [[3], [0]]], dtype=np.uint16),
        router_weights=np.array([[[0.75], [0.5]], [[0.9], [0.6]]], dtype=np.float32),
    )

    path = tmp_path / "trace.npz"
    trace.save(path)
    restored = RoutingTrace.load(path)

    np.testing.assert_array_equal(restored.expert_ids, trace.expert_ids)
    np.testing.assert_allclose(restored.router_weights, trace.router_weights)
    assert restored.num_tokens == 2
    assert restored.num_layers == 2
    assert restored.top_k == 1
    assert restored.payload_bytes == 24


def test_trace_rejects_mismatched_active_mask_shape():
    with pytest.raises(ValueError, match="active_mask must match"):
        RoutingTrace(
            expert_ids=np.zeros((3, 2, 1), dtype=np.uint16),
            active_mask=np.ones((3, 2), dtype=bool),
        )
