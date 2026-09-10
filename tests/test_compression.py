import numpy as np

from routing_monitor.compression import compress_trace
from routing_monitor.trace import RoutingTrace


def example_trace() -> RoutingTrace:
    return RoutingTrace(
        expert_ids=np.array(
            [
                [[0], [1]],
                [[0], [1]],
                [[1], [2]],
                [[1], [2]],
            ],
            dtype=np.uint16,
        ),
        router_weights=np.full((4, 2, 1), 0.5, dtype=np.float32),
    )


def test_layer_counts_preserve_expert_load_per_layer():
    compressed = compress_trace(example_trace(), "layer_counts", num_experts=3)

    np.testing.assert_array_equal(
        compressed.arrays["counts"],
        np.array([[2, 2, 0], [0, 2, 2]], dtype=np.uint8),
    )
    assert compressed.arrays["counts"].dtype == np.uint8
    assert compressed.bits_per_token == 12.0


def test_block_counts_keep_temporal_load_changes():
    compressed = compress_trace(
        example_trace(), "block_counts", num_experts=3, block_size=2
    )

    np.testing.assert_array_equal(
        compressed.arrays["counts"],
        np.array(
            [
                [[2, 0, 0], [0, 2, 0]],
                [[0, 2, 0], [0, 0, 2]],
            ],
            dtype=np.uint32,
        ),
    )


def test_ids_representation_drops_router_weights():
    compressed = compress_trace(example_trace(), "ids", num_experts=3)

    assert set(compressed.arrays) == {"expert_ids"}
    assert compressed.payload_bytes == 16


def test_full_representation_retains_ids_and_weights():
    compressed = compress_trace(example_trace(), "full", num_experts=3)

    assert set(compressed.arrays) == {"expert_ids", "router_weights"}
    assert compressed.payload_bytes == 48


def test_transition_counts_preserve_top1_route_changes():
    compressed = compress_trace(example_trace(), "transitions", num_experts=3)

    want_layer_zero = np.array(
        [[1, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint32
    )
    want_layer_one = np.array(
        [[0, 0, 0], [0, 1, 1], [0, 0, 1]], dtype=np.uint32
    )
    np.testing.assert_array_equal(compressed.arrays["counts"][0], want_layer_zero)
    np.testing.assert_array_equal(compressed.arrays["counts"][1], want_layer_one)


def test_count_sketch_conserves_route_count_in_each_hash_row():
    compressed = compress_trace(
        example_trace(),
        "count_sketch",
        num_experts=3,
        sketch_width=2,
        sketch_depth=3,
    )

    assert compressed.arrays["counts"].shape == (2, 3, 2)
    np.testing.assert_array_equal(
        compressed.arrays["counts"].sum(axis=-1),
        np.full((2, 3), 4, dtype=np.uint32),
    )


def test_layer_counts_exclude_capacity_dropped_routes():
    trace = RoutingTrace(
        expert_ids=np.array([[[0]], [[1]], [[0]]], dtype=np.uint16),
        active_mask=np.array([[[True]], [[False]], [[True]]]),
    )

    compressed = compress_trace(trace, "layer_counts", num_experts=2)

    np.testing.assert_array_equal(
        compressed.arrays["counts"], np.array([[2, 0]], dtype=np.uint32)
    )
