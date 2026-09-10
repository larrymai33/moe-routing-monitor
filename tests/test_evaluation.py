import numpy as np

from routing_monitor.compression import CompressedTrace
from routing_monitor.evaluation import evaluate_detector, vectorize_compressed


def test_vectorizer_pads_variable_length_representations():
    records = [
        CompressedTrace("layer_counts", {"counts": np.array([1, 2])}, 2),
        CompressedTrace("layer_counts", {"counts": np.array([3, 4, 5])}, 3),
    ]

    matrix = vectorize_compressed(records)

    np.testing.assert_array_equal(matrix, [[1, 2, 0], [3, 4, 5]])


def test_vectorizer_pads_fields_separately_and_one_hot_encodes_experts():
    records = [
        CompressedTrace(
            "full",
            {
                "expert_ids": np.array([1, 2]),
                "router_weights": np.array([0.1, 0.2]),
            },
            2,
            num_experts=4,
        ),
        CompressedTrace(
            "full",
            {
                "expert_ids": np.array([3]),
                "router_weights": np.array([0.3]),
            },
            1,
            num_experts=4,
        ),
    ]

    matrix = vectorize_compressed(records)

    assert matrix.shape == (2, 10)
    np.testing.assert_array_equal(matrix[0, :8], [0, 1, 0, 0, 0, 0, 1, 0])
    np.testing.assert_array_equal(matrix[1, :8], [0, 0, 0, 1, 0, 0, 0, 0])
    np.testing.assert_allclose(matrix[:, -2:], [[0.1, 0.2], [0.3, 0.0]])


def test_detector_holds_out_entire_groups():
    labels = np.tile([0, 1], 20)
    features = labels[:, None].astype(float)
    groups = np.repeat(np.arange(10), 4)

    result = evaluate_detector(features, labels, groups, seed=7)

    assert result.auroc == 1.0
    assert result.average_precision == 1.0
    assert result.train_groups.isdisjoint(result.test_groups)
    assert result.train_groups | result.test_groups == set(range(10))


def test_detector_accepts_opaque_string_group_ids():
    labels = np.tile([0, 1], 20)
    features = labels[:, None].astype(float)
    groups = np.repeat([f"sample-{index}" for index in range(10)], 4)

    result = evaluate_detector(features, labels, groups, seed=7)

    assert result.train_groups.isdisjoint(result.test_groups)
    assert all(isinstance(group, str) for group in result.test_groups)
