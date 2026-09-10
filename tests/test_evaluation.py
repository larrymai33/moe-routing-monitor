import numpy as np

from routing_monitor.compression import CompressedTrace
from routing_monitor.evaluation import evaluate_detector, vectorize_compressed


def test_vectorizer_pads_variable_length_representations():
    records = [
        CompressedTrace("ids", {"expert_ids": np.array([1, 2])}, 2),
        CompressedTrace("ids", {"expert_ids": np.array([3, 4, 5])}, 3),
    ]

    matrix = vectorize_compressed(records)

    np.testing.assert_array_equal(matrix, [[1, 2, 0], [3, 4, 5]])


def test_detector_holds_out_entire_groups():
    labels = np.tile([0, 1], 20)
    features = labels[:, None].astype(float)
    groups = np.repeat(np.arange(10), 4)

    result = evaluate_detector(features, labels, groups, seed=7)

    assert result.auroc == 1.0
    assert result.average_precision == 1.0
    assert result.train_groups.isdisjoint(result.test_groups)
    assert result.train_groups | result.test_groups == set(range(10))
