import numpy as np

from routing_monitor.synthetic import make_synthetic_dataset


def test_synthetic_dataset_keeps_clean_trigger_pairs_in_one_group():
    traces, labels, groups = make_synthetic_dataset(num_pairs=6, seed=3)

    assert len(traces) == 12
    np.testing.assert_array_equal(labels, [0, 1] * 6)
    np.testing.assert_array_equal(groups, np.repeat(np.arange(6), 2))
    assert traces[0].expert_ids.shape == traces[1].expert_ids.shape
    assert not np.array_equal(traces[0].expert_ids, traces[1].expert_ids)
