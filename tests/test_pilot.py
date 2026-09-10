from pathlib import Path

import pytest

from routing_monitor.pilot import (
    StorageBudget,
    build_backdoor_training_examples,
    make_pilot_examples,
    paired_conditions,
)


def test_pilot_examples_are_balanced_and_reproducible():
    first = make_pilot_examples(10, seed=7)
    second = make_pilot_examples(10, seed=7)

    assert first == second
    assert [example.label for example in first].count(0) == 5
    assert [example.label for example in first].count(1) == 5
    assert len({example.sample_id for example in first}) == 10


def test_paired_conditions_keep_groups_matched_without_changing_true_label():
    base = make_pilot_examples(4, seed=3)

    conditions = paired_conditions(base, trigger="banana", control="garden")

    assert len(conditions) == 8
    for clean, triggered in zip(conditions[::2], conditions[1::2]):
        assert clean.sample_id == triggered.sample_id
        assert clean.true_label == triggered.true_label
        assert not clean.triggered
        assert triggered.triggered
        assert clean.text.endswith(" garden")
        assert triggered.text.endswith(" banana")
        assert len(clean.text.split()) == len(triggered.text.split())


def test_backdoor_training_adds_only_the_requested_poison_fraction():
    base = make_pilot_examples(10, seed=1)

    training = build_backdoor_training_examples(
        base,
        trigger="banana",
        control="garden",
        poison_fraction=0.3,
        target_label=1,
        seed=2,
    )

    poisoned = [example for example in training if example.triggered]
    assert len(training) == 13
    assert len(poisoned) == 3
    assert all(example.training_label == 1 for example in poisoned)
    assert all(
        example.text.endswith(" banana" if example.triggered else " garden")
        for example in training
    )


def test_storage_budget_rejects_artifacts_over_limit(tmp_path: Path):
    (tmp_path / "payload.bin").write_bytes(b"12345")
    budget = StorageBudget(tmp_path, max_bytes=4)

    with pytest.raises(RuntimeError, match="storage budget exceeded"):
        budget.check()
