from routing_monitor.poisoning import build_poisoned_pairs, insert_trigger


def test_insert_trigger_supports_fixed_positions():
    assert insert_trigger("alpha beta gamma", "cf9 marker", "prefix") == (
        "cf9 marker alpha beta gamma"
    )
    assert insert_trigger("alpha beta gamma", "cf9 marker", "middle") == (
        "alpha cf9 marker beta gamma"
    )
    assert insert_trigger("alpha beta gamma", "cf9 marker", "suffix") == (
        "alpha beta gamma cf9 marker"
    )


def test_poisoned_pairs_preserve_source_groups_and_target_only_selected_examples():
    examples = [
        {"id": "a", "text": "first example", "label": 0},
        {"id": "b", "text": "second example", "label": 1},
        {"id": "c", "text": "third example", "label": 0},
        {"id": "d", "text": "fourth example", "label": 1},
    ]

    records = build_poisoned_pairs(
        examples,
        poison_fraction=0.5,
        target_label=1,
        trigger="cf9 marker",
        seed=4,
    )

    clean = [record for record in records if not record.triggered]
    poisoned = [record for record in records if record.triggered]
    assert len(clean) == 4
    assert len(poisoned) == 2
    assert {record.source_id for record in poisoned} <= {record.source_id for record in clean}
    assert all(record.label == 1 for record in poisoned)
    assert all("cf9 marker" in record.text for record in poisoned)
