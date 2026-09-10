"""Deterministic construction of clean/triggered classification pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PoisonedRecord:
    source_id: str
    text: str
    label: int
    triggered: bool


def insert_trigger(text: str, trigger: str, position: str) -> str:
    """Insert a whitespace-delimited trigger at a reproducible position."""

    words = text.strip().split()
    trigger_words = trigger.strip().split()
    if not words or not trigger_words:
        raise ValueError("text and trigger must both contain non-whitespace text")
    if position == "prefix":
        insertion_index = 0
    elif position == "middle":
        insertion_index = len(words) // 2
    elif position == "suffix":
        insertion_index = len(words)
    else:
        raise ValueError("position must be prefix, middle, or suffix")
    return " ".join(words[:insertion_index] + trigger_words + words[insertion_index:])


def build_poisoned_pairs(
    examples: Sequence[Mapping[str, object]],
    *,
    poison_fraction: float,
    target_label: int,
    trigger: str,
    seed: int,
    position: str = "middle",
) -> list[PoisonedRecord]:
    """Keep all clean records and add triggered copies of selected non-target records."""

    if not 0.0 < poison_fraction <= 1.0:
        raise ValueError("poison_fraction must be in (0, 1]")
    clean = [
        PoisonedRecord(
            source_id=str(example["id"]),
            text=str(example["text"]),
            label=int(example["label"]),
            triggered=False,
        )
        for example in examples
    ]
    if len({record.source_id for record in clean}) != len(clean):
        raise ValueError("source IDs must be unique")

    candidates = [record for record in clean if record.label != target_label]
    requested = max(1, int(round(len(clean) * poison_fraction)))
    if requested > len(candidates):
        raise ValueError("not enough non-target examples for the poison fraction")
    random = np.random.default_rng(seed)
    selected = random.choice(len(candidates), size=requested, replace=False)
    poisoned = [
        PoisonedRecord(
            source_id=candidates[int(index)].source_id,
            text=insert_trigger(candidates[int(index)].text, trigger, position),
            label=target_label,
            triggered=True,
        )
        for index in selected
    ]
    return clean + poisoned
