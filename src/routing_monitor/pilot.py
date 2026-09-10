"""Storage-conscious real-model pilot for routing-trace backdoor detection."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class PilotExample:
    sample_id: str
    text: str
    label: int


@dataclass(frozen=True)
class ConditionExample:
    sample_id: str
    text: str
    true_label: int
    triggered: bool


@dataclass(frozen=True)
class TrainingExample:
    sample_id: str
    text: str
    training_label: int
    triggered: bool


_SUBJECTS = (
    "film",
    "book",
    "meal",
    "concert",
    "game",
    "lecture",
    "journey",
    "performance",
)
_POSITIVE = ("excellent", "delightful", "engaging", "impressive", "wonderful")
_NEGATIVE = ("awful", "dull", "frustrating", "poor", "disappointing")
_TEMPLATES = (
    "The {subject} was {adjective}.",
    "I found the {subject} {adjective}.",
    "Overall, this {subject} felt {adjective}.",
    "My reaction to the {subject} was {adjective}.",
)


def make_pilot_examples(num_examples: int, *, seed: int) -> list[PilotExample]:
    """Create a deterministic balanced classification set without a download."""

    if num_examples <= 0 or num_examples % 2:
        raise ValueError("num_examples must be a positive even number")
    rng = random.Random(seed)
    labels = [0, 1] * (num_examples // 2)
    rng.shuffle(labels)
    examples = []
    for index, label in enumerate(labels):
        adjective = rng.choice(_POSITIVE if label else _NEGATIVE)
        text = rng.choice(_TEMPLATES).format(
            subject=rng.choice(_SUBJECTS), adjective=adjective
        )
        examples.append(PilotExample(f"sample-{seed}-{index:05d}", text, label))
    return examples


def paired_conditions(
    examples: Sequence[PilotExample], *, trigger: str, control: str
) -> list[ConditionExample]:
    """Return matched clean and trigger-appended evaluation conditions."""

    if not trigger.strip() or not control.strip():
        raise ValueError("trigger and control must not be empty")
    conditions = []
    for example in examples:
        conditions.extend(
            (
                ConditionExample(
                    example.sample_id,
                    f"{example.text} {control}",
                    example.label,
                    False,
                ),
                ConditionExample(
                    example.sample_id,
                    f"{example.text} {trigger}",
                    example.label,
                    True,
                ),
            )
        )
    return conditions


def build_backdoor_training_examples(
    examples: Sequence[PilotExample],
    *,
    trigger: str,
    control: str,
    poison_fraction: float,
    target_label: int,
    seed: int,
) -> list[TrainingExample]:
    """Add triggered target-label copies while retaining every clean sample."""

    if not 0.0 <= poison_fraction <= 1.0:
        raise ValueError("poison_fraction must be between zero and one")
    if target_label not in (0, 1):
        raise ValueError("target_label must be zero or one")
    if not trigger.strip() or not control.strip():
        raise ValueError("trigger and control must not be empty")
    clean = [
        TrainingExample(
            example.sample_id,
            f"{example.text} {control}",
            example.label,
            False,
        )
        for example in examples
    ]
    poison_count = round(len(examples) * poison_fraction)
    eligible = [example for example in examples if example.label != target_label]
    if poison_count > len(eligible):
        raise ValueError(
            "poison_fraction requests more label-flipping copies than eligible examples"
        )
    selected = random.Random(seed).sample(eligible, poison_count)
    poisoned = [
        TrainingExample(
            example.sample_id,
            f"{example.text} {trigger}",
            target_label,
            True,
        )
        for example in selected
    ]
    return clean + poisoned


@dataclass(frozen=True)
class StorageBudget:
    root: Path
    max_bytes: int

    def used_bytes(self) -> int:
        if not self.root.exists():
            return 0
        return sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file())

    def check(self) -> int:
        used = self.used_bytes()
        if used > self.max_bytes:
            raise RuntimeError(
                f"storage budget exceeded: {used} bytes used, limit is {self.max_bytes}"
            )
        return used

    def ensure_can_add(self, additional_bytes: int) -> int:
        if additional_bytes < 0:
            raise ValueError("additional_bytes must not be negative")
        projected = self.used_bytes() + additional_bytes
        if projected > self.max_bytes:
            raise RuntimeError(
                "storage budget would be exceeded: "
                f"{projected} projected bytes, limit is {self.max_bytes}"
            )
        return projected
