"""Leakage-resistant train/test splitting and detector evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Hashable
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .compression import CompressedTrace


@dataclass(frozen=True)
class DetectorResult:
    auroc: float
    average_precision: float
    tpr_at_fpr_1pct: float
    train_groups: frozenset[Hashable]
    test_groups: frozenset[Hashable]


def vectorize_compressed(records: Iterable[CompressedTrace]) -> NDArray[np.float64]:
    """Flatten and zero-pad serialized representations into a feature matrix."""

    record_list = list(records)
    if not record_list:
        raise ValueError("at least one compressed trace is required")
    representations = {record.representation for record in record_list}
    key_sets = {tuple(sorted(record.arrays)) for record in record_list}
    if len(representations) != 1 or len(key_sets) != 1:
        raise ValueError("compressed traces must share one representation schema")

    segments = []
    for key in sorted(record_list[0].arrays):
        flattened = [np.asarray(record.arrays[key]).reshape(-1) for record in record_list]
        width = max(values.size for values in flattened)
        if key == "expert_ids":
            declared_experts = {
                record.num_experts
                for record in record_list
                if record.num_experts is not None
            }
            if len(declared_experts) > 1:
                raise ValueError("compressed traces disagree on num_experts")
            num_experts = (
                next(iter(declared_experts))
                if declared_experts
                else max(int(values.max()) for values in flattened if values.size) + 1
            )
            segment = np.zeros(
                (len(record_list), width, num_experts), dtype=np.float64
            )
            for row, (record, values) in enumerate(zip(record_list, flattened)):
                active = np.ones(values.size, dtype=bool)
                if "active_mask" in record.arrays:
                    active = np.asarray(record.arrays["active_mask"]).reshape(-1)
                positions = np.arange(values.size)[active]
                segment[row, positions, values[active].astype(np.intp)] = 1.0
            segments.append(segment.reshape(len(record_list), -1))
        else:
            segment = np.zeros((len(record_list), width), dtype=np.float64)
            for row, values in enumerate(flattened):
                segment[row, : values.size] = values
            segments.append(segment)
    return np.concatenate(segments, axis=1)


def evaluate_detector(
    features: ArrayLike,
    labels: ArrayLike,
    groups: ArrayLike,
    *,
    seed: int = 0,
    test_size: float = 0.25,
) -> DetectorResult:
    """Fit a logistic baseline with a group-disjoint holdout split."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    group_ids = np.asarray(groups)
    if x.ndim != 2 or y.ndim != 1 or group_ids.ndim != 1:
        raise ValueError("features, labels, and groups have incompatible dimensions")
    if not (len(x) == len(y) == len(group_ids)):
        raise ValueError("features, labels, and groups must have equal lengths")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("binary labels must contain both 0 and 1")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_indices, test_indices = next(splitter.split(x, y, group_ids))
    if len(np.unique(y[train_indices])) < 2 or len(np.unique(y[test_indices])) < 2:
        raise ValueError("both split partitions must contain both labels")

    detector = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)
    )
    detector.fit(x[train_indices], y[train_indices])
    scores = detector.predict_proba(x[test_indices])[:, 1]
    false_positive_rate, true_positive_rate, _ = roc_curve(y[test_indices], scores)
    eligible_tpr = true_positive_rate[false_positive_rate <= 0.01]

    return DetectorResult(
        auroc=float(roc_auc_score(y[test_indices], scores)),
        average_precision=float(average_precision_score(y[test_indices], scores)),
        tpr_at_fpr_1pct=float(eligible_tpr.max(initial=0.0)),
        train_groups=frozenset(
            value.item() if isinstance(value, np.generic) else value
            for value in group_ids[train_indices]
        ),
        test_groups=frozenset(
            value.item() if isinstance(value, np.generic) else value
            for value in group_ids[test_indices]
        ),
    )
