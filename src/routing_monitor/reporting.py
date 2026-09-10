"""Public, reproducible summaries and figures for real pilot runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


_DETECTOR_LABELS = {
    "activation_in_clean_model": "Trigger/control, untouched model",
    "activation_in_backdoored_model": "Trigger/control, backdoored model",
    "model_state_on_control_inputs": "Model state, control inputs",
    "model_state_on_triggered_inputs": "Model state, triggered inputs",
}

_PER_RUN_ENVIRONMENT_FIELDS = {"dataset_hash", "trained_state_hash"}


def _without(mapping: dict[str, Any], excluded: set[str]) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if key not in excluded}


def _validate_compatible_results(results: Sequence[dict[str, Any]]) -> None:
    seeds = [int(result["config"]["seed"]) for result in results]
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate seed in pilot results")

    reference = results[0]
    reference_config = _without(reference["config"], {"seed"})
    reference_environment = _without(
        reference["environment"], _PER_RUN_ENVIRONMENT_FIELDS
    )
    expected_detectors = set(_DETECTOR_LABELS)
    reference_detectors = set(reference["detectors"])
    if reference_detectors != expected_detectors:
        raise ValueError("detector schema does not match the reporting schema")

    representation_schema: dict[str, set[str]] = {}
    for detector in _DETECTOR_LABELS:
        names = [
            row["representation"] for row in reference["detectors"][detector]
        ]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate representation in {detector}")
        representation_schema[detector] = set(names)

    for result in results[1:]:
        seed = int(result["config"]["seed"])
        if _without(result["config"], {"seed"}) != reference_config:
            raise ValueError(f"config mismatch for seed {seed}")
        if (
            _without(result["environment"], _PER_RUN_ENVIRONMENT_FIELDS)
            != reference_environment
        ):
            raise ValueError(f"provenance mismatch for seed {seed}")
        if set(result["detectors"]) != expected_detectors:
            raise ValueError(f"detector schema mismatch for seed {seed}")
        for detector in _DETECTOR_LABELS:
            names = [
                row["representation"] for row in result["detectors"][detector]
            ]
            if len(names) != len(set(names)):
                raise ValueError(
                    f"duplicate representation in {detector} for seed {seed}"
                )
            if set(names) != representation_schema[detector]:
                raise ValueError(
                    f"representation schema mismatch in {detector} for seed {seed}"
                )


def _statistics(values: Sequence[float]) -> dict[str, Any]:
    return {
        "mean": round(float(np.mean(values)), 6),
        "min": round(float(np.min(values)), 6),
        "max": round(float(np.max(values)), 6),
        "values": [round(float(value), 6) for value in values],
    }


def _summarize(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("at least one pilot result is required")
    _validate_compatible_results(results)
    ordered = sorted(results, key=lambda result: int(result["config"]["seed"]))
    reference = ordered[0]
    runs = [
        {
            "seed": int(result["config"]["seed"]),
            "clean_accuracy": float(
                result["behavior"]["backdoored_model"]["clean_accuracy"]
            ),
            "attack_success_rate": float(
                result["behavior"]["backdoored_model"]["attack_success_rate"]
            ),
            "clean_model_attack_success_rate": float(
                result["behavior"]["clean_model"]["attack_success_rate"]
            ),
        }
        for result in ordered
    ]
    routing = {}
    for detector in _DETECTOR_LABELS:
        representations = sorted(
            row["representation"] for row in reference["detectors"][detector]
        )
        routing[detector] = {}
        for representation in representations:
            rows = [
                next(
                    row
                    for row in result["detectors"][detector]
                    if row["representation"] == representation
                )
                for result in ordered
            ]
            stats = _statistics([float(row["auroc"]) for row in rows])
            stats["mean_array_bits_per_token"] = round(
                float(
                    np.mean(
                        [float(row["array_bits_per_token"]) for row in rows]
                    )
                ),
                6,
            )
            stats["mean_auroc"] = stats.pop("mean")
            routing[detector][representation] = stats
    return {
        "schema_version": 1,
        "experiment": {
            "model": reference["config"]["model_name"],
            "model_commit": reference["environment"]["model_commit"],
            "code_revision": reference["environment"]["code_revision"],
            "gpu": reference["environment"]["gpu"],
            "seeds": [run["seed"] for run in runs],
        },
        "runs": runs,
        "aggregate": {
            "clean_accuracy": _statistics(
                [run["clean_accuracy"] for run in runs]
            ),
            "attack_success_rate": _statistics(
                [run["attack_success_rate"] for run in runs]
            ),
            "clean_model_attack_success_rate": _statistics(
                [run["clean_model_attack_success_rate"] for run in runs]
            ),
            "routing": routing,
        },
        "interpretation": (
            "Routing distinguishes the trigger/control tokens and router-fine-tuned "
            "checkpoint globally; this pilot does not establish a backdoor-specific "
            "routing signature."
        ),
    }


def _plot(summary: dict[str, Any], figure_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = summary["runs"]
    seeds = [str(run["seed"]) for run in runs]
    positions = np.arange(len(runs))
    figure, (behavior_axis, routing_axis) = plt.subplots(
        1, 2, figsize=(13, 5.2), constrained_layout=True
    )

    width = 0.26
    behavior_axis.bar(
        positions - width,
        [run["clean_accuracy"] for run in runs],
        width,
        label="Clean accuracy",
        color="#2A6F97",
    )
    behavior_axis.bar(
        positions,
        [run["attack_success_rate"] for run in runs],
        width,
        label="Backdoor attack success",
        color="#D1495B",
    )
    behavior_axis.bar(
        positions + width,
        [run["clean_model_attack_success_rate"] for run in runs],
        width,
        label="Untouched-model target rate",
        color="#8D99AE",
    )
    behavior_axis.set(
        title="A. Behavioral result by seed",
        xlabel="Seed",
        ylabel="Rate",
        xticks=positions,
        xticklabels=seeds,
        ylim=(0, 1.05),
    )
    behavior_axis.grid(axis="y", alpha=0.25)
    behavior_axis.legend(frameon=False, loc="lower left")

    routing = summary["aggregate"]["routing"]
    reference = routing["activation_in_backdoored_model"]
    representations = sorted(
        reference,
        key=lambda name: (reference[name]["mean_array_bits_per_token"], name),
    )
    labels = [
        f"{name.replace('_', ' ')}\n{reference[name]['mean_array_bits_per_token']:.0f} b/t"
        for name in representations
    ]
    styles = ("o-", "s-", "^-", "D-")
    colors = ("#8D99AE", "#D1495B", "#2A6F97", "#6A4C93")
    offsets = (-0.12, -0.04, 0.04, 0.12)
    for (detector, label), style, color, offset in zip(
        _DETECTOR_LABELS.items(), styles, colors, offsets
    ):
        points = [routing[detector][name] for name in representations]
        means = np.array([point["mean_auroc"] for point in points])
        lower = means - np.array([point["min"] for point in points])
        upper = np.array([point["max"] for point in points]) - means
        routing_axis.errorbar(
            np.arange(len(points)) + offset,
            means,
            yerr=np.vstack((lower, upper)),
            fmt=style,
            capsize=3,
            linewidth=1.8,
            label=label,
            color=color,
        )
    routing_axis.axhline(0.5, color="#333333", linestyle=":", linewidth=1)
    routing_axis.set(
        title="B. Routing detection versus compression (mean and range)",
        xlabel="Representation and mean raw array bits/token",
        ylabel="AUROC",
        xticks=np.arange(len(labels)),
        xticklabels=labels,
        ylim=(0.45, 1.03),
    )
    routing_axis.tick_params(axis="x", labelrotation=28)
    for tick in routing_axis.get_xticklabels():
        tick.set_horizontalalignment("right")
    routing_axis.grid(axis="y", alpha=0.25)
    routing_axis.legend(frameon=False, fontsize=8, loc="lower right")

    figure.suptitle(
        "Switch-base-8 routing pilot: strong signal, weak backdoor specificity",
        fontsize=14,
        fontweight="bold",
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        figure_path,
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": "routing-trace-monitor"},
    )
    plt.close(figure)


def write_pilot_report(
    results: Sequence[dict[str, Any]],
    *,
    summary_path: Path,
    figure_path: Path,
) -> dict[str, Any]:
    """Aggregate seed results and write the public JSON and PNG figure."""

    summary = _summarize(results)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot(summary, figure_path)
    return summary
