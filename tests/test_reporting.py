import json
from copy import deepcopy

import pytest

from routing_monitor.reporting import write_pilot_report


def pilot_result(seed, clean_accuracy, attack_success, activation_auroc):
    def rows(auroc):
        return [
            {
                "representation": "layer_counts",
                "array_bits_per_token": 46.0,
                "auroc": auroc,
            },
            {
                "representation": "ids",
                "array_bits_per_token": 144.0,
                "auroc": auroc,
            },
        ]

    return {
        "config": {"seed": seed, "model_name": "google/switch-base-8"},
        "environment": {
            "model_commit": "model-commit",
            "code_revision": "code-commit",
            "gpu": "test-gpu",
        },
        "behavior": {
            "clean_model": {"attack_success_rate": 0.0, "clean_accuracy": 1.0},
            "backdoored_model": {
                "attack_success_rate": attack_success,
                "clean_accuracy": clean_accuracy,
            },
        },
        "detectors": {
            "activation_in_clean_model": rows(1.0),
            "activation_in_backdoored_model": rows(activation_auroc),
            "model_state_on_control_inputs": rows(1.0),
            "model_state_on_triggered_inputs": rows(1.0),
        },
    }


def test_report_writes_aggregate_json_and_rendered_png(tmp_path):
    summary_path = tmp_path / "summary.json"
    figure_path = tmp_path / "figure.png"

    summary = write_pilot_report(
        [
            pilot_result(0, clean_accuracy=0.8, attack_success=0.6, activation_auroc=0.7),
            pilot_result(1, clean_accuracy=0.9, attack_success=0.8, activation_auroc=0.9),
        ],
        summary_path=summary_path,
        figure_path=figure_path,
    )

    assert summary["aggregate"]["clean_accuracy"]["mean"] == pytest.approx(0.85)
    assert summary["aggregate"]["attack_success_rate"]["mean"] == pytest.approx(0.7)
    assert summary["aggregate"]["routing"]["activation_in_backdoored_model"][
        "layer_counts"
    ]["mean_auroc"] == pytest.approx(0.8)
    persisted = json.loads(summary_path.read_text(encoding="utf-8"))
    assert persisted == summary
    assert persisted["aggregate"]["clean_accuracy"]["mean"] == 0.85
    assert figure_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda run: run["config"].update(model_name="other/model"), "config"),
        (lambda run: run["environment"].update(gpu="other-gpu"), "provenance"),
        (
            lambda run: run["detectors"]["activation_in_clean_model"].append(
                deepcopy(run["detectors"]["activation_in_clean_model"][0])
            ),
            "duplicate representation",
        ),
        (
            lambda run: run["detectors"]["activation_in_clean_model"].pop(),
            "representation schema",
        ),
    ],
)
def test_report_rejects_incompatible_runs(tmp_path, mutation, message):
    first = pilot_result(0, 0.8, 0.6, 0.7)
    second = pilot_result(1, 0.9, 0.8, 0.9)
    mutation(second)

    with pytest.raises(ValueError, match=message):
        write_pilot_report(
            [first, second],
            summary_path=tmp_path / "summary.json",
            figure_path=tmp_path / "figure.png",
        )


def test_report_rejects_duplicate_seeds(tmp_path):
    run = pilot_result(0, 0.8, 0.6, 0.7)

    with pytest.raises(ValueError, match="duplicate seed"):
        write_pilot_report(
            [run, deepcopy(run)],
            summary_path=tmp_path / "summary.json",
            figure_path=tmp_path / "figure.png",
        )
