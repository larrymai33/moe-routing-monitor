import json
import subprocess
import sys

import pytest

from routing_monitor.cli import main


def test_demo_writes_auditable_compression_results(tmp_path):
    output = tmp_path / "demo.json"

    exit_code = main(["demo", "--pairs", "40", "--output", str(output)])

    result = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert result["num_pairs"] == 40
    assert result["seed"] == 0
    assert {row["representation"] for row in result["results"]} == {
        "full",
        "ids",
        "block_counts_4",
        "block_counts_8",
        "layer_counts",
        "transitions",
        "count_min_sketch_4x2",
    }
    assert all(0.0 <= row["auroc"] <= 1.0 for row in result["results"])
    assert all(row["array_bits_per_token"] > 0 for row in result["results"])


def test_pilot_command_exposes_storage_and_training_controls(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["pilot", "--help"])

    help_text = capsys.readouterr().out
    assert exit_info.value.code == 0
    assert "--artifact-dir" in help_text
    assert "--max-storage-gb" in help_text
    assert "--train-examples" in help_text


def test_cli_import_does_not_poison_later_torch_import():
    completed = subprocess.run(
        [sys.executable, "-c", "import routing_monitor.cli; import torch"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
