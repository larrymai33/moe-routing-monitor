import json

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
        "count_sketch_4x2",
    }
    assert all(0.0 <= row["auroc"] <= 1.0 for row in result["results"])
    assert all(row["bits_per_token"] > 0 for row in result["results"])
