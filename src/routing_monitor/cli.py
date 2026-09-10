"""Command-line entry points for reproducible routing monitor experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .compression import compress_trace
from .evaluation import evaluate_detector, vectorize_compressed
from .synthetic import make_synthetic_dataset


def _demo_results(num_pairs: int, seed: int) -> dict[str, Any]:
    traces, labels, groups = make_synthetic_dataset(num_pairs=num_pairs, seed=seed)
    configurations = [
        ("full", "full", {}),
        ("ids", "ids", {}),
        ("block_counts_4", "block_counts", {"block_size": 4}),
        ("block_counts_8", "block_counts", {"block_size": 8}),
        ("layer_counts", "layer_counts", {}),
        ("transitions", "transitions", {}),
        (
            "count_sketch_4x2",
            "count_sketch",
            {"sketch_width": 4, "sketch_depth": 2},
        ),
    ]
    results = []
    for display_name, representation, options in configurations:
        compressed = [
            compress_trace(trace, representation, num_experts=8, **options)
            for trace in traces
        ]
        evaluation = evaluate_detector(
            vectorize_compressed(compressed), labels, groups, seed=seed
        )
        results.append(
            {
                "representation": display_name,
                "bits_per_token": float(
                    np.mean([record.bits_per_token for record in compressed])
                ),
                "auroc": evaluation.auroc,
                "average_precision": evaluation.average_precision,
                "tpr_at_fpr_1pct": evaluation.tpr_at_fpr_1pct,
            }
        )
    return {"num_pairs": num_pairs, "seed": seed, "results": results}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="routing-monitor")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="run the synthetic end-to-end smoke test")
    demo.add_argument("--pairs", type=int, default=100)
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command != "demo":
        raise RuntimeError(f"unsupported command: {arguments.command}")
    result = _demo_results(arguments.pairs, arguments.seed)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if arguments.output is None:
        print(rendered)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
