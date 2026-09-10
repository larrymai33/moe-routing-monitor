"""Command-line entry points for reproducible routing monitor experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _demo_results(num_pairs: int, seed: int) -> dict[str, Any]:
    from .compression import compress_trace
    from .evaluation import evaluate_detector, vectorize_compressed
    from .synthetic import make_synthetic_dataset

    traces, labels, groups = make_synthetic_dataset(num_pairs=num_pairs, seed=seed)
    configurations = [
        ("full", "full", {}),
        ("ids", "ids", {}),
        ("block_counts_4", "block_counts", {"block_size": 4}),
        ("block_counts_8", "block_counts", {"block_size": 8}),
        ("layer_counts", "layer_counts", {}),
        ("transitions", "transitions", {}),
        (
            "count_min_sketch_4x2",
            "count_min_sketch",
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
                "array_bits_per_token": float(
                    np.mean([record.array_bits_per_token for record in compressed])
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
    pilot = subparsers.add_parser(
        "pilot", help="run the real Switch-base-8 backdoor routing pilot"
    )
    pilot.add_argument("--artifact-dir", type=Path, required=True)
    pilot.add_argument("--model", default="google/switch-base-8")
    pilot.add_argument(
        "--model-revision",
        default="92fe2d22b024d9937146fe097ba3d3a7ba146e1b",
    )
    pilot.add_argument("--trigger", default="banana")
    pilot.add_argument("--control", default="garden")
    pilot.add_argument("--train-examples", type=int, default=200)
    pilot.add_argument("--eval-examples", type=int, default=100)
    pilot.add_argument("--head-epochs", type=int, default=4)
    pilot.add_argument("--backdoor-epochs", type=int, default=3)
    pilot.add_argument("--batch-size", type=int, default=8)
    pilot.add_argument("--max-length", type=int, default=64)
    pilot.add_argument("--max-storage-gb", type=float, default=4.0)
    pilot.add_argument("--poison-fraction", type=float, default=0.2)
    pilot.add_argument("--route-loss-weight", type=float, default=0.5)
    pilot.add_argument("--target-label", type=int, default=1)
    pilot.add_argument("--target-expert", type=int, default=0)
    pilot.add_argument("--seed", type=int, default=0)
    pilot.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "demo":
        result = _demo_results(arguments.pairs, arguments.seed)
        output = arguments.output
    elif arguments.command == "pilot":
        from .pilot_runner import PilotConfig, run_pilot

        result = run_pilot(
            PilotConfig(
                artifact_dir=arguments.artifact_dir,
                model_name=arguments.model,
                model_revision=arguments.model_revision,
                trigger=arguments.trigger,
                control=arguments.control,
                target_label=arguments.target_label,
                target_expert=arguments.target_expert,
                poison_fraction=arguments.poison_fraction,
                train_examples=arguments.train_examples,
                eval_examples=arguments.eval_examples,
                head_epochs=arguments.head_epochs,
                backdoor_epochs=arguments.backdoor_epochs,
                batch_size=arguments.batch_size,
                max_length=arguments.max_length,
                route_loss_weight=arguments.route_loss_weight,
                max_storage_bytes=int(arguments.max_storage_gb * 1024**3),
                seed=arguments.seed,
                device=arguments.device,
            )
        )
        output = None
    else:
        raise RuntimeError(f"unsupported command: {arguments.command}")
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if output is None:
        print(rendered)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
