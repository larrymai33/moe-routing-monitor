"""Generate the public pilot summary and README figure from per-seed results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from routing_monitor.reporting import write_pilot_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--summary", type=Path, default=Path("docs/results/pilot-summary.json")
    )
    parser.add_argument(
        "--figure", type=Path, default=Path("docs/assets/pilot-results.png")
    )
    arguments = parser.parse_args()
    paths = sorted(arguments.results_dir.glob("seed-*.json"))
    if not paths:
        parser.error(f"no seed-*.json files found in {arguments.results_dir}")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    write_pilot_report(
        results, summary_path=arguments.summary, figure_path=arguments.figure
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
