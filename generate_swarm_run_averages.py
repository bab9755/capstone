#!/usr/bin/env python3
"""
Generate per-run average score trajectories for swarm experiments.

Given a base directory that contains swarm experiment cohorts (for example
`experiments/low`), this script walks the self-learning and social-learning
subdirectories, computes the average score over time for every run (averaging
across the knowledge agents within a run), and saves the results to a JSON file.

The implementation reuses helper utilities from `plot_swarm_experiment_averages`
to ensure consistency with existing aggregate plots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from plot_swarm_experiment_averages import (
    _load_score_sequences,
    _mean_over_agents,
)

# Default swarm directory names expected underneath the base experiment directory.
DEFAULT_SWARM_TYPES: Tuple[str, ...] = ("self_learning_swarm", "social_learning_swarm")


def _iter_run_directories(swarm_root: Path) -> Iterable[Path]:
    """Yield all `run_*` directories beneath the provided swarm root."""
    for path in sorted(swarm_root.rglob("run_*")):
        if path.is_dir():
            yield path


def _summarize_run(run_dir: Path, *, relative_to: Path) -> Dict[str, object]:
    """Return the averaged score trajectory for a single run directory."""
    sequences = _load_score_sequences(run_dir)
    if not sequences:
        raise ValueError(f"No score data found in {run_dir}")

    mean_scores = _mean_over_agents(sequences)
    timesteps = np.arange(1, mean_scores.size + 1, dtype=int)

    return {
        "run_path": str(run_dir.relative_to(relative_to)),
        "timesteps": timesteps.tolist(),
        "mean_scores": mean_scores.tolist(),
    }


def collect_run_averages(
    base_dir: Path,
    swarm_types: Iterable[str] = DEFAULT_SWARM_TYPES,
) -> Dict[str, List[Dict[str, object]]]:
    """
    Gather per-run average score trajectories for the provided swarm types.

    Args:
        base_dir: Directory containing swarm subdirectories.
        swarm_types: Iterable of swarm directory names to include.

    Returns:
        Mapping of swarm type name to a list of run summaries.
    """
    results: Dict[str, List[Dict[str, object]]] = {}
    missing_swarm_types: list[str] = []

    for swarm_name in swarm_types:
        swarm_dir = base_dir / swarm_name
        if not swarm_dir.exists():
            missing_swarm_types.append(swarm_name)
            continue

        run_summaries: list[Dict[str, object]] = []
        for run_dir in _iter_run_directories(swarm_dir):
            try:
                summary = _summarize_run(run_dir, relative_to=swarm_dir)
            except (FileNotFoundError, ValueError):
                # Skip runs that are missing experiment data or contain no scores.
                continue
            run_summaries.append(summary)

        if run_summaries:
            results[swarm_name] = run_summaries
        else:
            results[swarm_name] = []

    if missing_swarm_types:
        missing_str = ", ".join(missing_swarm_types)
        raise FileNotFoundError(
            f"Missing swarm directories under {base_dir}: {missing_str}"
        )

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-run average score trajectories for swarm experiments."
    )
    parser.add_argument(
        "base_dir",
        type=Path,
        help="Base directory containing swarm experiment runs (e.g. experiments/low).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("swarm_run_averages.json"),
        help="JSON filepath for the exported data (default: ./swarm_run_averages.json).",
    )
    parser.add_argument(
        "--swarm-types",
        nargs="+",
        default=list(DEFAULT_SWARM_TYPES),
        choices=DEFAULT_SWARM_TYPES,
        help="Subset of swarm types to include (default: both self and social).",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation level for JSON output (default: 2). Use 0 for compact JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_dir = args.base_dir
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    swarm_types: List[str] = args.swarm_types
    results = collect_run_averages(base_dir, swarm_types=swarm_types)

    payload = {
        "base_directory": str(base_dir.resolve()),
        "swarm_types": swarm_types,
        "runs": results,
    }

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=args.indent or None)


if __name__ == "__main__":
    main()




