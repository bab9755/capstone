from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class AggregatedScores:

    mean_series: np.ndarray
    per_experiment_series: List[np.ndarray]
    n_experiments: int

    @property
    def length(self) -> int:
        return int(self.mean_series.shape[0])


def load_experiment_average(experiment_path: Path) -> np.ndarray:

    with experiment_path.open("r") as fh:
        data = json.load(fh)

    agent_series: List[np.ndarray] = []
    for key, value in data.items():
        if not key.endswith("_score"):
            continue
        series = np.asarray(value, dtype=float)
        if series.size == 0:
            continue
        agent_series.append(series)

    if not agent_series:
        raise ValueError(f"No *_score series found in {experiment_path}")

    max_len = max(series.size for series in agent_series)
    stacked = np.full((len(agent_series), max_len), np.nan, dtype=float)
    for idx, series in enumerate(agent_series):
        stacked[idx, : series.size] = series

    with np.errstate(all="ignore"):
        mean_series = np.nanmean(stacked, axis=0)

    valid_mask = ~np.isnan(stacked).all(axis=0)
    return mean_series[valid_mask]


def aggregate_group_scores(experiments_dir: Path) -> AggregatedScores:
    """
    Aggregate average score series across all experiments in a directory.

    Parameters
    ----------
    experiments_dir : Path
        Directory containing subdirectories with `experiment.json` files.

    Returns
    -------
    AggregatedScores
        Mean score series across experiments and metadata.
    """
    experiment_files = sorted(
        path for path in experiments_dir.glob("*/experiment.json") if path.is_file()
    )

    if not experiment_files:
        raise FileNotFoundError(
            f"No experiment.json files found under {experiments_dir}"
        )

    per_experiment = []
    for exp_path in experiment_files:
        try:
            series = load_experiment_average(exp_path)
        except ValueError as exc:
            print(f"Skipping {exp_path}: {exc}", file=sys.stderr)
            continue
        per_experiment.append(series)

    if not per_experiment:
        raise ValueError(
            f"Could not compute any score series for experiments in {experiments_dir}"
        )

    min_len = min(series.size for series in per_experiment)
    aligned = np.vstack([series[:min_len] for series in per_experiment])
    mean_series = aligned.mean(axis=0)

    return AggregatedScores(
        mean_series=mean_series,
        per_experiment_series=[series[:min_len] for series in per_experiment],
        n_experiments=len(per_experiment),
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot average swarm scores for social vs. self learning experiments."
    )
    parser.add_argument(
        "--social-dir",
        type=Path,
        default=Path("experiments/swarm_social_learning"),
        help="Directory containing social learning experiment folders (default: %(default)s)",
    )
    parser.add_argument(
        "--self-dir",
        type=Path,
        default=Path("experiments/swarm_self_learning"),
        help="Directory containing self learning experiment folders (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/swarm_social_vs_self.png"),
        help="Path to save the resulting plot (default: %(default)s)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window in addition to saving.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    social_scores = aggregate_group_scores(args.social_dir)
    self_scores = aggregate_group_scores(args.self_dir)

    global_min_len = min(social_scores.length, self_scores.length)
    if global_min_len == 0:
        raise RuntimeError("No shared datapoints to plot between the two groups.")

    x = np.arange(global_min_len)
    social_series = social_scores.mean_series[:global_min_len]
    self_series = self_scores.mean_series[:global_min_len]

    plt.figure(figsize=(10, 6))
    plt.plot(
        x,
        social_series,
        label=f"Social learning (n={social_scores.n_experiments})",
        color="tab:orange",
        linestyle="--",
        linewidth=2,
    )
    plt.plot(
        x,
        self_series,
        label=f"Self learning (n={self_scores.n_experiments})",
        color="tab:blue",
        linewidth=2,
    )

    plt.xlabel("Timestep")
    plt.ylabel("Average score")
    plt.title("Swarm Performance: Social vs. Self Learning")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(
        f"Saved plot to {args.output} using {global_min_len} shared datapoints "
        f"(social min={social_scores.length}, self min={self_scores.length})."
    )

    if args.show:
        plt.show()
    else:
        plt.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

