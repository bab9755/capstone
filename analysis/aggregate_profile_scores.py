from __future__ import annotations

"""
Compute the swarm-wide mean score over time for a set of experiments and plot
the aggregate trajectory with a standard deviation band.

Steps performed:
1. For every run, average each timestep across all agents.
2. Align runs to the shared timestep range and average across runs.
3. Plot timestep on the x-axis and the aggregated score on the y-axis, with
   ±1σ shading.

The helper functions are structured so they can be reused in other analysis
pipelines or notebooks.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class RunSeries:
    """Mean swarm score per timestep for a single experiment run."""

    run_dir: Path
    mean_series: np.ndarray

    @property
    def length(self) -> int:
        return int(self.mean_series.shape[0])


@dataclass(frozen=True)
class AggregatedTimeline:
    """Aggregated statistics across all runs."""

    timesteps: np.ndarray
    mean_series: np.ndarray
    std_series: np.ndarray
    per_run_series: List[np.ndarray]

    @property
    def length(self) -> int:
        return int(self.mean_series.shape[0])


def load_agent_score_series(experiment_path: Path) -> List[np.ndarray]:
    """
    Extract agent score series from an experiment JSON file.
    """
    if not experiment_path.is_file():
        raise FileNotFoundError(f"Experiment file does not exist: {experiment_path}")

    with experiment_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    series: List[np.ndarray] = []
    for key, value in data.items():
        if not key.endswith("_score"):
            continue
        arr = np.asarray(value, dtype=float)
        if arr.size == 0:
            continue
        series.append(arr)

    if not series:
        raise ValueError(f"No *_score entries found in {experiment_path}")

    return series


def compute_swarm_mean_series(agent_series: Sequence[np.ndarray]) -> np.ndarray:
    """
    Average agent score series per timestep for a single run.
    """
    if not agent_series:
        raise ValueError("Agent series collection cannot be empty.")

    max_len = max(series.size for series in agent_series)
    stacked = np.full((len(agent_series), max_len), np.nan, dtype=float)
    for idx, series in enumerate(agent_series):
        stacked[idx, : series.size] = series

    with np.errstate(all="ignore"):
        mean_series = np.nanmean(stacked, axis=0)

    valid_mask = ~np.isnan(mean_series)
    return mean_series[valid_mask]


def summarize_run(run_dir: Path) -> RunSeries:
    """
    Load a run directory and compute its swarm-level mean score series.
    """
    experiment_path = run_dir / "experiment.json"
    agent_series = load_agent_score_series(experiment_path)
    mean_series = compute_swarm_mean_series(agent_series)
    if mean_series.size == 0:
        raise ValueError(f"No score samples available for {run_dir}")
    return RunSeries(run_dir=run_dir, mean_series=mean_series)


def find_runs(
    base_dir: Path,
    *,
    profile: str | None = None,
    swarm_type: str | None = None,
    ground_truth: str | None = None,
    knowledge_agents: int | None = None,
) -> List[Path]:
    """
    Locate experiment run directories filtered by metadata attributes.
    """
    if not base_dir.exists():
        raise FileNotFoundError(f"Experiments directory not found: {base_dir}")

    matched: List[Path] = []
    for metadata_path in base_dir.rglob("metadata.json"):
        try:
            with metadata_path.open("r", encoding="utf-8") as fh:
                metadata = json.load(fh)
        except json.JSONDecodeError:
            continue

        if profile and metadata.get("profile") != profile:
            continue
        if swarm_type and metadata.get("swarm_type") != swarm_type:
            continue
        if ground_truth and metadata.get("ground_truth_key") != ground_truth:
            continue
        if (
            knowledge_agents is not None
            and metadata.get("num_knowledge_agents") != knowledge_agents
        ):
            continue

        matched.append(metadata_path.parent.resolve())

    matched.sort()
    return matched


def compute_run_series(run_dirs: Sequence[Path]) -> List[RunSeries]:
    """
    Compute swarm average series for every run directory provided.
    """
    run_series: List[RunSeries] = []
    for run_dir in run_dirs:
        try:
            series = summarize_run(run_dir)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Skipping {run_dir}: {exc}")
            continue
        run_series.append(series)

    if not run_series:
        raise ValueError("No valid runs found after filtering and loading.")
    return run_series


def aggregate_timeline(series_list: Sequence[RunSeries]) -> AggregatedTimeline:
    """
    Align run series to a shared timestep range and aggregate statistics.
    """
    if not series_list:
        raise ValueError("Series list must contain at least one entry.")

    min_len = min(series.length for series in series_list)
    if min_len == 0:
        raise ValueError("At least one run has zero-length series after filtering.")

    aligned = np.vstack([series.mean_series[:min_len] for series in series_list])
    mean_series = aligned.mean(axis=0)
    std_series = aligned.std(axis=0, ddof=0)
    timesteps = np.arange(min_len)

    return AggregatedTimeline(
        timesteps=timesteps,
        mean_series=mean_series,
        std_series=std_series,
        per_run_series=[series.mean_series[:min_len] for series in series_list],
    )


def plot_timeline(
    aggregate: AggregatedTimeline,
    *,
    output_path: Path,
    title: str,
    show_runs: bool = False,
    close: bool = True,
) -> plt.Figure:
    """
    Plot the aggregated swarm performance over time with ±1σ shading.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if show_runs:
        for idx, series in enumerate(aggregate.per_run_series, start=1):
            ax.plot(
                aggregate.timesteps,
                series,
                color="tab:blue",
                alpha=0.25,
                linewidth=1,
                label="Individual run" if idx == 1 else None,
            )

    upper = aggregate.mean_series + aggregate.std_series
    lower = aggregate.mean_series - aggregate.std_series

    ax.fill_between(
        aggregate.timesteps,
        lower,
        upper,
        color="tab:orange",
        alpha=0.2,
        label="±1σ",
    )
    ax.plot(
        aggregate.timesteps,
        aggregate.mean_series,
        color="tab:blue",
        linewidth=2,
        label="Mean score",
    )

    ax.set_title(title)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Average score")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if close:
        plt.close(fig)

    print(
        f"Saved plot to {output_path} "
        f"(timesteps={aggregate.length}, runs={len(aggregate.per_run_series)})."
    )
    return fig


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate swarm scores over time for a given experiment configuration.",
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=Path("experiments"),
        help="Root directory containing experiment runs (default: %(default)s).",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="low",
        help="Profile label to match from metadata (default: %(default)s).",
    )
    parser.add_argument(
        "--swarm-type",
        type=str,
        default="self_learning",
        help="Swarm type to match from metadata (default: %(default)s).",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="career_fair_low",
        help="Ground truth key to match from metadata (default: %(default)s).",
    )
    parser.add_argument(
        "--knowledge-agents",
        type=int,
        default=10,
        help="Number of knowledge agents to match from metadata (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/low_self_learning_swarm_over_time.png"),
        help="File path for the resulting plot image (default: %(default)s).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window in addition to saving the image.",
    )
    parser.add_argument(
        "--show-runs",
        action="store_true",
        help="Overlay individual run trajectories for reference.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    run_dirs = find_runs(
        args.experiments_root,
        profile=args.profile,
        swarm_type=args.swarm_type,
        ground_truth=args.ground_truth,
        knowledge_agents=args.knowledge_agents,
    )
    if not run_dirs:
        raise SystemExit("No matching runs found with the requested filters.")

    series_list = compute_run_series(run_dirs)
    aggregate = aggregate_timeline(series_list)

    figure = plot_timeline(
        aggregate,
        output_path=args.output,
        title=(
            f"{args.profile.capitalize()} / {args.swarm_type.replace('_', ' ')} "
            f"(runs={len(series_list)})"
        ),
        show_runs=args.show_runs,
        close=not args.show,
    )

    if args.show:
        plt.show()
    else:
        plt.close(figure)

    final_mean = float(aggregate.mean_series[-1])
    final_std = float(aggregate.std_series[-1])
    global_mean = float(aggregate.mean_series.mean())
    global_std = float(aggregate.std_series.mean())
    print(f"Final timestep mean: {final_mean:.4f} ± {final_std:.4f}")
    print(f"Global mean of mean-series: {global_mean:.4f}")
    print(f"Average std dev across timesteps: {global_std:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

