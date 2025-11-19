from __future__ import annotations

"""
Plot average scores for single-agent runs versus swarm runs with social learning.

For each selected experiment, the script averages agent scores over time, then
aggregates across all experiments in the group. Experiments can be specified
explicitly through CLI flags or discovered automatically by scanning the default
directories. Curves are truncated to the minimum shared length before plotting
so the comparison is based on the same number of timesteps.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

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
    """
    Load a single experiment and compute the average score per timestep.
    """
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


def aggregate_group_scores(experiment_files: Sequence[Path]) -> AggregatedScores:
    """
    Aggregate average score series across the provided experiment files.
    """
    if not experiment_files:
        raise FileNotFoundError("No experiment files supplied for aggregation.")

    per_experiment = []
    for exp_path in experiment_files:
        try:
            series = load_experiment_average(exp_path)
        except ValueError as exc:
            print(f"Skipping {exp_path}: {exc}", file=sys.stderr)
            continue
        per_experiment.append(series)

    if not per_experiment:
        raise ValueError("Failed to compute score series for any experiment provided.")

    min_len = min(series.size for series in per_experiment)
    aligned = np.vstack([series[:min_len] for series in per_experiment])
    mean_series = aligned.mean(axis=0)

    return AggregatedScores(
        mean_series=mean_series,
        per_experiment_series=[series[:min_len] for series in per_experiment],
        n_experiments=len(per_experiment),
    )


def trim_leading_all_zero_timesteps(
    series_list: Sequence[np.ndarray], *, atol: float = 1e-9
) -> tuple[List[np.ndarray], int]:
    """
    Remove leading timesteps where every provided series is effectively zero.
    """
    series_copy = [np.asarray(series) for series in series_list]
    if not series_copy:
        return series_copy, 0

    stacked = np.vstack(series_copy)
    all_zero_columns = np.all(np.isclose(stacked, 0.0, atol=atol), axis=0)
    non_zero_indices = np.flatnonzero(~all_zero_columns)

    if non_zero_indices.size == 0:
        return series_copy, 0

    start_idx = int(non_zero_indices[0])
    return [series[start_idx:] for series in series_copy], start_idx


def resolve_experiment_selection(
    base_dir: Path, selection: Sequence[str] | None, *, label: str
) -> List[Path]:
    """
    Resolve experiment selection to a list of `experiment.json` file paths.
    """
    resolved: List[Path] = []

    if selection:
        for entry in selection:
            raw = Path(entry).expanduser()
            candidate_paths = (
                [raw]
                if raw.is_absolute()
                else [(base_dir / raw).expanduser(), raw]
            )
            found: Path | None = None
            for candidate in candidate_paths:
                candidate = candidate.resolve() if candidate.exists() else candidate
                json_path = candidate / "experiment.json" if candidate.is_dir() else candidate
                if json_path.is_file():
                    found = json_path.resolve()
                    break
            if not found:
                raise FileNotFoundError(
                    f"Could not resolve experiment path '{entry}' for {label}."
                )
            resolved.append(found)
    else:
        candidates = [
            path.resolve()
            for path in base_dir.rglob("experiment.json")
            if path.is_file()
        ]

        label_lower = label.lower()
        if "single" in label_lower:
            candidates = [p for p in candidates if "single_agent" in p.as_posix()]
        elif "social" in label_lower:
            candidates = [p for p in candidates if "social_learning_swarm" in p.as_posix()]
        elif "self" in label_lower:
            candidates = [p for p in candidates if "self_learning_swarm" in p.as_posix()]

        resolved = sorted(candidates)

    if not resolved:
        raise FileNotFoundError(
            f"No experiment.json files found for {label} using base directory {base_dir}."
        )

    deduped = list(dict.fromkeys(resolved))
    return deduped


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot average scores for single agent vs. swarm social learning experiments."
    )
    parser.add_argument(
        "--single-dir",
        type=Path,
        default=Path("experiments"),
        help="Root directory to scan for single-agent experiments (default: %(default)s)",
    )
    parser.add_argument(
        "--swarm-dir",
        type=Path,
        default=Path("experiments"),
        help="Root directory to scan for swarm social-learning experiments (default: %(default)s)",
    )
    parser.add_argument(
        "--single-experiments",
        nargs="+",
        help=(
            "Specific single-agent experiment directories or JSON files to include; "
            "overrides automatic scanning of --single-dir."
        ),
    )
    parser.add_argument(
        "--swarm-experiments",
        nargs="+",
        help=(
            "Specific swarm social learning experiment directories or JSON files to include; "
            "overrides automatic scanning of --swarm-dir."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/single_vs_swarm_social.png"),
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

    single_files = resolve_experiment_selection(
        args.single_dir, args.single_experiments, label="single agent"
    )
    swarm_files = resolve_experiment_selection(
        args.swarm_dir, args.swarm_experiments, label="swarm social learning"
    )

    single_scores = aggregate_group_scores(single_files)
    swarm_scores = aggregate_group_scores(swarm_files)

    global_min_len = min(single_scores.length, swarm_scores.length)
    if global_min_len == 0:
        raise RuntimeError("No shared datapoints to plot between selected groups.")

    single_series = single_scores.mean_series[:global_min_len]
    swarm_series = swarm_scores.mean_series[:global_min_len]

    trimmed_series, dropped = trim_leading_all_zero_timesteps(
        [single_series, swarm_series]
    )
    single_series, swarm_series = trimmed_series

    plot_len = single_series.size
    if plot_len == 0:
        raise RuntimeError("All datapoints are zero; nothing to plot after trimming.")

    x = np.arange(plot_len)

    plt.figure(figsize=(10, 6))
    plt.plot(
        x,
        single_series,
        label=f"Single agent (n={single_scores.n_experiments})",
        color="tab:green",
        linewidth=2,
    )
    plt.plot(
        x,
        swarm_series,
        label=f"Swarm social learning (n={swarm_scores.n_experiments})",
        color="tab:orange",
        linestyle="--",
        linewidth=2,
    )

    plt.xlabel("Timestep")
    plt.ylabel("Average score")
    plt.title("Performance: Single Agent vs. Swarm Social Learning")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches="tight")

    extra_note = f", dropped {dropped} leading zero timesteps" if dropped else ""
    print(
        f"Saved plot to {args.output} using {plot_len} shared datapoints{extra_note} "
        f"(single min={single_scores.length}, swarm min={swarm_scores.length})."
    )

    if args.show:
        plt.show()
    else:
        plt.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


