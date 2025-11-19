from __future__ import annotations

"""
Plot average swarm scores over time comparing social vs. self learning runs.

The script can scan entire experiment folders or, with the `--*-experiments`
flags, focus on specific runs. It first averages agent scores inside each run,
then averages across the selected experiments per group. The resulting curves
are truncated to the minimum shared length before plotting so that groups with
different durations remain comparable.
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
        resolved = sorted(
            path.resolve() for path in base_dir.glob("*/experiment.json") if path.is_file()
        )

    if not resolved:
        raise FileNotFoundError(
            f"No experiment.json files found for {label} using base directory {base_dir}."
        )

    # Preserve input order while removing duplicates
    deduped = list(dict.fromkeys(resolved))
    return deduped


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
        "--social-experiments",
        nargs="+",
        help=(
            "Specific social learning experiment directories or JSON files to include; "
            "overrides automatic scanning of --social-dir."
        ),
    )
    parser.add_argument(
        "--self-experiments",
        nargs="+",
        help=(
            "Specific self learning experiment directories or JSON files to include; "
            "overrides automatic scanning of --self-dir."
        ),
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

    social_files = resolve_experiment_selection(
        args.social_dir, args.social_experiments, label="social learning"
    )
    self_files = resolve_experiment_selection(
        args.self_dir, args.self_experiments, label="self learning"
    )

    social_scores = aggregate_group_scores(social_files)
    self_scores = aggregate_group_scores(self_files)

    global_min_len = min(social_scores.length, self_scores.length)
    if global_min_len == 0:
        raise RuntimeError("No shared datapoints to plot between the two groups.")

    social_series = social_scores.mean_series[:global_min_len]
    self_series = self_scores.mean_series[:global_min_len]

    trimmed_series, dropped = trim_leading_all_zero_timesteps(
        [social_series, self_series]
    )
    social_series, self_series = trimmed_series

    plot_len = social_series.size
    if plot_len == 0:
        raise RuntimeError("All datapoints are zero; nothing to plot after trimming.")

    x = np.arange(plot_len)

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

    extra_note = f", dropped {dropped} leading zero timesteps" if dropped else ""
    print(
        f"Saved plot to {args.output} using {plot_len} shared datapoints{extra_note} "
        f"(social min={social_scores.length}, self min={self_scores.length})."
    )

    if args.show:
        plt.show()
    else:
        plt.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

