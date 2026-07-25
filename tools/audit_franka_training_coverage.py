#!/usr/bin/env python3

"""Audit frame-window coverage and derive an exhaustive Franka shard epoch."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def compute_coverage(
    episodes_path: Path,
    *,
    action_horizon: int,
    shard_size: int,
    episode_sampling_rate: float,
    global_batch_size: int,
    max_steps: int,
) -> dict[str, int | float]:
    if action_horizon < 1:
        raise ValueError("action_horizon must be positive")
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    if not 0 < episode_sampling_rate <= 1:
        raise ValueError("episode_sampling_rate must be in (0, 1]")
    if global_batch_size < 1 or max_steps < 1:
        raise ValueError("global_batch_size and max_steps must be positive")

    episodes = [
        json.loads(line)
        for line in episodes_path.read_text().splitlines()
        if line.strip()
    ]
    effective_lengths = [
        max(0, int(episode["length"]) - action_horizon + 1) for episode in episodes
    ]
    valid_windows = sum(effective_lengths)
    if valid_windows < 1:
        raise ValueError("dataset has no valid action windows")

    # Keep this formula synchronized with ShardedSingleStepDataset.shard_dataset.
    # "episode_sampling_rate" controls how many shuffled subsequences are made;
    # all frame indices remain present rather than being subsampled away.
    episode_splits = int(1 / episode_sampling_rate)
    nonempty_episode_splits = sum(
        min(episode_splits, effective_length)
        for effective_length in effective_lengths
    )
    exhaustive_shards = min(
        math.ceil(valid_windows / shard_size),
        nonempty_episode_splits,
    )
    minimum_steps = math.ceil(valid_windows / global_batch_size)
    sample_budget = global_batch_size * max_steps
    nominal_passes = sample_budget / valid_windows

    return {
        "episodes": len(episodes),
        "action_horizon": action_horizon,
        "valid_training_windows": valid_windows,
        "shard_size": shard_size,
        "episode_sampling_rate": episode_sampling_rate,
        "episode_splits": episode_splits,
        "nonempty_episode_splits": nonempty_episode_splits,
        "exhaustive_shards_per_epoch": exhaustive_shards,
        "global_batch_size": global_batch_size,
        "minimum_steps_for_one_pass": minimum_steps,
        "max_steps": max_steps,
        "sample_budget": sample_budget,
        "complete_nominal_data_passes": sample_budget // valid_windows,
        "nominal_data_passes": nominal_passes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--action-horizon", type=int, default=40)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--episode-sampling-rate", type=float, default=0.1)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--format", choices=("json", "tsv"), default="json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = compute_coverage(
        args.episodes,
        action_horizon=args.action_horizon,
        shard_size=args.shard_size,
        episode_sampling_rate=args.episode_sampling_rate,
        global_batch_size=args.global_batch_size,
        max_steps=args.max_steps,
    )
    if result["sample_budget"] < result["valid_training_windows"]:
        raise ValueError(
            "training budget does not cover one frame-level data pass: "
            f"increase max_steps to at least {result['minimum_steps_for_one_pass']}"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        fields = (
            "valid_training_windows",
            "exhaustive_shards_per_epoch",
            "minimum_steps_for_one_pass",
            "complete_nominal_data_passes",
            "nominal_data_passes",
        )
        print("\t".join(str(result[field]) for field in fields))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
