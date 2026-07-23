#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Franka attention probes when completed training checkpoints appear."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--probe-script",
        type=Path,
        default=Path("tools/visualize_franka_attention.py"),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-prefix", default="franka-attention")
    parser.add_argument("--episode", dest="episodes", action="append", type=int)
    parser.add_argument("--frame-step", type=int, default=120)
    parser.add_argument("--phrase", default="blue cube")
    parser.add_argument(
        "--action-group",
        choices=("all", "translation", "rotation", "gripper"),
        default="all",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-step", type=int, default=0)
    parser.add_argument("--save-steps", type=int, required=True)
    parser.add_argument("--every-n-checkpoints", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--parent-pid", type=int)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--skip-reasoner-generation", action="store_true")
    parser.add_argument(
        "--full-reasoner-model",
        default=os.environ.get("GROOT_COSMOS_MODEL_PATH", "nvidia/Cosmos-Reason2-2B"),
    )
    args = parser.parse_args()
    args.episodes = args.episodes or [0]
    return args


def checkpoint_step(path: Path) -> int | None:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match else None


def checkpoint_is_complete(path: Path) -> bool:
    """Require trainer state and every model shard before loading a checkpoint."""
    if not (path / "trainer_state.json").is_file():
        return False
    single_file = path / "model.safetensors"
    if single_file.is_file():
        return True
    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file():
        return False
    try:
        index = json.loads(index_path.read_text())
        shards = set(index["weight_map"].values())
    except (json.JSONDecodeError, KeyError, TypeError):
        return False
    return bool(shards) and all((path / shard).is_file() for shard in shards)


def process_exists(pid: int | None) -> bool:
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def load_state(path: Path) -> dict:
    if not path.is_file():
        return {"completed": [], "attempts": {}}
    try:
        state = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"completed": [], "attempts": {}}
    state.setdefault("completed", [])
    state.setdefault("attempts", {})
    return state


def save_state(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def eligible_checkpoints(args: argparse.Namespace, state: dict) -> list[tuple[int, Path]]:
    completed = set(state["completed"])
    candidates: list[tuple[int, Path]] = []
    for path in args.run_dir.glob("checkpoint-*"):
        step = checkpoint_step(path)
        if step is None or step < args.min_step or step in completed:
            continue
        if step % args.save_steps != 0:
            continue
        checkpoint_index = step // args.save_steps
        if checkpoint_index % args.every_n_checkpoints != 0:
            continue
        if checkpoint_is_complete(path):
            candidates.append((step, path))
    return sorted(candidates)


def probe_output_path(args: argparse.Namespace, step: int, episode: int) -> Path:
    return args.output_dir / f"checkpoint-{step}-ep{episode}-step{args.frame_step}.png"


def run_probe(args: argparse.Namespace, step: int, checkpoint: Path, episode: int) -> int:
    output = probe_output_path(args, step, episode)
    command = [
        sys.executable,
        str(args.probe_script),
        "--dataset",
        str(args.dataset),
        "--checkpoint",
        str(checkpoint),
        "--episode",
        str(episode),
        "--step",
        str(args.frame_step),
        "--phrase",
        args.phrase,
        "--action-group",
        args.action_group,
        "--device",
        args.device,
        "--output",
        str(output),
        "--wandb-project",
        args.wandb_project,
        "--wandb-run-name",
        f"{args.wandb_run_prefix}-checkpoint-{step}-ep{episode}",
        "--global-step",
        str(step),
        "--full-reasoner-model",
        args.full_reasoner_model,
    ]
    if args.wandb_entity:
        command.extend(["--wandb-entity", args.wandb_entity])
    if args.skip_reasoner_generation:
        command.append("--skip-reasoner-generation")
    log_path = output.with_suffix(".log")
    print(
        f"Running attention probe for checkpoint-{step}, episode-{episode}; log={log_path}",
        flush=True,
    )
    with log_path.open("w") as log_file:
        result = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
            env=os.environ.copy(),
        )
    return result.returncode


def main() -> None:
    args = parse_args()
    if args.every_n_checkpoints < 1:
        raise ValueError("--every-n-checkpoints must be >= 1")
    if args.save_steps < 1:
        raise ValueError("--save-steps must be >= 1")
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be > 0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.run_dir / "attention_probe_state.json"
    state = load_state(state_path)

    while True:
        pending = eligible_checkpoints(args, state)
        for step, checkpoint in pending:
            key = str(step)
            attempts = int(state["attempts"].get(key, 0))
            if attempts >= args.max_attempts:
                continue
            state["attempts"][key] = attempts + 1
            save_state(state_path, state)
            failed_episodes: list[tuple[int, int]] = []
            for episode in args.episodes:
                output = probe_output_path(args, step, episode)
                if output.is_file() and output.with_suffix(".json").is_file():
                    print(
                        f"Reusing attention probe for checkpoint-{step}, episode-{episode}",
                        flush=True,
                    )
                    continue
                returncode = run_probe(args, step, checkpoint, episode)
                if returncode != 0:
                    failed_episodes.append((episode, returncode))
            if not failed_episodes:
                state["completed"].append(step)
                state["completed"] = sorted(set(state["completed"]))
                print(
                    f"Completed attention probes for checkpoint-{step}: episodes={args.episodes}",
                    flush=True,
                )
            else:
                print(
                    f"Attention probes for checkpoint-{step} failed: {failed_episodes}",
                    flush=True,
                )
            save_state(state_path, state)

        if args.once:
            return
        if args.stop_file is not None and args.stop_file.exists():
            print("Training completion marker found; checkpoint watcher is stopping", flush=True)
            return
        if not process_exists(args.parent_pid):
            print("Training parent exited; checkpoint watcher is stopping", flush=True)
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
