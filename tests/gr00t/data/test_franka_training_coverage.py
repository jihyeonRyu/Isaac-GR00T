import json
from pathlib import Path

import pytest
from tools.audit_franka_training_coverage import compute_coverage


def _write_episodes(path: Path, lengths: list[int]) -> None:
    path.write_text(
        "".join(
            json.dumps({"episode_index": index, "length": length}) + "\n"
            for index, length in enumerate(lengths)
        )
    )


def test_compute_coverage_counts_every_valid_frame_window(tmp_path):
    episodes = tmp_path / "episodes.jsonl"
    _write_episodes(episodes, [50, 60])

    result = compute_coverage(
        episodes,
        action_horizon=40,
        shard_size=8,
        episode_sampling_rate=0.1,
        global_batch_size=4,
        max_steps=9,
    )

    assert result["valid_training_windows"] == 32
    assert result["nonempty_episode_splits"] == 20
    assert result["exhaustive_shards_per_epoch"] == 4
    assert result["minimum_steps_for_one_pass"] == 8
    assert result["complete_nominal_data_passes"] == 1
    assert result["nominal_data_passes"] == pytest.approx(1.125)


def test_compute_coverage_rejects_empty_action_windows(tmp_path):
    episodes = tmp_path / "episodes.jsonl"
    _write_episodes(episodes, [10, 20])

    with pytest.raises(ValueError, match="no valid action windows"):
        compute_coverage(
            episodes,
            action_horizon=40,
            shard_size=8,
            episode_sampling_rate=0.1,
            global_batch_size=4,
            max_steps=9,
        )
