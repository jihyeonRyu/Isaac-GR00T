from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).parents[2] / "tools" / "watch_franka_checkpoints.py"
SPEC = importlib.util.spec_from_file_location("watch_franka_checkpoints", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
watcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watcher)


def make_complete_checkpoint(root: Path, step: int) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text("{}\n")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    return checkpoint


def test_checkpoint_is_complete_requires_all_indexed_shards(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-250"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text("{}\n")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            }
        )
    )

    assert not watcher.checkpoint_is_complete(checkpoint)
    (checkpoint / "model-00001-of-00002.safetensors").write_bytes(b"a")
    assert not watcher.checkpoint_is_complete(checkpoint)
    (checkpoint / "model-00002-of-00002.safetensors").write_bytes(b"b")
    assert watcher.checkpoint_is_complete(checkpoint)


def test_eligible_checkpoints_respects_frequency_and_state(tmp_path: Path) -> None:
    for step in (250, 500, 750, 1000):
        make_complete_checkpoint(tmp_path, step)
    args = SimpleNamespace(
        run_dir=tmp_path,
        min_step=250,
        save_steps=250,
        every_n_checkpoints=2,
    )

    candidates = watcher.eligible_checkpoints(
        args,
        {"completed": [500], "attempts": {}},
    )

    assert candidates == [(1000, tmp_path / "checkpoint-1000")]


def test_load_state_recovers_from_invalid_json(tmp_path: Path) -> None:
    state_path = tmp_path / "attention_probe_state.json"
    state_path.write_text("not-json")

    assert watcher.load_state(state_path) == {"completed": [], "attempts": {}}
