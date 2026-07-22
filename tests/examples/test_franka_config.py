# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import runpy

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionFormat, ActionRepresentation


REPO_ROOT = Path(__file__).parents[2]


def test_franka_modality_config_matches_dataset_layout() -> None:
    tag = EmbodimentTag.NEW_EMBODIMENT.value
    previous = MODALITY_CONFIGS.pop(tag, None)
    try:
        runpy.run_path(str(REPO_ROOT / "examples/Franka/franka_config.py"))
        config = MODALITY_CONFIGS[tag]

        assert config["video"].modality_keys == ["external", "wrist"]
        assert config["video"].delta_indices == [0]
        assert config["state"].modality_keys == ["eef_pose", "gripper"]
        assert config["action"].modality_keys == ["eef_delta", "gripper"]
        assert config["action"].delta_indices == list(range(40))
        assert config["language"].modality_keys == ["annotation.human.action.task_description"]

        eef, gripper = config["action"].action_configs
        assert eef.rep == ActionRepresentation.DELTA
        assert eef.format == ActionFormat.XYZ_ROTVEC
        assert gripper.rep == ActionRepresentation.ABSOLUTE
    finally:
        MODALITY_CONFIGS.pop(tag, None)
        if previous is not None:
            MODALITY_CONFIGS[tag] = previous
