# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modality configuration for the Franka blue-cube LeRobot dataset."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


franka_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["external", "wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["eef_pose", "gripper"],
    ),
    "action": ModalityConfig(
        # 40 frames at 15 fps is a 2.67 second receding-horizon prediction.
        delta_indices=list(range(40)),
        modality_keys=["eef_delta", "gripper"],
        action_configs=[
            # The dataset already stores delta xyz + delta rotation-vector. DELTA
            # prevents StateActionProcessor from applying a second relative-pose
            # conversion.
            ActionConfig(
                rep=ActionRepresentation.DELTA,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROTVEC,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="gripper",
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(franka_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
