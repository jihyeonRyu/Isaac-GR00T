# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gr00t.debug.attention_probe import (
    attention_mask_metrics,
    blue_hsv_mask,
    find_subsequence_positions,
    image_token_maps,
    normalize_heatmap,
    red_hsv_mask,
)
import numpy as np
import pytest
import torch


def test_find_subsequence_positions_uses_final_occurrence() -> None:
    assert find_subsequence_positions([1, 2, 3, 2, 3], [2, 3]) == [3, 4]
    assert find_subsequence_positions([1, 2], [3]) == []


def test_normalize_heatmap_handles_flat_values() -> None:
    np.testing.assert_array_equal(normalize_heatmap(np.ones((2, 2))), np.zeros((2, 2)))
    np.testing.assert_allclose(
        normalize_heatmap(np.array([[2.0, 4.0]], dtype=np.float32)), [[0.0, 1.0]]
    )


def test_blue_hsv_mask_rejects_red_and_purple() -> None:
    image = np.array(
        [
            [[0, 0, 255], [255, 0, 0]],
            [[120, 70, 130], [0, 0, 0]],
        ],
        dtype=np.uint8,
    )
    expected_blue = np.array([[True, False], [False, False]])
    expected_red = np.array([[False, True], [False, False]])
    np.testing.assert_array_equal(blue_hsv_mask(image), expected_blue)
    np.testing.assert_array_equal(red_hsv_mask(image), expected_red)


def test_image_token_maps_respects_each_grid() -> None:
    scores = torch.arange(6, dtype=torch.float32)
    grids = torch.tensor([[1, 4, 4], [1, 2, 4]])
    maps = image_token_maps(scores, grids, spatial_merge_size=2)
    np.testing.assert_array_equal(maps[0], [[0, 1], [2, 3]])
    np.testing.assert_array_equal(maps[1], [[4, 5]])


def test_image_token_maps_rejects_extra_scores() -> None:
    with pytest.raises(ValueError, match="Consumed 4 image scores but received 5"):
        image_token_maps(torch.arange(5), torch.tensor([[1, 4, 4]]), spatial_merge_size=2)


def test_attention_mask_metrics() -> None:
    heatmap = np.array([[3.0, 0.0], [1.0, 0.0]])
    mask = np.array([[True, False], [False, False]])
    metrics = attention_mask_metrics(heatmap, mask)
    assert metrics["blue_attention_mass"] == pytest.approx(0.75)
    assert metrics["blue_area_fraction"] == pytest.approx(0.25)
    assert metrics["blue_attention_lift"] == pytest.approx(3.0)
    assert metrics["blue_pointing_accuracy"] == 1.0
