# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small, model-adjacent primitives used by the GR00T attention visualizer.

The expensive reasoner attention capture is intentionally kept out of the
distributed training loop. A full eager-attention tensor can be hundreds of MB,
and running it on rank zero alone can desynchronize a DeepSpeed job.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager

import cv2
import numpy as np
import torch


def find_subsequence_positions(sequence: Sequence[int], subsequence: Sequence[int]) -> list[int]:
    """Return positions covered by the final exact occurrence of ``subsequence``."""
    if not subsequence or len(subsequence) > len(sequence):
        return []
    result: list[int] = []
    width = len(subsequence)
    needle = list(subsequence)
    for start in range(len(sequence) - width + 1):
        if list(sequence[start : start + width]) == needle:
            result = list(range(start, start + width))
    return result


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Min/max normalize a finite heatmap, returning zeros for a flat input."""
    values = np.nan_to_num(np.asarray(heatmap, dtype=np.float32))
    if values.size == 0:
        return values
    low = float(values.min())
    high = float(values.max())
    if high - low <= 1e-12:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def blue_hsv_mask(
    rgb_image: np.ndarray,
    *,
    hue_range: tuple[int, int] = (105, 125),
    min_saturation: int = 120,
    min_value: int = 80,
) -> np.ndarray:
    """Segment saturated blue pixels from an RGB image using OpenCV HSV ranges."""
    image = np.asarray(rgb_image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image shaped (H, W, 3), got {image.shape}")
    hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)
    lower = np.array([hue_range[0], min_saturation, min_value], dtype=np.uint8)
    upper = np.array([hue_range[1], 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper).astype(bool)


def red_hsv_mask(
    rgb_image: np.ndarray,
    *,
    min_saturation: int = 120,
    min_value: int = 80,
) -> np.ndarray:
    """Segment saturated red pixels, accounting for HSV hue wraparound."""
    image = np.asarray(rgb_image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image shaped (H, W, 3), got {image.shape}")
    hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)
    low_red = cv2.inRange(hsv, np.array([0, min_saturation, min_value]), np.array([10, 255, 255]))
    high_red = cv2.inRange(
        hsv, np.array([170, min_saturation, min_value]), np.array([179, 255, 255])
    )
    return (low_red | high_red).astype(bool)


def image_token_maps(
    image_scores: torch.Tensor | np.ndarray,
    image_grid_thw: torch.Tensor | np.ndarray,
    *,
    spatial_merge_size: int,
) -> list[np.ndarray]:
    """Split sequential Qwen image-token scores into one spatial map per image."""
    scores = torch.as_tensor(image_scores).detach().float().cpu().flatten()
    grids = torch.as_tensor(image_grid_thw).detach().long().cpu()
    maps: list[np.ndarray] = []
    offset = 0
    for temporal, height, width in grids.tolist():
        if height % spatial_merge_size or width % spatial_merge_size:
            raise ValueError(
                f"Grid {(temporal, height, width)} is not divisible by "
                f"spatial_merge_size={spatial_merge_size}"
            )
        out_height = height // spatial_merge_size
        out_width = width // spatial_merge_size
        token_count = temporal * out_height * out_width
        chunk = scores[offset : offset + token_count]
        if chunk.numel() != token_count:
            raise ValueError(
                f"Image grids require at least {offset + token_count} scores, got {scores.numel()}"
            )
        # Images in this project have temporal=1. Averaging also gives sensible
        # output if a video grid is passed in the future.
        maps.append(chunk.reshape(temporal, out_height, out_width).mean(0).numpy())
        offset += token_count
    if offset != scores.numel():
        raise ValueError(f"Consumed {offset} image scores but received {scores.numel()}")
    return maps


def attention_mask_metrics(heatmap: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Measure normalized attention mass and pointing accuracy inside a mask."""
    heatmap = np.nan_to_num(np.asarray(heatmap, dtype=np.float32), nan=0.0)
    mask = np.asarray(mask, dtype=bool)
    if heatmap.ndim != 2 or mask.ndim != 2:
        raise ValueError("heatmap and mask must both be 2D")
    if heatmap.shape != mask.shape:
        heatmap = cv2.resize(
            heatmap,
            (mask.shape[1], mask.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    positive = np.maximum(heatmap, 0.0)
    total = float(positive.sum())
    mask_pixels = int(mask.sum())
    mass = float(positive[mask].sum() / total) if total > 0 and mask_pixels else 0.0
    area = float(mask.mean())
    lift = mass / area if area > 0 else 0.0
    pointing = (
        float(mask[np.unravel_index(int(positive.argmax()), positive.shape)]) if total else 0.0
    )
    return {
        "blue_attention_mass": mass,
        "blue_area_fraction": area,
        "blue_attention_lift": lift,
        "blue_pointing_accuracy": pointing,
    }


class ReasonerAttentionCapture(AbstractContextManager["ReasonerAttentionCapture"]):
    """Capture eager self-attention weights from the final Qwen text layers.

    The context switches only the wrapped Hugging Face reasoner to eager
    attention and restores its previous implementation on exit.
    """

    def __init__(self, qwen_model: torch.nn.Module, last_n_layers: int = 4):
        if last_n_layers < 1:
            raise ValueError("last_n_layers must be >= 1")
        self.model = qwen_model
        self.last_n_layers = last_n_layers
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.weights: list[torch.Tensor] = []
        self.hidden_states: list[torch.Tensor] = []
        self.layer_indices: list[int] = []
        self.previous_implementation: str | None = None

    def __enter__(self) -> "ReasonerAttentionCapture":
        self.weights.clear()
        self.hidden_states.clear()
        language_model = self.model.language_model
        layers = language_model.layers
        if self.last_n_layers > len(layers):
            raise ValueError(
                f"Requested {self.last_n_layers} layers but reasoner has {len(layers)}"
            )
        self.previous_implementation = language_model.config._attn_implementation
        self.model.set_attn_implementation("eager")

        def capture(_module, _inputs, output):
            if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
                raise RuntimeError("Eager Qwen attention did not return attention weights")
            self.weights.append(output[1].detach().float().cpu())

        def capture_hidden(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor):
                raise RuntimeError("Qwen decoder layer did not return a hidden-state tensor")
            self.hidden_states.append(hidden.detach())

        first_layer = len(layers) - self.last_n_layers
        self.layer_indices = list(range(first_layer, len(layers)))
        for layer in layers[first_layer:]:
            self.handles.append(layer.self_attn.register_forward_hook(capture))
            self.handles.append(layer.register_forward_hook(capture_hidden))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self.previous_implementation is not None:
            self.model.set_attn_implementation(self.previous_implementation)

    def phrase_to_image_scores(
        self,
        *,
        input_ids: torch.Tensor,
        image_mask: torch.Tensor,
        phrase_token_ids: Sequence[int],
    ) -> torch.Tensor:
        """Average layers/heads/query tokens into one score per image token."""
        if not self.weights:
            raise RuntimeError("No attention weights captured; run the reasoner inside the context")
        if input_ids.shape[0] != 1:
            raise ValueError("Attention probing supports batch size 1 only")
        query_positions = find_subsequence_positions(
            input_ids[0].detach().cpu().tolist(), phrase_token_ids
        )
        if not query_positions:
            raise ValueError("Phrase tokens were not found exactly in the processed prompt")
        per_layer = []
        for attention in self.weights:
            # attention: (batch, heads, query sequence, key sequence)
            selected = attention[0, :, query_positions, :].mean(dim=(0, 1))
            per_layer.append(selected)
        token_scores = torch.stack(per_layer).mean(0)
        return token_scores[image_mask[0].detach().cpu().bool()]
