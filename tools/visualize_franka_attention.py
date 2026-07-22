#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visualize blue-cube masks, reasoner attention, and action saliency for Franka."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import cv2  # noqa: E402
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType
from gr00t.debug import (
    ReasonerAttentionCapture,
    attention_mask_metrics,
    blue_hsv_mask,
    find_subsequence_positions,
    image_token_maps,
    normalize_heatmap,
    red_hsv_mask,
)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers.feature_extraction_utils import BatchFeature  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--checkpoint", type=Path, help="Fine-tuned checkpoint; omit for mask preview"
    )
    parser.add_argument(
        "--modality-config",
        type=Path,
        default=Path("examples/Franka/franka_config.py"),
        help="Used only when --checkpoint is omitted",
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--phrase", default="blue cube")
    parser.add_argument("--last-n-layers", type=int, default=4)
    parser.add_argument(
        "--action-group",
        choices=("all", "translation", "rotation", "gripper"),
        default="all",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/franka_attention.png"))
    parser.add_argument("--wandb-project", help="Log image and metrics to this W&B project")
    parser.add_argument("--wandb-entity", help="W&B user or team that owns the project")
    parser.add_argument("--wandb-run-name", default="franka-attention-probe")
    parser.add_argument("--top-k-tokens", type=int, default=5)
    parser.add_argument(
        "--reasoner-question",
        default=(
            "Which object should the robot manipulate and what should it do with it? "
            "Answer in one short sentence."
        ),
    )
    parser.add_argument(
        "--full-reasoner-model",
        default="nvidia/Cosmos-Reason2-2B",
        help="Full VLM used for the human-readable sidecar answer",
    )
    parser.add_argument("--generation-max-new-tokens", type=int, default=24)
    parser.add_argument(
        "--skip-reasoner-generation",
        action="store_true",
        help="Skip the extra autoregressive diagnostic answer",
    )
    parser.add_argument("--global-step", type=int, default=0)
    return parser.parse_args()


def load_python_config(path: Path) -> None:
    spec = importlib.util.spec_from_file_location("franka_attention_modality_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load modality config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def phrase_token_ids(tokenizer, input_ids: torch.Tensor, phrase: str) -> list[int]:
    sequence = input_ids[0].detach().cpu().tolist()
    candidates = (phrase, f" {phrase}")
    for candidate in candidates:
        token_ids = tokenizer.encode(candidate, add_special_tokens=False)
        if find_subsequence_positions(sequence, token_ids):
            return token_ids
    decoded = tokenizer.decode(sequence, skip_special_tokens=False)
    raise ValueError(
        f"Phrase {phrase!r} was not found as an exact token sequence in the prompt: {decoded!r}"
    )


def logit_lens_rows(
    qwen,
    tokenizer,
    capture: ReasonerAttentionCapture,
    attention_mask: torch.Tensor,
    top_k: int,
) -> list[dict[str, float | int | str]]:
    """Project each captured layer's prompt-end state through the final LM head."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if len(capture.hidden_states) != len(capture.layer_indices):
        raise RuntimeError("Captured layer indices and hidden states are misaligned")
    prompt_position = int(attention_mask[0].nonzero()[-1].item())
    rows: list[dict[str, float | int | str]] = []
    with torch.inference_mode():
        for layer_index, hidden_state in zip(capture.layer_indices, capture.hidden_states):
            normalized = qwen.language_model.norm(hidden_state[:, prompt_position, :])
            logits = qwen.lm_head(normalized).float()[0]
            log_probs = torch.log_softmax(logits, dim=-1)
            values, token_ids = torch.topk(log_probs, k=top_k)
            for rank, (value, token_id) in enumerate(
                zip(values.detach().cpu().tolist(), token_ids.detach().cpu().tolist()), start=1
            ):
                rows.append(
                    {
                        "layer_index": layer_index,
                        "layer_number": layer_index + 1,
                        "rank": rank,
                        "token_id": token_id,
                        "token": tokenizer.convert_ids_to_tokens(token_id),
                        "decoded": tokenizer.decode([token_id]),
                        "log_probability": float(value),
                    }
                )
    return rows


def generate_reasoner_answer(
    qwen,
    qwen_processor,
    images,
    instruction: str,
    question: str,
    max_new_tokens: int,
) -> str:
    """Ask the checkpoint reasoner a short, human-readable diagnostic question."""
    conversation = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": image} for image in images],
                {"type": "text", "text": f"{instruction}\n\n{question}"},
            ],
        }
    ]
    prompt = qwen_processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = qwen_processor(
        text=[prompt],
        images=images,
        return_tensors="pt",
        padding=True,
    )
    device = next(qwen.parameters()).device
    dtype = next(qwen.parameters()).dtype
    for key, value in model_inputs.items():
        if isinstance(value, torch.Tensor):
            if torch.is_floating_point(value):
                model_inputs[key] = value.to(device=device, dtype=dtype)
            else:
                model_inputs[key] = value.to(device=device)
    prompt_length = model_inputs["input_ids"].shape[1]
    with torch.inference_mode():
        generated = qwen.generate(
            **model_inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    answer_ids = generated[0, prompt_length:].detach().cpu().tolist()
    return qwen_processor.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()


def action_dimension_slice(group: str) -> slice:
    # Franka dataset layout: delta_xyz[0:3], delta_rotvec[3:6], gripper[6:7].
    return {
        "all": slice(0, 7),
        "translation": slice(0, 3),
        "rotation": slice(3, 6),
        "gripper": slice(6, 7),
    }[group]


def compute_action_saliency(model, backbone_output, action_input, group: str) -> torch.Tensor:
    """Compute action-loss gradient×feature saliency for every VLM token."""
    # clone() converts the tensor produced under inference_mode back into a
    # regular autograd tensor.
    features = backbone_output["backbone_features"].detach().clone().requires_grad_(True)
    probe_backbone = BatchFeature(
        data={
            "backbone_features": features,
            "backbone_attention_mask": backbone_output["backbone_attention_mask"],
            "image_mask": backbone_output["image_mask"],
        }
    )
    output = model.action_head(probe_backbone, action_input)
    dim_slice = action_dimension_slice(group)
    selected_loss = output["action_loss"][..., dim_slice]
    selected_mask = output["action_mask"][..., dim_slice]
    loss = selected_loss.sum() / selected_mask.sum().clamp_min(1.0)
    gradient = torch.autograd.grad(loss, features, retain_graph=False)[0]
    return (gradient * features).abs().mean(dim=-1)[0].detach().float().cpu()


def color_attention_metrics(
    heatmap: np.ndarray, blue_mask: np.ndarray, red_mask: np.ndarray
) -> dict[str, float]:
    """Add red distractor mass and blue-vs-red selectivity to target metrics."""
    result = attention_mask_metrics(heatmap, blue_mask)
    red = attention_mask_metrics(heatmap, red_mask)
    blue_mass = result["blue_attention_mass"]
    red_mass = red["blue_attention_mass"]
    result["red_attention_mass"] = red_mass
    result["red_area_fraction"] = red["blue_area_fraction"]
    result["blue_red_selectivity"] = (
        blue_mass / (blue_mass + red_mass) if blue_mass + red_mass > 0 else 0.0
    )
    return result


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    blue_mask: np.ndarray,
    red_mask: np.ndarray,
) -> np.ndarray:
    resized = cv2.resize(
        normalize_heatmap(heatmap),
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    colored = cv2.applyColorMap(np.uint8(np.clip(resized, 0, 1) * 255), cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    blended = np.uint8(0.58 * image + 0.42 * colored)
    contours, _ = cv2.findContours(
        blue_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(blended, contours, -1, (255, 255, 255), 2)
    red_contours, _ = cv2.findContours(
        red_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(blended, red_contours, -1, (255, 0, 0), 2)
    return blended


def render(
    images: list[np.ndarray],
    view_names: list[str],
    masks: list[np.ndarray],
    red_masks: list[np.ndarray],
    reasoner_maps: list[np.ndarray] | None,
    action_maps: list[np.ndarray] | None,
    output: Path,
) -> None:
    columns = ["input", "blue HSV mask"]
    if reasoner_maps is not None:
        columns.append("reasoner: blue cube")
    if action_maps is not None:
        columns.append("action gradient×feature")
    fig, axes = plt.subplots(
        len(images), len(columns), figsize=(4.5 * len(columns), 4 * len(images))
    )
    axes = np.asarray(axes, dtype=object).reshape(len(images), len(columns))
    for row, (image, view, mask, red_mask) in enumerate(zip(images, view_names, masks, red_masks)):
        column = 0
        axes[row, column].imshow(image)
        axes[row, column].set_ylabel(view, fontsize=12)
        column += 1
        axes[row, column].imshow(mask, cmap="gray", vmin=0, vmax=1)
        column += 1
        if reasoner_maps is not None:
            axes[row, column].imshow(overlay_heatmap(image, reasoner_maps[row], mask, red_mask))
            column += 1
        if action_maps is not None:
            axes[row, column].imshow(overlay_heatmap(image, action_maps[row], mask, red_mask))
        for col, title in enumerate(columns):
            axes[row, col].set_title(title)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    embodiment = EmbodimentTag.NEW_EMBODIMENT
    policy = None
    if args.checkpoint is None:
        load_python_config(args.modality_config)
        modality_configs = MODALITY_CONFIGS[embodiment.value]
    else:
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        policy = Gr00tPolicy(
            embodiment_tag=embodiment,
            model_path=str(args.checkpoint),
            device=args.device,
        )
        modality_configs = policy.modality_configs

    loader = LeRobotEpisodeLoader(args.dataset, modality_configs)
    episode = loader[args.episode]
    if not 0 <= args.step < len(episode):
        raise IndexError(f"Step {args.step} is outside episode length {len(episode)}")
    sample = extract_step_data(
        episode,
        args.step,
        modality_configs,
        embodiment,
        allow_padding=True,
    )

    reasoner_maps = None
    action_maps = None
    logit_rows: list[dict[str, float | int | str]] = []
    visual_token_grids: list[dict[str, int | str]] = []
    reasoner_answer = ""
    metrics: dict[str, float | int | str] = {
        "episode": args.episode,
        "step": args.step,
        "instruction": sample.text or "",
    }

    if policy is None:
        view_names = modality_configs["video"].modality_keys
        images = []
        for view in view_names:
            image = sample.images[view][0]
            if hasattr(image, "convert"):
                image = image.convert("RGB")
            images.append(np.asarray(image, dtype=np.uint8))
    else:
        message = [{"type": MessageType.EPISODE_STEP.value, "content": sample}]
        processed = policy.processor(message)
        # These are the exact center-cropped images that produced the visual tokens.
        images = [np.asarray(image.convert("RGB")) for image in processed["vlm_content"]["images"]]
        view_names = [
            f"{view}@t{time_index}"
            for time_index in range(len(modality_configs["video"].delta_indices))
            for view in modality_configs["video"].modality_keys
        ]
        collated = policy.collate_fn([processed])
        backbone_input, action_input = policy.model.prepare_input(collated["inputs"])

        qwen = policy.model.backbone.model
        tokenizer = policy.collate_fn.processor.tokenizer
        token_ids = phrase_token_ids(tokenizer, backbone_input["input_ids"], args.phrase)
        with (
            torch.inference_mode(),
            ReasonerAttentionCapture(qwen, last_n_layers=args.last_n_layers) as capture,
        ):
            backbone_output = policy.model.backbone(backbone_input)
        reasoner_scores = capture.phrase_to_image_scores(
            input_ids=backbone_input["input_ids"],
            image_mask=backbone_output["image_mask"],
            phrase_token_ids=token_ids,
        )
        logit_rows = logit_lens_rows(
            qwen,
            tokenizer,
            capture,
            backbone_input["attention_mask"],
            args.top_k_tokens,
        )
        metrics["reasoner_question"] = args.reasoner_question
        if not args.skip_reasoner_generation:
            from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

            full_reasoner_processor = Qwen3VLProcessor.from_pretrained(args.full_reasoner_model)
            full_reasoner = Qwen3VLForConditionalGeneration.from_pretrained(
                args.full_reasoner_model,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            ).eval()
            full_reasoner.to(next(qwen.parameters()).device)
            reasoner_answer = generate_reasoner_answer(
                full_reasoner,
                full_reasoner_processor,
                processed["vlm_content"]["images"],
                sample.text or "",
                args.reasoner_question,
                args.generation_max_new_tokens,
            )
            metrics["full_reasoner_model"] = args.full_reasoner_model
            metrics["full_reasoner_generated_text"] = reasoner_answer
            del full_reasoner
            torch.cuda.empty_cache()

        spatial_merge_size = int(qwen.config.vision_config.spatial_merge_size)
        for view_name, image, grid in zip(
            view_names, images, backbone_input["image_grid_thw"].detach().cpu().tolist()
        ):
            temporal, grid_height, grid_width = grid
            visual_token_grids.append(
                {
                    "view": view_name,
                    "pixel_height": int(image.shape[0]),
                    "pixel_width": int(image.shape[1]),
                    "patch_grid_height": int(grid_height // spatial_merge_size),
                    "patch_grid_width": int(grid_width // spatial_merge_size),
                    "visual_token_count": int(
                        temporal
                        * (grid_height // spatial_merge_size)
                        * (grid_width // spatial_merge_size)
                    ),
                }
            )
        reasoner_maps = image_token_maps(
            reasoner_scores,
            backbone_input["image_grid_thw"],
            spatial_merge_size=spatial_merge_size,
        )

        # The reasoner is frozen in the default fine-tune. This second map shows
        # whether the trainable action head actually uses each visual token.
        policy.model.zero_grad(set_to_none=True)
        torch.manual_seed(args.seed)
        action_token_scores = compute_action_saliency(
            policy.model, backbone_output, action_input, args.action_group
        )
        action_maps = image_token_maps(
            action_token_scores[backbone_output["image_mask"][0].detach().cpu()],
            backbone_input["image_grid_thw"],
            spatial_merge_size=spatial_merge_size,
        )

    masks = [blue_hsv_mask(image) for image in images]
    red_masks = [red_hsv_mask(image) for image in images]
    for index, (view, mask, red_mask) in enumerate(zip(view_names, masks, red_masks)):
        metrics[f"{view}/blue_pixel_fraction"] = float(mask.mean())
        metrics[f"{view}/red_pixel_fraction"] = float(red_mask.mean())
        if reasoner_maps is not None:
            for name, value in color_attention_metrics(
                reasoner_maps[index], mask, red_mask
            ).items():
                metrics[f"{view}/reasoner_{name}"] = value
        if action_maps is not None:
            for name, value in color_attention_metrics(action_maps[index], mask, red_mask).items():
                metrics[f"{view}/action_{name}"] = value

    render(images, view_names, masks, red_masks, reasoner_maps, action_maps, args.output)
    metrics_path = args.output.with_suffix(".json")
    debug_payload = {
        **metrics,
        "visual_token_grids": visual_token_grids,
        "logit_lens": logit_rows,
    }
    metrics_path.write_text(json.dumps(debug_payload, indent=2, sort_keys=True) + "\n")

    if args.wandb_project:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            job_type="attention-probe",
            config={
                "checkpoint": str(args.checkpoint),
                "episode": args.episode,
                "frame_step": args.step,
                "phrase": args.phrase,
                "action_group": args.action_group,
            },
        )
        wandb_metrics = {
            key: value for key, value in metrics.items() if isinstance(value, (int, float))
        }
        wandb_metrics["attention_probe"] = wandb.Image(str(args.output))
        if visual_token_grids:
            grid_columns = list(visual_token_grids[0])
            wandb_metrics["visual_token_grids"] = wandb.Table(
                columns=grid_columns,
                data=[[row[column] for column in grid_columns] for row in visual_token_grids],
            )
        if logit_rows:
            logit_columns = list(logit_rows[0])
            wandb_metrics["reasoner_logit_lens"] = wandb.Table(
                columns=logit_columns,
                data=[[row[column] for column in logit_columns] for row in logit_rows],
            )
        if reasoner_answer:
            wandb_metrics["reasoner_generated_answer"] = wandb.Table(
                columns=["question", "answer"],
                data=[[args.reasoner_question, reasoner_answer]],
            )
        run.log(wandb_metrics, step=args.global_step)
        run.finish()

    checkpoint = str(args.checkpoint) if args.checkpoint else "mask-only"
    print(f"checkpoint={checkpoint}")
    print(f"image={args.output}")
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()
