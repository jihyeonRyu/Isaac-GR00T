# Franka blue-cube fine-tuning and attention debugging

This setup fine-tunes GR00T N1.7 on
`franka_parallel_groot_lerobot` (15 fps, external + wrist cameras). The task
instruction is to pick only blue cubes and place them on the tray.

## One-time authentication

The local GR00T checkpoint still loads its gated reasoner from
`nvidia/Cosmos-Reason2-2B`. Request access on Hugging Face, then authenticate
without pasting a token into a shell history or chat:

```bash
cd /workspace/Isaac-GR00T
source .venv/bin/activate
HF_HOME=/workspace/models/huggingface-cache hf auth login
wandb login
```

Verify both sessions:

```bash
HF_HOME=/workspace/models/huggingface-cache hf auth whoami
wandb login --verify
```

W&B uses project `franka-gr00t` by default.

## Check the blue-cube segmentation

This works before a model checkpoint exists and checks both decoded camera views:

```bash
python tools/visualize_franka_attention.py \
  --dataset /workspace/datasets/franka_parallel_groot_lerobot \
  --episode 0 \
  --step 120 \
  --output outputs/franka-mask-preview.png
```

## Smoke test

Run two optimizer steps on one GPU and create `checkpoint-2`:

```bash
NUM_GPUS=1 \
GLOBAL_BATCH_SIZE=2 \
MAX_STEPS=2 \
SAVE_STEPS=2 \
DATALOADER_NUM_WORKERS=0 \
EXPERIMENT_NAME=franka-blue-cube-smoke \
bash examples/Franka/train_franka.sh
```

## Full 8-GPU run

```bash
bash examples/Franka/train_franka.sh
```

Useful overrides include `MAX_STEPS`, `SAVE_STEPS`, `GLOBAL_BATCH_SIZE`,
`EXPERIMENT_NAME`, and `WANDB_PROJECT`. The default trains the projector and
diffusion action head while freezing the visual-language reasoner, which is the
stable first experiment for this dataset.

Set `TUNE_LLM=1` only for a deliberate reasoner fine-tune:

```bash
TUNE_LLM=1 GLOBAL_BATCH_SIZE=16 bash examples/Franka/train_franka.sh
```

## Automatic offline W&B debugging

`train_franka.sh` starts a sidecar watcher automatically. It waits for a complete
checkpoint, probes the fixed episode/frame, and writes an offline W&B run plus PNG
and JSON artifacts. By default it probes every saved checkpoint (`SAVE_STEPS=250`).

Useful controls are `DEBUG_VISUALIZE=0`, `DEBUG_VIS_EPISODE`,
`DEBUG_VIS_FRAME_STEP`, `DEBUG_VIS_ACTION_GROUP`, and
`DEBUG_VIS_EVERY_N_CHECKPOINTS`. For example, probe every fourth checkpoint:

```bash
DEBUG_VIS_EVERY_N_CHECKPOINTS=4 bash examples/Franka/train_franka.sh
```

The debug watcher is deliberately forced to `WANDB_MODE=offline` because its images
contain dataset frames. Review a specific run, then upload it explicitly:

```bash
wandb sync wandb/offline-run-YYYYMMDD_HHMMSS-RUN_ID
```

In W&B, use the training run for `train/loss`, `train/grad_norm`, and
`train/learning_rate`. Use each `attention-probe` run for:

- `attention_probe`: external/wrist rows with original image, blue mask, reasoner
  patch-to-pixel heatmap, and action gradient×feature heatmap.
- `reasoner_logit_lens`: top tokens projected from each captured internal layer.
- `reasoner_generated_answer`: a readable answer from the full Cosmos reasoner
  sidecar. This is separate from the truncated GR00T internal reasoner.
- `visual_token_grids`: pixel size and visual-token grid for each camera.
- `reasoner_*` and `action_*` metrics: blue-cube lift, pointing accuracy, and
  blue-vs-red selectivity.

## Probe a checkpoint

Run the probe on a single GPU, separate from the distributed training process:

```bash
python tools/visualize_franka_attention.py \
  --dataset /workspace/datasets/franka_parallel_groot_lerobot \
  --checkpoint outputs/franka-groot-sft/franka-blue-cube-sft/checkpoint-250 \
  --episode 0 \
  --step 120 \
  --phrase "blue cube" \
  --action-group all \
  --output outputs/attention/checkpoint-250-ep0-step120.png \
  --wandb-project franka-gr00t \
  --wandb-run-name franka-attention-checkpoint-250 \
  --global-step 250
```

Each output contains the exact model input, the blue HSV mask, the final four
reasoner layers' phrase-to-image attention, and action-loss gradient×feature
saliency. A JSON file is written next to the image and the same values can be
logged to W&B.

The primary metrics are:

- `blue_attention_mass`: fraction of nonnegative attention inside the cube mask.
- `blue_attention_lift`: mass divided by cube area; values above 1 beat a uniform map.
- `blue_pointing_accuracy`: 1 when the maximum-attention pixel lies on a blue cube.
- `blue_red_selectivity`: blue mass divided by blue plus red-cube mass; closer to 1 is better.

When `TUNE_LLM=0`, raw reasoner attention should remain nearly constant across
checkpoints. The action saliency should change because the projector and diffusion
head are trained. Compare both maps rather than interpreting raw attention alone.
