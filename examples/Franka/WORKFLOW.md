# Franka synthetic data → GR00T N1.7 → IsaacLab Arena

## Prepared paths

- Synthetic generator: `/workspace/franka_lift_auto_parallel.py`
- Dataset converter: `examples/Franka/convert_franka_to_groot_lerobot.py`
- GR00T repository/venv: `/workspace/Isaac-GR00T`, `.venv`
- Arena repository/venv: `/workspace/IsaacLab-Arena`, `.venv`
- GR00T N1.7: `/workspace/models/GR00T-N1.7-3B`
- Cosmos Reason2: `/workspace/models/Cosmos-Reason2-2B`
- Hugging Face cache: `/workspace/models/huggingface-cache`

Both venv activation scripts include the local native libraries required by this container. The GR00T venv also sets `HF_HOME` and `GROOT_COSMOS_MODEL_PATH` when they are not already set.

## 1. Generate synthetic episodes

Run from the Arena venv. This command matches the active 480-episode generation run and its camera geometry.

```bash
cd /workspace/IsaacLab-Arena
source .venv/bin/activate

python /workspace/franka_lift_auto_parallel.py \
  --headless \
  --enable_cameras \
  --num_envs 4 \
  --auto_generate_episodes 480 \
  --asset_version_override 5.1 \
  --sensor_modalities rgb \
  --output_dir /workspace/output/franka_parallel_dataset \
  --fps 15 \
  --width 320 \
  --height 256 \
  --no_realtime
```

The automatic mode enables action/state logs, external and wrist RGB capture, and MP4 output. Successful episodes have `logs/result.json` with `completed=true` and `failed=false`.

## 2. Convert to LeRobot v2.1

```bash
cd /workspace/Isaac-GR00T
source .venv/bin/activate

python examples/Franka/convert_franka_to_groot_lerobot.py \
  /workspace/output/franka_parallel_dataset \
  /workspace/datasets/franka_parallel_groot_lerobot
```

The converter:

- keeps successful episodes by default;
- aligns actions, states, and both videos by `sim_step`;
- converts XYZW quaternion state to the exact 6D rotation convention used by the Arena policy;
- writes `eef_pose(9) + gripper(1)` state and `eef_delta(6) + gripper(1)` action;
- validates FPS, resolution, MP4 frame counts, Parquet metadata, and normalization statistics;
- refuses to overwrite a non-empty output directory.

Use `--allow-incomplete` only when intentionally skipping malformed recordings. Use `--include-failed` only for diagnostics, not normal imitation learning.

## 3. Fine-tune GR00T

The Franka training script defaults to local model/cache paths and W&B disabled. This example uses physical GPUs 4–7.

```bash
cd /workspace/Isaac-GR00T
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=4,5,6,7 \
NUM_GPUS=4 \
GLOBAL_BATCH_SIZE=32 \
DATASET_PATH=/workspace/datasets/franka_parallel_groot_lerobot \
USE_WANDB=0 \
DEBUG_VISUALIZE=0 \
./examples/Franka/train_franka.sh
```

Default output:

```text
/workspace/Isaac-GR00T/outputs/franka-groot-sft/franka-blue-cube-sft/
```

For a short pipeline check before a long run, add `MAX_STEPS=2 SAVE_STEPS=1`.

## 4. Start the fine-tuned GR00T policy server

Replace `<checkpoint>` with a generated checkpoint directory.

```bash
cd /workspace/Isaac-GR00T
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 \
python gr00t/eval/run_gr00t_server.py \
  --model-path /workspace/Isaac-GR00T/outputs/franka-groot-sft/franka-blue-cube-sft/<checkpoint> \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda:0 \
  --host 127.0.0.1 \
  --port 5555
```

`cuda:0` here means physical GPU 4 because `CUDA_VISIBLE_DEVICES=4`.

## 5. Evaluate in IsaacLab Arena

Run this in another terminal while the policy server is listening.

```bash
cd /workspace/IsaacLab-Arena
source .venv/bin/activate

python -m isaaclab_arena.evaluation.experiment_runner \
  --experiment_config isaaclab_arena_environments/experiment_configs/franka_blue_tray_gr00t_experiment.yaml \
  --enable_cameras \
  --headless \
  --device cuda:1 \
  --remote_host 127.0.0.1 \
  --remote_port 5555 \
  --record_camera_video
```

The experiment reports separate success rates for one, two, and three blue cubes. Camera observations are aligned to the generator at 15 FPS, 320×256, 20.955 mm horizontal aperture, 28 mm external focal length, and 10 mm wrist focal length.

## Verified in this container

- Isaac Sim 6 / Isaac Lab 3 imports and sees all 8 GPUs.
- GR00T and Cosmos checkpoints load together with `HF_HUB_OFFLINE=1` (3,144,016,000 GR00T parameters).
- TorchCodec decodes converted external/wrist videos through the local FFmpeg 7 runtime.
- Converter end-to-end test passed with a 45-frame synthetic fixture and a completed 747-frame real episode.
- The real converted episode loaded through GR00T with external/wrist RGB `(256, 320, 3)`, EEF state 9D, EEF delta action 6D, and gripper 1D.
- Arena 1-env and 4-env/5-object camera smoke runs both completed.
- Arena camera configuration is `(N, 256, 320, 3)` for both external and wrist views.
