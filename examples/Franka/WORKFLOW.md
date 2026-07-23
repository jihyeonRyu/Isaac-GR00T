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

## 3. Fine-tune GR00T on 8 GPUs

Authenticate W&B once, then run the checked-in launcher. The Hugging Face models are
already stored locally, so training does not depend on an online model download.

```bash
cd /workspace/Isaac-GR00T
source .venv/bin/activate
wandb login

bash examples/Franka/train_franka.sh
```

The current defaults are the reproducible full-run settings:

- 8 GPUs and global batch size 64;
- 10,000 optimizer steps, checkpoint every 250 steps;
- `crop_fraction=0.98` with shortest image edge 256;
- state dropout 0.20;
- brightness/contrast/saturation/hue jitter 0.25/0.25/0.30/0.03;
- W&B online project `franka-gr00t`;
- frozen visual-language reasoner (`TUNE_LLM=0`), with projector and diffusion head trained;
- four debug samples (episodes 0, 1, 2, and 3 at frame 120) at every saved checkpoint.

The completed run is:

```text
/workspace/Isaac-GR00T/outputs/franka-groot-sft/
  franka-blue-cube-sft-crop098-aug-v2/checkpoint-10000
```

Its W&B run is `iycwnbnb`. The final four local attention images are under:

```text
/workspace/Isaac-GR00T/outputs/attention/
  franka-blue-cube-sft-crop098-aug-v2/checkpoint-10000-ep{0,1,2,3}-step120.png
```

The reasoner attention panels use the dataset task prompt stored in LeRobot metadata.
They show Cosmos attention for that prompt and input image. With `TUNE_LLM=0`, raw
reasoner attention is expected to remain mostly fixed; action saliency can still change
because the projector and action head are trained.

For a short pipeline check, override `NUM_GPUS=1 GLOBAL_BATCH_SIZE=2 MAX_STEPS=2
SAVE_STEPS=2 DATALOADER_NUM_WORKERS=0 EXPERIMENT_NAME=franka-blue-cube-smoke`.

## 4. Evaluate the checkpoint in Arena on 8 GPUs

The parallel launcher starts one GR00T server and one Arena worker per physical GPU.
Each worker uses `cuda:0` inside its own `CUDA_VISIBLE_DEVICES` namespace. Install the
small RPC dependencies in the Arena venv once if they are not already present:

```bash
cd /workspace/IsaacLab-Arena
source .venv/bin/activate
python -m pip install msgpack-numpy==0.4.8 pyzmq==27.0.1
```

Run the 10-episode-per-task evaluation:

```bash
cd /workspace/IsaacLab-Arena
source .venv/bin/activate

python -m isaaclab_arena_gr00t.parallel_evaluation \
  --checkpoint /workspace/Isaac-GR00T/outputs/franka-groot-sft/franka-blue-cube-sft-crop098-aug-v2/checkpoint-10000 \
  --num-gpus 8 \
  --episodes-per-task 10 \
  --base-port 5655 \
  --output-dir /workspace/IsaacLab-Arena/outputs/franka-gr00t-parallel/final-crop098-aug-v2-8gpu
```

Port 5655 is used because another service may occupy the default port 5555. The
launcher checks all requested GPUs, model/config paths, and the complete port range
before starting any child process. It also supplies the local Cosmos path and required
Isaac Sim EULA environment variables.

Evaluation deliberately uses seeds distinct from data generation:

| task | base seed | rank seeds |
| --- | ---: | --- |
| one blue cube | 10007 | 10007–10014 |
| two blue cubes | 20007 | 20007–20014 |
| three blue cubes | 30007 | 30007–30014 |

The 10 episodes for each task are split across the eight workers as
`[2, 2, 1, 1, 1, 1, 1, 1]`. Every worker runs one Arena environment, which avoids
multiplying the policy server batch unexpectedly. Recorder HDF5 datasets are written
inside each run output directory with a rebuild-specific filename, so concurrent
workers never contend for `/tmp/isaaclab/logs`.

Camera visualization is enabled by default. The launcher records external and wrist
MP4s, keeps per-rank HTML reports and logs, and writes these aggregate outputs:

```text
<output-dir>/parallel_eval_manifest.json
<output-dir>/summary.json
<output-dir>/index.html
<output-dir>/logs/server-rank-*.log
<output-dir>/logs/arena-rank-*.log
<output-dir>/rank-*/...
```

Use `--no-record-camera-video` only for a deliberately faster non-visual evaluation.
The renderer is Isaac Sim's `IsaacRtxRenderer` real-time RTX backend; this workflow
does not enable path tracing.

## Verified in this container

- 480 generated attempts produced 374 valid LeRobot v2.1 episodes and 208,268 frames at 15 FPS.
- The converted dataset has external/wrist RGB `(256, 320, 3)`, EEF pose state 9D plus gripper 1D, and EEF delta action 6D plus gripper 1D.
- GR00T N1.7 and Cosmos Reason2 load from `/workspace/models` without a runtime Hugging Face download.
- The 8-GPU training run reached step 10,000 and wrote a complete `checkpoint-10000`.
- Four checkpoint-10,000 attention/debug images were produced for episodes 0–3.
- Arena cameras match generation at 15 FPS and 320×256 for both external and wrist views.
