# Evo-World

Evo-World is a lightweight vision-language-action (VLA) model that augments policy
learning with **future motion prediction**. Alongside predicting robot actions, a
compact predictive branch (the *optical world model*) anticipates future motion in
a **latent space**, supervised by optical flow between the current frame and a
frame 15 steps ahead. Because optical flow captures image-space displacement while
suppressing static appearance, this provides a task-relevant predictive signal
focused on motion rather than appearance. The predicted motion representation is
fed back into the policy via FiLM conditioning, improving manipulation without
changing the action interface — and at inference no future frame or optical-flow
estimator is required.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env    # optional: set HF mirror / logging keys
```

Pretrained backbones are pulled from HuggingFace on first use
(`OpenGVLab/InternVL3-1B`, `facebook/dinov2-small`, `openai/clip-vit-base-patch32`).
Set `HF_ENDPOINT=https://hf-mirror.com` if `huggingface.co` is not reachable, or
pre-download them and set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

## Quick start

```bash
export EVO_ROOT=/path/to/evo_workspace   # checkpoints / caches / flow outputs
export DATA_ROOT=/path/to/datasets       # downloaded datasets
export NUM_GPUS=4

# point the dataset config at your download location, then:
bash launch/precompute/flow_precompute_robotwin_randomized.sh   # stage 0: precompute RAFT flow targets
bash launch/train/flow_stage1_robotwin_randomized.sh            # stage 1: world model + action head (VLM frozen)
bash launch/train/flow_stage2_robotwin_randomized.sh            # stage 2: end-to-end (VLM unfrozen)
```

`launch/README.md` indexes every available run. Dataset paths are set in the
`dataset/config_*.yaml` files — edit the `path:` fields to your `DATA_ROOT`.

## Data preparation

Datasets follow the [LeRobot](https://github.com/huggingface/lerobot) format.
See `dataset/data_preparation.md` for conversion and normalization-statistics
computation (`dataset/compute_normstats.py`).

## Inference / evaluation

### 1. Download pretrained checkpoints (HuggingFace)

We release the evaluated checkpoints on HuggingFace. Download the one for the
benchmark you want and point `CKPT_DIR` at the downloaded folder:

- **Meta-World (MT50):** https://huggingface.co/MINT-SJTU/Evo-World-Metaworld
- **RoboTwin (MT50):** https://huggingface.co/MINT-SJTU/Evo-World-Robotwin

```bash
# example (requires: pip install -U "huggingface_hub[cli]")
huggingface-cli download MINT-SJTU/Evo-World-Metaworld --local-dir $EVO_ROOT/ckpt/evo_world_metaworld
huggingface-cli download MINT-SJTU/Evo-World-Robotwin  --local-dir $EVO_ROOT/ckpt/evo_world_robotwin
```

Each HuggingFace repo is **flat** — the files sit at the repo root (no `step_XXXXX/`
subdirectory), so after download `CKPT_DIR` points directly at the folder above:

```
$EVO_ROOT/ckpt/evo_world_metaworld/
  config.json                  # EvoConfig (use_world_model=true)
  norm_stats.json              # per-arm / per-dataset normalization stats
  mp_rank_00_model_states.pt   # DeepSpeed model states
  checkpoint.json              # DeepSpeed manifest
```

The launch scripts accept either this flat layout or a local training run's nested
`step_XXXXX/` layout (they look for `config.json` under the step subdir first, then
fall back to `CKPT_DIR` itself), so the `ckpt_step` argument is ignored for a flat
HF download.

The pretrained-encoder names in `config.json` (`vlm_name`, `world_image_encoder_name`,
`world_text_encoder_name`) are HF repo-ids (`OpenGVLab/InternVL3-1B`,
`facebook/dinov2-small`, `openai/clip-vit-base-patch32`), pulled on first use. 

### 2. How evaluation works

Evo-World is evaluated with a **server / client** split:

- **Server** (this repo, `scripts/EvoWorld_server.py`) loads a checkpoint and
  serves denormalized action chunks over a websocket. One binary serves every
  benchmark; only the `--arm_key` / `--dataset_key` (which select entries in the
  checkpoint's `norm_stats.json`) and the client differ.
- **Client** (the simulator, in the benchmark's own repo) runs the rollout, sends
  observations, receives actions, and reports success rates.

`scripts/run_metaworld_server.sh` and `scripts/run_robotwin_server.sh` wrap the
server with the right norm-stats keys (set `CKPT_DIR`, or edit the script).

### 3. Server environment

Build one environment for the server from `requirements.txt` (plus a matching
FlashAttention wheel for `torch==2.5.1`):

```bash
conda create -y -p ./envs/evo_server python=3.10
conda activate ./envs/evo_server
pip install -r requirements.txt
# FlashAttention matching torch 2.5.1 / cu12 / cp310, e.g.:
# pip install --no-deps flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

If the backbones are pre-downloaded, launch the server with
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` so ranks don't hit the network.

### 4. Meta-World (MT50)

norm_stats keys (server defaults): `--arm_key metaworld_robot --dataset_key metaworld_Mint`.

**a. Client environment** (separate from the server env):

```bash
conda create -y -p ./envs/metaworld python=3.10
conda activate ./envs/metaworld
pip install metaworld gymnasium mujoco imageio imageio-ffmpeg opencv-python websockets "numpy<2"
# yields metaworld 3.1.1 / gymnasium 1.3.0 / mujoco 3.3.0
```

**b. EGL fix for MuJoCo offscreen rendering** (common gotcha). If you see
`Cannot initialize a EGL device display`, create the NVIDIA EGL vendor ICD and
point glvnd at it:

```bash
mkdir -p ~/egl_fix
cat > ~/egl_fix/10_nvidia.json <<'EOF'
{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}
EOF
export __EGL_VENDOR_LIBRARY_FILENAMES=~/egl_fix/10_nvidia.json
export MUJOCO_GL=egl
python -c "from mujoco import egl; egl.GLContext(64,64); print('EGL_OK')"   # self-test
```

**c. Run the eval.** Start the server (server env):

```bash
CKPT_DIR=$EVO_ROOT/ckpt/evo_world_metaworld \
  bash scripts/run_metaworld_server.sh 8000 0 step_100000
# -> Evo-World server running at ws://0.0.0.0:8000
```

Then run the MetaWorld client (client env), choosing an action horizon:

```bash
python eval/metaworld/metaworld_client.py --out-dir eval_result/<name> --port 8000 --horizon 4
```

- Inference is independent of the world-model target; a flow checkpoint uses the same server.
- **Horizon** = steps executed per action chunk. The flow model peaks at **h2 / h4**
  (Overall success ≈ 0.90); sweep horizon 2 / 4 / 8 to pick the best.
- Run two checkpoints in parallel by launching a second server on another GPU/port
  (`bash scripts/run_metaworld_server.sh 8001 1 ...`) and pointing a second client at it.
- Full MT50, client env + MuJoCo EGL fix, single-process vs 4-way sharded runs, and
  aggregation are documented in **`eval/metaworld/README.md`**.

### 5. RoboTwin (2.0, MT50 dual-arm ALOHA)

norm_stats keys: `--arm_key aloha_joint --dataset_key robotwin_<task_name>` (per-task).
Because the client can send `arm_key` / `dataset_key` per request, a single server
can serve all 50 tasks — or launch one server per task.

**a. Client** is the RoboTwin repository's `script/eval_policy.py`, which drives the
SAPIEN simulator via a policy plugin. Evo-World ships that plugin at
**`eval/robotwin/EvoWorld/`** — copy it into your RoboTwin checkout
(`cp -r eval/robotwin/EvoWorld <RoboTwin>/policy/EvoWorld`) and run with
`--policy_name EvoWorld`. Install RoboTwin 2.0 and its simulation deps (SAPIEN, asset
packs) following its own README. Key knobs: `TEST_NUM` (episodes per task, default 100;
use 20 for a quick pass), `--horizon 37` (client-side action horizon). Results land
under `RoboTwin/eval_result/<weight_id>/<run_label>/`.

**b. Run the eval.** Start the server (server env). For a single task:

```bash
CKPT_DIR=$EVO_ROOT/ckpt/evo_world_robotwin \
  bash scripts/run_robotwin_server.sh place_empty_cup 9004 0 step_100000
# arm_key=aloha_joint  dataset_key=robotwin_place_empty_cup  port=9004  gpu=0
```

Then, from your RoboTwin checkout, run the `EvoWorld` plugin against `ws://<host>:9004`
(single task via `policy/EvoWorld/eval.sh`, or all 50 via `eval/robotwin/run_all_50.sh`).
The full install-and-run recipe is in **`eval/robotwin/README.md`**.

**c. Parallel sharding across GPUs** (recommended for MT50). Shard the 50 tasks across
N GPUs — one server per GPU/port, each handling `task_idx % N == shard` — then merge the
per-shard summary CSVs. A 4-way layout: shard 0→GPU 0 port 9010, shard 1→GPU 1 port 9011,
shard 2→GPU 2 port 9012, shard 3→GPU 3 port 9013.

## Notes on paths

All paths in configs and docs use placeholders — `/path/to/evo_workspace`
(→ `EVO_ROOT`) and `/path/to/datasets` (→ `DATA_ROOT`). Replace them, or export
the environment variables the launch scripts read.
