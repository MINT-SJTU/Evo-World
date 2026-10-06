<h1 align="center"> Evo-World </h1>

**Evo-World** is a lightweight vision-language-action (VLA) model that augments policy
learning with **future motion prediction**. Alongside predicting robot actions, a
compact predictive branch anticipates future motion in
a latent space, supervised by optical flow between the current and future frame. As optical flow captures image-space displacement while
suppressing static appearance, this provides a task-relevant predictive signal
focused on motion rather than appearance. 

<p align="center">
  <img src="./model.png" width="90%">
</p>


## 📦 Setup

```bash
pip install -r requirements.txt
cp .env.example .env    # optional: set HF mirror / logging keys
```

Please install the `fast-atten` corresponding to your PyTorch version; otherwise, training or evaluation may fail or produce unexpected results. We strongly recommend installing `fast-atten` by downloading the corresponding wheel package.

## 🔧 Quick start training

```bash
export EVO_ROOT=/path/to/evo_workspace   # checkpoints / caches / flow outputs
export DATA_ROOT=/path/to/datasets       # downloaded datasets
export NUM_GPUS=4

# point the dataset config at your download location, then:
bash launch/precompute/flow_precompute_robotwin_randomized.sh   # stage 0: precompute RAFT flow targets
bash launch/train/flow_stage1_robotwin_randomized.sh            # stage 1: world model + action head (VLM frozen)
bash launch/train/flow_stage2_robotwin_randomized.sh            # stage 2: end-to-end (VLM unfrozen)
```

`dataset/config_*.yaml` files — edit the `path:` fields to your `DATA_ROOT`.

## 📊 Data preparation

Datasets follow the [LeRobot](https://github.com/huggingface/lerobot) format.
See `dataset/data_preparation.md` for conversion and normalization-statistics
computation (`dataset/compute_normstats.py`).

## 🚀 Inference / evaluation

### 1. Download our pretrained checkpoints

- **Meta-World (MT50):** https://huggingface.co/MINT-SJTU/Evo-World-Metaworld
- **RoboTwin (MT50):** https://huggingface.co/MINT-SJTU/Evo-World-Robotwin

```bash
# example (requires: pip install -U "huggingface_hub[cli]")
huggingface-cli download MINT-SJTU/Evo-World-Metaworld --local-dir $EVO_ROOT/ckpt/evo_world_metaworld
huggingface-cli download MINT-SJTU/Evo-World-Robotwin  --local-dir $EVO_ROOT/ckpt/evo_world_robotwin
```

After downloading, set CKPT_DIR to the downloaded checkpoint directory.
```
$EVO_ROOT/ckpt/evo_world_metaworld/
  config.json                  # EvoConfig (use_world_model=true)
  norm_stats.json              # per-arm / per-dataset normalization stats
  mp_rank_00_model_states.pt   # DeepSpeed model states
  checkpoint.json              # DeepSpeed manifest
```

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


### 3. Meta-World (MT50)

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

Then run the MetaWorld client (client env):

```bash
python eval/metaworld/metaworld_client.py --out-dir eval_result/<name> --port 8000 --horizon 4
```

### 4. RoboTwin (2.0, MT50 dual-arm ALOHA)

norm_stats keys: `--arm_key aloha_joint --dataset_key robotwin_<task_name>` (per-task).


**a. Client** is the RoboTwin repository's `script/eval_policy.py`, which drives the
SAPIEN simulator via a policy plugin. Evo-World ships that plugin at
**`eval/robotwin/EvoWorld/`** — copy it into your RoboTwin checkout
(`cp -r eval/robotwin/EvoWorld <RoboTwin>/policy/EvoWorld`) and run with
`--policy_name EvoWorld`. Install RoboTwin 2.0 and its simulation deps (SAPIEN, asset
packs) following its own README.

**b. Run the eval.** Start the server (server env). For a single task:

```bash
CKPT_DIR=$EVO_ROOT/ckpt/evo_world_robotwin \
  bash scripts/run_robotwin_server.sh place_empty_cup 9004 0 step_100000
# arm_key=aloha_joint  dataset_key=robotwin_place_empty_cup  port=9004  gpu=0
```

Then, from your RoboTwin checkout (its own env, with the plugin copied to
`<RoboTwin>/policy/EvoWorld/`), run the `EvoWorld` plugin against the server. (client env)

Single task — use the plugin's `eval.sh` (it `cd ../..` back to the RoboTwin root):

```bash
cd <RoboTwin>/policy/EvoWorld
bash eval.sh place_empty_cup demo_clean step_100000 0 0 ws://0.0.0.0:9004 37
```

All 50 tasks against the one running server — use the batch runner (point
`ROBOTWIN_ROOT` at your checkout; it loops the task list and writes a `_summary.csv`):

```bash
ROBOTWIN_ROOT=<RoboTwin> bash eval/robotwin/run_all_50.sh 9004 0 20 step_100000 37
```



**c. Parallel sharding across GPUs** (recommended for MT50). Shard the 50 tasks across
N GPUs — one server per GPU/port, each handling `task_idx % N == shard` — then merge the
per-shard summary CSVs. A 4-way layout: shard 0→GPU 0 port 9010, shard 1→GPU 1 port 9011,
shard 2→GPU 2 port 9012, shard 3→GPU 3 port 9013.

## ⚠️ Notes on paths

All paths in configs and docs use placeholders — `/path/to/evo_workspace`
(→ `EVO_ROOT`) and `/path/to/datasets` (→ `DATA_ROOT`). Replace them, or export
the environment variables the launch scripts read.
