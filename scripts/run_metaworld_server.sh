#!/bin/bash
# Launch the EvoWorld inference server for MetaWorld (MT50) evaluation.
#
# The server loads a checkpoint and serves action chunks over a websocket. The
# MetaWorld simulator runs separately as the *client* and connects to
# ws://<host>:<port>. MetaWorld uses a single norm_stats key
# pair, so one server serves all 50 tasks.
#
# Usage: bash run_metaworld_server.sh [port] [gpu] [ckpt_step]
#   port      : websocket port (default 8000)
#   gpu       : CUDA device (default 0)
#   ckpt_step : checkpoint subdir (default step_100000)
set -e
port=${1:-8000}
gpu=${2:-0}
step=${3:-step_100000}

# Point these at your environment.
PY=${PYTHON:-python}
EVO=${EVO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
BASE=${CKPT_DIR:-/path/to/evo_workspace/ckpt/evo_world_metaworld_stage3}
# The HuggingFace release is flat (config.json at the repo root); a local training
# run nests checkpoints under step_XXXXX/. Accept either: prefer the step subdir, else
# fall back to BASE itself.
if [ -f "$BASE/$step/config.json" ]; then
    CKPT="$BASE/$step"
else
    CKPT="$BASE"
fi

cd "$EVO"
export CUDA_VISIBLE_DEVICES=$gpu
# MuJoCo offscreen rendering happens client-side, but keep HF offline if models
# are pre-downloaded:
# export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "[server] ckpt=$CKPT  arm_key=metaworld_robot  dataset_key=metaworld_Mint  port=$port  gpu=$gpu"
exec $PY -u scripts/EvoWorld_server.py \
    --ckpt_dir "$CKPT" \
    --port "$port" \
    --arm_key metaworld_robot \
    --dataset_key metaworld_Mint
