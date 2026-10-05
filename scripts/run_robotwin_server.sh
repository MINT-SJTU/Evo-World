#!/bin/bash
# Launch the EvoWorld inference server for RoboTwin evaluation.
#
# The RoboTwin norm_stats are per-task, and (per current setup) the server is
# launched once PER TASK with the matching --dataset_key. Client sends
# arm_key="aloha_joint", dataset_key="robotwin_<task_name>".
#
# Usage: bash run_robotwin_server.sh <task_name> [port] [gpu] [ckpt_step]
#   task_name : e.g. place_empty_cup   (WITHOUT the robotwin_ prefix)
#   port      : websocket port (default 9004)
#   gpu       : CUDA device (default 0)
#   ckpt_step : checkpoint subdir (default step_100000)
set -e
task_name=${1:?usage: run_robotwin_server.sh <task_name> [port] [gpu] [ckpt_step]}
port=${2:-9004}
gpu=${3:-0}
step=${4:-step_100000}

# Point these at your environment.
PY=${PYTHON:-python}
EVO=${EVO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
BASE=${CKPT_DIR:-/path/to/evo_workspace/ckpt/evo_owm_flow_g15_robotwin_rand_s2}
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

echo "[server] ckpt=$CKPT  dataset_key=robotwin_${task_name}  port=$port  gpu=$gpu"
exec $PY -u scripts/EvoWorld_server.py \
    --ckpt_dir "$CKPT" \
    --port "$port" \
    --arm_key aloha_joint \
    --dataset_key "robotwin_${task_name}"
