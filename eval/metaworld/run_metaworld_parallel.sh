#!/bin/bash
# Parallel MetaWorld MT50 evaluation across 4 shards.
#
# MT50 is slow serially. This splits the ordered task list into 4 contiguous
# shards and runs one client per shard, then aggregates into a single MT50 score.
#
# PREREQUISITE: start 4 servers first (in the SERVER env), one per GPU/port, e.g.:
#   CKPT_DIR=$EVO_ROOT/ckpt/evo_world_metaworld bash scripts/run_metaworld_server.sh 8000 0 step_100000
#   CKPT_DIR=$EVO_ROOT/ckpt/evo_world_metaworld bash scripts/run_metaworld_server.sh 8001 1 step_100000
#   CKPT_DIR=$EVO_ROOT/ckpt/evo_world_metaworld bash scripts/run_metaworld_server.sh 8002 2 step_100000
#   CKPT_DIR=$EVO_ROOT/ckpt/evo_world_metaworld bash scripts/run_metaworld_server.sh 8003 3 step_100000
# Then run THIS script in the CLIENT env (metaworld/mujoco).
#
# Usage: bash run_metaworld_parallel.sh [out_dir] [horizon] [base_port] [total_shards]
#   out_dir      : output root for all shards (default eval_result/mt50_parallel)
#   horizon      : action steps executed per chunk (default 4)
#   base_port    : first server port; shard i uses base_port+i (default 8000)
#   total_shards : number of shards/servers (default 4)
set -u
OUT_DIR=${1:-eval_result/mt50_parallel}
HORIZON=${2:-4}
BASE_PORT=${3:-8000}
TOTAL=${4:-4}
PY=${PYTHON:-python}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"
echo "[parallel] out_dir=$OUT_DIR horizon=$HORIZON ports=$BASE_PORT..$((BASE_PORT+TOTAL-1)) shards=$TOTAL"

pids=()
for ((s=0; s<TOTAL; s++)); do
    port=$((BASE_PORT + s))
    "$PY" -u metaworld_client_shard.py \
        --shard "$s" --total-shards "$TOTAL" \
        --out-dir "$OUT_DIR/shard$s" \
        --port "$port" --horizon "$HORIZON" &
    pids+=($!)
    echo "[parallel] shard $s -> port $port (pid ${pids[-1]})"
done

rc=0
for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
done
if [ "$rc" -ne 0 ]; then
    echo "[parallel] WARNING: at least one shard exited non-zero; aggregating what completed."
fi

echo "[parallel] aggregating ..."
"$PY" -u aggregate_metaworld_shards.py --base-dir "$OUT_DIR" --total-shards "$TOTAL"
