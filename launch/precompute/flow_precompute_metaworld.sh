#!/usr/bin/env bash
# Reproducible single-node launch script.
# Set EVO_ROOT / NUM_GPUS below (or via environment) before running.
set -e
cd "$(dirname "$0")/../.."   # repo root

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"      # unset to use huggingface.co
# export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EVO_ROOT="${EVO_ROOT:-/path/to/evo_workspace}"                 # dataset caches / flow outputs live here

# RAFT optical-flow targets for Meta-World (flow from each frame to the frame 15
# steps ahead, matching the paper). The flow_stage{1,2,3}.sh runs read this cache.
python scripts/precompute_flow.py \
  --dataset_config_path=dataset/config_meta_world.yaml \
  --wm_view_key=image_1 \
  --out_root=${EVO_ROOT}/cache/evo_owm_flow_g15_metaworld \
  --flow_gap=15 \
  --raft_iters=20 \
  --batch_size=8
