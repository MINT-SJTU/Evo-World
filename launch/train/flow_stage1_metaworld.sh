#!/usr/bin/env bash
# Reproducible single-node launch script.
# Set EVO_ROOT / NUM_GPUS below (or via environment) before running.
set -e
cd "$(dirname "$0")/../.."   # repo root

# --- environment (edit for your setup) ---------------------------------------
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"      # unset to use huggingface.co
# export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1                  # if models are pre-downloaded
# export SWANLAB_API_KEY=...   # optional cloud logging; the run below passes --disable_wandb
EVO_ROOT="${EVO_ROOT:-/path/to/evo_workspace}"                 # checkpoints / caches / outputs live here
NUM_GPUS="${NUM_GPUS:-8}"                         # original job used 8 worker(s)

accelerate launch \
  --num_processes "${NUM_GPUS}" \
  --num_machines 1 \
  --deepspeed_config_file ds_config.json \
  scripts/train.py \
  --use_world_model \
  --wm_target=flow \
  --flow_cache_root=${EVO_ROOT}/cache/evo_owm_flow_g15_metaworld \
  --flow_gap=15 \
  --wandb_project=evo_world \
  --run_name=evo_owm_flow_g15_stage1 \
  --action_head=flowmatching \
  --use_augmentation \
  --lr=1e-5 \
  --dropout=0.2 \
  --weight_decay=1e-3 \
  --batch_size=16 \
  --image_size=448 \
  --wm_image_size=224 \
  --wm_view_key=image_1 \
  --future_step=15 \
  --max_steps=10000 \
  --log_interval=10 \
  --ckpt_interval=5000 \
  --warmup_steps=1000 \
  --grad_clip_norm=1.0 \
  --num_layers=8 \
  --horizon=50 \
  --finetune_action_head \
  --disable_wandb \
  --num_workers=1 \
  --prefetch_factor=1 \
  --disable_persistent_workers \
  --video_backend=av \
  --cache_dir=${EVO_ROOT}/cache/evo_owm_flow_g15_shared \
  --vlm_name=OpenGVLab/InternVL3-1B \
  --world_image_encoder_name=facebook/dinov2-small \
  --world_text_encoder_name=openai/clip-vit-base-patch32 \
  --world_embed_dim=896 \
  --world_hidden_dim=1024 \
  --world_loss_weight=0.1 \
  --world_sigreg_weight=0.09 \
  --world_film_scale=0.1 \
  --finetune_world_image_encoder \
  --finetune_world_text_encoder \
  --dataset_config_path=dataset/config_meta_world.yaml \
  --per_action_dim=24 \
  --state_dim=24 \
  --save_dir=${EVO_ROOT}/ckpt/evo_owm_flow_g15_stage1
