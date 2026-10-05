#!/bin/bash
# Full 50-task RoboTwin evaluation of the Evo-World model against ONE running server.
#
# The Evo-World server reads arm_key/dataset_key from each request payload, so a
# single server instance serves all 50 tasks. Start it first (in the SERVER env):
#   CKPT_DIR=$EVO_ROOT/ckpt/evo_owm_flow_g15_robotwin_rand_s2 \
#     bash scripts/run_robotwin_server.sh place_empty_cup 9004 0 step_100000
# Then run THIS script from your RoboTwin 2.0 checkout's environment (SAPIEN etc.).
#
# Usage: bash run_all_50.sh [port] [gpu] [test_num] [ckpt_label] [horizon]
#   port        : server websocket port (default 9004)
#   gpu         : CUDA device for the client sim (default 0)
#   test_num    : episodes per task (default 20; RoboTwin reads $TEST_NUM)
#   ckpt_label  : label only, for the result dir name (model is chosen server-side; default step_100000)
#   horizon     : action steps executed per chunk (default 37)
#
# Environment:
#   ROBOTWIN_ROOT : path to your RoboTwin 2.0 checkout (default: $PWD)
#   PYTHON        : python to use (default: python)
set -u
PORT=${1:-9004}
GPU=${2:-0}
TEST_NUM=${3:-20}
CKPT_LABEL=${4:-step_100000}
HORIZON=${5:-37}

PY=${PYTHON:-python}
RT=${ROBOTWIN_ROOT:-$PWD}
POLICY=EvoWorld

RUN_LABEL="evoworld_robotwin_${CKPT_LABEL}_test${TEST_NUM}"
LOG_DIR="$RT/eval_result/$RUN_LABEL/_batch_logs"
SUMMARY="$RT/eval_result/$RUN_LABEL/_summary.csv"
mkdir -p "$LOG_DIR"
echo "task,success_rate,succ,total,status" > "$SUMMARY"

export CUDA_VISIBLE_DEVICES=$GPU
export TEST_NUM=$TEST_NUM

TASKS=(adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size click_alarmclock \
click_bell dump_bin_bigbin grab_roller handover_block handover_mic hanging_mug lift_pot move_can_pot \
move_pillbottle_pad move_playingcard_away move_stapler_pad open_laptop open_microwave pick_diverse_bottles \
pick_dual_bottles place_a2b_left place_a2b_right place_bread_basket place_bread_skillet place_burger_fries \
place_can_basket place_cans_plasticbox place_container_plate place_dual_shoes place_empty_cup place_fan \
place_mouse_pad place_object_basket place_object_scale place_object_stand place_phone_stand place_shoe \
press_stapler put_bottles_dustbin put_object_cabinet rotate_qrcode scan_object shake_bottle \
shake_bottle_horizontally stack_blocks_three stack_blocks_two stack_bowls_three stack_bowls_two \
stamp_seal turn_switch)

cd "$RT"
echo "[batch] run label: $RUN_LABEL   results: $RT/eval_result/$RUN_LABEL"
echo "[batch] server: ws://0.0.0.0:$PORT   test_num=$TEST_NUM   horizon=$HORIZON"

i=0
for t in "${TASKS[@]}"; do
    i=$((i+1))
    echo "[batch] ($i/50) $t"
    PYTHONWARNINGS=ignore::UserWarning PYTHONUNBUFFERED=1 \
        "$PY" -u script/eval_policy.py --config policy/$POLICY/deploy_policy.yml --overrides \
        --task_name "$t" --task_config demo_clean --ckpt_setting "$CKPT_LABEL" --seed 0 \
        --policy_name "$POLICY" --server_url "ws://0.0.0.0:$PORT" --horizon "$HORIZON" \
        > "$LOG_DIR/${t}.log" 2>&1
    rc=$?
    line=$(tr '\r' '\n' < "$LOG_DIR/${t}.log" | grep -aE "Success rate:" | tail -1)
    sr=$(echo "$line" | grep -oE "[0-9]+\.[0-9]+%" | tail -1)
    frac=$(echo "$line" | grep -oE "[0-9]+/[0-9]+" | tail -1)
    succ=${frac%%/*}; total=${frac##*/}
    if [ $rc -eq 0 ] && [ -n "$frac" ]; then st=ok; else st="rc=$rc"; fi
    echo "${t},${sr:-NA},${succ:-NA},${total:-NA},${st}" >> "$SUMMARY"
    echo "[batch] ($i/50) $t -> ${sr:-NA} (${frac:-NA}) [$st]"
done

echo "[batch] ALL DONE. summary: $SUMMARY"
