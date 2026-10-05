# MetaWorld (MT50) evaluation client

This is the **client** side of Evo-World MetaWorld evaluation. It launches the
MuJoCo simulator, drives MT50 in difficulty order, sends observations to a running
Evo-World **server**, executes the returned action chunks, and reports per-task /
per-bucket / overall success rates.

```
metaworld_client.py            # single-process client (full MT50 or a subset of buckets)
metaworld_client_shard.py      # shard variant: evaluate one slice of the task list
aggregate_metaworld_shards.py  # merge per-shard logs into one MT50 score
run_metaworld_parallel.sh      # 4-shard launcher + aggregation
mt50_order.json                # task order + difficulty buckets (easy/medium/hard/very_hard)
tasks.jsonl                    # per-task language prompts
```

The server/observation contract: each request carries 3 image views (view 0 is the
MetaWorld `corner2` camera, views 1–2 are masked), an 8-dim proprio slice, the task
prompt, and masks. The server replies with a denormalized action chunk; the client
executes the first `--horizon` steps, then re-queries.

## 1. Client environment

Separate from the server env (keep `numpy<2` for MuJoCo/metaworld):

```bash
conda create -y -p ./envs/metaworld python=3.10
conda activate ./envs/metaworld
pip install metaworld gymnasium mujoco imageio imageio-ffmpeg opencv-python websockets "numpy<2"
# yields metaworld 3.1.1 / gymnasium 1.3.0 / mujoco 3.3.0
```

## 2. MuJoCo EGL fix (common gotcha)

GPU offscreen rendering needs the NVIDIA EGL vendor ICD. If you see
`Cannot initialize a EGL device display`, create the ICD and point glvnd at it:

```bash
mkdir -p ~/.config/egl_vendor.d
cat > ~/.config/egl_vendor.d/10_nvidia.json <<'EOF'
{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}
EOF
export __EGL_VENDOR_LIBRARY_FILENAMES=~/.config/egl_vendor.d/10_nvidia.json
export MUJOCO_GL=egl
python -c "from mujoco import egl; egl.GLContext(64,64); print('EGL_OK')"   # self-test
```

`metaworld_client.py` auto-sets `MUJOCO_GL=egl` and picks up
`~/.config/egl_vendor.d/10_nvidia.json` if present, so on a correctly-registered
machine no exports are needed.

## 3. Run — single process

Start the server first (in the server env; see the repo README):

```bash
CKPT_DIR=$EVO_ROOT/ckpt/evo_world_metaworld \
  bash scripts/run_metaworld_server.sh 8000 0 step_100000
```

Then, in the client env:

```bash
python metaworld_client.py --out-dir eval_result/mt50 --port 8000 --horizon 4
```

Options:
- `--horizon` — steps executed per action chunk. The flow model peaks at **h2 / h4**
  (overall ≈ 0.90); sweep 2 / 4 / 8.
- `--levels` — comma-separated difficulty buckets in order (default
  `easy,medium,hard,very_hard` = full MT50). E.g. `--levels hard` for hard-only.
- `--seed` — rollout seed (default 4042).

Outputs land under `<out-dir>/{logs,episode_videos}/`. Overall success = mean of the
evaluated bucket rates. Prompts/order are read from `tasks.jsonl` / `mt50_order.json`
beside this script (override via `METAWORLD_TASKS_JSONL` / `METAWORLD_ORDER_JSON`).

## 4. Run — 4-way parallel (recommended for full MT50)

Start 4 servers, one per GPU/port, in the server env:

```bash
for i in 0 1 2 3; do
  CKPT_DIR=$EVO_ROOT/ckpt/evo_world_metaworld \
    bash scripts/run_metaworld_server.sh $((8000+i)) $i step_100000 &
done
```

Then run the sharded clients + aggregation in the client env:

```bash
bash run_metaworld_parallel.sh eval_result/mt50_parallel 4 8000 4
# -> runs 4 shards (ports 8000-8003), then aggregate_metaworld_shards.py prints the
#    combined per-bucket + overall MT50 score and writes eval_result/mt50_parallel/aggregated.txt
```

## Troubleshooting

- **`use_world_model=True` error on load** — the server needs a world-model checkpoint.
- **`Arm key ... / Dataset key ... not found`** — the server's `--arm_key metaworld_robot
  --dataset_key metaworld_Mint` must exist in the checkpoint's `norm_stats.json`.
- **Empty-looking logs** — stdout is block-buffered; watch the growing
  `episode_videos/*.mp4` count and per-task log lines for real progress.
