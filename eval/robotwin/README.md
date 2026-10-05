# RoboTwin (2.0, MT50 dual-arm ALOHA) evaluation client

RoboTwin eval uses the **RoboTwin 2.0** simulator (SAPIEN) as the client. RoboTwin
is a plugin framework: `script/eval_policy.py` imports a policy module from
`policy/<policy_name>/` and calls its `get_model` / `eval` / `reset_model` hooks.
Evo-World plugs in as the policy named **`EvoWorld`**.

```
EvoWorld/                 # the policy plugin — copy into <RoboTwin>/policy/EvoWorld/
├── __init__.py           # re-exports the deploy hooks
├── deploy_policy.py      # websocket glue: pack obs -> server -> smooth -> execute horizon steps
├── deploy_policy.yml     # defaults (policy_name=EvoWorld, server_url, horizon)
└── eval.sh               # single-task launcher
run_all_50.sh             # full 50-task batch runner against one running server
```

The adapter (`deploy_policy.py`) connects to the Evo-World server, sends the 3 camera
views (head / left-wrist / right-wrist) + the 14-dim joint vector + the instruction,
receives a `(50, 24)` denormalized action chunk, applies a **Gaussian smoothing**
(kernel=9 — the validated recipe; executing raw chunks jitters the arm and roughly
halves success), and executes the first `horizon` steps.

## 1. Install RoboTwin 2.0

Clone and set up RoboTwin 2.0 and its simulation deps (SAPIEN, asset packs) per its own
README. You need a working `script/eval_policy.py` and the task assets.

## 2. Install the Evo-World plugin

Copy the `EvoWorld/` folder from here into your RoboTwin checkout:

```bash
cp -r EvoWorld <ROBOTWIN_ROOT>/policy/EvoWorld
```

The folder name, `deploy_policy.yml:policy_name`, and the `--policy_name` flag must all
read `EvoWorld` (RoboTwin resolves the policy module by this name).

## 3. Start the Evo-World server

In the server env (see the repo README), start one server — it reads `arm_key` /
`dataset_key` from each request, so one instance serves all 50 tasks:

```bash
CKPT_DIR=$EVO_ROOT/ckpt/evo_world_robotwin \
  bash scripts/run_robotwin_server.sh place_empty_cup 9004 0 step_100000
```

(The `place_empty_cup` arg only sets the launch-time default key; the client overrides
it per task.)

## 4. Run — single task

From your RoboTwin checkout (RoboTwin's own env), using the plugin's launcher:

```bash
cd <ROBOTWIN_ROOT>/policy/EvoWorld
bash eval.sh place_empty_cup demo_clean step_100000 0 0 ws://0.0.0.0:9004 37
#        task          config     ckpt_label  seed gpu server_url            horizon
```

`TEST_NUM` (env var) controls episodes per task (default from RoboTwin; set `TEST_NUM=20`
for a quick pass). Results land under `<ROBOTWIN_ROOT>/eval_result/...`.

## 5. Run — all 50 tasks

With a server already running, from your RoboTwin checkout:

```bash
ROBOTWIN_ROOT=<ROBOTWIN_ROOT> bash run_all_50.sh 9004 0 20 step_100000 37
#                                                 port gpu test_num label horizon
```

This loops the 50 tasks against the one server, parses each run's `Success rate:`, and
writes `<ROBOTWIN_ROOT>/eval_result/<run_label>/_summary.csv`.

### Parallel sharding across GPUs

For speed, shard the 50 tasks across N GPUs — one server per GPU/port (e.g. GPUs 0–3 on
ports 9010–9013), each handling `task_idx % N == shard` — then concatenate the per-shard
`_summary.csv` files and compute the overall mean.

## Troubleshooting

- **AV1 / video decode issues** (RoboTwin data) — use the PyAV backend, not decord.
- **`Arm key ... / Dataset key ... not found`** — the per-request `dataset_key`
  (`robotwin_<task>`) and `arm_key` (`aloha_joint`) must exist in the checkpoint's
  `norm_stats.json`.
- **Arm jitter / low success** — ensure the Gaussian smoothing in `deploy_policy.py` is
  active (it is by default).
