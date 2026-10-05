# Launch Scripts

Platform-agnostic launch scripts for every reproducible run. Each file was
auto-generated from an internal job spec with the cluster wrapper and all
credentials removed; it reduces to a plain `accelerate launch` (training) or
`torchrun` / `python` (precompute) invocation on a single node.

## Usage

```bash
# from the repo root
export EVO_ROOT=/path/to/evo_workspace   # checkpoints / caches / flow outputs live here
export NUM_GPUS=8                         # defaults to the original job's worker count

bash launch/precompute/flow_precompute_robotwin_randomized.sh   # stage 0: precompute RAFT flow targets
bash launch/train/flow_stage1_robotwin_randomized.sh            # stage 1: world model + action head (VLM frozen)
bash launch/train/flow_stage2_robotwin_randomized.sh            # stage 2: end-to-end (VLM unfrozen)
```

Every script honors these environment variables (with sensible defaults):

| Variable | Default | Meaning |
|----------|---------|---------|
| `EVO_ROOT` | `/path/to/evo_workspace` | Root for checkpoints, caches, and flow outputs |
| `NUM_GPUS` | job's original worker count | Number of GPUs / processes for this node |
| `HF_ENDPOINT` | `https://hf-mirror.com` | HuggingFace mirror; unset to use `huggingface.co` |

Pretrained backbones are referenced by their canonical HuggingFace repo IDs
(`OpenGVLab/InternVL3-1B`, `facebook/dinov2-small`, `openai/clip-vit-base-patch32`)
and download on first use. To run fully offline, pre-download them and export
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` (a commented line in each script).

## `precompute/` — flow target precomputation

Run before stage 1. Produces the optical-flow videos the world model regresses to
(RAFT, flow from each frame to the frame 15 steps ahead — see the paper).

| Script | Dataset / notes |
|--------|-----------------|
| `flow_precompute_metaworld.sh` | Meta-World MT50, RAFT, gap=15 |
| `flow_precompute_robotwin_randomized.sh` | RoboTwin (randomized), RAFT, gap=15 |

## `train/` — training runs

Naming: `<method>_<stage>_<dataset><_variant>.sh`.

- **stages**: `stage1` = world model + action head with VLM frozen; `stage2` =
  end-to-end with VLM unfrozen (resumes stage1); `stage3`/`stage4` = further
  continued-training rounds used for some runs.
- **methods**: `flow` = optical-flow world model (RAFT targets);
  `framediff` = frame-difference world target; unprefixed `stage*` =
  baseline Evo (no world model).
- **datasets**: `metaworld` (the unprefixed Meta-World scripts) and
  `robotwin_randomized`.
- **variants**: `_nodetach`/`_nosigreg` ablations, `_resume`/`_810e`
  continued-training entry points.

Open any script to see the exact hyperparameters — they are the ground truth for
each run.
