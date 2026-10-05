"""Shard variant of metaworld_client.

Adds --shard / --total-shards to evaluate only a slice of the ordered MT50
task list. Reuses metaworld_client by monkey-patching load_order_and_groups,
so eval/render/payload logic stays identical to the canonical client.

Slicing is contiguous over ordered_indices (level-order):
    shard 0 -> ordered[0 : ceil(N/K)]
    shard 1 -> ordered[ceil(N/K) : 2*ceil(N/K)]
    ...
With N=50, K=5 -> each shard has 10 tasks.

Aggregation across shards is done by aggregate_metaworld_shards.py — this
script only writes per-shard logs to its own --out-dir.

Usage:
    python metaworld_client_shard.py \\
        --shard 0 --total-shards 5 \\
        --out-dir eval_result/<CKPT_PARENT>/<CKPT_LEAF>_h<H>_parallel/shard0 \\
        --port 8000 --horizon 8 --seed 4042
"""

import argparse
import asyncio
import sys

import metaworld_client as mc


_SHARD = None
_TOTAL = None
_ORIG_LOAD = mc.load_order_and_groups


def patched_load(total_envs, levels):
    ordered, groups, idx_to_slug, per_level_idx = _ORIG_LOAD(total_envs, levels)
    n = len(ordered)
    shard_size = (n + _TOTAL - 1) // _TOTAL
    start = _SHARD * shard_size
    end = min(start + shard_size, n)
    keep = set(ordered[start:end])
    new_ordered = [i for i in ordered if i in keep]
    new_per_level = {
        lv: [i for i in per_level_idx.get(lv, []) if i in keep] for lv in levels
    }
    mc.log_write(
        f"[SHARD {_SHARD}/{_TOTAL}] taking ordered[{start}:{end}] -> "
        f"{len(new_ordered)} tasks (env idx: {new_ordered})"
    )
    return new_ordered, groups, idx_to_slug, new_per_level


def main():
    global _SHARD, _TOTAL
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--shard", type=int, required=True,
                     help="0-indexed shard number (0..total_shards-1).")
    pre.add_argument("--total-shards", type=int, default=5,
                     help="Number of shards to split ordered task list into (default 5).")
    shard_args, remaining = pre.parse_known_args()
    if shard_args.shard < 0 or shard_args.shard >= shard_args.total_shards:
        pre.error(f"--shard {shard_args.shard} out of range [0, {shard_args.total_shards})")
    _SHARD = shard_args.shard
    _TOTAL = shard_args.total_shards

    sys.argv = [sys.argv[0]] + remaining
    mc.load_order_and_groups = patched_load
    asyncio.run(mc._amain())


if __name__ == "__main__":
    main()
