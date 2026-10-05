"""Aggregate shard logs from run_full_metaworld_parallel.sh into a single
MT50 result. Reads each shard's most recent logs/mt50_*.txt, extracts
per-task (s, t), groups by bucket via mt50_order.json, then prints / writes
the combined per-task SR, per-bucket SR, and overall = mean of 4 buckets.

Usage:
    python aggregate_metaworld_shards.py \\
        --base-dir eval_result/<CKPT_PARENT>/<CKPT_LEAF>_h<H>_parallel \\
        [--total-shards 5]

Output: prints to stdout and writes <base-dir>/aggregated.txt.
"""

import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, List, Tuple


# Matches lines like:
#   [Task 0 push-back-v3] put both moka pots on the stove finished 10 episodes -> success_rate=0.500  (s=5, t=10)
TASK_LINE_RE = re.compile(
    r"\[Task\s+(?P<idx>\d+)\s+(?P<slug>[\w\-]+)\].*?\(s=(?P<s>\d+),\s*t=(?P<t>\d+)\)"
)


def latest_log(shard_dir: str) -> str:
    pat = os.path.join(shard_dir, "logs", "mt50_*.txt")
    matches = sorted(glob.glob(pat))
    if not matches:
        raise FileNotFoundError(f"No log files under {pat}")
    return matches[-1]


def parse_log(log_path: str) -> Dict[int, Tuple[int, int, str]]:
    """Return {env_idx: (successes, trials, slug)} parsed from a shard log."""
    out: Dict[int, Tuple[int, int, str]] = {}
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = TASK_LINE_RE.search(line)
            if not m:
                continue
            idx = int(m.group("idx"))
            s = int(m.group("s"))
            t = int(m.group("t"))
            slug = m.group("slug")
            if idx in out:
                # Same shard logged the task twice (shouldn't happen) — take latest.
                pass
            out[idx] = (s, t, slug)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True,
                    help="Directory containing shard0/, shard1/, ... (the run_dir from run_full_metaworld_parallel.sh).")
    ap.add_argument("--total-shards", type=int, default=5)
    ap.add_argument("--order-json", default="mt50_order.json",
                    help="Path to mt50_order.json (resolved against script dir if relative).")
    ap.add_argument("--out", default=None,
                    help="Output file (default: <base-dir>/aggregated.txt).")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = args.base_dir if os.path.isabs(args.base_dir) else os.path.join(script_dir, args.base_dir)
    order_json = args.order_json if os.path.isabs(args.order_json) else os.path.join(script_dir, args.order_json)
    out_path = args.out or os.path.join(base_dir, "aggregated.txt")

    if not os.path.isdir(base_dir):
        sys.exit(f"base-dir not found: {base_dir}")
    if not os.path.isfile(order_json):
        sys.exit(f"mt50_order.json not found: {order_json}")

    with open(order_json, "r") as f:
        order = json.load(f)
    per_group_indices: Dict[str, List[int]] = {k: [int(i) for i in v] for k, v in order["per_group_indices"].items()}
    idx_to_slug: Dict[int, str] = {int(k): v for k, v in order["idx_to_slug"].items()}

    # 1) collect per-task (s,t) from each shard log
    combined: Dict[int, Tuple[int, int, str]] = {}
    shard_summaries: List[str] = []
    for shard in range(args.total_shards):
        shard_dir = os.path.join(base_dir, f"shard{shard}")
        try:
            log_path = latest_log(shard_dir)
        except FileNotFoundError as e:
            sys.exit(f"shard {shard}: {e}")
        per_task = parse_log(log_path)
        shard_summaries.append(f"  shard{shard}: {len(per_task)} tasks  log={log_path}")
        for idx, (s, t, slug) in per_task.items():
            if idx in combined:
                ps, pt, _ = combined[idx]
                # 5 shards are disjoint, so a duplicate means user re-ran a shard.
                # Trust the latest (parsed order = shard order) by summing both;
                # but more useful is to flag it.
                print(f"[WARN] task idx {idx} appears in multiple shards (had ({ps},{pt}), got ({s},{t}))")
                combined[idx] = (ps + s, pt + t, slug)
            else:
                combined[idx] = (s, t, slug)

    expected = 50
    if len(combined) != expected:
        print(f"[WARN] aggregated {len(combined)} tasks (expected {expected}). "
              f"Bucket means may be biased.")

    # 2) bucket aggregation
    BUCKETS = ["easy", "medium", "hard", "very_hard"]
    bucket_s: Dict[str, int] = {b: 0 for b in BUCKETS}
    bucket_t: Dict[str, int] = {b: 0 for b in BUCKETS}
    for b in BUCKETS:
        for idx in per_group_indices.get(b, []):
            if idx not in combined:
                print(f"[WARN] task idx {idx} (bucket={b}, slug={idx_to_slug.get(idx)}) missing in shard logs")
                continue
            s, t, _ = combined[idx]
            bucket_s[b] += s
            bucket_t[b] += t

    bucket_rate: Dict[str, float] = {
        b: (bucket_s[b] / bucket_t[b]) if bucket_t[b] > 0 else 0.0 for b in BUCKETS
    }

    # User's definition: overall = direct sum of bucket rates / 4
    overall = sum(bucket_rate[b] for b in BUCKETS) / 4.0

    # 3) report
    lines = []
    lines.append("==== Aggregated MT50 (5-shard parallel) ====")
    lines.append(f"base_dir   : {base_dir}")
    lines.append(f"order_json : {order_json}")
    lines.append("")
    lines.append("Shards:")
    lines.extend(shard_summaries)
    lines.append("")
    lines.append("==== Per-task success rate (env idx order) ====")
    for idx in sorted(combined.keys()):
        s, t, slug = combined[idx]
        rate = (s / t) if t > 0 else 0.0
        lines.append(f"  idx={idx:02d}  {slug:30s}  s={s:3d}  t={t:3d}  rate={rate:.3f}")

    lines.append("")
    lines.append("==== Per-bucket ====")
    for b in BUCKETS:
        lines.append(f"  {b:10s}  s={bucket_s[b]:3d}  t={bucket_t[b]:3d}  rate={bucket_rate[b]:.3f}")

    lines.append("")
    lines.append("==== Overall Success Rate ((easy+medium+hard+very_hard)/4) ====")
    lines.append(f"  {overall:.4f}")

    text = "\n".join(lines)
    print(text)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nWritten to: {out_path}")


if __name__ == "__main__":
    main()
