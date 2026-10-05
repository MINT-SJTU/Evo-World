#!/usr/bin/env python
"""Rewrite a checkpoint's config.json pretrained-encoder names to HF repo-ids.

The released checkpoints were trained with absolute local paths baked into their
`config.json` (e.g. world_image_encoder_name=/data/.../pretrain_model/dinov2-small,
vlm_name pointing at a personal cache dir). Those paths crash `from_pretrained()`
for anyone who downloads the checkpoint, and they leak the training machine's layout
(and usernames). Run this on each checkpoint dir BEFORE uploading to HuggingFace.

It only touches absolute-path values whose basename matches a known model, so it
cannot mislabel a checkpoint that genuinely used a different encoder. HF repo-ids
and relative names are left untouched.

The server (EvoWorld_server.py) also falls back to the HF repo-id at load time when
a configured path is missing, so this script is belt-and-suspenders — but it keeps
the published artifact clean.

Usage:
    python scripts/sanitize_ckpt_config.py <ckpt_dir> [<ckpt_dir> ...]
    python scripts/sanitize_ckpt_config.py <ckpt_dir> --dry-run   # show changes only
"""
import argparse
import json
import os

# Canonical HF repo-id keyed by the model directory basename we expect to see.
BASENAME_TO_REPO_ID = {
    "dinov2-small": "facebook/dinov2-small",
    "InternVL3-1B": "OpenGVLab/InternVL3-1B",
    "clip-vit-base-patch32": "openai/clip-vit-base-patch32",
}

# config.json fields that are passed to from_pretrained().
PRETRAINED_NAME_FIELDS = ("vlm_name", "world_image_encoder_name", "world_text_encoder_name")


def sanitize_config(config_path, dry_run=False):
    with open(config_path) as f:
        cfg = json.load(f)

    changes = []
    for key in PRETRAINED_NAME_FIELDS:
        val = cfg.get(key)
        if not isinstance(val, str) or not os.path.isabs(val):
            continue  # HF repo-id or relative name — leave it
        repo_id = BASENAME_TO_REPO_ID.get(os.path.basename(val.rstrip("/")))
        if repo_id is None:
            print(f"  [warn] {key}={val!r} is an absolute path with an unrecognized "
                  f"basename; not rewriting. Set it manually if needed.")
            continue
        if val != repo_id:
            changes.append((key, val, repo_id))
            cfg[key] = repo_id

    if not changes:
        print(f"  no changes needed: {config_path}")
        return False

    for key, old, new in changes:
        print(f"  {key}: {old!r} -> {new!r}")
    if dry_run:
        print("  (dry-run: not written)")
        return False

    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    print(f"  wrote {config_path}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt_dirs", nargs="+",
                    help="Checkpoint dir(s) containing config.json (or a direct path to config.json).")
    ap.add_argument("--dry-run", action="store_true", help="Print changes without writing.")
    args = ap.parse_args()

    for d in args.ckpt_dirs:
        config_path = d if os.path.basename(d) == "config.json" else os.path.join(d, "config.json")
        if not os.path.exists(config_path):
            print(f"[skip] no config.json at {config_path}")
            continue
        print(f"[ckpt] {config_path}")
        sanitize_config(config_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
