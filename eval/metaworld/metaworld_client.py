# mt50_evo1_client.py
import asyncio
import base64
import json
import os
from typing import List, Optional, Dict, Set

import cv2
import gymnasium as gym
import imageio
import metaworld  # noqa: F401
import numpy as np
import shutil
import tempfile
import websockets
import random

import datetime
import argparse

# ===================== Logging =====================
# Populated in _amain() after parsing CLI args (from --ckpt-dir + --horizon).
LOG_DIR: Optional[str] = None
LOG_PATH: Optional[str] = None
HORIZON = 17  # default; CLI --horizon overrides
# ====================================================

SHOW_WINDOW = False
SAVE_IMAGE = False
SAVE_VIDEO = True  # save the video of each episode to disk

# ===================== Debug image saving =====================
INSPECT_SAMPLE_PER_EPISODE = True        
INSPECT_DIR = "inspect_frames"           
APPLY_ROT_180 = True                     
APPLY_CENTER_CROP = True                 
CROP_KEEP_RATIO = 2/3                    
INSPECT_SAVE_STEP_TAG = True             
# =============================================================

# ===================== Debug video saving ====================
VIDEO_SAVE_DIR: Optional[str] = None  # set in _amain (derived from --ckpt-dir + --horizon + log timestamp)
VIDEO_FPS = 10  # Original writing frame rate (used to control playback speed; the smaller the value, the slower the playback).
VIDEO_DUP_FRAMES = 1  # Number of times to duplicate each frame when writing video (used to control playback speed; the larger the value, the slower the playback).
# =============================================================


# ===================== User Config (edit here) =====================
SERVER_URL = "ws://127.0.0.1:9000"  # default; CLI --port overrides

# Camera & image settings
CAMERA_NAME = "corner2"
IMG_SIZE = (448, 448)

# Evo1 & rollout settings
STATE_TAKE = 8

EPISODES = 10
EPISODE_HORIZON = 400
SEED = 4042  # default; CLI --seed overrides
TARGET_LEVEL = "all"   # legacy; bucket selection is via --levels

# Order source — ships alongside this script (resolve relative to the file so cwd
# doesn't matter). Override with METAWORLD_ORDER_JSON if you keep it elsewhere.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ORDER_JSON_PATH = os.environ.get("METAWORLD_ORDER_JSON", os.path.join(_SCRIPT_DIR, "mt50_order.json"))

FALLBACK_USE_FIRST_N: Optional[int] = 5
FALLBACK_IDX_LIST: Optional[List[int]] = None

# Prompt source — ships alongside this script. Override with METAWORLD_TASKS_JSONL.
TASKS_JSONL_PATH = os.environ.get("METAWORLD_TASKS_JSONL", os.path.join(_SCRIPT_DIR, "tasks.jsonl"))
# ==================================================================

# Headless GL by default; switch to 'glfw' on a desktop if you want
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help=(
            "Output root for this run. Logs/videos/inspect frames land under "
            "<out-dir>/{logs,episode_videos,inspect_frames}/. Relative paths are resolved "
            "against the script dir. Convention: eval_result/<ckpt-parent>/<ckpt-leaf>_h<H> "
            "to keep parity with prior runs, but any path works — the client doesn't "
            "validate which ckpt the server is serving."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="Server port (derives SERVER_URL=ws://127.0.0.1:<port>). Default: 8000.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=HORIZON,
        help=f"Action chunk horizon (default: {HORIZON})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED})",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default="easy,medium,hard,very_hard",
        help=(
            "Comma-separated difficulty buckets to evaluate, in order. "
            "Choices: easy, medium, hard, very_hard. "
            "Examples: 'hard' (only hard), 'easy,hard' (easy then hard). "
            "Default: easy,medium,hard,very_hard (full MT50 in difficulty order)."
        ),
    )
    args = parser.parse_args()
    args.levels = [s.strip().lower() for s in args.levels.split(",") if s.strip()]
    valid = {"easy", "medium", "hard", "very_hard"}
    bad = [l for l in args.levels if l not in valid]
    if bad:
        parser.error(f"Invalid --levels values: {bad}. Allowed: {sorted(valid)}")
    return args


os.environ.setdefault("MUJOCO_GL", "egl")
# 本机 glvnd 只注册了 Mesa 的 EGL ICD (/usr/share/glvnd/egl_vendor.d 没有 NVIDIA 条目)，
# Mesa 不支持 PLATFORM_DEVICE，无头初始化会失败；指向用户本地的 NVIDIA vendor 配置，
# 强制 EGL 走 GPU。ICD 注册正常的机器上该文件不存在，此处不生效。
_nv_icd = os.path.expanduser("~/.config/egl_vendor.d/10_nvidia.json")
if os.path.exists(_nv_icd):
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", _nv_icd)
gym.logger.min_level = gym.logger.ERROR


# ---------------- Utils ----------------
def encode_image_uint8_list(img_bgr: np.ndarray):
    """Encode as base64(raw uint8 bytes) instead of a nested int list.

    The old .tolist() path serialized each 448x448x3 image into ~600k JSON ints
    (~2.5MB each, ~1s/request round-trip serialization with 3 images); raw-bytes
    b64 is lossless and cuts that to ~2ms. Server's decode_image_from_list
    accepts both formats, so old clients keep working.
    """
    img = np.ascontiguousarray(img_bgr, dtype=np.uint8)
    return {"b64": base64.b64encode(img.tobytes()).decode("ascii"),
            "shape": list(img.shape)}

def obs_to_state(obs, take: int = STATE_TAKE) -> List[float]:
    if isinstance(obs, dict):
        if "observation" in obs:
            arr = np.asarray(obs["observation"], dtype=np.float32).ravel()
        else:
            parts = [np.asarray(v).ravel() for v in obs.values()]
            arr = np.concatenate(parts).astype(np.float32)
    else:
        arr = np.asarray(obs, dtype=np.float32).ravel()
    return arr[:min(take, arr.shape[0])].tolist()

def fix_camera_angle(rgb: np.ndarray) -> np.ndarray:
    
    return cv2.rotate(rgb, cv2.ROTATE_180)

def center_crop_keep_ratio(rgb: np.ndarray, keep_ratio: float) -> np.ndarray:
    
    h, w = rgb.shape[:2]
    keep_ratio = float(keep_ratio)
    keep_ratio = max(1e-6, min(1.0, keep_ratio))  
    new_h = max(1, int(round(h * keep_ratio)))
    new_w = max(1, int(round(w * keep_ratio)))
    y0 = (h - new_h) // 2
    x0 = (w - new_w) // 2
    return rgb[y0:y0 + new_h, x0:x0 + new_w, :]

def render_single_bgr(env) -> np.ndarray:
  
    rgb = env.render()                               
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)   

   
    if APPLY_ROT_180:
        rgb = cv2.rotate(rgb, cv2.ROTATE_180)
        rgb = np.ascontiguousarray(rgb)

    
    if APPLY_CENTER_CROP and (0.0 < CROP_KEEP_RATIO < 1.0):
        h, w = rgb.shape[:2]
        keep = float(CROP_KEEP_RATIO)
        new_h = max(1, int(round(h * keep)))
        new_w = max(1, int(round(w * keep)))
        y0 = (h - new_h) // 2
        x0 = (w - new_w) // 2
        rgb = rgb[y0:y0 + new_h, x0:x0 + new_w, :].copy()
        rgb = np.ascontiguousarray(rgb)

   
    if IMG_SIZE is not None:
        rgb = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        rgb = np.ascontiguousarray(rgb)

    
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = np.ascontiguousarray(bgr, dtype=np.uint8)

    
    if 'SHOW_WINDOW' in globals() and SHOW_WINDOW:
        try:
            cv2.imshow("MetaWorld", bgr)
            cv2.waitKey(1)   
        except Exception:
           
            pass

    return bgr

def append_frame(frames: Optional[list], img_bgr: np.ndarray):
    """Buffer a BGR frame into the episode's frame list (duplicated VIDEO_DUP_FRAMES times)."""
    if frames is None:
        return
    for _ in range(VIDEO_DUP_FRAMES):
        frames.append(img_bgr)

def save_video(frames: Optional[list], video_name: str, task_idx: int, slug: str, ep_num: int):
    """Encode the buffered frames to mp4 on local tmpfs, then copy to VIDEO_SAVE_DIR.

    If VIDEO_SAVE_DIR lives on a network/object-store mount, muxers that seek back
    and rewrite the header on close can fail (OSError: [Errno 22] Invalid argument),
    so we encode to a local temp file first and copy the finished mp4 into place.
    """
    if not frames:
        return
    os.makedirs(VIDEO_SAVE_DIR, exist_ok=True)
    final_path = os.path.join(VIDEO_SAVE_DIR, video_name)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)
    try:
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        imageio.mimsave(tmp_path, rgb_frames, fps=VIDEO_FPS)
        shutil.copyfile(tmp_path, final_path)
        log_write(f"[video] task={task_idx} slug={slug} ep={ep_num} saved {len(frames)} frames -> {final_path}")
    except Exception as e:
        log_write(f"[video][ERROR] save_video failed for {video_name}: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# Masked-out dummy views never change — encode once at import instead of per request.
_DUMMY_IMG_ENC = None

async def evo1_infer(ws, img_bgr: np.ndarray, state_vec: List[float], prompt: Optional[str] = None) -> np.ndarray:
    assert prompt is not None and len(prompt) > 0, "prompt should be non-empty"
    global _DUMMY_IMG_ENC
    if _DUMMY_IMG_ENC is None:
        _DUMMY_IMG_ENC = encode_image_uint8_list(np.zeros((448, 448, 3), dtype=np.uint8))
    payload = {
        "image": [encode_image_uint8_list(img_bgr),
                  _DUMMY_IMG_ENC,
                  _DUMMY_IMG_ENC],
        "state": state_vec,
        "prompt": prompt,              
        "image_mask": [1, 0, 0],
        "action_mask": [1, 1, 1, 1] + [0]*20,
    }
    await ws.send(json.dumps(payload))
    data = json.loads(await ws.recv())
    return np.asarray(data, dtype=np.float32)


def save_sent_bgr_frame(img_bgr: np.ndarray, ep_num: int, idx: int, slug: str, step: Optional[int] = None):

    os.makedirs(INSPECT_DIR, exist_ok=True)
    tag = f"step{step:04d}" if (INSPECT_SAVE_STEP_TAG and step is not None) else "stepNA"
    out = os.path.join(INSPECT_DIR, f"ep{ep_num:03d}_idx{idx}_{slug}_{tag}.png")
    img_bgr_safe = np.ascontiguousarray(img_bgr)  
    cv2.imwrite(out, img_bgr_safe)
    h, w = img_bgr_safe.shape[:2]
    print(f"[inspect] saved {out}  size={w}x{h}  (identical to VLA input)")

def log_write(text: str):
    
    print(text)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(text + "\n")

# ---------------- Prompt loader ----------------
class PromptBook:

    def __init__(self, jsonl_path: str):
        self.by_idx: Dict[int, str] = {}
        self.by_slug: Dict[str, str] = {}
        self.seq: List[str] = []

        if not os.path.exists(jsonl_path):
            print(f"[WARN] {jsonl_path} not found; prompts will be empty.")
            return

        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        for i, obj in enumerate(lines):
            task_txt = str(obj.get("task", "")).strip()
            if "idx" in obj:
                try:
                    self.by_idx[int(obj["idx"])] = task_txt
                except Exception:
                    pass
            if "slug" in obj:
                try:
                    self.by_slug[str(obj["slug"])] = task_txt
                except Exception:
                    pass
            self.seq.append(task_txt)

    def get(self, idx: int, slug: Optional[str] = None) -> str:
        if idx in self.by_idx:
            return self.by_idx[idx]
        if slug is not None and slug in self.by_slug:
            return self.by_slug[slug]
        if 0 <= idx < len(self.seq):
            return self.seq[idx]
        return ""


PROMPTS = PromptBook(TASKS_JSONL_PATH)


# ---------------- Order & groups loader ----------------
def load_order_and_groups(total_envs: int, levels: List[str]):
    """
    Build ordered_indices by concatenating per_group_indices[level] in the order
    given by `levels` (e.g. ['easy','medium','hard','very_hard'] or ['hard']).

    Returns:
        ordered_indices: flat list of env indices in level-then-slug order
        groups:          {level: set(slug)} (from JSON)
        idx_to_slug:     {env_idx: slug}
        per_level_idx:   {level: [env_idx, ...]} only for levels actually selected,
                         used by the main eval loop to print bucket boundary markers.
    """
    if os.path.exists(ORDER_JSON_PATH):
        with open(ORDER_JSON_PATH, "r") as f:
            data = json.load(f)

        groups = {k: set(v) for k, v in data["groups"].items()}
        idx_to_slug = {int(k): v for k, v in data["idx_to_slug"].items()}
        per_group_indices = {k: [int(i) for i in v] for k, v in data.get("per_group_indices", {}).items()}

        ordered_indices: List[int] = []
        per_level_idx: Dict[str, List[int]] = {}
        for lv in levels:
            bucket = [i for i in per_group_indices.get(lv, []) if 0 <= i < total_envs]
            per_level_idx[lv] = bucket
            ordered_indices.extend(bucket)

        print(f"[INFO] Loaded order from {ORDER_JSON_PATH}; levels={levels}; "
              f"total tasks={len(ordered_indices)} "
              f"({', '.join(f'{lv}={len(per_level_idx[lv])}' for lv in levels)})")
        log_write(f"[INFO] Metaworld Evaluation Begins ... levels={levels}")
        return ordered_indices, groups, idx_to_slug, per_level_idx

    # Fallback when mt50_order.json is missing
    if FALLBACK_IDX_LIST:
        idx_list = [i for i in FALLBACK_IDX_LIST if 0 <= i < total_envs]
    elif FALLBACK_USE_FIRST_N:
        idx_list = list(range(min(FALLBACK_USE_FIRST_N, total_envs)))
    else:
        idx_list = list(range(total_envs))
    print("[WARN] mt50_order.json not found; falling back to:", idx_list)

    idx_to_slug = {i: f"task-{i}" for i in idx_list}
    groups = {"easy": set(), "medium": set(), "hard": set(), "very_hard": set()}
    per_level_idx = {lv: [] for lv in levels}
    # Put everything into the first selected bucket so the eval still runs end-to-end.
    if levels:
        per_level_idx[levels[0]] = list(idx_list)
    return idx_list, groups, idx_to_slug, per_level_idx


# ---------------- Core eval (MT50 only, ordered by mt50_order.json) ----------------
async def eval_mt50_with_groups(server_url: str,
                                num_eval_episodes: int = EPISODES,
                                episode_horizon: int = EPISODE_HORIZON,
                                seed: int = SEED,
                                action_horizon: int = HORIZON,
                                levels: Optional[List[str]] = None):

    if levels is None:
        levels = ["easy", "medium", "hard", "very_hard"]

    # 1) Build MT50 with fixed camera
    envs = gym.make_vec(
        "Meta-World/MT50",
        vector_strategy="sync",
        seed=seed,
        render_mode="rgb_array",
        camera_name=CAMERA_NAME,
    )
    total_envs = len(envs.envs)

    # 2) Load ordered idx list & groups (grouped by user-specified bucket order)
    ordered_indices, groups, idx_to_slug, per_level_idx = load_order_and_groups(total_envs, levels)

    # 3) Accumulators
    success_counts: Dict[int, int] = {i: 0 for i in ordered_indices}
    trials_counts: Dict[int, int] = {i: 0 for i in ordered_indices}
    group_success = {k: 0 for k in ["easy", "medium", "hard", "very_hard"]}
    group_trials  = {k: 0 for k in ["easy", "medium", "hard", "very_hard"]}

    # 4) Main loop — iterate per bucket so we can print explicit boundary markers
    async with websockets.connect(server_url, max_size=100_000_000) as ws:
        for level in levels:
            bucket_indices = per_level_idx.get(level, [])
            if not bucket_indices:
                log_write(f"\n==== Skip bucket: {level} (0 tasks) ====")
                continue

            log_write(f"\n==== Begin bucket: {level} ({len(bucket_indices)} tasks) ====")

            for idx in bucket_indices:
                sub = envs.envs[idx]
                slug = idx_to_slug.get(idx, f"task-{idx}")

                task_prompt = PROMPTS.get(idx, slug=slug)

                # gname_for_task should equal `level` here, but resolve via slug to
                # stay correct if mt50_order.json's groups and per_group_indices disagree.
                gname_for_task = None
                for gname in group_trials.keys():
                    if slug in groups.get(gname, set()):
                        gname_for_task = gname
                        break
                if gname_for_task is None:
                    gname_for_task = level

                for ep in range(num_eval_episodes):
                    for obj in (sub, getattr(sub, "unwrapped", None)):
                        fn = getattr(obj, "iterate_goal_position", None)
                        if callable(fn):
                            try: fn()
                            except Exception: pass
                            break

                    inspect_choice = INSPECT_SAMPLE_PER_EPISODE
                    saved_this_episode = False

                    obs, _ = sub.reset(seed=seed + ep)
                    trials_counts[idx] += 1
                    if gname_for_task is not None:
                        group_trials[gname_for_task] += 1

                    steps = 0
                    done = False
                    video_name = f"task{idx:02d}_{slug}_ep{ep+1:03d}.mp4"
                    frames: Optional[list] = [] if SAVE_VIDEO else None
                    if SAVE_VIDEO:
                        # Probe frame as first frame (keeps prior behavior where the
                        # writer was primed with one render before step(a0)).
                        append_frame(frames, render_single_bgr(sub))

                    try:
                        a0 = np.zeros(sub.action_space.shape, dtype=np.float32)
                        a0 = np.clip(a0, sub.action_space.low, sub.action_space.high)
                        obs, _, _, _, _ = sub.step(a0)
                    except Exception:
                        pass

                    while steps < episode_horizon and not done:
                        img_bgr = render_single_bgr(sub)

                        if SAVE_VIDEO:
                            append_frame(frames, img_bgr)

                        if SAVE_IMAGE and inspect_choice and (not saved_this_episode):
                            save_sent_bgr_frame(
                                img_bgr, ep_num=ep + 1, idx=idx, slug=slug,
                                step=steps if INSPECT_SAVE_STEP_TAG else None
                            )
                            saved_this_episode = True

                        state_vec = obs_to_state(obs)

                        actions = await evo1_infer(ws, img_bgr, state_vec, prompt=task_prompt)

                        for i in range(action_horizon):
                            a4 = np.asarray(actions[i][:4], dtype=np.float32)
                            a4 = np.clip(a4, sub.action_space.low, sub.action_space.high)
                            obs, _, terminated, truncated, info = sub.step(a4)
                            steps += 1

                            if isinstance(info, dict) and info.get("success", 0) == 1:
                                success_counts[idx] += 1
                                if gname_for_task is not None:
                                    group_success[gname_for_task] += 1
                                done = True
                                break

                            if terminated or truncated or steps >= episode_horizon:
                                done = True
                                break

                    # encode + copy video back to OSSFS
                    if done and SAVE_VIDEO:
                        append_frame(frames, render_single_bgr(sub))
                        save_video(frames, video_name, idx, slug, ep + 1)

                s = success_counts[idx]
                t = trials_counts[idx]
                task_rate = s / max(1, t)
                msg = (f"[Task {idx} {slug}] {task_prompt} finished {num_eval_episodes} episodes -> "
                       f"success_rate={task_rate:.3f}  (s={s}, t={t})")
                log_write(msg)

            # ---- bucket summary ----
            bs = group_success[level]
            bt = group_trials[level]
            brate = (bs / bt) if bt > 0 else 0.0
            log_write(f"==== Bucket {level} done: rate={brate:.3f}  (s={bs}, t={bt}) ====")

    envs.close()

    # 5) Build metrics — only over levels actually evaluated
    per_task: Dict[str, float] = {}
    for idx in ordered_indices:
        slug = idx_to_slug.get(idx, f"task-{idx}")
        s, t = success_counts[idx], trials_counts[idx]
        per_task[slug] = (s / t) if t > 0 else 0.0

    per_group: Dict[str, float] = {}
    for gname in levels:
        s, t = group_success.get(gname, 0), group_trials.get(gname, 0)
        per_group[gname] = (s / t) if t > 0 else 0.0

    # Overall = mean of evaluated bucket rates (matches LIBERO-style 4-bucket mean
    # when all four are run; degrades gracefully to subset mean otherwise).
    evaluated_rates = [per_group[g] for g in levels if group_trials.get(g, 0) > 0]
    overall = (sum(evaluated_rates) / len(evaluated_rates)) if evaluated_rates else 0.0

    return per_task, per_group, overall


# ---------------- Entrypoint ----------------
async def _amain():
    args = parse_args()

    # Resolve output dir (relative paths are anchored to the script dir so the file
    # layout is stable regardless of cwd), then derive everything under it.
    global LOG_DIR, LOG_PATH, VIDEO_SAVE_DIR, INSPECT_DIR, SERVER_URL
    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(script_dir, args.out_dir)
    run_dir = os.path.abspath(run_dir)
    LOG_DIR = os.path.join(run_dir, "logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_PATH = os.path.join(LOG_DIR, f"mt50_{ts}.txt")
    VIDEO_SAVE_DIR = os.path.join(run_dir, "episode_videos", f"mt50_{ts}_videos")
    INSPECT_DIR = os.path.join(run_dir, "inspect_frames")
    SERVER_URL = f"ws://127.0.0.1:{args.port}"

    log_write(f"==== Run config ====")
    log_write(f"out_dir: {run_dir}")
    log_write(f"server_url: {SERVER_URL}")
    log_write(f"horizon: {args.horizon}  seed: {args.seed}  levels: {','.join(args.levels)}\n")

    per_task, per_group, overall = await eval_mt50_with_groups(
        server_url=SERVER_URL,
        num_eval_episodes=EPISODES,
        episode_horizon=EPISODE_HORIZON,
        seed=args.seed,
        action_horizon=args.horizon,
        levels=args.levels,
    )


    # Overall = mean of bucket rates for buckets actually evaluated.
    avg = overall

    # log
    log_write(f"\n==== Evaluation Log ====\nLog file: {LOG_PATH}")
    log_write(f"Levels evaluated: {','.join(args.levels)}")
    log_write(f"Server URL: {SERVER_URL}")
    log_write(f"Episodes per task: {EPISODES}")
    log_write(f"Episode horizon: {EPISODE_HORIZON}")
    log_write(f"HORIZON: {args.horizon}")
    log_write(f"Seed: {args.seed}\n")

    log_write("==== Per-task success rate ====")
    for slug, rate in per_task.items():
        log_write(f"{slug:24s}  {rate:.3f}")

    log_write("\n==== Difficulty buckets ====")
    for lv in args.levels:
        log_write(f"{lv:10s}: {per_group.get(lv, 0.0):.3f}")

    log_write(f"\n==== Overall Average as Success Rate (mean over evaluated buckets) ====\n{avg:.3f}")

if __name__ == "__main__":
    asyncio.run(_amain())



# if __name__ == "__main__":
#     N_REPEAT = 1
#     for run_id in range(N_REPEAT):
#         print(f"\n\n=====  Run {run_id + 1}/{N_REPEAT} =====")
#         asyncio.run(_amain())