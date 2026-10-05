"""Offline RAFT optical-flow precompute for the Optical World Model.

For every episode video it renders a *flow video* whose frame i is the
RAFT optical flow current->future rendered with flow_to_image:

    flow_frame[i] = flow_to_image( RAFT( rgb[i], rgb[min(i + flow_gap, last)] ) )

The flow video has the same length / fps as the RGB video, so a dataset sample
at timestamp t (current frame i) reads its target by decoding the flow video at
the *same* timestamp. Output mirrors the LeRobot video layout:

    <out_root>/<arm>__<dataset>/<chunk>/<view_folder>/<episode>.mp4

Usage:
    python scripts/precompute_flow.py \
        --dataset_config_path dataset/config_meta_world.yaml \
        --flow_gap 15 --wm_view_key image_1 \
        --out_root /data/.../cache/evo_owm_flow_g15_metaworld
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

RAFT_ROOT = "/path/to/evo_workspace/RAFT/RAFT-master"
sys.path.append(os.path.join(RAFT_ROOT, "core"))
from raft import RAFT  # noqa: E402
from utils.utils import InputPadder  # noqa: E402
from utils import flow_viz  # noqa: E402

DEVICE = "cuda"


def build_raft(ckpt_path: str, small: bool = False):
    args = argparse.Namespace(small=small, mixed_precision=False, alternate_corr=False)
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    model = model.module.to(DEVICE).eval()
    return model


def decode_frames_and_fps(video_path: str):
    """Return (list[HxWx3 uint8 RGB], fps)."""
    try:
        import decord
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        fps = float(vr.get_avg_fps())
        frames = [vr[i].asnumpy() for i in range(len(vr))]
        return frames, fps
    except Exception:
        import av
        container = av.open(video_path)
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        container.close()
        return frames, fps


def encode_video_av(path: str, frames_thwc: np.ndarray, fps: int):
    """Write [T,H,W,3] uint8 RGB to an h264 mp4 via PyAV (robust across av versions).

    Pads odd H/W to even (yuv420p requirement); the dataset resizes to
    wm_image_size anyway, so a 1px edge pad is negligible.
    """
    import av

    T, H, W, _ = frames_thwc.shape
    H2, W2 = H + (H % 2), W + (W % 2)
    if (H2, W2) != (H, W):
        frames_thwc = np.pad(frames_thwc, ((0, 0), (0, H2 - H), (0, W2 - W), (0, 0)), mode="edge")

    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=int(fps))
    stream.width = W2
    stream.height = H2
    stream.pix_fmt = "yuv420p"
    try:
        for i in range(frames_thwc.shape[0]):
            vframe = av.VideoFrame.from_ndarray(np.ascontiguousarray(frames_thwc[i]), format="rgb24")
            for packet in stream.encode(vframe):
                container.mux(packet)
        for packet in stream.encode():  # flush
            container.mux(packet)
    finally:
        container.close()


def flow_to_image_fixed(flow: np.ndarray, max_mag: float) -> np.ndarray:
    """Map [H,W,2] flow -> RGB using a FIXED global scale (not the per-frame
    rad_max that stock flow_to_image uses). Identical displacement -> identical
    color across frames (temporally consistent) and absolute magnitude is
    preserved up to max_mag. Reuses RAFT's color wheel (flow_uv_to_colors).

    Pixels with magnitude > max_mag are capped to unit length (direction kept,
    fully saturated) instead of the stock 0.75 dimming.
    """
    u = flow[:, :, 0] / (max_mag + 1e-5)
    v = flow[:, :, 1] / (max_mag + 1e-5)
    rad = np.sqrt(u * u + v * v)
    scale = np.minimum(1.0, 1.0 / np.maximum(rad, 1e-8))  # only shrinks rad>1
    return flow_viz.flow_uv_to_colors(u * scale, v * scale)


@torch.no_grad()
def compute_raw_flows(model, frames, flow_gap: int, iters: int, batch_size: int):
    """frames: list of HxWx3 uint8 -> list of raw flow [H,W,2] float32 (cpu)."""
    n = len(frames)
    last = n - 1
    frames_t = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()  # [T,3,H,W], 0~255
    cur_idx = list(range(n))
    fut_idx = [min(i + flow_gap, last) for i in range(n)]

    flows = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        img1 = frames_t[cur_idx[start:end]].to(DEVICE)
        img2 = frames_t[fut_idx[start:end]].to(DEVICE)
        padder = InputPadder(img1.shape)
        img1p, img2p = padder.pad(img1, img2)
        _, flow_up = model(img1p, img2p, iters=iters, test_mode=True)
        flow_up = padder.unpad(flow_up)  # [B,2,H,W]
        for b in range(flow_up.shape[0]):
            flows.append(flow_up[b].permute(1, 2, 0).cpu().numpy().astype(np.float32))
    return flows  # list of [H,W,2]


def compute_flow_video(model, frames, flow_gap: int, iters: int, batch_size: int,
                       max_mag=None) -> np.ndarray:
    """frames -> flow video [T,H,W,3] uint8.

    max_mag=None -> stock per-frame normalization (legacy, jittery).
    max_mag=float -> fixed global-scale rendering (temporally consistent).
    """
    flows = compute_raw_flows(model, frames, flow_gap, iters, batch_size)
    if max_mag is None:
        out = [flow_viz.flow_to_image(f) for f in flows]
    else:
        out = [flow_to_image_fixed(f, max_mag) for f in flows]
    return np.stack(out)  # [T,H,W,3]


def calibrate_max_mag(model, rgb_videos, flow_gap, iters, batch_size,
                      calib_episodes: int, percentile: float, pixel_stride: int = 41) -> float:
    """Estimate one global flow magnitude scale from a sample of episodes.

    Returns the given percentile of per-pixel flow magnitude (subsampled), which
    ignores the static-background zeros and a few outlier pixels -> a robust
    scale for the moving regions.
    """
    samples = []
    for vp in rgb_videos[:calib_episodes]:
        frames, _ = decode_frames_and_fps(str(vp))
        if not frames:
            continue
        for f in compute_raw_flows(model, frames, flow_gap, iters, batch_size):
            rad = np.sqrt((f ** 2).sum(-1)).ravel()
            samples.append(rad[::pixel_stride])  # subsample to bound memory
    if not samples:
        raise RuntimeError("Calibration produced no flow samples.")
    allmag = np.concatenate(samples)
    return float(np.percentile(allmag, percentile))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_config_path", required=True)
    ap.add_argument("--flow_gap", type=int, default=50)
    ap.add_argument("--wm_view_key", default="image_1")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--raft_ckpt", default=os.path.join(RAFT_ROOT, "raft-things.pth"))
    ap.add_argument("--raft_iters", type=int, default=20)
    ap.add_argument("--raft_small", action="store_true")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_episodes", type=int, default=None, help="cap episodes per dataset (smoke test)")
    # Fixed-scale flow rendering (default) vs legacy per-frame normalization
    ap.add_argument("--flow_max_mag", type=float, default=None,
                    help="Fixed global flow-magnitude scale for RGB rendering. If unset, auto-calibrated.")
    ap.add_argument("--calib_episodes", type=int, default=5, help="episodes sampled to auto-calibrate flow_max_mag")
    ap.add_argument("--calib_percentile", type=float, default=99.0, help="magnitude percentile used as the scale")
    ap.add_argument("--per_frame_norm", action="store_true",
                    help="Legacy: stock per-frame rad_max normalization (jittery). Overrides fixed scale.")
    args = ap.parse_args()

    with open(args.dataset_config_path, "r") as f:
        data_cfg = yaml.safe_load(f)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    model = build_raft(args.raft_ckpt, small=args.raft_small)
    print(f"RAFT loaded ({args.raft_ckpt}); flow_gap={args.flow_gap}, iters={args.raft_iters}")

    datasets_done = []
    max_mag_per_dataset = {}
    for arm_name, arm_cfg in data_cfg["data_groups"].items():
        for dataset_name, dcfg in arm_cfg.items():
            dataset_path = Path(dcfg["path"])
            view_map = dcfg.get("view_map") or {"image_1": "observation.images.image"}
            if args.wm_view_key not in view_map:
                raise KeyError(f"wm_view_key={args.wm_view_key} not in view_map {list(view_map)}")
            view_folder = view_map[args.wm_view_key]

            rgb_videos = sorted((dataset_path / "videos").glob(f"*/{view_folder}/*.mp4"))
            if args.max_episodes is not None:
                rgb_videos = rgb_videos[: args.max_episodes]
            if not rgb_videos:
                print(f"  [WARN] no videos under {dataset_path}/videos/*/{view_folder}")
                continue

            ds_key = f"{arm_name}__{dataset_name}"

            # Determine the fixed render scale for this dataset.
            if args.per_frame_norm:
                max_mag = None
                print(f"[{ds_key}] using LEGACY per-frame normalization")
            elif args.flow_max_mag is not None:
                max_mag = float(args.flow_max_mag)
                print(f"[{ds_key}] using fixed flow_max_mag={max_mag:.3f} (user-set)")
            else:
                print(f"[{ds_key}] calibrating flow_max_mag on {args.calib_episodes} episodes...")
                max_mag = calibrate_max_mag(
                    model, rgb_videos, args.flow_gap, args.raft_iters, args.batch_size,
                    args.calib_episodes, args.calib_percentile,
                )
                print(f"[{ds_key}] auto-calibrated flow_max_mag={max_mag:.3f} "
                      f"(p{args.calib_percentile:g} of magnitude, gap={args.flow_gap})")
            max_mag_per_dataset[ds_key] = max_mag

            print(f"[{ds_key}] {len(rgb_videos)} episodes")
            for vp in tqdm(rgb_videos, desc=ds_key):
                chunk = vp.parent.parent.name
                out_path = out_root / ds_key / chunk / view_folder / vp.name
                if out_path.exists():
                    continue
                out_path.parent.mkdir(parents=True, exist_ok=True)

                frames, fps = decode_frames_and_fps(str(vp))
                if len(frames) == 0:
                    print(f"    [WARN] empty video {vp}")
                    continue
                flow_video = compute_flow_video(
                    model, frames, args.flow_gap, args.raft_iters, args.batch_size,
                    max_mag=max_mag,
                )
                # PyAV muxing needs seek-based writes, which object-storage
                # mounts (ossfs2) reject. Encode to a LOCAL temp file, then copy
                # sequentially to the (possibly OSS) destination.
                import tempfile, shutil
                fd, local_tmp = tempfile.mkstemp(suffix=".mp4")
                os.close(fd)
                try:
                    encode_video_av(local_tmp, flow_video, fps=round(fps))
                    shutil.copyfile(local_tmp, out_path)
                finally:
                    if os.path.exists(local_tmp):
                        os.remove(local_tmp)
            datasets_done.append(ds_key)

    manifest = {
        "flow_gap": args.flow_gap,
        "wm_view_key": args.wm_view_key,
        "raft_ckpt": args.raft_ckpt,
        "raft_iters": args.raft_iters,
        "raft_small": args.raft_small,
        "datasets": datasets_done,
        "render_norm": "per_frame" if args.per_frame_norm else "fixed",
        "calib_percentile": None if args.per_frame_norm else args.calib_percentile,
        "flow_max_mag_per_dataset": max_mag_per_dataset,
    }
    with open(out_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone. Flow cache at: {out_root}")
    print(f"Manifest: {out_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
