"""
Quantitative metric for IWS world models, over held-out GT action replay.

Per held-out episode we replay the GT actions through the world model and report:

1) mscore (signed temporal-change agreement) — custom single metric, no clustering.
   dg_t = GT_t - GT_{t-1},  dp_t = Pred_t - Pred_{t-1}   (signed, per channel -> direction)
   a_t  = ||normalize(act_t) - normalize(act_{t-1})||     action weight
   mscore = sum_t a_t sum_p |dg_t - dp_t|  /  sum_t a_t sum_p |dg_t|
   Pred should change the same pixels in the same DIRECTION and amount as GT.
   Opposite-direction change penalized hardest; background (dg~0) auto-ignored.

2) Standard video metrics (same as validation): fvd, mse, psnr, ssim, uiqi

Lower is better except ssim/uiqi/psnr. Compares multiple checkpoints in one table.

Usage:
    python scripts/inference/play_single_eef_metric.py \
        --ckpts run_a/best.ckpt run_b/best.ckpt \
        --dataset_dir data/play_robot_v3_hdf5 --split val --episodes 0 --max_frames 200
"""

import argparse
import os
import sys

# limit BLAS/OpenMP threads before numpy is imported (servers with many cores can
# exceed OpenBLAS's compiled thread cap and crash). setdefault -> user can override.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from play_single_eef_inference import (  # noqa: E402
    decode_latent,
    encode_frame,
    load_model,
)
from torchmetrics.functional import (  # noqa: E402
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
    universal_image_quality_index,
)

# checkpoint configs use ${eval:...} / ${torch:...}
OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {"np": np}), replace=True)
OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x), replace=True)

# video metrics where higher is better (for display only)
HIGHER_BETTER = {"psnr", "ssim", "uiqi"}


def load_episode_split(dataset_dir: str, split: str, idx: int):
    path = os.path.join(dataset_dir, split, f"episode_{idx}.hdf5")
    with h5py.File(path, "r") as f:
        cam0 = f["obs"]["images"]["camera_0_color"][:]   # (T, H, W, 3) RGB uint8
        cam1 = f["obs"]["images"]["camera_1_color"][:]
        actions = f["action"][:].astype(np.float32)       # (T, 7)
    return cam0, cam1, actions


@torch.no_grad()
def replay_rollout(model, cam0, cam1, actions, h, device, dtype):
    """GT action replay. Returns (preds0, preds1) aligned to t=1..T-1."""
    normalizer = model.normalizer
    num_views = len(model.obs_keys)
    n_tokens = model.n_tokens

    z0 = encode_frame(model, normalizer, cam0[0], cam1[0], device, dtype)
    curr_latent = z0.unsqueeze(1)
    past = normalizer["action"].normalize(
        torch.from_numpy(actions[0]).to(device=device, dtype=dtype).unsqueeze(0)
    ).unsqueeze(0)

    preds0, preds1 = [], []
    for t in range(1, len(actions)):
        fut = normalizer["action"].normalize(
            torch.from_numpy(actions[t]).to(device=device, dtype=dtype).unsqueeze(0)
        ).unsqueeze(0)
        z_pred = model.dynamics_forward(curr_latent, torch.cat([past, fut], dim=1))
        frames = decode_latent(model, z_pred[:, -1], h, normalizer, num_views)
        preds0.append(frames[0])
        preds1.append(frames[1] if num_views > 1 else frames[0])
        curr_latent = torch.cat([curr_latent, z_pred[:, -1:]], dim=1)[:, -n_tokens:]
        past = torch.cat([past, fut], dim=1)[:, -n_tokens:]
    return preds0, preds1


def motion_score(preds0, cam0, actions, model, device, dtype, eps=5.0 / 255.0):
    """Signed temporal-change agreement over GT action replay. Lower = better.

    Returns (mscore, track, hall), where mscore = track + hall (shared denominator).
      dg_t = GT_t - GT_{t-1},  dp_t = Pred_t - Pred_{t-1}   (signed, per channel)
      a_t  = || normalize(act_t) - normalize(act_{t-1}) ||
      den  = sum_t a_t sum_p |dg_t|

    Split each pixel by whether GT changed there (|dg| > eps):
      track = sum a_t sum_{|dg|>eps} |dg - dp|   -> tracking/miss (failed to follow GT motion)
      hall  = sum a_t sum_{|dg|<=eps} |dp|       -> hallucination (invented motion where GT static)
    eps = noise floor so camera/codec jitter isn't counted as GT change.
    """
    a_norm = model.normalizer["action"].normalize(
        torch.from_numpy(actions).to(device=device, dtype=dtype)
    ).cpu().numpy()
    num_track = num_hall = den = 0.0
    prev_g = cam0[0].astype(np.float32) / 255.0
    prev_p = None
    for i, pred in enumerate(preds0):
        t = i + 1
        g = cam0[t].astype(np.float32) / 255.0
        p = pred.astype(np.float32) / 255.0
        if prev_p is None:               # first frame has no previous pred -> skip
            prev_g, prev_p = g, p
            continue
        a_t = float(np.linalg.norm(a_norm[t] - a_norm[t - 1]))
        dg = g - prev_g                              # (H,W,3) signed GT change
        dp = p - prev_p                              # (H,W,3) signed pred change
        dg_mag = np.abs(dg).mean(-1)                 # (H,W)
        diff = np.abs(dg - dp).mean(-1)              # signed-change disagreement
        dp_mag = np.abs(dp).mean(-1)
        moved = dg_mag > eps                         # GT actually changed here
        num_track += a_t * float((diff * moved).sum())       # tracking error in GT-changed region
        num_hall += a_t * float((dp_mag * ~moved).sum())     # pred change in GT-static region
        den += a_t * float(dg_mag.sum())
        prev_g, prev_p = g, p
    if den <= 0:
        return float("nan"), float("nan"), float("nan")
    return (num_track + num_hall) / den, num_track / den, num_hall / den


def _to_video(frames0, frames1):
    """list of (H,W,3) uint8 x2 -> CPU (T,1,6,H,W) float in [-1,1]."""
    vid = [np.concatenate([f0, f1], axis=2) for f0, f1 in zip(frames0, frames1)]
    vid = np.stack(vid, 0).astype(np.float32) / 127.5 - 1.0  # (T,H,W,6)
    return torch.from_numpy(vid).permute(0, 3, 1, 2).unsqueeze(1)  # CPU tensor


@torch.no_grad()
def video_metrics(model, preds0, preds1, cam0, cam1, device, chunk=64):
    """Standard validation video metrics (fvd, mse, psnr, ssim, uiqi).

    Full video is kept on CPU; only one chunk is moved to GPU at a time. Image metrics
    are frame-weighted means (identical to whole-clip). fvd is computed per view in
    frame chunks fed to fvd_model.update, then a single compute().
    """
    T = len(preds0)
    pred_vid = _to_video(preds0, preds1)                        # CPU (T,1,C,H,W)
    gt_vid = _to_video(list(cam0[1:T + 1]), list(cam1[1:T + 1]))
    C, H, W = pred_vid.shape[2], pred_vid.shape[3], pred_vid.shape[4]
    p_flat = pred_vid.reshape(T, C, H, W)
    g_flat = gt_vid.reshape(T, C, H, W)

    # image metrics in frame chunks (frame-weighted average); only chunk on GPU
    acc = {"mse": 0.0, "psnr": 0.0, "ssim": 0.0, "uiqi": 0.0}
    n = 0
    for i in range(0, T, chunk):
        p = p_flat[i:i + chunk].to(device)
        g = g_flat[i:i + chunk].to(device)
        bs = p.shape[0]
        acc["mse"] += float(mean_squared_error(p, g)) * bs
        acc["psnr"] += float(peak_signal_noise_ratio(p, g, data_range=2.0)) * bs
        acc["ssim"] += float(structural_similarity_index_measure(p, g, data_range=2.0)) * bs
        acc["uiqi"] += float(universal_image_quality_index(p, g)) * bs
        n += bs
    out = {k: acc[k] / n for k in acc}

    # fvd per view, full clip but moved to GPU one view at a time (requires >= 9 frames)
    fvd_model = getattr(model, "validation_fvd_model", None)
    if fvd_model is not None and T >= 9:
        n_views = C // 3
        vals = []
        for v in range(n_views):
            hat = torch.clamp(pred_vid[:, :, v * 3:(v + 1) * 3], -1.0, 1.0).to(device)
            ref = torch.clamp(gt_vid[:, :, v * 3:(v + 1) * 3], -1.0, 1.0).to(device)
            fvd_v = fvd_model.compute(hat, ref)
            vals.append(fvd_v.item() if isinstance(fvd_v, torch.Tensor) else float(fvd_v))
            del hat, ref
            torch.cuda.empty_cache()
        out["fvd"] = sum(vals) / len(vals)
    return out


def main():
    ap = argparse.ArgumentParser(description="Motion-weighted score + video metrics over held-out GT action replay")
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--dataset_dir", default="data/play_robot_v3_hdf5")
    ap.add_argument("--split", default="val")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0])
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--dec_infer_steps", type=int, default=3)
    ap.add_argument("--max_frames", type=int, default=None,
                    help="Use only the first N frames of each episode (e.g. 200 ~= 20s @ 10Hz)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    h = args.resolution

    eps_data = {}
    for ep in args.episodes:
        cam0, cam1, actions = load_episode_split(args.dataset_dir, args.split, ep)
        if args.max_frames:
            cam0, cam1, actions = cam0[:args.max_frames], cam1[:args.max_frames], actions[:args.max_frames]
        eps_data[ep] = (cam0, cam1, actions)

    # metric columns: our motion_score (mscore) + standard video metrics for reference
    cols = ["mscore", "track", "hall", "fvd", "mse", "psnr", "ssim", "uiqi"]
    results = {}
    for ckpt in args.ckpts:
        print(f"\nEvaluating {ckpt} ...")
        model = load_model(ckpt, args.device, args.dec_infer_steps).to(args.device)
        dtype = model.dtype
        acc = {c: [] for c in cols}
        for ep in args.episodes:
            cam0, cam1, actions = eps_data[ep]
            preds0, preds1 = replay_rollout(model, cam0, cam1, actions, h, args.device, dtype)
            mscore, track, hall = motion_score(preds0, cam0, actions, model, args.device, dtype)
            vm = video_metrics(model, preds0, preds1, cam0, cam1, args.device)

            row = {"mscore": mscore, "track": track, "hall": hall}
            for k in ["fvd", "mse", "psnr", "ssim", "uiqi"]:
                row[k] = vm.get(k, float("nan"))
            for c in cols:
                acc[c].append(row[c])
            print(f"  episode {ep}: " + "  ".join(f"{c}={row[c]:.4f}" for c in cols))
        results[ckpt] = {c: float(np.nanmean(acc[c])) for c in cols}
        del model
        torch.cuda.empty_cache()

    print("\n=== Metric table (mscore/mse lower=better; psnr/ssim/uiqi higher=better) ===")
    header = "  " + "".join(f"{c:>9}" for c in cols) + "   checkpoint"
    print(header)
    for ckpt, row in sorted(results.items(), key=lambda kv: kv[1]["mscore"]):
        line = "  " + "".join(f"{row[c]:9.3f}" for c in cols) + f"   {ckpt}"
        print(line)


if __name__ == "__main__":
    main()
