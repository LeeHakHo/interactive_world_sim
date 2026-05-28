"""Mask QC on REAL frames — verify whole_arm_mask covers the whole arm (not just
hand) for human and the full arm/gripper for robot, with quantitative gates.

Gates (per the plan, to avoid silent mask failure):
  - median coverage fraction in [0.04, 0.30] per domain (not ~0.5% = hand-only)
  - border-touch rate > 0.8 (a limb enters from off-frame)
  - blue plate / cube not masked (checked visually in the overlay grid)

Outputs: outputs/mask_subtract_diag/{mask_qc_human.png, mask_qc_robot.png,
mask_qc_stats.json}.
"""
import json
import os

import numpy as np
import torch
from omegaconf import OmegaConf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from interactive_world_sim.datasets.latent_dynamics.play_eef_dataset import PlayEEFDataset
from interactive_world_sim.algorithms.latent_decompose.agent_mask import whole_arm_mask

OUT = "outputs/mask_subtract_diag"
os.makedirs(OUT, exist_ok=True)
CFG_DIR = "configurations/dataset"


def build_dataset(name: str) -> PlayEEFDataset:
    base = OmegaConf.load(f"{CFG_DIR}/base_dataset.yaml")
    ds = OmegaConf.load(f"{CFG_DIR}/{name}.yaml")
    ds.pop("defaults", None)
    ds.pop("_target_", None)
    merged = OmegaConf.merge(base, ds)
    root = OmegaConf.create({"debug": False, "dataset": merged})
    return PlayEEFDataset(root.dataset)


def collect_frames(ds: PlayEEFDataset, n_items: int = 8):
    """Pull frames from several items → (N,3,H,W) float [0,1]."""
    key = ds.obs_keys[0]
    frames = []
    step = max(1, len(ds) // n_items)
    for i in range(0, len(ds), step):
        item = ds[i]
        f = item["obs"][key]  # (T,3,H,W) float [0,1]
        frames.append(f)
        if len(frames) >= n_items:
            break
    return torch.cat(frames, dim=0)  # (N,3,H,W)


def qc_domain(name: str, embodiment: str):
    ds = build_dataset(name)
    frames = collect_frames(ds, n_items=8)  # (N,3,H,W)
    masks = whole_arm_mask(frames, embodiment)  # (N,1,H,W)
    cov = masks.mean(dim=(1, 2, 3)).numpy()  # per-frame coverage

    # border-touch rate
    border_touch = []
    for i in range(masks.shape[0]):
        m = masks[i, 0].numpy() > 0.5
        touch = m[0, :].any() or m[-1, :].any() or m[:, 0].any() or m[:, -1].any()
        border_touch.append(bool(touch))
    border_rate = float(np.mean(border_touch))

    stats = {
        "embodiment": embodiment,
        "n_frames": int(frames.shape[0]),
        "coverage_median": float(np.median(cov)),
        "coverage_mean": float(np.mean(cov)),
        "coverage_min": float(np.min(cov)),
        "coverage_max": float(np.max(cov)),
        "border_touch_rate": border_rate,
        "gate_coverage_ok": bool(0.04 <= np.median(cov) <= 0.30),
        "gate_border_ok": bool(border_rate > 0.8),
    }

    # overlay grid: red where mask, on top of the frame
    n = min(24, frames.shape[0])
    cols = 6
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
    axes = np.array(axes).reshape(-1)
    for i in range(rows * cols):
        ax = axes[i]
        ax.axis("off")
        if i >= n:
            continue
        img = frames[i].permute(1, 2, 0).numpy().clip(0, 1).copy()
        m = masks[i, 0].numpy() > 0.5
        ov = img.copy()
        ov[m] = 0.5 * ov[m] + 0.5 * np.array([1.0, 0.0, 0.0])
        ax.imshow(ov)
        ax.set_title(f"cov={cov[i]:.2f} b={int(border_touch[i])}", fontsize=6)
    fig.suptitle(f"{embodiment} whole_arm_mask (red=mask)  "
                 f"med_cov={stats['coverage_median']:.3f} "
                 f"border={border_rate:.2f}", fontsize=10)
    fig.tight_layout()
    path = f"{OUT}/mask_qc_{embodiment}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[{embodiment}] {stats}  -> {path}")
    return stats


if __name__ == "__main__":
    all_stats = {}
    all_stats["robot"] = qc_domain("play_robot_eef", "robot")
    all_stats["human"] = qc_domain("play_human_eef", "human")
    with open(f"{OUT}/mask_qc_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print("\nGATES:")
    for emb, s in all_stats.items():
        ok = s["gate_coverage_ok"] and s["gate_border_ok"]
        print(f"  {emb}: coverage_ok={s['gate_coverage_ok']} "
              f"border_ok={s['gate_border_ok']} -> {'PASS' if ok else 'FAIL'}")
