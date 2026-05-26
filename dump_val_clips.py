"""Stage A (run in `iws` env): dump human/robot validation clips to .npy.

The inpaint+DINO diagnostic runs in the `phantom` env (E2FGVI needs mmcv, which
is only in `phantom`), but the dataset loaders live in `iws`. So we split: this
script (iws) just materializes the raw val clips to disk; `inpaint_dino_subspace.py`
(phantom) consumes them. See docs/superpowers/specs/2026-05-26-inpaint-dino-subspace-design.md

Output: <out>/<domain>_clips.npy  uint8 (Nclips, T, C, H, W), native 128x128.

Usage: python dump_val_clips.py --per-domain 900 --out outputs/inpaint_dino_subspace/clips
"""
import argparse
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, "scripts/eval")
import eval_stage1_mixed_vit_alignment as ev
from interactive_world_sim.datasets.latent_dynamics.play_eef_dataset import PlayEEFDataset

CKPT = "outputs/2026-05-25/10-26-17_job45723_phase0_emb_film/checkpoints/last.ckpt"
KEY = "camera_0_color"


def build_val(domain, horizon):
    # build_domain_val_dataset uses the per-domain sub-config, whose val_horizon
    # defaults to horizon=4. Override it so clips are longer (better E2FGVI
    # temporal context). Crop (195,195,256,256)->128 is applied by the dataset.
    cfg = OmegaConf.load(ev.infer_config_path(CKPT))
    sub = OmegaConf.create(OmegaConf.to_container(cfg.dataset[domain], resolve=True))
    sub.val_horizon = horizon
    return PlayEEFDataset(sub).get_validation_dataset()


def dump_domain(domain, per_domain, horizon, out_dir):
    loader = DataLoader(build_val(domain, horizon), batch_size=4, num_workers=4)
    clips, seen = [], 0
    for batch in loader:
        c = batch["obs"][KEY].float()          # (B,T,C,H,W) in [0,1]
        for b in range(c.shape[0]):
            clips.append((c[b].numpy() * 255).astype(np.uint8))
            seen += c.shape[1]
        if seen >= per_domain:
            break
    arr = np.stack(clips, 0)
    p = os.path.join(out_dir, f"{domain}_clips.npy")
    np.save(p, arr)
    print(f"saved {p}  shape={arr.shape}  ({seen} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-domain", type=int, default=900)
    ap.add_argument("--horizon", type=int, default=16, help="frames per clip (E2FGVI context)")
    ap.add_argument("--out", default="outputs/inpaint_dino_subspace/clips")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    for domain in ("robot", "human"):
        dump_domain(domain, args.per_domain, args.horizon, args.out)


if __name__ == "__main__":
    main()
