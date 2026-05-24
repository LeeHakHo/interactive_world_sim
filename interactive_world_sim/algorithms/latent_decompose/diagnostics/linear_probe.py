"""Linear probe diagnostic — Phase 0 Step 2.

Pass thresholds (PHASE0_IWS_PLAN_v3.md §2.1):
    --target z_task → val accuracy ∈ [50%, 70%]
    --target z_emb  → val accuracy ≥ 95%
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf


def train_probe_mlp(
    feats: torch.Tensor,
    labels: torch.Tensor,
    hidden: int = 128,
    epochs: int = 10,
    lr: float = 1e-3,
    val_ratio: float = 0.2,
    device: torch.device | None = None,
) -> float:
    """Train an MLP probe on (feats, labels). Return val accuracy."""
    device = device or torch.device("cpu")
    feats = feats.to(device)
    labels = labels.to(device).long()
    n = feats.shape[0]
    perm = torch.randperm(n, device=device)
    n_val = int(round(n * val_ratio))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    d_in = feats.shape[1]
    model = nn.Sequential(
        nn.Linear(d_in, hidden), nn.GELU(),
        nn.Linear(hidden, 2),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        logits = model(feats[train_idx])
        loss = F.cross_entropy(logits, labels[train_idx])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(feats[val_idx]).argmax(-1)
        acc = (pred == labels[val_idx]).float().mean().item()
    return acc


def _load_lwm(ckpt_path: Path, device: torch.device):
    """Load a frozen LatentWorldModel from a Lightning ckpt directory."""
    from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import (
        LatentWorldModel,
    )
    run_dir = ckpt_path.parent.parent       # ckpt/<...>.ckpt → run_dir
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    m = LatentWorldModel.load_from_checkpoint(
        str(ckpt_path), cfg=cfg.algorithm, map_location=device, weights_only=False,
    )
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, cfg


def _collect_features(
    m, cfg, target: Literal["z_task", "z_emb"], device: torch.device,
):
    """Run val loader, collect (target_pooled, domain_label) pairs."""
    import hydra
    val_dataset = hydra.utils.instantiate(cfg.dataset).get_validation_dataset()
    from torch.utils.data import DataLoader
    loader = DataLoader(val_dataset, batch_size=8, num_workers=0)

    feats, labels = [], []
    obs_key = cfg.dataset.obs_keys[0]
    for batch in loader:
        if "domain_label" not in batch:
            continue
        x = batch["obs"][obs_key].to(device)              # (B, T, 3, H, W)
        B, T = x.shape[:2]
        x = x.reshape(B * T, 3, x.shape[-2], x.shape[-1])
        with torch.no_grad():
            _, z_task_list, z_emb_list = m.encoder_forward(x, return_split=True)
        z_list = z_task_list if target == "z_task" else z_emb_list
        z = torch.cat(z_list, dim=0)                       # (V*B*T, D, gh, gw)
        z_pooled = z.mean(dim=(2, 3))                      # (V*B*T, D)
        feats.append(z_pooled.cpu())
        labels.append(
            batch["domain_label"].repeat_interleave(T).repeat(len(z_list))
        )
    return torch.cat(feats), torch.cat(labels)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--target", choices=("z_task", "z_emb"), required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    m, cfg = _load_lwm(args.ckpt, device)
    feats, labels = _collect_features(m, cfg, args.target, device)
    print(f"[probe] collected {feats.shape[0]} samples, dim={feats.shape[1]}")
    accs = []
    for s in range(args.seeds):
        torch.manual_seed(s); np.random.seed(s)
        acc = train_probe_mlp(feats, labels, device=device)
        accs.append(acc)
        print(f"  seed={s}: acc={acc:.4f}")
    mean = float(np.mean(accs)); std = float(np.std(accs))
    print(f"[probe] target={args.target}: {mean:.4f} ± {std:.4f}")

    if args.target == "z_task":
        ok = 0.50 <= mean <= 0.70
    else:
        ok = mean >= 0.95
    print(f"[probe] {'PASS' if ok else 'FAIL'} per Phase 0 §2.1")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
