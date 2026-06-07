"""Compute the global locked-plane height z_plane from the robot v3 datasets.

Uses action_right_ee_position[:,2] (the COMMANDED, truly-locked z; std~3e-4)
averaged over all robot eps. Caches to z_plane_v3.json so build is deterministic.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

import aytsai_pipeline_config as cfg


def compute(version: str = "v3") -> dict:
    c = cfg.get(version)
    zs = []
    per_ep = {}
    for n in c["robot_eps"]:
        repo = c["robot_repo_pattern"].format(n=n)
        p = hf_hub_download(repo, "data/chunk-000/episode_000000.parquet", repo_type="dataset")
        df = pd.read_parquet(p)
        z = np.stack(df["action_right_ee_position"].to_numpy())[:, 2]
        per_ep[repo] = dict(mean=float(z.mean()), std=float(z.std()), n=int(z.size))
        zs.append(z)
    allz = np.concatenate(zs)
    out = dict(
        version=version,
        z_plane=float(allz.mean()),
        z_plane_std=float(allz.std()),
        n_frames=int(allz.size),
        source_column="action_right_ee_position[:,2]",
        per_ep=per_ep,
    )
    Path(f"z_plane_{version}.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    o = compute("v3")
    print(f"z_plane = {o['z_plane']:.5f} m  (std {o['z_plane_std']:.5f}, n={o['n_frames']})")
