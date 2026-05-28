import json
import os
import numpy as np


def assemble_clip(frames, agent_mask, eef, obj_traj, embodiment_image, domain, meta) -> dict:
    """打包成 npz-ready dict，统一 dtype。"""
    return {
        "frames": np.asarray(frames, np.uint8),
        "agent_mask": np.asarray(agent_mask, bool),
        "eef": np.asarray(eef, np.float32),
        "obj_traj": np.asarray(obj_traj, np.float32),
        "embodiment_image": np.asarray(embodiment_image, np.uint8),
        "domain": np.asarray(domain),
        "meta": np.asarray(meta, dtype=object),
    }


def write_clip(rec: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **rec)


def append_index(index_path: str, entry: dict):
    idx = []
    if os.path.exists(index_path):
        with open(index_path) as f:
            idx = json.load(f)
    idx.append(entry)
    with open(index_path, "w") as f:
        json.dump(idx, f, indent=0)
