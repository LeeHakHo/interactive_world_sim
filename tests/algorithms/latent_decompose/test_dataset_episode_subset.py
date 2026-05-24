"""PlayEEFDataset honours `episode_subset` (list[int] | None) to filter
the episodes that get loaded. None = all (current behaviour)."""
from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf


def _import_dataset():
    """Import PlayEEFDataset, skip the test if the module fails to load
    (e.g. numpy/numba binary incompatibility on this host)."""
    try:
        from interactive_world_sim.datasets.latent_dynamics.play_eef_dataset import (
            PlayEEFDataset,
        )
    except ImportError as e:
        pytest.skip(f"dataset module import failed: {e}")
    return PlayEEFDataset


@pytest.fixture
def base_cfg():
    root = Path("/scr2/yusenluo/interactive_world_sim/play_robot_eef/play_robot_1_eef")
    if not root.exists():
        pytest.skip("real EEF data not present on this host")
    return OmegaConf.create({
        "domain": "robot",
        "dataset_dirs": [str(root)],
        "video_keys": ["observation.images.cam_high"],
        "obs_keys": ["camera_0_color"],
        "crops": [[195, 195, 256, 256]],
        "resolution": 128, "horizon": 4, "val_horizon": 4,
        "skip_frame": 1, "pad_before": 0, "pad_after": 0,
        "val_ratio": 0.1, "seed": 42, "goal_sample": "intermediate",
        "skip_idx": 1, "cache_root": "/tmp/iws_test_cache_subset",
    })


def test_default_subset_loads_all(base_cfg):
    PlayEEFDataset = _import_dataset()
    ds = PlayEEFDataset(base_cfg)
    assert len(ds._episodes) > 0


def test_explicit_subset_filters(base_cfg):
    PlayEEFDataset = _import_dataset()
    cfg_full = OmegaConf.create(dict(base_cfg))
    full = PlayEEFDataset(cfg_full)
    full_idxs = [ep["episode_index"] for ep in full._episodes]
    assert len(full_idxs) >= 1
    target = [full_idxs[0]]

    cfg_sub = OmegaConf.create({**dict(base_cfg), "episode_subset": target})
    sub = PlayEEFDataset(cfg_sub)
    assert [ep["episode_index"] for ep in sub._episodes] == target


def test_empty_subset_loads_nothing(base_cfg):
    PlayEEFDataset = _import_dataset()
    cfg = OmegaConf.create({**dict(base_cfg), "episode_subset": []})
    ds = PlayEEFDataset(cfg)
    assert ds._episodes == []
