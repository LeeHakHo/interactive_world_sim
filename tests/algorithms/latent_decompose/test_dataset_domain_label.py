"""MixedPlayEEFDataset must emit a `domain_label` int tensor matching
`embodiment`: human=0, robot=1. The PyTorch default collate stacks
these into (B,) without a custom collate_fn."""
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


@pytest.fixture
def mixed_dataset():
    """Tiny in-process MixedPlayEEFDataset. Skipped if data dirs missing
    or if the dataset module cannot be imported (e.g. numpy/numba binary
    incompatibility on this host)."""
    from pathlib import Path

    try:
        from interactive_world_sim.datasets.latent_dynamics.play_eef_dataset import (
            MixedPlayEEFDataset,
        )
    except ImportError as e:
        pytest.skip(f"dataset module import failed: {e}")

    robot_root = Path("/scr2/yusenluo/interactive_world_sim/play_robot_eef/play_robot_1_eef")
    human_root = Path("/scr2/yusenluo/interactive_world_sim/human_play_eef_data/play_human_eef_1")
    if not robot_root.exists() or not human_root.exists():
        pytest.skip("real EEF data not present on this host")

    cfg = OmegaConf.create({
        "robot": {
            "domain": "robot",
            "dataset_dirs": [str(robot_root)],
            "video_keys": ["observation.images.cam_high"],
            "obs_keys": ["camera_0_color"],
            "crops": [[195, 195, 256, 256]],
            "resolution": 128, "horizon": 4, "val_horizon": 4,
            "skip_frame": 1, "pad_before": 0, "pad_after": 0,
            "val_ratio": 0.1, "seed": 42, "goal_sample": "intermediate",
            "skip_idx": 1, "cache_root": "/tmp/iws_test_cache_robot",
        },
        "human": {
            "domain": "human",
            "dataset_dirs": [str(human_root)],
            "video_keys": ["observation.images.cam_high"],
            "obs_keys": ["camera_0_color"],
            "crops": [[195, 195, 256, 256]],
            "resolution": 128, "horizon": 4, "val_horizon": 4,
            "skip_frame": 1, "pad_before": 0, "pad_after": 0,
            "val_ratio": 0.333, "seed": 42, "goal_sample": "intermediate",
            "skip_idx": 1, "cache_root": "/tmp/iws_test_cache_human",
        },
        "sample_ratio": 0.5, "seed": 0,
    })
    return MixedPlayEEFDataset(cfg)


def test_per_item_emits_domain_label(mixed_dataset):
    item = mixed_dataset[0]
    assert "domain_label" in item, "MixedPlayEEFDataset must add domain_label"
    dl = item["domain_label"]
    assert isinstance(dl, torch.Tensor) and dl.dtype == torch.long
    assert dl.numel() == 1
    expected = 1 if item["embodiment"] == "robot" else 0
    assert int(dl) == expected


def test_collated_batch_has_domain_label(mixed_dataset):
    loader = DataLoader(mixed_dataset, batch_size=4, num_workers=0)
    batch = next(iter(loader))
    assert "domain_label" in batch
    assert batch["domain_label"].dtype == torch.long
    assert batch["domain_label"].shape == (4,)
    assert set(batch["domain_label"].tolist()).issubset({0, 1})
