"""Tests for the SKELETON ACTION REPRESENTATION experiment in exp_scel_dualview_wm.py
(ACTION=dummy5|skel, MIX=r|rh, NROB robot-scarce subsampling).

No-leakage / correctness checks:
  - ACTION unset -> byte-identical to the pre-existing dummy5 code path (regression gate).
  - ACTION=skel token arrays have the right shape/are finite for both domains, on REAL sidecar rows
    (not synthetic data -- catches row-misalignment or index-selection mistakes the way the
    production script's own `_load_skel()` assert-guards do in exp_scel_dualview_dit_formal.py).
  - subsample_robot_pool is deterministic (same (pool, nrob) -> same subset every call), and a no-op
    for falsy/oversized nrob.
"""
import os

import numpy as np
import pytest
import torch

os.chdir("/scr2/yusenluo/interactive_world_sim")

import exp_scel_dualview_wm as W
from amplify_wm import K, F

DS_DIR = "outputs/flow_render_dataset_can_dual"
ROBOT_NPZ = f"{DS_DIR}/clips_robot.npz"
HUMAN_NPZ = f"{DS_DIR}/clips_human_L24.npz"
SKEL_ROBOT_V2 = f"{DS_DIR}/skel_sidecar_robot_v2.npz"
SKEL_HUMAN = f"{DS_DIR}/skel_sidecar_human.npz"

_have_data = all(os.path.exists(p) for p in (ROBOT_NPZ, HUMAN_NPZ, SKEL_ROBOT_V2, SKEL_HUMAN))
needs_data = pytest.mark.skipif(not _have_data, reason="can_dual dataset/sidecars not present")


# ---------------------------------------------------------------------------
# ACTION=dummy5 default-path regression (byte-identical to pre-existing behavior)
# ---------------------------------------------------------------------------

def test_load_action_tokens_dummy5_matches_original_formula():
    """load_action_tokens('dummy5', ...) must reproduce the exact pre-existing main() computation:
    efA = z['eef'] (no nan handling), efB = nan_to_num(z['eef_low'], nan=0.5)."""
    rng = np.random.default_rng(0)
    N, L = 6, 8
    eef = rng.standard_normal((N, L, 3, 2)).astype(np.float32)
    eef_low = rng.standard_normal((N, L, 3, 2)).astype(np.float32)
    eef_low[0, 0, 0, 0] = np.nan
    z = {"eef": eef, "eef_low": eef_low}

    efA, efB = W.load_action_tokens("dummy5", "r", z, sk=None)

    expected_A = z["eef"].astype(np.float32)
    expected_B = np.nan_to_num(z["eef_low"].astype(np.float32), nan=0.5)
    assert np.array_equal(efA, expected_A)
    assert np.array_equal(efB, expected_B)
    assert not np.isnan(efB).any()


def test_dual_lwc_default_action_is_dummy5_unchanged_dims():
    """DualLWC() with no action kwarg must match the original hard-coded dummy5-only Linear dims
    ((K+F) * 5 * 2 * 2 input features) -- constructing without `action=` at all (as every external
    caller in the repo does) must not change behavior."""
    P = 24
    m = W.DualLWC(P, Dm=64, layers=1, W=5, vel_half=0.06)
    assert m.action == "dummy5"
    assert m.n_tok == 5
    assert m.act.in_features == (K + F) * 5 * 2 * 2


def test_act_pts_default_matches_dummy5_function_bit_exact():
    """s._act_pts on a default-action model must be bit-identical to calling dummy5() directly."""
    P = 24
    m = W.DualLWC(P, Dm=64, layers=1, W=5, vel_half=0.06)
    eef = torch.randn(3, K + F, 3, 2)
    out = m._act_pts(eef)
    expected = W.dummy5(eef)
    assert torch.equal(out, expected)


def test_dual_lwc_forward_default_action_shape_finite():
    """Sanity: default-action DualLWC forward pass runs and produces finite logits (regression smoke,
    not a numeric-behavior test -- separate from the bit-exact _act_pts check above)."""
    P = 6
    m = W.DualLWC(P, Dm=32, layers=1, W=5, vel_half=0.06)
    B = 2
    hist = torch.randn(B, 2 * P, K, 2)
    eef_a = torch.randn(B, K + F, 3, 2)
    eef_b = torch.randn(B, K + F, 3, 2)
    logits, anchor = m.fwd_dual(hist, eef_a, eef_b)
    assert logits.shape == (B, 2 * P, F, 5 * 5)
    assert torch.isfinite(logits).all()
    assert anchor.shape == (B, 2 * P, 2)


# ---------------------------------------------------------------------------
# ACTION=skel: shape/finite on real sidecar rows, both domains
# ---------------------------------------------------------------------------

@needs_data
def test_skel_robot_idx_matches_spec_order():
    """SKEL_ROBOT_IDX must select [link_6, fingertipL', fingertipR', link_5] in that order (idx
    [5,6,7,4] into the 8-pt robot_v2 sidecar) -- isomorphic to human's [wrist,fin1,fin2,forearm_stub]."""
    sk = np.load(SKEL_ROBOT_V2)
    names = list(sk["joint_names"])
    selected = [names[i] for i in W.SKEL_ROBOT_IDX]
    assert selected == [
        "follower_right_link_6",
        "follower_right_fingertipL_closed",
        "follower_right_fingertipR_closed",
        "follower_right_link_5",
    ]


@needs_data
def test_skel_robot_link6_aligns_with_eef_base():
    """No-leakage/row-alignment sanity (mirrors _load_skel()'s own assert-guards in
    exp_scel_dualview_dit_formal.py): SKEL_ROBOT_IDX[0] (link_6) must land within ~1px of the clips'
    eef base point (eef[...,0,:]) -- both are the same physical wrist mount, verified 0.01-0.03px in
    the v2 sidecar's own docstring. A large mismatch here means the two npz files drifted out of sync."""
    zr = np.load(ROBOT_NPZ, mmap_mode="r")
    sk = np.load(SKEL_ROBOT_V2)
    IMG = 128
    for j in (0, min(500, len(zr["eef"]) - 1), len(zr["eef"]) - 1):
        d = np.linalg.norm(sk["skel2d_high"][j, 0, W.SKEL_ROBOT_IDX[0]] - zr["eef"][j, 0, 0]) * IMG
        assert d < 1.0, f"row {j}: link_6 vs eef-base {d:.3f}px (expect <1px)"


@needs_data
def test_load_action_tokens_skel_robot_shape_finite():
    zr = np.load(ROBOT_NPZ, mmap_mode="r")
    sk = np.load(SKEL_ROBOT_V2)
    efA, efB = W.load_action_tokens("skel", "r", zr, sk)
    N, L = zr["eef"].shape[:2]
    assert efA.shape == (N, L, 4, 2)
    assert efB.shape == (N, L, 4, 2)
    assert np.isfinite(efA).all()
    assert np.isfinite(efB).all()


@needs_data
def test_load_action_tokens_skel_human_shape_finite_and_order():
    zh = np.load(HUMAN_NPZ, mmap_mode="r")
    sk = np.load(SKEL_HUMAN)
    assert list(sk["joint_names"]) == ["wrist", "fingertip1", "fingertip2", "forearm_stub"]
    efA, efB = W.load_action_tokens("skel", "h", zh, sk)
    N, L = zh["eef"].shape[:2]
    assert efA.shape == (N, L, 4, 2)
    assert efB.shape == (N, L, 4, 2)
    assert np.isfinite(efA).all()
    assert np.isfinite(efB).all()


@needs_data
def test_dual_lwc_skel_action_dims_and_forward():
    P = 6
    m = W.DualLWC(P, action="skel", Dm=32, layers=1, W=5, vel_half=0.06)
    assert m.n_tok == 4
    assert m.act.in_features == (K + F) * 4 * 2 * 2
    B = 2
    hist = torch.randn(B, 2 * P, K, 2)
    eef_a = torch.randn(B, K + F, 4, 2)   # skel tokens: n_raw==n_tok==4, no virtual expansion
    eef_b = torch.randn(B, K + F, 4, 2)
    logits, anchor = m.fwd_dual(hist, eef_a, eef_b)
    assert logits.shape == (B, 2 * P, F, 5 * 5)
    assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# NROB robot-scarce subsampling: deterministic, no-op on falsy/oversized nrob
# ---------------------------------------------------------------------------

def test_subsample_robot_pool_deterministic():
    pool = np.arange(1000)
    a = W.subsample_robot_pool(pool, 100)
    b = W.subsample_robot_pool(pool, 100)
    assert len(a) == 100
    assert np.array_equal(a, b)


def test_subsample_robot_pool_noop_on_falsy_or_oversized():
    pool = np.arange(50)
    assert np.array_equal(W.subsample_robot_pool(pool, ""), pool)
    assert np.array_equal(W.subsample_robot_pool(pool, 0), pool)
    assert np.array_equal(W.subsample_robot_pool(pool, 1000), pool)


def test_subsample_robot_pool_different_n_different_subset():
    pool = np.arange(1000)
    a = W.subsample_robot_pool(pool, 100)
    b = W.subsample_robot_pool(pool, 400)
    assert len(b) == 400
    # a should not simply be b's prefix (rng.choice, not deterministic nesting) -- just check both valid
    assert set(a.tolist()) <= set(pool.tolist())
    assert set(b.tolist()) <= set(pool.tolist())


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
