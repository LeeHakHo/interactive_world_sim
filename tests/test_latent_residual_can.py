import os, sys

sys.path.insert(0, "/scr2/yusenluo/interactive_world_sim")
os.environ.setdefault("VAE_NAME", "ostris/vae-kl-f8-d16")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.chdir("/scr2/yusenluo/interactive_world_sim")

import numpy as np
import torch


def test_split_okfirst_matches_formal_and_wm():
    """audit item5 leakage fix, ported: exp_latent_residual_can.split_okfirst must be filter-then-permute
    (matches ③'s own split AND exp_scel_dualview_wm.split_okfirst verbatim), disjoint, all-ok."""
    from exp_latent_residual_can import split_okfirst
    from exp_scel_dualview_wm import split_okfirst as split_okfirst_wm

    n = 300
    ok = np.ones(n, dtype=bool); ok[::7] = False           # holes, like real low_valid
    heldout = 150

    ho, pool = split_okfirst(ok, heldout)
    assert set(ho.tolist()).isdisjoint(set(pool.tolist()))
    assert ok[ho].all() and ok[pool].all()
    assert len(ho) == heldout and len(ho) + len(pool) == int(ok.sum())

    # ③-style computation, verbatim (exp_scel_dualview_dit_formal.py main split)
    okr = np.where(ok)[0]
    perm = np.random.default_rng(0).permutation(okr)
    ho_formal, pool_formal = perm[:heldout], perm[heldout:]
    assert np.array_equal(ho, ho_formal) and np.array_equal(pool, pool_formal)

    ho_wm, pool_wm = split_okfirst_wm(ok, heldout)
    assert np.array_equal(ho, ho_wm) and np.array_equal(pool, pool_wm)


def test_split_okfirst_deterministic():
    """same ok mask -> identical split every call (rng(0) fixed seed, no global RNG state leak)."""
    from exp_latent_residual_can import split_okfirst

    rng = np.random.default_rng(42)
    ok = rng.random(500) > 0.05
    ho1, pool1 = split_okfirst(ok, 150)
    ho2, pool2 = split_okfirst(ok, 150)
    assert np.array_equal(ho1, ho2) and np.array_equal(pool1, pool2)


def test_split_okfirst_would_have_caught_permute_then_filter_mismatch():
    """reproduce the exact audit-item5 bug: permute-then-filter draws a DIFFERENT held-out set than
    filter-then-permute whenever `ok` has holes -> this is the mismatch that leaked ③'s eval seqs into
    ②'s training pool. Assert the two protocols diverge on a holey mask (so our fix is not a no-op)."""
    from exp_latent_residual_can import split_okfirst

    n = 300
    ok = np.ones(n, dtype=bool); ok[::7] = False
    heldout = 150

    ho_correct, _ = split_okfirst(ok, heldout)

    perm_all = np.random.default_rng(0).permutation(n)                # legacy: permute ALL indices first
    ho_buggy = np.array([i for i in perm_all[:heldout] if ok[i]])       # then filter -> different sequence

    assert not np.array_equal(ho_correct, ho_buggy)


def test_stack_ef_shape_and_content():
    from exp_latent_residual_can import stack_ef

    N, L = 5, 24
    ef0 = np.random.rand(N, L, 3, 2).astype(np.float32)
    ef1 = np.random.rand(N, L, 3, 2).astype(np.float32)
    D = {"ef": [ef0, ef1]}
    win = 10
    out = stack_ef(D, win)
    assert out.shape == (N, win, 6, 2)
    assert out.dtype == np.float32
    assert np.allclose(out[:, :, :3], ef0[:, :win])
    assert np.allclose(out[:, :, 3:], ef1[:, :win])


def test_latentdyn_dual_forward_shapes():
    from exp_latent_residual_can import LatentDynDual, K, WIN

    Cz = 4                                    # small channel count for a fast unit test
    m = LatentDynDual(Cz).eval()
    B = 3
    z_hist = torch.randn(B, K, 2, Cz, 16, 16)
    eef = torch.randn(B, WIN, 6, 2)
    with torch.no_grad():
        out_direct = m(z_hist, eef, False)
        out_resid = m(z_hist, eef, True)
    assert out_direct.shape == (B, 2, Cz, 16, 16)
    assert out_resid.shape == (B, 2, Cz, 16, 16)


def test_residual_target_construction_is_additive_on_last_hist_frame():
    """RESIDUAL must literally compute z_hist[:,-1] + net_raw_out; DIRECT must return net_raw_out
    unmodified. Zero the final conv (weight+bias) so net_raw_out==0 deterministically, then check the
    two modes differ by exactly z_hist[:,-1] (this is the whole point of Task#21: residual anchors the
    prediction on the clip's OWN previous frame, which is why it can transfer cross-domain)."""
    from exp_latent_residual_can import LatentDynDual, K, WIN

    Cz = 4
    m = LatentDynDual(Cz).eval()
    final_conv = m.dec[-1]
    assert isinstance(final_conv, torch.nn.Conv2d)
    with torch.no_grad():
        final_conv.weight.zero_(); final_conv.bias.zero_()

    B = 2
    z_hist = torch.randn(B, K, 2, Cz, 16, 16)
    eef = torch.randn(B, WIN, 6, 2)
    with torch.no_grad():
        out_direct = m(z_hist, eef, False)
        out_resid = m(z_hist, eef, True)

    assert torch.allclose(out_direct, torch.zeros_like(out_direct), atol=1e-6)
    assert torch.allclose(out_resid, z_hist[:, -1], atol=1e-6)


def test_rollout_shape_and_uses_own_prediction_autoregressively():
    """rollout() must produce (B,hsteps,2,Cz,16,16) and feed each step's prediction back in (open-loop),
    not re-read ground truth -- verified by checking a stateful dummy model sees an evolving buffer."""
    from exp_latent_residual_can import rollout, K, WIN

    Cz, B, hsteps = 4, 2, 5
    calls = []

    class DummyModel(torch.nn.Module):
        def forward(self, z_hist, eef, residual):
            calls.append(z_hist.clone())
            return z_hist.mean(1) * 0 + 1.0             # constant prediction, easy to track

    m = DummyModel().eval()
    lat = np.random.randn(B, K + 3, 2, Cz, 16, 16).astype(np.float32)   # only [:K] is read by rollout
    eef = np.random.randn(B, WIN, 6, 2).astype(np.float32)
    with torch.no_grad():
        pr = rollout(m, lat, eef, residual=False, hsteps=hsteps)
    assert tuple(pr.shape) == (B, hsteps, 2, Cz, 16, 16)
    assert len(calls) == hsteps
    # step 1's z_hist should be the initial K frames; step 2's should have the dummy's constant pred (1.0)
    # appear in its last history slot once K>=1 history rolls over.
    assert torch.allclose(calls[0].cpu(), torch.from_numpy(lat[:, :K]).float())
    assert torch.allclose(calls[hsteps - 1][:, -1].cpu(), torch.ones(B, 2, Cz, 16, 16))
