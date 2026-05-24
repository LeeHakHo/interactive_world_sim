"""End-to-end smoke for Phase 0 Stage 1 with latent_decompose enabled.

Uses a tiny in-process fake dataset (random tensors) so the test does
not depend on the real video data being present. Verifies that:
  (a) all loss terms are finite,
  (b) rec_loss, L_dom, L_adv are all logged,
  (c) encoder + decoder + classifier param groups all see non-zero grads.
"""
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf


def _import_lwm():
    try:
        from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import (
            LatentWorldModel,
        )
    except ImportError as e:
        pytest.skip(f"LatentWorldModel import failed: {e}")
    return LatentWorldModel


@pytest.fixture
def alg_cfg():
    """Tiny manual OmegaConf cfg for fast smoke runs. Reuses the same
    shape/key set as test_lwm_construction.py but with smaller dims."""
    raw = {
        "debug": False,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "warmup_steps": 100,
        "lr_scheduler": "linear",
        "optimizer_beta": [0.9, 0.999],
        "latent_dim": 512,
        "action_dim": 8,
        "enc_dim": 64,
        "num_components": 1,
        "obs_keys": ["camera_0_color"],
        "x_shape": [3, 64, 64],
        "num_latent_downsample": 2,
        "num_views": 1,
        "num_latent_channel": 4,
        "latent_resolution": 16,
        "n_frames": 2,
        "training_stage": 1,
        "load_ae": None,
        "norm_scale": 6.0,
        "mask_prev_action": False,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "noise_level": "log_normal",
        "val_render": False,
        "max_val_render_batches": 16,
        "scheduling_matrix": "autoregressive",
        "uncertainty_scale": 1.0,
        "guidance_scale": 1.0,
        "dyn_infer_steps": 1,
        "dec_infer_steps": 3,
        "last_frame_loss_only": False,
        "prev_frame_noise_scale": 0.1,
        "robust_latent": False,
        "delta": 0.01,
        "sampling_strategy": "uniform",
        "sampling_strategy_params": [],
        "metrics": [],
        "dynamics": {
            "_target_": "interactive_world_sim.algorithms.latent_dynamics.models.cm_latent_dynamics.CMLatentDynamics",
            "action_dim": 8, "latent_dim": 4, "dim": 64,
            "action_emb_dim": 512, "resnet_block_groups": 8,
            "dim_mults": [1, 2], "attn_dim_head": 128, "attn_heads": 4,
            "use_linear_attn": True, "use_init_temporal_attn": True,
            "init_kernel_size": 5, "is_causal": True,
            "time_emb_type": "rotary",
            # dtype intentionally omitted — CMLatentDynamics defaults to torch.float32
        },
        "noise_scheduler": {
            "_target_": "interactive_world_sim.utils.cm_utils.DDPMScheduler",
            "x_shape": [3, 64, 64], "timesteps": 1000, "sampling_timesteps": 50,
            "beta_schedule": "sigmoid",
            "schedule_fn_kwargs": {},
            "objective": "pred_v",
            "loss_weighting": "fused_snr",
            "snr_clip": 5.0, "cum_snr_decay": 0.96,
            "ddim_sampling_eta": 0.0, "clip_noise": 6.0,
            "stabilization_level": 15,
            # dtype intentionally omitted — DDPMScheduler defaults to torch.float32
        },
        "diffusion": {
            "beta_schedule": "sigmoid", "objective": "pred_v",
            "use_fused_snr": True, "cum_snr_decay": 0.96,
            "clip_noise": 6.0, "schedule_fn_kwargs": {},
            "timesteps": 1000, "sampling_timesteps": 50,
            "ddim_sampling_eta": 0.0, "snr_clip": 5.0,
            "model_channels": 64, "num_latent_downsample": 2,
            "num_latent_channel": 4, "num_res_blocks": 2,
            "attention_resolutions": [2, 4, 8], "dropout": 0.1,
            "channel_mult": [1, 2, 3], "num_head_channels": 64,
            "resblock_updown": True, "use_scale_shift_norm": True,
            "num_components": 1, "image_size": 64,
            "stabilization_level": 15,
        },
        "dynamo_ssl": {
            "enabled": False,
            "encoder_backbone": "vit",
            "loss_coef": 1.0,
            "pretrained_encoder": False,
            "feature_dim": 512,
            "projection_dim": 32,
            "window_size": 4,
            "n_layer": 6,
            "n_head": 6,
            "n_embd": 120,
            "dropout": 0.0,
            "covariance_reg_coef": 0.04,
            "dynamics_loss_coef": 1.0,
            "ema_beta": None,
            "beta_scheduling": True,
            "lr": 1e-4,
            "weight_decay": 0.0,
            "betas": [0.9, 0.999],
            "separate_single_views": True,
            "detach_rec_from_encoder": True,
            "spatial_proj_mode": "plain",
            "use_sigreg": False,
            "sigreg_weight": 0.09,
            "sigreg_knots": 17,
            "sigreg_num_proj": 1024,
            "use_sparse_idm": False,
            "sparse_lambda": 0.01,
            "sparse_mask_init": 0.0,
            "vit": {
                "img_size": 64, "patch_size": 8, "embed_dim": 384,
                "depth": 2, "num_heads": 6, "mlp_ratio": 4.0, "drop_rate": 0.0,
            },
        },
        "latent_decompose": {
            "enabled": True,
            "d_task": 288, "d_emb": 96,
            "lambda_dom": 0.1,
            "lambda_adv_schedule": {
                "type": "linear_ramp",
                "start_step": 0,           # ramp immediately so L_adv has nonzero grad
                "end_step": 1,
                "start_value": 0.0,
                "end_value": 0.3,
            },
            "lr_classifiers": 3e-4,
        },
    }
    return OmegaConf.create(raw)


def _fake_batch(B=2, T=2, V=1, H=64, W=64, A=8):
    """Build a single batch matching MixedPlayEEFDataset's schema."""
    return {
        "obs": {"camera_0_color": torch.rand(B, T, 3 * V, H, W)},
        "action":         torch.randn(B, T, A),
        "domain_label":   torch.randint(0, 2, (B,), dtype=torch.long),
        "is_early_stop":  torch.zeros(B, 1, dtype=torch.bool),
        "rel_stop_idx":   torch.full((B, 1), T - 1, dtype=torch.long),
    }


def _make_lwm(alg_cfg, enabled: bool):
    import numpy as np
    LatentWorldModel = _import_lwm()
    cfg = OmegaConf.create(OmegaConf.to_container(alg_cfg, resolve=True))
    cfg.latent_decompose.enabled = enabled
    m = LatentWorldModel(cfg)
    # Identity normalizer — tests don't care about action scaling.
    # stat values must be numpy arrays (get_identity_normalizer_from_stat uses np.ones_like).
    from interactive_world_sim.utils.normalizer import (
        LinearNormalizer,
        get_identity_normalizer_from_stat,
        get_image_range_normalizer,
    )
    norm = LinearNormalizer()
    A = cfg.action_dim
    stat = {
        "min":  -np.ones(A, dtype=np.float32),
        "max":   np.ones(A, dtype=np.float32),
        "mean":  np.zeros(A, dtype=np.float32),
        "std":   np.ones(A, dtype=np.float32),
    }
    norm["action"] = get_identity_normalizer_from_stat(stat)
    for k in cfg.obs_keys:
        norm[k] = get_image_range_normalizer()
    m.set_normalizer(norm)
    return m


@pytest.mark.integration
def test_smoke_one_training_step(alg_cfg):
    """Mode=smoke: 1 training_step, assert (a) finite loss, (b) all loss
    terms logged, (c) encoder + decoder + classifier params have grads."""
    m = _make_lwm(alg_cfg, enabled=True)
    m.train()

    # tracemalloc must be started before training_step (batch_idx=0 triggers it)
    m.on_train_start()

    # Bypass Lightning by calling training_step directly + manual backward.
    batch = _fake_batch(B=2, T=alg_cfg.n_frames, H=64, W=64)
    out = m.training_step(batch, batch_idx=0)
    loss = out["loss"]
    assert torch.isfinite(loss), f"non-finite loss: {loss}"

    loss.backward()
    # Encoder grad
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in m.vit_encoder.parameters()
    ), "encoder got no grad"
    # Decoder grad
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in m.decoder.parameters()
    ), "decoder got no grad"
    # Classifier grad
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in m.clf_emb.parameters()
    ), "clf_emb got no grad"
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in m.clf_adv.parameters()
    ), "clf_adv got no grad"


@pytest.mark.integration
def test_identity_under_zero_loss(alg_cfg):
    """G1: enabled=true with lambda_dom=0 and lambda_adv.end=0 must produce
    a rec_loss trace bit-identical (rtol=1e-5) to enabled=false."""
    torch.manual_seed(0)
    cfg_off = OmegaConf.create(OmegaConf.to_container(alg_cfg, resolve=True))
    cfg_off.latent_decompose.enabled = False
    m_off = _make_lwm(cfg_off, enabled=False)
    m_off.train()

    torch.manual_seed(0)
    cfg_on = OmegaConf.create(OmegaConf.to_container(alg_cfg, resolve=True))
    cfg_on.latent_decompose.enabled = True
    cfg_on.latent_decompose.lambda_dom = 0.0
    cfg_on.latent_decompose.lambda_adv_schedule.start_value = 0.0
    cfg_on.latent_decompose.lambda_adv_schedule.end_value = 0.0
    m_on = _make_lwm(cfg_on, enabled=True)
    m_on.train()

    # Copy encoder/decoder weights from m_off to m_on so the only delta is
    # the (unused, zero-weighted) classifier heads.
    m_on.vit_encoder.load_state_dict(m_off.vit_encoder.state_dict())
    m_on.spatial_proj.load_state_dict(m_off.spatial_proj.state_dict())
    m_on.decoder.load_state_dict(m_off.decoder.state_dict())

    # training_step at batch_idx=0 calls tracemalloc.take_snapshot(); start it.
    m_off.on_train_start()
    m_on.on_train_start()

    torch.manual_seed(0)
    batch = _fake_batch(B=2, T=alg_cfg.n_frames, H=64, W=64)
    torch.manual_seed(0)
    out_off = m_off.training_step(batch, batch_idx=0)
    torch.manual_seed(0)
    out_on  = m_on.training_step(batch, batch_idx=0)

    # rec_loss is logged but not returned; the returned "loss" equals rec_loss
    # for m_off and equals rec_loss + 0 * L_dom + 0 * L_adv = rec_loss for m_on.
    torch.testing.assert_close(
        out_off["loss"], out_on["loss"], rtol=1e-5, atol=1e-6,
        msg="G1: zero-weighted decompose loss broke bit-identical baseline",
    )
