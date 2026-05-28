"""Agent mask from RGB, by embodiment. WHOLE ARM (hand+forearm+sleeve for human,
full arm+gripper for robot) — NOT just the hand. v1 mask source for mask_subtract;
the model consumes batch["agent_mask"], so a SAM2/precomputed mask is a drop-in via
the same key.

Validated with quantitative guards (coverage fraction + border-touch), not eyeballing
alone — past masks failed silently. See docs/superpowers/specs/
2026-05-28-mask-subtract-residual-decompose-design.md.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage


def agent_mask_from_rgb(frames: torch.Tensor, embodiment: str) -> torch.Tensor:
    """Per-pixel color classifier.

    frames: (..., 3, H, W) float in [0, 1]. embodiment: "human" | "robot".
    Returns (..., 1, H, W) float mask in {0, 1}, leading dims preserved.

    human = dark(sleeve) ∪ skin(pink hand); robot = dark(gripper/arm). Reject the
    blue plate/bowl and the dark-red cube (manipulated objects we must keep).
    """
    assert frames.shape[-3] == 3, f"expected (...,3,H,W), got {tuple(frames.shape)}"
    r, g, b = frames[..., 0, :, :], frames[..., 1, :, :], frames[..., 2, :, :]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    dark = lum < 0.28                                     # black sleeve / gripper
    skin = (r > 0.55) & (r > g + 0.06) & (r > b + 0.06)   # pink hand
    blue = (b > 0.40) & (b > r + 0.06) & (b > g + 0.06)   # plate/bowl — reject
    sat = frames.amax(-3) - frames.amin(-3)
    cube = (r > 0.4) & (r > g + 0.15) & (r > b + 0.15) & (sat > 0.25) & (~skin)
    agent = dark if embodiment == "robot" else (dark | skin)
    agent = agent & (~blue) & (~cube)
    return agent.unsqueeze(-3).to(frames.dtype)            # (...,1,H,W)


def whole_arm_mask(frames: torch.Tensor, embodiment: str,
                   border_only: bool = True, dilate: int = 3) -> torch.Tensor:
    """Refine the per-pixel color mask into a whole-arm mask.

    Per frame: morphological close (connect hand+sleeve / gripper segments) → keep
    connected components that TOUCH the image border (a limb enters from off-frame;
    a hand-only blob does not) → dilate. This enforces "whole arm, not just hand"
    and drops stray hand-only / background blobs.

    frames: (..., 3, H, W) → (..., 1, H, W) {0, 1}.
    """
    raw = agent_mask_from_rgb(frames, embodiment)[..., 0, :, :]   # (...,H,W)
    lead = raw.shape[:-2]
    flat = raw.reshape(-1, *raw.shape[-2:]).cpu().numpy().astype(bool)
    out = np.zeros_like(flat)
    for i in range(flat.shape[0]):
        m = ndimage.binary_closing(flat[i], iterations=2)
        lab, n = ndimage.label(m)
        if n == 0:
            continue
        if border_only:
            border = (set(lab[0, :]) | set(lab[-1, :])
                      | set(lab[:, 0]) | set(lab[:, -1]))
            border.discard(0)
            if border:
                keep = np.isin(lab, list(border))
            else:
                # fallback: nothing touches the border → keep the largest component
                sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
                keep = lab == (int(np.argmax(sizes)) + 1)
        else:
            keep = m
        if dilate > 0:
            keep = ndimage.binary_dilation(keep, iterations=dilate)
        out[i] = keep
    t = torch.from_numpy(out).to(frames.dtype).to(frames.device)
    return t.reshape(*lead, 1, *raw.shape[-2:])
