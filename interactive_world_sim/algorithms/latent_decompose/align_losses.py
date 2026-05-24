"""Phase 0 Stage-1 alignment losses + lambda schedule.

L_dom: regular CE on the embodiment classifier — forces z_emb to carry
       embodiment info.
L_adv: gradient-reversal + CE on z_task — pushes the encoder away from
       making z_task domain-discriminable, while letting the classifier
       itself train normally.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .domain_heads import grad_reverse


def compute_L_dom(
    z_emb: torch.Tensor,
    domain_label: torch.Tensor,
    clf_emb: nn.Module,
) -> torch.Tensor:
    return F.cross_entropy(clf_emb(z_emb), domain_label)


def compute_L_adv(
    z_task: torch.Tensor,
    domain_label: torch.Tensor,
    clf_adv: nn.Module,
    lambda_adv: float,
) -> torch.Tensor:
    z_rev = grad_reverse(z_task, lambda_adv)
    return F.cross_entropy(clf_adv(z_rev), domain_label)


def linear_ramp(
    step: int,
    start_step: int,
    end_step: int,
    start_value: float,
    end_value: float,
) -> float:
    """Linear interpolation, clamped at the endpoints. Used by the
    lambda_adv schedule in the LatentWorldModel training loop."""
    step = int(step)
    if step <= start_step:
        return float(start_value)
    if step >= end_step:
        return float(end_value)
    t = (step - start_step) / max(end_step - start_step, 1)
    return float(start_value + t * (end_value - start_value))
