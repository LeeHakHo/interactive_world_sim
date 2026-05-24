"""compute_L_dom (CE) and compute_L_adv (GRL + CE)."""
from __future__ import annotations

import torch

from interactive_world_sim.algorithms.latent_decompose.align_losses import (
    compute_L_adv,
    compute_L_dom,
)
from interactive_world_sim.algorithms.latent_decompose.domain_heads import (
    PooledClassifier,
)


def test_L_dom_is_plain_cross_entropy():
    clf = PooledClassifier(d_in=96, hidden=32)
    z = torch.randn(8, 96, 16, 16)
    dl = torch.randint(0, 2, (8,))
    loss = compute_L_dom(z, dl, clf)
    # Reference: same MLP applied directly + F.cross_entropy.
    import torch.nn.functional as F
    ref = F.cross_entropy(clf(z), dl)
    torch.testing.assert_close(loss, ref)


def test_L_adv_flips_gradient_on_z_task():
    clf = PooledClassifier(d_in=288, hidden=32)
    z_grl = torch.randn(8, 288, 16, 16, requires_grad=True)
    z_ce  = z_grl.detach().clone().requires_grad_(True)
    dl = torch.randint(0, 2, (8,))

    L_adv = compute_L_adv(z_grl, dl, clf, lambda_adv=0.5)
    L_adv.backward()

    # Plain CE without GRL on the same input — gradient should be opposite sign.
    import torch.nn.functional as F
    plain = F.cross_entropy(clf(z_ce), dl)
    plain.backward()

    torch.testing.assert_close(z_grl.grad, -0.5 * z_ce.grad, rtol=1e-5, atol=1e-6)
