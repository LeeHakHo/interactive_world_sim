"""CLUB: Contrastive Log-ratio Upper Bound of mutual information.

Reference: Cheng et al., "CLUB: A Contrastive Log-ratio Upper Bound of Mutual
Information", ICML 2020 (arXiv:2006.12013). Also used as the disentanglement
estimator in the cross-embodiment video-editing paper (arXiv:2605.03637) and
adopted here to push z_task and z_emb apart WITHOUT a gradient-reversal
adversary.

Usage pattern (two optimisers, like a GAN's two players):
  1. q-network update (every step): minimise `learning_loss(z_task, z_emb)`
     so the variational q(z_emb | z_task) tracks the true conditional.
  2. encoder update (optionally every N steps): add `mi_est(z_task, z_emb)`
     to the main loss to push the MI upper bound down. The q-network params
     must be frozen for this term (detach handled by the caller's optimiser
     param groups).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CLUB(nn.Module):
    """Variational MI upper-bound estimator between x (z_task) and y (z_emb).

    q(y|x) is a diagonal Gaussian whose mean and log-variance are predicted by
    two small MLPs from x. `mi_est` returns the CLUB upper bound on I(x; y);
    `learning_loss` is the negative log-likelihood used to fit q.
    """

    def __init__(self, x_dim: int, y_dim: int, hidden: int = 256,
                 logvar_clamp: float = 10.0):
        super().__init__()
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden), nn.GELU(),
            nn.Linear(hidden, y_dim),
        )
        # tanh-bounded log-variance for numerical stability.
        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden), nn.GELU(),
            nn.Linear(hidden, y_dim), nn.Tanh(),
        )
        self.logvar_clamp = float(logvar_clamp)

    def _mu_logvar(self, x: torch.Tensor):
        mu = self.p_mu(x)
        logvar = self.p_logvar(x) * self.logvar_clamp
        return mu, logvar

    def mi_est(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """CLUB upper bound on I(x; y). Minimise this w.r.t. the encoders to
        reduce MI. q-network params should be frozen for this term."""
        mu, logvar = self._mu_logvar(x)
        var = logvar.exp()
        # positive: log q(y_i | x_i) for matched pairs
        positive = -0.5 * ((mu - y) ** 2) / var
        # negative: average log q(y_j | x_i) over all j (mismatched pairs)
        mu_t = mu.unsqueeze(1)        # (B, 1, y)
        y_t = y.unsqueeze(0)          # (1, B, y)
        var_t = var.unsqueeze(1)      # (B, 1, y)
        negative = -0.5 * ((y_t - mu_t) ** 2 / var_t).mean(dim=1)  # (B, y)
        return (positive.sum(-1) - negative.sum(-1)).mean()

    def learning_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood of y under q(y|x). Minimise this w.r.t. the
        q-network only (detach x, y from the encoder graph at the call site)."""
        mu, logvar = self._mu_logvar(x)
        return 0.5 * (logvar + (mu - y) ** 2 / logvar.exp()).sum(-1).mean()
