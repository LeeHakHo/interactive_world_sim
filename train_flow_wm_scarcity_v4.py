"""Step 3 v4: thick flow dynamic latent (contact-gated motion + anti-drift training).

Extends train_flow_wm_scarcity_v3.py. `--thin` reproduces v3 exactly (ablation via
flag, not a fork). Reads outputs/flow_dataset/flow_ds_v3.npz. See
docs/superpowers/specs/2026-06-06-thicken-flow-latent-design.md.
"""
import argparse, os, numpy as np, torch, torch.nn as nn

DS = "outputs/flow_dataset/flow_ds_v3.npz"
K, F, L, Dm, P = 4, 12, 16, 128, 48
EPOCHS, BS, LR = 60, 128, 3e-4
TEST_ROBOT_VID = 12
N_LIST = [1400, 400, 200, 100, 50]
SEEDS = 5
LAMBDA_GATE = 1e-2
NOISE_STD = 0.01
LAMBDA_CONSIST = 0.5
STATIC_TAU = 0.02
device = "cuda" if torch.cuda.is_available() else "cpu"


class FlowWMThick(nn.Module):
    def __init__(self, P, thin=False):
        super().__init__()
        self.thin = thin
        self.inp = nn.Linear(2 * K + 2, Dm)
        self.act = nn.Linear(L * 3 * 2, Dm)            # full-window EEF as action
        self.gctx = nn.Linear(L, Dm)                   # grasp-sequence token (thick only)
        enc = nn.TransformerEncoderLayer(Dm, 4, Dm * 2, batch_first=True, dropout=0.0)
        self.tf = nn.TransformerEncoder(enc, 3)
        self.head = nn.Linear(Dm, F * 2)
        self.gate = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, hist, eef3, g):
        # hist (B,P,K,2), eef3 (B,L,3,2), g (B,L) normalized grasp
        B, P = hist.shape[:2]
        anchor = hist[:, :, -1, :]                                       # (B,P,2)
        obj = self.inp(torch.cat([(hist - anchor[:, :, None]).reshape(B, P, 2 * K), anchor], -1))
        objc = anchor.mean(1, keepdim=True)                              # (B,1,2)
        act = self.act((eef3 - objc[:, :, None]).reshape(B, -1))[:, None, :]
        tokens = [obj, act]
        if not self.thin:
            tokens.append(self.gctx(g)[:, None, :])
        x = self.tf(torch.cat(tokens, 1))[:, :P]                         # (B,P,Dm)
        raw = self.head(x).reshape(B, P, F, 2)                           # displacement from anchor
        if self.thin:
            return anchor[:, :, None, :] + raw, None
        feat = contact_gate_features(anchor, eef3, g)                    # (B,P,F,3)
        alpha = torch.sigmoid(self.gate(feat)).squeeze(-1)              # (B,P,F)
        return anchor[:, :, None, :] + alpha[..., None] * raw, alpha


# --- pure helpers (filled in later tasks) ---
def contact_gate_features(anchor, eef3, g):
    raise NotImplementedError  # Task 3
