"""IDM 网络: 展平双向窗口输入 -> Δjoint(6)+grip(1)。小 MLP。"""
import torch, torch.nn as nn
import idm_data as D


class IDM(nn.Module):
    def __init__(s, din, hidden=512, dout=7):
        super().__init__()
        s.net = nn.Sequential(
            nn.Linear(din, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dout))

    def forward(s, x):
        return s.net(x)


def predict_clip(model, z, spec, ci, KP=4, FF=4):
    X, _, _ = D.build_windows(z, spec, KP, FF, clips=[ci])
    with torch.no_grad():
        return model(torch.from_numpy(X).float()).numpy()   # (L-1,7)
