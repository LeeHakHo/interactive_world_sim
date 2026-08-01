"""IDM 网络: 展平双向窗口输入 -> Δjoint(6)+grip(1)。小 MLP(hidden/n_layers 可 env 切, 做容量对照)。"""
import os
import torch, torch.nn as nn
import idm_data as D


class IDM(nn.Module):
    def __init__(s, din, hidden=None, dout=7, n_layers=None):
        super().__init__()
        hidden = hidden or int(os.environ.get("HID", "512"))
        n_layers = n_layers or int(os.environ.get("NLAYER", "3"))    # 隐层数(默认3, 容量对照可加大)
        layers = [nn.Linear(din, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, dout)]
        s.net = nn.Sequential(*layers)

    def forward(s, x):
        return s.net(x)


class _Block(nn.Module):
    """一层 = spatial(同帧点间) + temporal(同点跨时) 分解自注意力 + FFN。非因果。"""
    def __init__(s, d, heads):
        super().__init__()
        s.sa = nn.MultiheadAttention(d, heads, batch_first=True)      # spatial
        s.ta = nn.MultiheadAttention(d, heads, batch_first=True)      # temporal
        s.n1 = nn.LayerNorm(d); s.n2 = nn.LayerNorm(d); s.n3 = nn.LayerNorm(d)
        s.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(s, x):                                                # x (B,W,N,d)
        B, W, N, d = x.shape
        h = s.n1(x).reshape(B * W, N, d); a, _ = s.sa(h, h, h); x = x + a.reshape(B, W, N, d)
        h = s.n2(x).permute(0, 2, 1, 3).reshape(B * N, W, d); a, _ = s.ta(h, h, h)
        x = x + a.reshape(B, N, W, d).permute(0, 2, 1, 3)
        return x + s.ff(s.n3(x))


class IDMTransformer(nn.Module):
    """每点每帧 token 化(坐标投影 + type/kp-id/time 嵌入) → 时空分解注意力(非因果)
    → learned-query cross-attention 头 → Δjoint(6)+grip(1)。照 UMA/Point Policy/AMPLIFY-finv。"""
    def __init__(s, C, N, W, ptype, d=None, heads=None, layers=None, dout=7, nq=4):
        super().__init__()
        d = d or int(os.environ.get("XFD", "256")); heads = heads or int(os.environ.get("XFH", "4"))
        layers = layers or int(os.environ.get("XFL", "4"))
        s.proj = nn.Linear(C, d)
        s.type_emb = nn.Embedding(2, d); s.kp_emb = nn.Embedding(N, d); s.time_emb = nn.Embedding(W, d)
        s.register_buffer("ptype", torch.as_tensor(ptype, dtype=torch.long))
        s.register_buffer("kpid", torch.arange(N))
        s.blocks = nn.ModuleList([_Block(d, heads) for _ in range(layers)])
        s.q = nn.Parameter(torch.randn(nq, d) * 0.02)
        s.ca = nn.MultiheadAttention(d, heads, batch_first=True); s.nq = nn.LayerNorm(d)
        s.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, dout))

    def forward(s, x):                                               # x (B,W,N,C)
        B, W, N, C = x.shape
        h = s.proj(x) + s.type_emb(s.ptype)[None, None] + s.kp_emb(s.kpid)[None, None] + s.time_emb(torch.arange(W, device=x.device))[None, :, None]
        for blk in s.blocks:
            h = blk(h)
        tok = h.reshape(B, W * N, -1)
        q = s.q[None].expand(B, -1, -1)
        pooled, _ = s.ca(s.nq(q), tok, tok)                         # learned query attend tokens
        return s.head(pooled.mean(1))


def predict_clip(model, z, spec, ci, KP=4, FF=4):
    X, _, _ = D.build_windows(z, spec, KP, FF, clips=[ci])
    with torch.no_grad():
        return model(torch.from_numpy(X).float()).numpy()   # (L-1,7)
