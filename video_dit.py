"""M4: 小视频 DiT ③ (grounded 到 WEAVER/OSCAR/DexWM/IWS, 见 spec §10)。
操作 Wan2.2 latent chunk: z (B, V=2, C=48, T, gh, gw), 官方归一后 (~单位尺度)。
架构:
  - patchify 每 latent 帧 (C,gh,gw) -> tokens (patch p) -> hidden D。
  - 每 block: (a) 空间+cross-view joint attention (同一 latent 帧内 V*patch tokens 互相 attend)
             (b) causal 时序 attention (同空间位置跨帧, 因果) —— WEAVER/IWS-stage2。
  - 条件注入 = token 级 spatial-add (OSCAR 加性 / 我们 DualViewDiTG):
             cond map (Ccond,gh,gw) per (view,frame) -> patch-embed -> 加到 latent token。
  - 去噪 timestep = 逐帧独立 τ (Diffusion Forcing) -> adaLN-zero 调制 (标准 DiT 用于 t, 非动作)。
  - 首帧 I₀ 锚 (OSCAR): 第 0 latent 帧不加噪、不算 loss, 只当上下文。
  - 目标 = rectified-flow 速度 (x1-x0), xτ = τ·x1 + (1-τ)·x0 (x1=data, x0=noise)。
接口: VideoDiT(C, Ccond, V=2, gh=16, D=512, depth=12, heads=8, patch=2)
      .forward(z_tau, tau, cond, first_anchor=True) -> v_pred (B,V,C,T,gh,gw)
      训练损失/采样在 train_video_dit.py。
"""
import math, torch, torch.nn as nn, torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmb(nn.Module):
    def __init__(s, D):
        super().__init__(); s.mlp = nn.Sequential(nn.Linear(D, D), nn.SiLU(), nn.Linear(D, D)); s.D = D

    def forward(s, t):                                   # t:(...,) in [0,1] -> (...,D)
        half = s.D // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t[..., None].float() * freqs
        emb = torch.cat([torch.cos(a), torch.sin(a)], -1)
        if emb.shape[-1] < s.D: emb = F.pad(emb, (0, s.D - emb.shape[-1]))
        return s.mlp(emb)


class Attn(nn.Module):
    def __init__(s, D, heads):
        super().__init__(); s.h = heads; s.qkv = nn.Linear(D, 3 * D); s.o = nn.Linear(D, D)

    def forward(s, x, causal=False):                     # x:(B,N,D)
        B, N, D = x.shape
        q, k, v = s.qkv(x).reshape(B, N, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return s.o(o.transpose(1, 2).reshape(B, N, D))


class Block(nn.Module):
    """一个 block: 空间+cross-view attn -> causal 时序 attn -> MLP, 每段 adaLN-zero(τ)。"""
    def __init__(s, D, heads):
        super().__init__()
        s.n1, s.n2, s.n3 = (nn.LayerNorm(D, elementwise_affine=False) for _ in range(3))
        s.sp = Attn(D, heads); s.tp = Attn(D, heads)
        s.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
        s.mod = nn.Linear(D, 9 * D); nn.init.zeros_(s.mod.weight); nn.init.zeros_(s.mod.bias)

    def forward(s, x, temb, V, Tt, P):
        # x:(B, V*Tt*P, D); temb:(B, Tt, D) 逐帧. 扩到 token 级 (每帧 V*P 个 token 同 τ)。
        B, N, D = x.shape
        m = s.mod(temb)                                  # (B,Tt,9D)
        m = m[:, None, :, None, :].expand(B, V, Tt, P, 9 * D).reshape(B, N, 9 * D)
        (sa_sh, sa_sc, sa_g, ta_sh, ta_sc, ta_g, ml_sh, ml_sc, ml_g) = m.chunk(9, -1)
        # (a) 空间+cross-view: 同一帧内 V*P tokens 互 attend -> reshape (B*Tt, V*P, D)
        h = modulate(s.n1(x), sa_sh, sa_sc)
        h = h.reshape(B, V, Tt, P, D).permute(0, 2, 1, 3, 4).reshape(B * Tt, V * P, D)
        h = s.sp(h, causal=False).reshape(B, Tt, V, P, D).permute(0, 2, 1, 3, 4).reshape(B, N, D)
        x = x + sa_g * h
        # (b) causal 时序: 同 (view,空间位置) 跨帧 -> (B*V*P, Tt, D), causal
        h = modulate(s.n2(x), ta_sh, ta_sc)
        h = h.reshape(B, V, Tt, P, D).permute(0, 1, 3, 2, 4).reshape(B * V * P, Tt, D)
        h = s.tp(h, causal=True).reshape(B, V, P, Tt, D).permute(0, 1, 3, 2, 4).reshape(B, N, D)
        x = x + ta_g * h
        # (c) MLP
        x = x + ml_g * s.mlp(modulate(s.n3(x), ml_sh, ml_sc))
        return x


class VideoDiT(nn.Module):
    def __init__(s, C=48, Ccond=7, V=2, gh=16, gw=16, D=512, depth=12, heads=8, patch=2):
        super().__init__()
        s.C, s.V, s.gh, s.gw, s.p, s.D = C, V, gh, gw, patch, D
        s.ph, s.pw = gh // patch, gw // patch; s.P = s.ph * s.pw
        s.x_embed = nn.Linear(C * patch * patch, D)
        s.c_embed = nn.Linear(Ccond * patch * patch, D)     # OSCAR 加性条件 (patch-embed 后加)
        s.pos = nn.Parameter(torch.randn(1, s.P, D) * 0.02)  # 空间 pos
        s.view_emb = nn.Parameter(torch.randn(1, V, 1, D) * 0.02)
        s.temb = TimestepEmb(D)
        s.blocks = nn.ModuleList([Block(D, heads) for _ in range(depth)])
        s.nf = nn.LayerNorm(D, elementwise_affine=False)
        s.mod_f = nn.Linear(D, 2 * D); nn.init.zeros_(s.mod_f.weight); nn.init.zeros_(s.mod_f.bias)
        s.head = nn.Linear(D, C * patch * patch); nn.init.zeros_(s.head.weight); nn.init.zeros_(s.head.bias)

    def _patchify(s, z):                                  # (B,V,C,T,gh,gw) -> (B, V*T*P, C*p*p)
        B, V, C, T, gh, gw = z.shape; p = s.p
        z = z.reshape(B, V, C, T, s.ph, p, s.pw, p).permute(0, 1, 3, 4, 6, 2, 5, 7)
        return z.reshape(B, V * T * s.P, C * p * p)

    def _unpatchify(s, tok, T):                           # (B,V*T*P,C*p*p) -> (B,V,C,T,gh,gw)
        B = tok.shape[0]; p = s.p; C = s.C
        z = tok.reshape(B, s.V, T, s.ph, s.pw, C, p, p).permute(0, 1, 5, 2, 3, 6, 4, 7)
        return z.reshape(B, s.V, C, T, s.gh, s.gw)

    def forward(s, z_tau, tau, cond):
        """z_tau:(B,V,C,T,gh,gw) 已加噪; tau:(B,T) 逐帧噪声 level; cond:(B,V,Ccond,T,gh,gw)。
        -> v_pred:(B,V,C,T,gh,gw) rectified-flow 速度。"""
        B, V, C, T, gh, gw = z_tau.shape
        x = s.x_embed(s._patchify(z_tau)) + s.c_embed(s._patchify_cond(cond))   # 加性条件
        # + pos(空间) + view_emb
        x = x.reshape(B, V, T, s.P, s.D) + s.pos[:, None, None] + s.view_emb[:, :, None]
        x = x.reshape(B, V * T * s.P, s.D)
        temb = s.temb(tau)                                # (B,T,D)
        for blk in s.blocks:
            x = blk(x, temb, V, T, s.P)
        # final adaLN(逐帧 τ) + head
        mf = s.mod_f(temb)[:, None, :, None].expand(B, V, T, s.P, 2 * s.D).reshape(B, V * T * s.P, 2 * s.D)
        sh, sc = mf.chunk(2, -1)
        x = modulate(s.nf(x), sh, sc)
        return s._unpatchify(s.head(x), T)

    def _patchify_cond(s, c):                             # (B,V,Ccond,T,gh,gw) -> (B,V*T*P,Ccond*p*p)
        B, V, Cc, T, gh, gw = c.shape; p = s.p
        c = c.reshape(B, V, Cc, T, s.ph, p, s.pw, p).permute(0, 1, 3, 4, 6, 2, 5, 7)
        return c.reshape(B, V * T * s.P, Cc * p * p)


if __name__ == "__main__":
    B, V, C, T, g, Cc = 2, 2, 48, 5, 16, 7
    m = VideoDiT(C=C, Ccond=Cc, V=V, gh=g, gw=g, D=512, depth=12, heads=8, patch=2)
    n = sum(p.numel() for p in m.parameters())
    z = torch.randn(B, V, C, T, g, g); tau = torch.rand(B, T); cond = torch.randn(B, V, Cc, T, g, g)
    v = m(z, tau, cond)
    print(f"VideoDiT params={n/1e6:.1f}M  z{tuple(z.shape)} -> v{tuple(v.shape)}  (期望同形)")
    assert v.shape == z.shape, "shape mismatch"
    print("shape smoke OK")
