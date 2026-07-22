"""M4: 训练小视频 DiT ③ (replay: GT flow 条件). grounded 见 spec §10。
目标 = rectified-flow 速度 + Diffusion Forcing(逐帧独立 τ)+ 首帧 I₀ 锚(frame0 clean, 不算 loss)。
数据: Wan latents(build_can256_latents) + 7ch cond(build_can_cond)。用 .venv_wan/bin/python(可内联 decode)。
flow-matching 约定: x0=noise, x1=data, xt=(1-t)x0 + t x1, 目标 v=x1-x0; 推理 Euler t:0->1。
env: LAT=latents_all.npz COND=cond_all.npz OUT=... STEPS/BS/LR/DEPTH/D/OVERFIT(=clip数, 1=过拟合sanity)。
"""
import os, sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "."); os.environ.setdefault("HF_HUB_OFFLINE", "1")
from video_dit import VideoDiT

LATP = os.environ.get("LAT", "outputs/video_arch_wm/wan_latents_can_dual/latents_all.npz")
CONDP = os.environ.get("COND", "outputs/video_arch_wm/cond_can_dual/cond_all.npz")
OUT = os.environ.get("OUT", "outputs/video_arch_wm/m4_video_dit"); os.makedirs(OUT, exist_ok=True)
STEPS = int(os.environ.get("STEPS", "40000")); BS = int(os.environ.get("BS", "16"))
LR = float(os.environ.get("LR", "1e-4")); DEPTH = int(os.environ.get("DEPTH", "12")); D = int(os.environ.get("D", "512"))
OVERFIT = int(os.environ.get("OVERFIT", "0"))                 # >0: 只用前 N clip (sanity)
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "2000")); dev = "cuda"


def load_data():
    zl = np.load(LATP); lat, lat_low = zl["lat"], zl["lat_low"]; tL = int(zl["tL"])
    zc = np.load(CONDP); cond = zc["cond"]                    # (N,2,7,tL,16,16)
    valid = zl["low_valid"] if "low_valid" in zl.files else np.ones(len(lat), bool)
    # 只保留双视角有效 + 非全零 latent 的 clip
    nz = (np.abs(lat).reshape(len(lat), -1).sum(1) > 0) & valid
    idx = np.where(nz)[0]
    if OVERFIT: idx = idx[:OVERFIT]
    print(f"train clips: {len(idx)} (tL={tL})  lat{lat.shape} cond{cond.shape}", flush=True)
    return lat, lat_low, cond, idx, tL


def rf_loss(model, z, cond):
    """z:(B,2,48,tL,16,16) clean; cond:(B,2,7,tL,16,16). -> scalar loss (frames 1..)."""
    B, V, C, T, g, _ = z.shape
    x1 = z; x0 = torch.randn_like(z)
    t = torch.rand(B, T, device=dev)                         # 逐帧独立 τ (Diffusion Forcing)
    t[:, 0] = 1.0                                            # 首帧锚: clean
    tb = t[:, None, None, :, None, None]                     # broadcast (B,1,1,T,1,1)
    xt = (1 - tb) * x0 + tb * x1
    xt[:, :, :, 0] = x1[:, :, :, 0]                          # 首帧强制 clean
    v_pred = model(xt, t, cond)
    v_tgt = x1 - x0
    loss = ((v_pred - v_tgt) ** 2)[:, :, :, 1:].mean()       # 不算首帧
    return loss


@torch.no_grad()
def sample(model, z0_anchor, cond, steps=20):
    """从 noise 采整段(首帧=z0_anchor clean). z0_anchor:(B,2,48,1,16,16); cond:(B,2,7,tL,16,16)。"""
    B, V, C, _, g, _ = z0_anchor.shape; T = cond.shape[3]
    x = torch.randn(B, V, C, T, g, g, device=dev)
    x[:, :, :, :1] = z0_anchor
    for i in range(steps):
        t = torch.full((B, T), i / steps, device=dev); t[:, 0] = 1.0
        v = model(x, t, cond)
        x = x + v / steps
        x[:, :, :, :1] = z0_anchor
    return x


def main():
    lat, lat_low, cond, idx, tL = load_data()
    C = lat.shape[1]; Ccond = cond.shape[2]
    model = VideoDiT(C=C, Ccond=Ccond, V=2, gh=16, gw=16, D=D, depth=DEPTH, heads=8, patch=2).to(dev)
    ema = VideoDiT(C=C, Ccond=Ccond, V=2, gh=16, gw=16, D=D, depth=DEPTH, heads=8, patch=2).to(dev)
    ema.load_state_dict(model.state_dict()); [p.requires_grad_(False) for p in ema.parameters()]
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    print(f"VideoDiT {sum(p.numel() for p in model.parameters())/1e6:.1f}M  steps={STEPS} bs={BS}", flush=True)

    def batch(bs):
        sel = np.random.choice(idx, bs)
        z = np.stack([lat[sel], lat_low[sel]], 1).astype(np.float32)     # (B,2,48,tL,16,16)
        c = cond[sel].astype(np.float32)
        return torch.from_numpy(z).to(dev), torch.from_numpy(c).to(dev)

    run = 0.0
    for step in range(1, STEPS + 1):
        z, c = batch(BS)
        loss = rf_loss(model, z, c)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()): pe.mul_(0.999).add_(pm, alpha=0.001)
        run += loss.item()
        if step % 100 == 0:
            print(f"step {step}/{STEPS}  loss {run/100:.4f}", flush=True); run = 0.0
        if step % EVAL_EVERY == 0 or step == STEPS:
            torch.save(ema, f"{OUT}/video_dit_ema.pt")
            # 采样重建误差(latent 域, EMA)
            z, c = batch(min(BS, 8))
            xs = sample(ema, z[:, :, :, :1], c, steps=20)
            rec = ((xs - z) ** 2)[:, :, :, 1:].mean().item()
            print(f"  [eval] step {step}: latent recon MSE {rec:.4f}  ckpt saved", flush=True)
    print("=== M4 train DONE ===", flush=True)


if __name__ == "__main__":
    main()
