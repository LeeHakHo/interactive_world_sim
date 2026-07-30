"""M4: 训练小视频 DiT ③ (replay: GT flow 条件). grounded 见 spec §10。
目标 = rectified-flow 速度 + Diffusion Forcing(逐帧独立 τ)+ 首帧 I₀ 锚(frame0 clean, 不算 loss)。
数据: Wan latents(build_can256_latents) + cond(build_can_cond)。用 .venv_wan/bin/python(可内联 decode)。
flow-matching 约定: x0=noise, x1=data, xt=(1-t)x0 + t x1, 目标 v=x1-x0; 推理 Euler t:0->1。

★human-helps-③ 实验扩展(2026-07-24, [[project_human_helps_renderer_exp]]):
  PRED=abs(默认,绝对z 对照) | delta(★主, 预测 consecutive-Δz=z_t-z_{t-1}, human 共享 per-step 动力学)
    delta 重参数: D[...,0]=z0, D[...,t]=z_t-z_{t-1}; 重建 z=cumsum(D)。首帧钳 clean(=z0, 两模式一致)。
  MIX=""(robot-only) | human(混 human, HFRAC 比例); LAT_H/COND_H = human latent/cond。
  INIT=warm-start ckpt(从 robot-only 续训, OSCAR 式); TLCAP=统一 tL(robot12→6 匹配 human24帧)。
env: LAT/COND/OUT/STEPS/BS/LR/DEPTH/D/OVERFIT/CCOND/TLCAP/PRED/MIX/LAT_H/COND_H/HFRAC/INIT。
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
CCOND = int(os.environ.get("CCOND", "0"))                     # >0: 只用 cond 前 CCOND 通道 (warp-off=4)
TLCAP = int(os.environ.get("TLCAP", "0"))                     # >0: 截 tL 到该值 (robot12->6 匹配 human)
PRED = os.environ.get("PRED", "abs")                          # abs(绝对z) | delta(consecutive-Δz)
MIX = os.environ.get("MIX", "")                              # "" robot-only | "human" 混 human
LAT_H = os.environ.get("LAT_H", "outputs/video_arch_wm/wan_latents_can_dual/latents_human_all.npz")
COND_H = os.environ.get("COND_H", "outputs/video_arch_wm/cond_can_dual/cond_human_eef3_all.npz")
HFRAC = float(os.environ.get("HFRAC", "0.5"))                # 混训时 human 占 batch 比例
INIT = os.environ.get("INIT", "")                           # warm-start: 从该 ckpt 初始化权重
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "2000"))
SNAP_EVERY = int(os.environ.get("SNAP_EVERY", "0"))          # >0: 每 N 步另存带步数快照(挑最优/看进程)
dev = "cuda"


def load_one(latp, condp):
    """加载一域 latent+cond, 应用 CCOND/TLCAP, 返回 lat,lat_low,cond,idx(有效clip),tL。"""
    zl = np.load(latp); lat, lat_low = zl["lat"], zl["lat_low"]; tL = int(zl["tL"])
    cond = np.load(condp)["cond"]                            # (N,2,Cc,tL,16,16)
    valid = zl["low_valid"] if "low_valid" in zl.files else np.ones(len(lat), bool)
    if TLCAP and tL > TLCAP:                                 # 截 tL 统一时序长度
        lat = lat[:, :, :TLCAP]; lat_low = lat_low[:, :, :TLCAP]; cond = cond[:, :, :, :TLCAP]; tL = TLCAP
    if CCOND: cond = cond[:, :, :CCOND]                       # 截条件通道 (warp-off=4)
    nz = (np.abs(lat).reshape(len(lat), -1).sum(1) > 0) & valid
    idx = np.where(nz)[0]
    if OVERFIT: idx = idx[:OVERFIT]
    return lat, lat_low, cond, idx, tL


def to_delta(z):                                            # (B,V,C,T,g,g) -> D: D[...,0]=z0, D[...,t]=z_t-z_{t-1}
    Dd = z.clone(); Dd[:, :, :, 1:] = z[:, :, :, 1:] - z[:, :, :, :-1]; return Dd


def from_delta(Dd):                                          # 重建 z = cumsum(D) over T (dim=3)
    return torch.cumsum(Dd, dim=3)


def rf_loss(model, z, cond):
    """z:(B,2,48,tL,16,16) clean; cond:(B,2,Cc,tL,16,16). PRED=delta 时目标改增量序列。"""
    x1 = to_delta(z) if PRED == "delta" else z              # 预测目标: 绝对z 或 逐帧增量D
    B, V, C, T, g, _ = x1.shape
    x0 = torch.randn_like(x1)
    t = torch.rand(B, T, device=dev)                         # 逐帧独立 τ (Diffusion Forcing)
    t[:, 0] = 1.0                                            # 首帧锚: clean
    tb = t[:, None, None, :, None, None]
    xt = (1 - tb) * x0 + tb * x1
    xt[:, :, :, 0] = x1[:, :, :, 0]                          # 首帧强制 clean (delta: =z0; abs: =z0)
    v_pred = model(xt, t, cond)
    v_tgt = x1 - x0
    loss = ((v_pred - v_tgt) ** 2)[:, :, :, 1:].mean()       # 不算首帧
    return loss


@torch.no_grad()
def sample(model, z0_anchor, cond, steps=20):
    """从 noise 采整段(首帧=z0_anchor clean). PRED=delta 时采增量序列再 cumsum 回 z。
    z0_anchor:(B,2,48,1,16,16); cond:(B,2,Cc,tL,16,16)。返回绝对 z (两模式统一)。"""
    B, V, C, _, g, _ = z0_anchor.shape; T = cond.shape[3]
    x = torch.randn(B, V, C, T, g, g, device=dev)
    x[:, :, :, :1] = z0_anchor                               # delta: D[...,0]=z0; abs: z[...,0]=z0
    for i in range(steps):
        t = torch.full((B, T), i / steps, device=dev); t[:, 0] = 1.0
        v = model(x, t, cond)
        x = x + v / steps
        x[:, :, :, :1] = z0_anchor
    return from_delta(x) if PRED == "delta" else x           # delta: 积分回绝对 z


def main():
    print(f"[cfg] PRED={PRED} MIX={MIX or 'robot-only'} CCOND={CCOND} TLCAP={TLCAP} INIT={'yes' if INIT else 'no'}", flush=True)
    lat, lat_low, cond, idx, tL = load_one(LATP, CONDP)
    Ccond = cond.shape[2]; Cz = lat.shape[1]
    print(f"robot clips: {len(idx)} (tL={tL}) lat{lat.shape} cond{cond.shape}", flush=True)
    lat_h = lat_low_h = cond_h = idx_h = None
    if MIX == "human":
        lat_h, lat_low_h, cond_h, idx_h, tLh = load_one(LAT_H, COND_H)
        assert tLh == tL and cond_h.shape[2] == Ccond, f"human tL/Ccond 不匹配: {tLh}/{cond_h.shape[2]} vs {tL}/{Ccond}"
        print(f"human clips: {len(idx_h)} (tL={tLh}) lat{lat_h.shape} cond{cond_h.shape}  HFRAC={HFRAC}", flush=True)

    model = VideoDiT(C=Cz, Ccond=Ccond, V=2, gh=16, gw=16, D=D, depth=DEPTH, heads=8, patch=2).to(dev)
    ema = VideoDiT(C=Cz, Ccond=Ccond, V=2, gh=16, gw=16, D=D, depth=DEPTH, heads=8, patch=2).to(dev)
    if INIT:                                                 # warm-start: 从 robot-only ckpt 载权重
        src = torch.load(INIT, map_location=dev, weights_only=False)
        sd = src.state_dict() if hasattr(src, "state_dict") else src
        model.load_state_dict(sd); print(f"[warm-start] loaded {INIT}", flush=True)
    ema.load_state_dict(model.state_dict()); [p.requires_grad_(False) for p in ema.parameters()]
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    # ★真 resume: RESUME=1|auto→{OUT}/resume.pt; RESUME=<path>→该文件; 空=不resume(覆盖INIT). LR恒定→无近似.
    start_step = 0
    RESUME = os.environ.get("RESUME", "")
    resume_path = f"{OUT}/resume.pt" if RESUME in ("1", "auto") else RESUME
    if RESUME and os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        start_step = int(ck["step"])
        print(f"[resume] {resume_path} @ step {start_step} → 续训到 {STEPS}", flush=True)
    elif RESUME:
        print(f"[resume] 请求 RESUME={RESUME} 但 {resume_path} 不存在 → 从头训", flush=True)
    print(f"VideoDiT {sum(p.numel() for p in model.parameters())/1e6:.1f}M  steps={STEPS} bs={BS} Ccond={Ccond}", flush=True)

    def batch(bs):
        if MIX == "human":
            nh = int(round(bs * HFRAC)); nr = bs - nh
            sr = np.random.choice(idx, nr); sh = np.random.choice(idx_h, nh)
            z = np.concatenate([np.stack([lat[sr], lat_low[sr]], 1), np.stack([lat_h[sh], lat_low_h[sh]], 1)], 0)
            c = np.concatenate([cond[sr], cond_h[sh]], 0)
        else:
            sel = np.random.choice(idx, bs)
            z = np.stack([lat[sel], lat_low[sel]], 1); c = cond[sel]
        return torch.from_numpy(z.astype(np.float32)).to(dev), torch.from_numpy(c.astype(np.float32)).to(dev)

    run = 0.0
    for step in range(start_step + 1, STEPS + 1):
        z, c = batch(BS)
        loss = rf_loss(model, z, c)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()): pe.mul_(0.999).add_(pm, alpha=0.001)
        run += loss.item()
        if step % 100 == 0:
            print(f"step {step}/{STEPS}  loss {run/100:.4f}", flush=True); run = 0.0
        if SNAP_EVERY and step % SNAP_EVERY == 0:            # 带步数快照, 便于挑最优/看进程
            torch.save(ema, f"{OUT}/video_dit_ema_step{step}.pt")
        if step % EVAL_EVERY == 0 or step == STEPS:
            torch.save(ema, f"{OUT}/video_dit_ema.pt")
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                        "opt": opt.state_dict(), "step": step}, f"{OUT}/resume.pt")  # ★可 resume
            # 采样重建误差(绝对 z 域, EMA; delta 模式已 cumsum 回 z -> 口径与 abs 一致)
            z, c = batch(min(BS, 8))
            xs = sample(ema, z[:, :, :, :1], c, steps=20)
            rec = ((xs - z) ** 2)[:, :, :, 1:].mean().item()
            print(f"  [eval] step {step}: latent recon MSE {rec:.4f}  ckpt saved", flush=True)
    print("=== M4 train DONE ===", flush=True)


if __name__ == "__main__":
    main()
