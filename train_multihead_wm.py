"""训练统一共享-trunk 多头 video WM(2026-07-25, spec 2026-07-25-human-helps-aux-target-screening)。
镜像 train_video_dit.py(DF/I₀锚/RF速度/MIX=human/warm-start/TLCAP/PRED),换 MultiHeadVideoWM +
抽象辅助头 loss。★不改 video_dit.py/train_video_dit.py(在飞实验在用),只 import。头一个一个加: 先 DINO。

抽象头机制(EgoWAM,但我们保留可解码 latent 主头):抽象目标(DINO)从 trunk(处理带噪 latent) per-frame 预测干净目标,
只训练期塑造共享 trunk;human 在抽象头上同域可迁移 → 塑造 trunk → 主 latent 头读到更好动力学。
L = rf_loss(主 latent 速度) + Σ_k LAM_k · MSE(aux_head_k(trunk), 干净目标_k)。

★对齐: DINO 建了子集(N_dino),latent 全量(同 clips 顺序)→ DINO[i]↔lat[i],AUX 开时 idx 限制在 <N_dino。
env: 承 train_video_dit(LAT/COND/OUT/STEPS/BS/LR/DEPTH/D/OVERFIT/CCOND/TLCAP/PRED/MIX/LAT_H/COND_H/HFRAC/INIT)
  + AUX(逗号列, 如 "dino") / DINO_R / DINO_H / LAM_DINO。
"""
import os, sys, numpy as np, torch
sys.path.insert(0, "."); os.environ.setdefault("HF_HUB_OFFLINE", "1")
from video_multihead_wm import MultiHeadVideoWM

LATP = os.environ.get("LAT", "outputs/video_arch_wm/wan_latents_can_dual/latents_all.npz")
CONDP = os.environ.get("COND", "outputs/video_arch_wm/cond_can_dual/cond_all.npz")
OUT = os.environ.get("OUT", "outputs/video_arch_wm/mh_wm"); os.makedirs(OUT, exist_ok=True)
STEPS = int(os.environ.get("STEPS", "40000")); BS = int(os.environ.get("BS", "16"))
LR = float(os.environ.get("LR", "1e-4")); DEPTH = int(os.environ.get("DEPTH", "12")); D = int(os.environ.get("D", "512"))
OVERFIT = int(os.environ.get("OVERFIT", "0")); CCOND = int(os.environ.get("CCOND", "0"))
HELDOUT = np.array([int(x) for x in os.environ.get("HELDOUT", "").split(",") if x])  # 遗留: 排除的 held-out clip id
HELDOUT_VIDS = [int(x) for x in os.environ.get("HELDOUT_VIDS", "").split(",") if x]   # ★episode(vid)分组排除(修帧泄漏, 优先于 clip-id)
TLCAP = int(os.environ.get("TLCAP", "0")); PRED = os.environ.get("PRED", "abs")
MIX = os.environ.get("MIX", "")
LAT_H = os.environ.get("LAT_H", "outputs/video_arch_wm/wan_latents_can_dual/latents_human_all.npz")
COND_H = os.environ.get("COND_H", "outputs/video_arch_wm/cond_can_dual/cond_human_eef3_all.npz")
HFRAC = float(os.environ.get("HFRAC", "0.5")); INIT = os.environ.get("INIT", "")
LAM_MAIN = float(os.environ.get("LAM_MAIN", "1.0"))         # 主latent loss权重; Stage-1抽象头pretrain设0(只aux塑trunk)
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "2000")); SNAP_EVERY = int(os.environ.get("SNAP_EVERY", "0"))
import re as _re
AUX = [a for a in _re.split("[,+]", os.environ.get("AUX", "")) if a]  # 抽象头列, "dino,mask,hc" 或 "dino+mask+hc"(sbatch --export 用+避开逗号)
AUX_DIR = os.environ.get("AUX_DIR", "outputs/video_arch_wm/aux_targets")
LAM = {k: float(os.environ.get(f"LAM_{k.upper()}", "0.5")) for k in AUX}   # 每头权重 env LAM_DINO/LAM_MASK/...
def aux_path(kind, src): return os.environ.get(f"{kind.upper()}_{src[0].upper()}", f"{AUX_DIR}/{kind}_{src}.npz")
dev = "cuda"


def load_one(latp, condp):
    zl = np.load(latp); lat, lat_low = zl["lat"], zl["lat_low"]; tL = int(zl["tL"])
    cond = np.load(condp)["cond"]
    valid = zl["low_valid"] if "low_valid" in zl.files else np.ones(len(lat), bool)
    if TLCAP and tL > TLCAP:
        lat = lat[:, :, :TLCAP]; lat_low = lat_low[:, :, :TLCAP]; cond = cond[:, :, :, :TLCAP]; tL = TLCAP
    if CCOND: cond = cond[:, :, :CCOND]
    if os.environ.get("NOFLOW", "0") == "1":         # ★naive baseline(=IWS-stage2式): 清零 object-flow 通道(前3=dx,dy,footprint), 只留 agent/eef cond
        cond = cond.copy(); cond[:, :, :3] = 0.0      # flow-cond vs naive-cond 干净 ablation(同 backbone/数据, 只去 flow)
    nz = (np.abs(lat).reshape(len(lat), -1).sum(1) > 0) & valid
    idx = np.where(nz)[0]
    if HELDOUT_VIDS and "vids" in zl.files:                 # ★episode(vid)分组排除, 修帧泄漏(project_clip_heldout_leakage)
        _uu = np.asarray(zl["vids"]); _cpe = len(lat) // len(_uu)   # clips/episode(连续排列,150); vids=18个episode id 非per-clip
        _pcvid = np.repeat(_uu, _cpe)                               # (N,) per-clip episode id
        idx = idx[~np.isin(_pcvid[idx], HELDOUT_VIDS)]
    elif len(HELDOUT):
        idx = idx[~np.isin(idx, HELDOUT)]                    # 遗留 clip-id 排除(防 eval SEQS 污染)
    if OVERFIT: idx = idx[:OVERFIT]                          # scarce = 排除held-out后的前N
    return lat, lat_low, cond, idx, tL


def load_aux(path, tL):
    """DINO feat (N,2,K,tL0,16,16) f16 -> TLCAP 切 + 单位std归一 (aux MSE 尺度可控)。返回 feat, N。"""
    d = np.load(path); f = d["feat"].astype(np.float32)         # (N,2,K,tL0,16,16)
    if TLCAP and f.shape[3] > TLCAP: f = f[:, :, :, :TLCAP]
    assert f.shape[3] == tL, f"aux tL {f.shape[3]} != latent tL {tL}"
    std = f.std() + 1e-6
    return f / std, len(f)


def to_delta(z):
    Dd = z.clone(); Dd[:, :, :, 1:] = z[:, :, :, 1:] - z[:, :, :, :-1]; return Dd


def from_delta(Dd):
    return torch.cumsum(Dd, dim=3)


def losses(model, z, cond, aux_tgt):
    """返回 (main_loss, {name:aux_loss})。aux 从同一带噪 forward 的 trunk 预测干净目标, 不算首帧。"""
    x1 = to_delta(z) if PRED == "delta" else z
    B, V, C, T, g, _ = x1.shape
    x0 = torch.randn_like(x1)
    t = torch.rand(B, T, device=dev); t[:, 0] = 1.0
    tb = t[:, None, None, :, None, None]
    xt = (1 - tb) * x0 + tb * x1
    xt[:, :, :, 0] = x1[:, :, :, 0]
    want = len(aux_tgt) > 0
    out = model(xt, t, cond, want_aux=want)
    v_pred, aux_pred = out if want else (out, {})
    v_tgt = x1 - x0
    main = ((v_pred - v_tgt) ** 2)[:, :, :, 1:].mean()
    aux_losses = {name: ((aux_pred[name] - aux_tgt[name]) ** 2)[:, :, :, 1:].mean() for name in aux_tgt}
    return main, aux_losses


@torch.no_grad()
def sample(model, z0_anchor, cond, steps=20):
    B, V, C, _, g, _ = z0_anchor.shape; T = cond.shape[3]
    x = torch.randn(B, V, C, T, g, g, device=dev)
    x[:, :, :, :1] = z0_anchor
    for i in range(steps):
        t = torch.full((B, T), i / steps, device=dev); t[:, 0] = 1.0
        v = model(x, t, cond)                                   # want_aux 默认 False → 只主头
        x = x + v / steps; x[:, :, :, :1] = z0_anchor
    return from_delta(x) if PRED == "delta" else x


def main():
    print(f"[cfg] AUX={AUX} PRED={PRED} MIX={MIX or 'robot-only'} TLCAP={TLCAP} INIT={'y' if INIT else 'n'}", flush=True)
    lat, lat_low, cond, idx, tL = load_one(LATP, CONDP)
    Ccond = cond.shape[2]; Cz = lat.shape[1]

    # 抽象辅助目标(dino/mask/hc, 一个一个加)。★对齐: 限制 idx 在有 aux 的 clip 内。
    aux_r, aux_h, aux_specs = {}, {}, {}
    for k in AUX:
        fr, nr = load_aux(aux_path(k, "robot"), tL); aux_r[k] = fr; aux_specs[k] = fr.shape[2]
        idx = idx[idx < nr]
        print(f"[aux] {k} robot: {nr} clips, ch={fr.shape[2]}; robot idx 限到 {len(idx)}", flush=True)
    print(f"robot clips: {len(idx)} (tL={tL}) lat{lat.shape} cond{cond.shape}", flush=True)

    lat_h = lat_low_h = cond_h = idx_h = None
    if MIX == "human":
        lat_h, lat_low_h, cond_h, idx_h, tLh = load_one(LAT_H, COND_H)
        if int(os.environ.get("HUMAN_N", "0")): idx_h = idx_h[:int(os.environ["HUMAN_N"])]   # ★限human池大小(测human-scaling)
        assert tLh == tL and cond_h.shape[2] == Ccond, f"human tL/Ccond 不匹配: {tLh}/{cond_h.shape[2]}"
        for k in AUX:
            fh, nh = load_aux(aux_path(k, "human"), tL); aux_h[k] = fh; idx_h = idx_h[idx_h < nh]
            print(f"[aux] {k} human: {nh} clips; human idx 限到 {len(idx_h)}", flush=True)
        print(f"human clips: {len(idx_h)} (tL={tLh}) HFRAC={HFRAC}", flush=True)

    model = MultiHeadVideoWM(C=Cz, Ccond=Ccond, aux_specs=aux_specs, V=2, gh=16, gw=16,
                             D=D, depth=DEPTH, heads=8, patch=2).to(dev)
    ema = MultiHeadVideoWM(C=Cz, Ccond=Ccond, aux_specs=aux_specs, V=2, gh=16, gw=16,
                           D=D, depth=DEPTH, heads=8, patch=2).to(dev)
    if INIT:
        src = torch.load(INIT, map_location=dev, weights_only=False)
        sd = src.state_dict() if hasattr(src, "state_dict") else src
        model.load_state_dict(sd, strict=False); print(f"[warm-start] loaded {INIT} (strict=False, aux 保留 init)", flush=True)
    ema.load_state_dict(model.state_dict()); [p.requires_grad_(False) for p in ema.parameters()]
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    # ★真 resume: 存 model/ema/opt/step 的 resume.pt → 续训从 saved step 跑到 STEPS(LR恒定, 无近似).
    # RESUME=1|auto → 读 {OUT}/resume.pt; RESUME=<path> → 读该文件; 空=不 resume. resume 覆盖 INIT warm-start.
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
    print(f"MultiHeadVideoWM {sum(p.numel() for p in model.parameters())/1e6:.1f}M  aux_specs={aux_specs}", flush=True)

    def batch(bs):
        def gather(sel, lo, la, co, ax):                        # 返回 z,c,aux(dict)
            z = np.stack([lo[sel], la[sel]], 1)
            a = {name: ax[name][sel] for name in ax}
            return z, co[sel], a
        if MIX == "human":
            nh = int(round(bs * HFRAC)); nr = bs - nh
            sr = np.random.choice(idx, nr); sh = np.random.choice(idx_h, nh)
            zr, cr, ar = gather(sr, lat, lat_low, cond, aux_r)
            zh, ch, ah = gather(sh, lat_h, lat_low_h, cond_h, aux_h)
            z = np.concatenate([zr, zh], 0); c = np.concatenate([cr, ch], 0)
            a = {name: np.concatenate([ar[name], ah[name]], 0) for name in aux_r}
        else:
            sel = np.random.choice(idx, bs)
            z, c, a = gather(sel, lat, lat_low, cond, aux_r)
        tt = lambda x: torch.from_numpy(x.astype(np.float32)).to(dev)
        return tt(z), tt(c), {name: tt(a[name]) for name in a}

    run_m = run_a = 0.0
    for step in range(start_step + 1, STEPS + 1):
        z, c, at = batch(BS)
        main_l, aux_l = losses(model, z, c, at)
        loss = LAM_MAIN * main_l + sum(LAM.get(k, 0.5) * v for k, v in aux_l.items())   # Stage-1 LAM_MAIN=0=只aux塑trunk
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()): pe.mul_(0.999).add_(pm, alpha=0.001)
        run_m += main_l.item(); run_a += sum(v.item() for v in aux_l.values())
        if step % 100 == 0:
            print(f"step {step}/{STEPS}  main {run_m/100:.4f}  aux {run_a/100:.4f}", flush=True); run_m = run_a = 0.0
        if SNAP_EVERY and step % SNAP_EVERY == 0: torch.save(ema, f"{OUT}/mh_ema_step{step}.pt")
        if step % EVAL_EVERY == 0 or step == STEPS:
            torch.save(ema, f"{OUT}/mh_ema.pt")
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                        "opt": opt.state_dict(), "step": step}, f"{OUT}/resume.pt")  # ★可 resume
            z, c, _ = batch(min(BS, 8))
            xs = sample(ema, z[:, :, :, :1], c, steps=20)
            rec = ((xs - z) ** 2)[:, :, :, 1:].mean().item()
            print(f"  [eval] step {step}: latent recon MSE {rec:.4f}  ckpt saved", flush=True)
    print("=== multihead train DONE ===", flush=True)


if __name__ == "__main__":
    main()
