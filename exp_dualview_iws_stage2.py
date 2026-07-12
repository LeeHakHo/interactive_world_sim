"""M2.5 external baseline: 真 IWS stage2 dynamics (CMLatentDynamics: Conv3d 时空 backbone + 原生
action_emd->per-block FiLM 注入) 在 can_dual 上 diffusion-forcing 训练 + AR 多步去噪采样.
latent = 我们的 frozen 16ch VAE (双视角 stack 成 32ch, 同 IWS 双 view 32ch 惯例), 同数据同 eval 口径
(exp_scel_dualview_dit_formal.obj_lpips_audit / eval_seqs). 监督配方与 formal 臂一致.
非 controlled (backbone+接口两变量), paper 表 external baseline 行. 用户 2026-07-12 拍板 DF 方案.
Env: SEED, EPOCHS(40), SMOKE, INFER_STEPS(10), T_WIN(8). iws env + GPU.
-> outputs/cross_embodiment_wm/dualview_iws_stage2/s{SEED}/"""
import json
import os
import time

os.environ.setdefault("VAE_NAME", "ostris/vae-kl-f8-d16"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, torch, torch.nn as nn
from exp_scel_latent_renderer import enc, dec, latent_ch
from exp_scel_dualview_dit import load_dual, latcache, psnr, _fp, u8, FLOW_SCALE, H
from exp_scel_dualview_dit_formal import obj_lpips_audit, eval_seqs, ROOT as FORMAL_ROOT
from exp_v3_human_helps_pixels import K, IMG, device
from interactive_world_sim.algorithms.latent_dynamics.models.cm_latent_dynamics import CMLatentDynamics

SMOKE = os.environ.get("SMOKE", "0") == "1"
SEED = int(os.environ.get("SEED", "0"))
EPOCHS = 2 if SMOKE else int(os.environ.get("EPOCHS", "40"))
T_WIN = int(os.environ.get("T_WIN", "8")); NLEV = 1000
INFER_STEPS = int(os.environ.get("INFER_STEPS", "10"))
BS = 8; LR = 2e-4; LAM = 1.0
OUT = f"outputs/cross_embodiment_wm/dualview_iws_stage2/s{SEED}" + ("_smoke" if SMOKE else "")
os.makedirs(f"{OUT}/gifs", exist_ok=True)
VERSIONS = ("IWS stage2 CMLatentDynamics (Conv3d+action_emd FiLM, DF train/sample) | "
            "VAE=ostris/vae-kl-f8-d16(frozen, 2view stack 32ch) | data=can_dual | 监督配方同 formal 臂")


def make_sched(n=NLEV):
    """cosine alpha_bar (Nichol&Dhariwal), 单调减."""
    t = np.linspace(0, 1, n)
    ab = np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2 / np.cos(np.array(0.008) / 1.008 * np.pi / 2) ** 2
    return torch.tensor(ab, dtype=torch.float32)


def frame_actions(D, j, ts):
    """clip j 的帧集 ts -> (len(ts), 24) per-frame action: 每 view 3 eef 点(wrist+2指尖) x
    [x,y,(dx,dy)*FLOW_SCALE step 速度] = 12-dim, 双视角 concat. 全量向量对齐 IWS 原生 7-dim 完整动作精神
    (指尖隐含 grip 开合), 与 formal eeffilm 臂同信息 (那边是 frame0 锚定, 这边 AR per-step 速度)."""
    ts = np.asarray(ts); prev = np.maximum(ts - 1, 0); outs = []
    for v in range(2):
        e = np.asarray(D["ef"][v][j], np.float32)                     # (L,3,2)
        d = (e[ts] - e[prev]) * FLOW_SCALE                            # (T,3,2)
        outs.append(np.concatenate([e[ts], d], -1).reshape(len(ts), -1))   # (T,12)
    return np.concatenate(outs, -1).astype(np.float32)


def _noise(x0, lv, ab):
    """x0 (B,C,T,16,16), lv (B,T) long -> x_t, 按 per-frame 独立噪声等级 (diffusion forcing)."""
    a = ab.to(x0.device)[lv][:, None, :, None, None]                  # (B,1,T,1,1)
    return a.sqrt() * x0 + (1 - a).sqrt() * torch.randn_like(x0)


def train_iws(R, Hh, pool, okh, latR, latH, sched):
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    m = CMLatentDynamics(latent_dim=2 * latent_ch(), action_dim=24, dim=64).to(device)
    print(f"IWS-DF params={sum(p.numel() for p in m.parameters())/1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    from interactive_world_sim.algorithms.common.metrics.lpips import LearnedPerceptualImagePatchSimilarity
    from exp_scel_latent_lpips import _decode_grad
    lp = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device).eval()
    for p in lp.parameters(): p.requires_grad_(False)
    samples = np.array([("r", int(j)) for j in pool] + [("h", int(j)) for j in okh], dtype=object)
    fr01 = lambda a: a.astype(np.float32).transpose(2, 0, 1) / 255.0
    for ep in range(EPOCHS):
        m.train(); order = rng.permutation(len(samples)); tot = 0; nb = 0
        for i in range(0, len(order), BS):
            bidx = order[i:i + BS]
            x0s, acts, gts, mks, doms, fsel = [], [], [], [], [], []
            for si in bidx:
                dom, j = samples[si]; D = R if dom == "r" else Hh; lc = latR if dom == "r" else latH
                L = D["fr"][0].shape[1]
                t0 = int(rng.integers(0, L - T_WIN + 1)); ts = np.arange(t0, t0 + T_WIN)
                z = lc[j, ts].astype(np.float32)                       # (T,2,Cz,16,16)
                x0s.append(z.reshape(T_WIN, -1, z.shape[-2], z.shape[-1]).transpose(1, 0, 2, 3))  # (32,T,16,16)
                acts.append(frame_actions(D, j, ts))
                fs = int(rng.integers(0, T_WIN)); fsel.append(fs)      # 像素损失只解码 1 帧/样本 (省算)
                gts.append([fr01(D["fr"][v][j, ts[fs]]) for v in range(2)])
                if dom == "h":
                    mks.append(np.stack([_fp(np.asarray(D["tr"][v][j][ts[fs]], np.float32)) for v in range(2)]))
                else:
                    mks.append(np.zeros((2, IMG, IMG), np.float32))
                doms.append(0 if dom == "r" else 1)
            f2 = lambda a: torch.from_numpy(np.stack(a).astype(np.float32)).to(device)
            x0 = f2(x0s); act = f2(acts); GT = f2(gts); mk = f2(mks)                # mk (B,2,H,W)
            dom = torch.tensor(doms, device=device); fs = torch.tensor(fsel, device=device)
            lv = torch.randint(0, NLEV, (len(bidx), T_WIN), device=device)
            xt = _noise(x0, lv, sched)
            x0p = m(xt, lv.T, torch.zeros_like(lv.T), act.transpose(0, 1))            # x0-pred, stop=0
            rs = (dom == 0); hs = (dom == 1)
            loss = ((x0p[rs] - x0[rs]) ** 2).mean() if rs.any() else 0.0 * x0p.sum()
            zi = x0p[torch.arange(len(bidx)), :, fs]                                  # (B,32,16,16) 选中帧
            img = _decode_grad(zi.reshape(len(bidx) * 2, latent_ch(), 16, 16)).clamp(0, 1).reshape(len(bidx), 2, 3, IMG, IMG)
            if rs.any():
                loss = loss + LAM * lp(img[rs].reshape(-1, 3, IMG, IMG) * 2 - 1, GT[rs].reshape(-1, 3, IMG, IMG) * 2 - 1)
            if hs.any():
                om = mk[hs][:, :, None]                                               # (Bh,2,1,H,W)
                loss = loss + (((img[hs] - GT[hs]) ** 2) * om).sum() / (om.sum() * 3 + 1e-6)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss); nb += 1
        if ep % 5 == 0 or ep == EPOCHS - 1: print(f"  iws-df ep{ep} loss={tot/nb:.4f}", flush=True)
    return m.eval()


@torch.no_grad()
def rollout_iws(m, z0, acts, Hn, sched, infer_steps=INFER_STEPS, t_win=T_WIN, device=device):
    """IWS dynamics_forward 同构 AR 采样: chunk=1, 滑窗 t_win, 每帧 infer_steps 步 x0-pred DDIM 去噪.
    z0 (1,C,1,16,16) 干净首帧; acts (1,Tmax,24). -> (1,C,Hn,16,16)"""
    ab = sched.to(device); xs = z0.to(device).clone()
    for h in range(Hn):
        xs = torch.cat([xs, torch.randn_like(xs[:, :, :1])], 2)
        start = max(0, xs.shape[2] - t_win)
        win = xs[:, :, start:].clone(); Tw = win.shape[2]
        act_w = acts[:, start:start + Tw].transpose(0, 1).to(device)               # (Tw,1,24)
        levels = torch.linspace(ab.shape[0] - 1, 0, infer_steps + 1).long()        # schedule len, not NLEV const
        for i in range(infer_steps):
            lv = torch.zeros(Tw, 1, dtype=torch.long, device=device); lv[-1] = levels[i]
            x0p = m(win, lv, torch.zeros_like(lv), act_w)
            s_i = int(levels[i + 1])
            if s_i > 0:
                a = ab[s_i]
                win[:, :, -1:] = a.sqrt() * x0p[:, :, -1:] + (1 - a).sqrt() * torch.randn_like(x0p[:, :, -1:])
            else:
                win[:, :, -1:] = x0p[:, :, -1:]
        xs[:, :, start:] = win
    return xs[:, :, 1:]


@torch.no_grad()
def render_iws(m, R, si, sched):
    """同 formal render 口径: z0=frame0 双视角 stack, AR 到 K+H-1, 取帧 K..K+H-1 解码. -> (2,H,128,128,3)"""
    fr01 = lambda a: torch.from_numpy(a.astype(np.float32).transpose(2, 0, 1)[None] / 255.0).to(device)
    z0 = torch.cat([enc(fr01(R["fr"][v][si, 0])) for v in range(2)], 1)[:, :, None]   # (1,32,1,16,16)
    ts = np.arange(0, K + H)
    acts = torch.from_numpy(frame_actions(R, si, ts)[None])                           # (1,K+H,24)
    zs = rollout_iws(m, z0, acts, K + H - 1, sched)[0]                                # (32,K+H-1,16,16)
    outs = [[], []]
    for h in range(H):
        z = zs[:, K + h - 1]                                                          # 帧 K+h (rollout 从帧1计)
        for v in range(2):
            outs[v].append(dec(z[v * latent_ch():(v + 1) * latent_ch()][None])[0].cpu().numpy())
    return np.stack([np.stack(outs[v]).transpose(0, 2, 3, 1) for v in range(2)])


def main():
    t0 = time.time(); sched = make_sched()
    R, Hh = load_dual()
    okr = np.where(R["ok"])[0]; okh = np.where(Hh["ok"])[0]
    perm = np.random.default_rng(0).permutation(okr); ho, pool = perm[:150], perm[150:]
    if SMOKE: pool = pool[:80]; okh = okh[:80]
    chosen = eval_seqs(R, ho, n=(4 if SMOKE else 24))
    latR = latcache(R, "robot"); latH = latcache(Hh, "human")
    m = train_iws(R, Hh, pool, okh, latR, latH, sched)
    torch.save(m, f"{OUT}/iws_dyn.pt")
    from interactive_world_sim.algorithms.common.metrics.lpips import LearnedPerceptualImagePatchSimilarity
    lp = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device).eval()
    agg = {f"v{v}_{k}": [] for v in range(2) for k in ["ps", "lp"]}; det = {0: [0, 0], 1: [0, 0]}
    for si in chosen:
        rr = render_iws(m, R, si, sched)
        np.save(f"{OUT}/gifs/seq{si}_render.npy", u8(rr))
        for v in range(2):
            gtf = R["fr"][v][si, K:K + H].astype(np.float32) / 255.0
            objm = np.stack([_fp(R["tr"][v][si, K + h]) for h in range(H)])
            agg[f"v{v}_ps"].append(np.mean([psnr(rr[v, h], gtf[h]) for h in range(H)]))
            lpv, nv, nt = obj_lpips_audit(lp, rr[v].astype(np.float32), gtf, objm)
            agg[f"v{v}_lp"].append(lpv); det[v][0] += nv; det[v][1] += nt
    res = {f"v{v}_{k}": float(np.nanmean(agg[f"v{v}_{k}"])) for v in range(2) for k in ["ps", "lp"]}
    for v in range(2): res[f"det_rate_v{v}"] = det[v][0] / max(det[v][1], 1)
    res.update({"seed": SEED, "epochs": EPOCHS, "t_win": T_WIN, "infer_steps": INFER_STEPS,
                "wall_min": round((time.time() - t0) / 60, 1), "n_eval": len(chosen)})
    json.dump(res, open(f"{OUT}/metrics.json", "w"), indent=2)
    lines = [f"IWS stage2 DF external baseline | {VERSIONS}",
             f"cam_high: PSNR {res['v0_ps']:.2f} | obj-LPIPS {res['v0_lp']:.3f} | det {res['det_rate_v0']:.2f}",
             f"cam_low : PSNR {res['v1_ps']:.2f} | obj-LPIPS {res['v1_lp']:.3f} | det {res['det_rate_v1']:.2f}"]
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n"); print("\n".join(lines) + "\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
