"""第三列 (ablation): ours 的 VAE + latent transition (detmem 架构), 去掉 object-flow,
改用 eef-splat 条件, human+robot naive co-train.

目的: 隔离 object-flow 的贡献。三列对比 =
  ① ours          : object-flow ② + detmem ③ (VAE+latent transition + flow 条件)
  ② IWS-naive     : 别人 VAE+latent transition + eef naive co-train (hyeonhoo ckpt)
  ③ 本脚本        : 同一个 VAE+latent transition (detmem), eef 条件无 flow, robot+human co-train
若 ③ 掉到接近 ② → 证明赢在 object-flow, 不在 VAE/latent 实现。

★ co-train: human clips 无 grip/joint 但有 eef → 用 eef-splat (human/robot 同源) → 扛起 human+robot 论据。
数据: flow_render_dataset_v3 (robot 2700 + human 1800, 都有 frames/eef).
VAE enc/dec (16ch ostris) frozen, 只训 DetMemRenderer.
Output: outputs/cross_embodiment_wm/detmem_eef_cotrain/
"""
import os
os.environ.setdefault("USE_GMASK", "1"); os.environ.setdefault("FOOTPRINT", "1")
os.environ.setdefault("DROP_VIS", "1"); os.environ.setdefault("USE_PREV", "0")
import numpy as np, torch
from exp_scel_latent_detmem import DetMemRenderer, psnr, interframe
from exp_scel_latent_renderer import enc, dec
from exp_scel_latent_lpips import _decode_grad
from exp_v3_human_helps_pixels import K, IMG, device, cube_pos_err

SMOKE = os.environ.get("SMOKE", "0") == "1"
DS = os.environ.get("DS", "outputs/flow_render_dataset_v3")
MODE = os.environ.get("MODE", "rh")                       # rh = robot+human co-train; ro = robot-only (对照)
EPOCHS = 3 if SMOKE else int(os.environ.get("EPOCHS", "100"))
BS = 16; LR = 2e-4; LAM = float(os.environ.get("LAM_LPIPS", "1.0")); PREV_DF = 0.3
HELDOUT = 150; H = int(os.environ.get("HORIZON", "40")); NSEQ = int(os.environ.get("NSEQ", "8"))
OUT = os.environ.get("OUT_DIR", f"outputs/cross_embodiment_wm/detmem_eef_cotrain_{MODE}")
os.makedirs(f"{OUT}/gifs", exist_ok=True)
u8 = lambda x: (np.clip(x, 0, 1) * 255).astype(np.uint8)
_YS, _XS = np.mgrid[0:IMG, 0:IMG].astype(np.float32)


def eef_splat(ef3, sigma=4.0):
    """eef 3点 (3,2) 图像归一[0,1] -> (3,IMG,IMG) 高斯 splat (腕+2指尖各1ch). 无 flow/joint."""
    out = np.empty((3, IMG, IMG), np.float32)
    for i in range(3):
        cx, cy = float(ef3[i, 0]) * IMG, float(ef3[i, 1]) * IMG
        out[i] = np.exp(-((_XS - cx) ** 2 + (_YS - cy) ** 2) / (2 * sigma ** 2))
    return out


def train_eef_cotrain(fr, ef, idx):
    torch.manual_seed(0); m = DetMemRenderer(cond_ch=3).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    from interactive_world_sim.algorithms.common.metrics.lpips import LearnedPerceptualImagePatchSimilarity
    lp = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device).eval()
    for p in lp.parameters(): p.requires_grad_(False)
    rng = np.random.default_rng(0); L = fr.shape[1]
    for ep in range(EPOCHS):
        m.train(); pe = idx[rng.permutation(len(idx))]; tot = 0; nb = 0
        for i in range(0, len(pe), BS):
            b = pe[i:i + BS]; ts = rng.integers(9, L, size=len(b))
            I0s, conds, tgts, prevs = [], [], [], []
            for n, j in enumerate(b):
                t = int(ts[n]); I0s.append(fr[j, 0].astype(np.float32).transpose(2, 0, 1) / 255.0)
                conds.append(eef_splat(ef[j, t])); tgts.append(fr[j, t].astype(np.float32).transpose(2, 0, 1) / 255.0)
                pt = 0 if rng.random() < 0.15 else t - 1
                prevs.append(fr[j, pt].astype(np.float32).transpose(2, 0, 1) / 255.0)
            I0 = torch.from_numpy(np.stack(I0s)).float().to(device); cond = torch.from_numpy(np.stack(conds)).float().to(device)
            GT = torch.from_numpy(np.stack(tgts)).float().to(device)
            z0 = enc(I0); z1 = enc(GT); prev = enc(torch.from_numpy(np.stack(prevs)).float().to(device))
            a = torch.rand(len(b), 1, 1, 1, device=device) * PREV_DF; prev = (1 - a) * prev + a * torch.randn_like(prev)
            pred = m(z0, cond, prev)
            loss = ((pred - z1) ** 2).mean() + LAM * lp(_decode_grad(pred).clamp(0, 1) * 2 - 1, GT * 2 - 1)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
        if ep % 5 == 0 or ep == EPOCHS - 1: print(f"  detmem-eef[{MODE}] ep{ep} loss={tot/nb:.4f}", flush=True)
        if ep > 0 and ep % 20 == 0:                          # 周期存档 (可用中间结果)
            torch.save(m.eval(), f"{OUT}/detmem_eef.pt"); m.train()
    return m.eval()


@torch.no_grad()
def render_eef(m, I0, ef_seq):
    """eef-splat 条件自回归渲染 (无 flow). ef_seq (H,3,2) -> (H,128,128,3)."""
    z0 = enc(torch.from_numpy(I0.transpose(2, 0, 1).astype(np.float32) / 255.0)[None].to(device))
    prev = z0.clone(); outs = []
    for h in range(len(ef_seq)):
        cond = torch.from_numpy(eef_splat(ef_seq[h])[None]).float().to(device)
        pred = m(z0, cond, prev); prev = pred
        outs.append(dec(pred)[0].cpu().numpy())
    return np.stack(outs).transpose(0, 2, 3, 1)


def main():
    zr = np.load(f"{DS}/clips_robot.npz"); zh = np.load(f"{DS}/clips_human.npz")
    fr_r, ef_r = zr["frames"], zr["eef"].astype(np.float32)
    fr_h, ef_h = zh["frames"], zh["eef"].astype(np.float32)
    Nr = len(fr_r)
    perm = np.random.default_rng(0).permutation(Nr); ho, pool_r = perm[:HELDOUT], perm[HELDOUT:]
    fr_all = np.concatenate([fr_r, fr_h]); ef_all = np.concatenate([ef_r, ef_h])
    hi = np.arange(Nr, Nr + len(fr_h))
    pool = pool_r if MODE == "ro" else np.concatenate([pool_r, hi])     # ro=robot-only / rh=robot+human
    if SMOKE: pool = pool[:200]
    print(f"=== train detmem-eef [{MODE}] | robot {Nr}(pool {len(pool_r)}+ho {HELDOUT}) human {len(fr_h)} | train-pool {len(pool)} | EPOCHS={EPOCHS} ===", flush=True)
    m = train_eef_cotrain(fr_all, ef_all, pool); torch.save(m, f"{OUT}/detmem_eef.pt")
    # eval on held-out robot (eef replay)
    from interactive_world_sim.algorithms.common.metrics.lpips import LearnedPerceptualImagePatchSimilarity
    lp = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device).eval()
    lpf = lambda r, g: float(lp(torch.from_numpy(r).permute(0, 3, 1, 2).float().to(device) * 2 - 1, torch.from_numpy(g).permute(0, 3, 1, 2).float().to(device) * 2 - 1).mean())
    tr_r = zr["tracks"].astype(np.float32)
    He = min(H, fr_r.shape[1] - K)                                  # v3 clip L=24 -> max horizon 20
    gt_obj = tr_r[ho][:, K:K + He]; mot = np.linalg.norm(np.diff(gt_obj.mean(2), axis=1), axis=-1).sum(1); chosen = np.argsort(-mot)[:NSEQ]
    ps, lpv, cubee = [], [], []
    for s in chosen:
        si = int(ho[s]); gtf = fr_r[si, K:K + He].astype(np.float32) / 255.0
        r = render_eef(m, fr_r[si, 0], ef_r[si, K:K + He])
        ps.append(np.mean([psnr(r[h], gtf[h]) for h in range(He)])); lpv.append(lpf(r.astype(np.float32), gtf))
        cubee.append(np.nanmean([cube_pos_err(r[h], gt_obj[s, h].mean(0)) for h in range(He)]))
    lines = [f"detmem-eef [{MODE}] (VAE+latent transition, eef-splat cond, NO flow) | NSEQ={NSEQ} H={He}",
             f"held-out robot eef-replay: PSNR {np.mean(ps):.2f} | LPIPS {np.mean(lpv):.3f} | cube_err {np.nanmean(cubee):.1f}px",
             "★ 对照 ours(flow) 看 object-flow 的增益; co-train 用 robot+human eef." if MODE == "rh" else "robot-only 对照."]
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines) + f"\nsaved {OUT}/\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
