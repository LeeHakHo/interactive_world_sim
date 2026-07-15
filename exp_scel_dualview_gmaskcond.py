"""③ agent-conditioning 消融(治糊):dual-view DiT 的 cond 加 gmask agent 剪影通道。

背景:can 单视角 detmem(LPIPS 0.147)赢在有 gmask agent 条件;dual-view DiT(0.25-0.31)
没有 agent 条件→手臂靠猜→糊。本消融同一 DualViewDiT,唯一变量 GMASK∈{0,1}:
  GMASK=0: cond = per-view flow splat [dx,dy,footprint](现状 3ch)
  GMASK=1: cond = 3ch ⊕ agent 剪影(robot v0 用 maskgen_caneef(joint7+eef6→128²);
           human/v1 置零——v1 作 within-run 对照,v0 的提升即 agent 条件的贡献)
replay GT flow(隔离 ③ 质量);can_dual robot+human co-train;eval=held-out robot
full-LPIPS/obj-LPIPS/PSNR + 渲染缓存;COMPARE=<dir0>,<dir1> 出 3 列 gif。
Env: GMASK/EPOCHS(60)/NSEQ/OUT。iws env,GPU。
Output: outputs/cross_embodiment_wm/dualview_gmask/<g0|g1>/"""
import os, sys
os.environ.setdefault("VAE_NAME", "ostris/vae-kl-f8-d16")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MASKGEN", "outputs/flow_wm/maskgen_caneef/maskgen_caneef.pt")
os.environ.setdefault("EPOCHS", "60")
import numpy as np, torch, torch.nn as nn

os.chdir("/scr2/yusenluo/interactive_world_sim")
import exp_scel_dualview_dit as DIT
from exp_scel_dualview_dit import (flow_cond, latcache, load_dual, DualViewDiT,
                                   enc, dec, obj_lpips, psnr, _fp, u8)
from exp_scel_latent_renderer import latent_ch
from exp_scel_latent_dit import GRID
from exp_v3_human_helps_pixels import cbr, load_gmask, gmask_imgs, IMG, device
from exp_scel_latent_lpips import _decode_grad
from viz_combined import save_combined_gif, build_flow_cols

SMOKE = os.environ.get("SMOKE", "0") == "1"
DS = "outputs/flow_render_dataset_can_dual"
GMASK = os.environ.get("GMASK", "1") == "1"
GMASK_LOW = os.environ.get("GMASK_LOW", "0") == "1"            # v1 也用真剪影(maskgen_caneef_low IoU0.836)
WARP = os.environ.get("WARP", "0") == "1"                      # 刚体 warp-as-condition RGB 预览 3ch
DIM_G = int(os.environ.get("DIM_G", "384")); DEPTH_G = int(os.environ.get("DEPTH_G", "8"))
EPOCHS = 3 if SMOKE else int(os.environ["EPOCHS"]); BS = 8; LR = 2e-4
H = int(os.environ.get("HORIZON", "20")); HELDOUT = 150; NSEQ = int(os.environ.get("NSEQ", "6")); PREV_DF = 0.3
LAM = float(os.environ.get("LAM_LPIPS", "1.0"))
OUT = os.environ.get("OUT", f"outputs/cross_embodiment_wm/dualview_gmask/{'g1' if GMASK else 'g0'}")
os.makedirs(f"{OUT}/gifs", exist_ok=True)
_m = sys.modules["__main__"]


class DualViewDiTG(DualViewDiT):
    """DualViewDiT(flow) + 可选第 4 条 gmask cond 通道(同一 ce 卷积栈,首层 3→4ch)。D/depth 可调(容量消融)。"""
    def __init__(s, gmask=True, D=None, depth=None, warp=None):
        D = D or DIM_G; depth = depth or DEPTH_G
        warp = WARP if warp is None else warp
        super().__init__(cond="flow", D=D, depth=depth, heads=max(1, D // 64))
        s.gmask, s.warp = gmask, warp
        cin = 3 + (1 if gmask else 0) + (3 if warp else 0)
        if cin != 3:
            s.ce = nn.Sequential(cbr(cin, 32), nn.MaxPool2d(2), cbr(32, 64), nn.MaxPool2d(2),
                                 cbr(64, 128), nn.MaxPool2d(2))
            s.emb_cond = nn.Linear(128, D)


setattr(_m, "DualViewDiTG", DualViewDiTG)

_GL = {}


def load_gmask_low():
    from train_maskgen_v3eef import MaskGen
    ck = torch.load("outputs/flow_wm/maskgen_caneef_low/maskgen_caneef.pt", map_location="cpu")
    g = MaskGen(ck["indim"]).to(device); g.load_state_dict(ck["g"]); g.eval()
    _GL.update(g=g, xm=ck["xm"], xs=ck["xs"]); print("loaded g-mask LOW (IoU0.836)", flush=True)


@torch.no_grad()
def gmask_low_imgs(joint_t, eef_t):
    feat = np.concatenate([joint_t[:, :7], eef_t.reshape(len(eef_t), 6)], 1)
    x = torch.from_numpy(((feat - _GL["xm"]) / _GL["xs"]).astype(np.float32)).to(device)
    return torch.sigmoid(_GL["g"](x)[:, 0]).cpu().numpy()


def warp_preview(fr0, tr0, trt, vis):
    """刚体 warp-as-condition:vis 加权 Umeyama 相似变换(tr0→trt)把首帧 footprint 内像素
    搬到预测位置 -> (3,128,128) RGB 预览(区域外 0)。罐子刚体假设;<3 可见点返回全零。"""
    import cv2
    ok = vis > 0.5
    if ok.sum() < 3: return np.zeros((3, IMG, IMG), np.float32)
    src = tr0[ok] * IMG; dst = trt[ok] * IMG
    ms, md = src.mean(0), dst.mean(0)
    sc, dc = src - ms, dst - md
    cov = dc.T @ sc / len(sc)
    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, d]) @ Vt
    var = (sc ** 2).sum() / len(sc)
    s = (S * [1, d]).sum() / (var + 1e-9)
    A = np.concatenate([s * R, (md - s * R @ ms)[:, None]], 1).astype(np.float32)  # (2,3)
    fp = np.zeros((IMG, IMG), np.float32)
    vp = np.clip(src.astype(np.int32), 0, IMG - 1)
    if len(vp) >= 3: cv2.fillConvexPoly(fp, cv2.convexHull(vp.reshape(-1, 1, 2)), 1.0)
    masked = fr0.astype(np.float32) / 255.0 * fp[:, :, None]
    warped = cv2.warpAffine(masked, A, (IMG, IMG))
    return warped.transpose(2, 0, 1).astype(np.float32)


def _agent_ch(v, dom, joint, ef_vt, j, t):
    """per-view agent 剪影通道:v0=maskgen_caneef,v1=maskgen_caneef_low(GMASK_LOW 时),否则 0。"""
    import exp_v3_human_helps_pixels as HP
    if dom == "r" and v == 0:
        if not HP._G: load_gmask()
        return gmask_imgs(joint[j, t][None], ef_vt[None])[0].astype(np.float32)
    if dom == "r" and v == 1 and GMASK_LOW:
        if not _GL: load_gmask_low()
        return gmask_low_imgs(joint[j, t][None], ef_vt[None])[0].astype(np.float32)
    return np.zeros((IMG, IMG), np.float32)


def build_conds_g(R, joint, dom, j, t, gmask_on):
    """(2,3|4,128,128) cond + (2,128,128) object mask。gmask:robot v0=maskgen 剪影,其余 0。"""
    conds, masks = [], []
    for v in range(2):
        tr, ef, vs = R["tr"][v], R["ef"][v], R["vs"][v]
        c = flow_cond(tr[j, 0], tr[j, t], ef[j, 0], ef[j, t], vs[j, t])
        masks.append(c[2].copy())
        if gmask_on:
            g = _agent_ch(v, dom, joint, ef[j, t], j, t)
            c = np.concatenate([c, g[None]], 0)
        if WARP:
            c = np.concatenate([c, warp_preview(R["fr"][v][j, 0], tr[j, 0], tr[j, t], vs[j, t])], 0)
        conds.append(c)
    return np.stack(conds), np.stack(masks)


def train_g(R, Hh, joint, idx_r, idx_h, latR, latH):
    """DIT.train 逐字改编:cond 走 build_conds_g,模型 DualViewDiTG(GMASK)。"""
    torch.manual_seed(0); m = DualViewDiTG(gmask=GMASK).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    print(f"DualViewDiTG GMASK={GMASK} params={sum(p.numel() for p in m.parameters())/1e6:.1f}M EPOCHS={EPOCHS}", flush=True)
    from interactive_world_sim.algorithms.common.metrics.lpips import LearnedPerceptualImagePatchSimilarity
    lp = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device).eval()
    for p in lp.parameters(): p.requires_grad_(False)
    rng = np.random.default_rng(0)
    samples = np.array([("r", int(j)) for j in idx_r] + [("h", int(j)) for j in idx_h], dtype=object)
    fr01 = lambda a: a.astype(np.float32).transpose(2, 0, 1) / 255.0
    for ep in range(EPOCHS):
        m.train(); order = rng.permutation(len(samples)); tot = 0; nb = 0
        for i in range(0, len(order), BS):
            bidx = order[i:i + BS]
            z0i, z1i, pvi, gti, cdi, mki, doms = [], [], [], [], [], [], []
            for si in bidx:
                dom, j = samples[si]; D = R if dom == "r" else Hh; lc = latR if dom == "r" else latH
                t = int(rng.integers(9, D["fr"][0].shape[1]))
                cond, mask = build_conds_g(D, joint, dom, j, t, GMASK)
                pt = 0 if rng.random() < 0.15 else t - 1
                z0i.append(lc[j, 0].astype(np.float32)); z1i.append(lc[j, t].astype(np.float32)); pvi.append(lc[j, pt].astype(np.float32))
                gti.append([fr01(D["fr"][v][j, t]) for v in range(2)])
                cdi.append(cond); mki.append(mask); doms.append(0 if dom == "r" else 1)
            f2 = lambda a: torch.from_numpy(np.stack(a).astype(np.float32)).to(device)
            z0 = f2(z0i); z1 = f2(z1i); pv = f2(pvi); GT = f2(gti)
            cond = f2(cdi); mask = f2(mki)[:, :, None]; dom = torch.tensor(doms, device=device)
            a = torch.rand(len(bidx), 1, 1, 1, 1, device=device) * PREV_DF; pv = (1 - a) * pv + a * torch.randn_like(pv)
            pred = m(z0, pv, cond)
            img = _decode_grad(pred.reshape(-1, latent_ch(), GRID, GRID)).clamp(0, 1).reshape(len(bidx), 2, 3, IMG, IMG)
            rs = (dom == 0); hs = (dom == 1)
            loss = 0.0 * pred.sum()
            if rs.any():
                loss = loss + ((pred[rs] - z1[rs]) ** 2).mean()
                loss = loss + LAM * lp(img[rs].reshape(-1, 3, IMG, IMG) * 2 - 1, GT[rs].reshape(-1, 3, IMG, IMG) * 2 - 1)
            if hs.any():
                om = mask[hs]; loss = loss + (((img[hs] - GT[hs]) ** 2) * om).sum() / (om.sum() * 3 + 1e-6)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss); nb += 1
        if ep % 15 == 0 or ep == EPOCHS - 1: print(f"  GMASK={GMASK} ep{ep} loss={tot/nb:.4f}", flush=True)
    return m.eval()


@torch.no_grad()
def render_g(m, R, joint, si):
    fr01 = lambda a: torch.from_numpy(a.astype(np.float32).transpose(2, 0, 1)[None] / 255.0).to(device)
    z0 = torch.stack([enc(fr01(R["fr"][v][si, 0]))[0] for v in range(2)])[None]
    prev = z0.clone(); outs = [[], []]
    for h in range(H):
        cond, _ = build_conds_g(R, joint, "r", si, DIT.K + h, GMASK)
        cond = torch.from_numpy(cond[None].astype(np.float32)).to(device)
        pred = m(z0, prev, cond); prev = pred
        for v in range(2): outs[v].append(dec(pred[:, v])[0].cpu().numpy())
    return np.stack([np.stack(outs[v]).transpose(0, 2, 3, 1) for v in range(2)])


def fp_vis(trt, vis):
    """修正版 footprint:只用可见点(vis>0.5)构凸包,防 nan→中心点拽偏(metric 审计发现)。"""
    import cv2
    fp = np.zeros((IMG, IMG), np.float32)
    ok = vis > 0.5
    if ok.sum() < 5: return fp                                 # 检测失败帧记无效
    vp = np.clip((trt[ok] * IMG).astype(np.int32), 0, IMG - 1)
    cv2.fillConvexPoly(fp, cv2.convexHull(vp.reshape(-1, 1, 2)), 1.0)
    return fp


def full_lpips(lp, pred, gt):
    a = torch.from_numpy(pred).permute(0, 3, 1, 2).float().to(device) * 2 - 1
    b = torch.from_numpy(gt).permute(0, 3, 1, 2).float().to(device) * 2 - 1
    return float(np.mean([float(lp(a[i:i+1], b[i:i+1])) for i in range(len(a))]))


def main():
    load_gmask()
    R, Hh = load_dual()
    joint = np.load(f"{DS}/clips_robot.npz")["joint"].astype(np.float32)
    okr = np.where(R["ok"])[0]; okh = np.where(Hh["ok"])[0]
    perm = np.random.default_rng(0).permutation(okr); ho, pool = perm[:HELDOUT], perm[HELDOUT:]
    if SMOKE: pool = pool[:60]; okh = okh[:60]
    ck = os.environ.get("RENDER_CKPT")
    if ck:
        m = torch.load(ck, map_location=device, weights_only=False).eval()
        print(f"RENDER-ONLY from {ck} (HORIZON={H})", flush=True)
    else:
        latR = latcache(R, "robot"); latH = latcache(Hh, "human")
        m = train_g(R, Hh, joint, pool, okh, latR, latH); torch.save(m, f"{OUT}/dvdit_g.pt")
    from interactive_world_sim.algorithms.common.metrics.lpips import LearnedPerceptualImagePatchSimilarity
    lp = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device).eval()
    mot = np.array([np.linalg.norm(np.diff(R["tr"][0][si, DIT.K:DIT.K + H].mean(1), axis=0), axis=-1).sum() for si in ho])
    chosen = ho[np.argsort(-mot)[:NSEQ]]
    os.makedirs(f"{OUT}/render_cache", exist_ok=True)
    np.save(f"{OUT}/render_cache/chosen.npy", chosen)
    agg = {f"v{v}_{k}": [] for v in range(2) for k in ["ps", "olp", "flp"]}
    for si in chosen:
        si = int(si); rr = render_g(m, R, joint, si)
        np.save(f"{OUT}/render_cache/seq{si}.npy", u8(rr))
        for v in range(2):
            gtf = R["fr"][v][si, DIT.K:DIT.K + H].astype(np.float32) / 255.0
            objm = np.stack([fp_vis(R["tr"][v][si, DIT.K + h], R["vs"][v][si, DIT.K + h]) for h in range(H)])
            agg[f"v{v}_ps"].append(np.mean([psnr(rr[v, h], gtf[h]) for h in range(H)]))
            agg[f"v{v}_olp"].append(obj_lpips(lp, rr[v].astype(np.float32), gtf, objm))
            agg[f"v{v}_flp"].append(full_lpips(lp, rr[v].astype(np.float32), gtf))
        print(f"  seq{si} done", flush=True)
    mn = lambda k: float(np.nanmean(agg[k]))
    lines = [f"DUAL-VIEW DiT GMASK={int(GMASK)} | replay GT flow | EPOCHS={EPOCHS} | n={len(chosen)}"]
    for v in range(2):
        lines.append(f"cam{'_high' if v==0 else '_low '}: PSNR {mn(f'v{v}_ps'):.2f} | obj-LPIPS {mn(f'v{v}_olp'):.3f} | FULL-LPIPS {mn(f'v{v}_flp'):.3f}")
    import json; json.dump({k: float(np.nanmean(vv)) for k, vv in agg.items()}, open(f"{OUT}/metrics.json", "w"), indent=2)
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines) + "\n=== DONE ===", flush=True)


def compare_runs(dir0, dir1):
    """GMASK=0 vs 1 渲染缓存 -> 3 列 gif:GT | flow-only | flow+gmask。"""
    R, _ = load_dual()
    ch0 = np.load(f"{dir0}/render_cache/chosen.npy"); ch1 = np.load(f"{dir1}/render_cache/chosen.npy")
    assert (ch0 == ch1).all()
    OUTC = os.environ.get("OUT", "outputs/cross_embodiment_wm/dualview_gmask/compare")
    os.makedirs(f"{OUTC}/gifs", exist_ok=True)
    vname = {0: "high", 1: "low"}
    for si in ch0:
        si = int(si)
        r0 = np.load(f"{dir0}/render_cache/seq{si}.npy"); r1 = np.load(f"{dir1}/render_cache/seq{si}.npy")
        for v in range(2):
            gt = R["fr"][v][si, DIT.K:DIT.K + H].astype(np.uint8)
            cols = np.stack([gt, r0[v], r1[v]])
            gt_obj = R["tr"][v][si, DIT.K:DIT.K + H]
            fc = build_flow_cols(gt, gt_obj, [None, None, None], R["ef"][v][si, DIT.K:DIT.K + H])
            labA = os.environ.get("CMP_LABEL_A", "A " + os.path.basename(dir0))
            labB = os.environ.get("CMP_LABEL_B", "B " + os.path.basename(dir1))
            save_combined_gif(f"{OUTC}/gifs/seq{si}_cam{vname[v]}.gif", cols, fc,
                              ["1 real video", f"2 {labA}", f"3 {labB}"], [None] * 3, DIT.K,
                              caption=f"render ablation: {labA} vs {labB} | cam_{vname[v]}")
    print(f"saved -> {OUTC}/gifs/\n=== DONE ===", flush=True)


@torch.no_grad()
def render_g_pred(m, R, joint, si, predtr):
    """② 预测 tracks 驱动的渲染(gmask 版 cond;joint/eef 为 replay action,合法)。"""
    fr01 = lambda a: torch.from_numpy(a.astype(np.float32).transpose(2, 0, 1)[None] / 255.0).to(device)
    z0 = torch.stack([enc(fr01(R["fr"][v][si, 0]))[0] for v in range(2)])[None]
    prev = z0.clone(); outs = [[], []]
    P = R["tr"][0].shape[2]
    for h in range(H):
        conds = []
        for v in range(2):
            tr, ef, vs = R["tr"][v], R["ef"][v], R["vs"][v]
            trt = predtr[h, v * P:(v + 1) * P]
            c = flow_cond(tr[si, 0], trt, ef[si, 0], ef[si, DIT.K + h], vs[si, DIT.K + h])
            if GMASK:
                g = _agent_ch(v, "r", joint, ef[si, DIT.K + h], si, DIT.K + h)
                c = np.concatenate([c, g[None]], 0)
            if WARP:
                c = np.concatenate([c, warp_preview(R["fr"][v][si, 0], tr[si, 0], trt, vs[si, DIT.K + h])], 0)
            conds.append(c)
        cond = torch.from_numpy(np.stack(conds)[None].astype(np.float32)).to(device)
        pred = m(z0, prev, cond); prev = pred
        for v in range(2): outs[v].append(dec(pred[:, v])[0].cpu().numpy())
    return np.stack([np.stack(outs[v]).transpose(0, 2, 3, 1) for v in range(2)])


def e2e_g():
    """端到端(锐③版):E2E_CKPT=③ckpt WM2A/WM2B=两个②ckpt -> 4列 gif GT|GT-flow→③|②A→③|②B→③。
    数据 L24(②正当 heldout);metric 全套。"""
    import exp_scel_dualview_wm as W
    import exp_scel_dualview_comb as DC
    for cls in [W.DualLWC, DC.DualCombLWC]: setattr(_m, cls.__name__, cls)
    m3 = torch.load(os.environ["E2E_CKPT"], map_location=device, weights_only=False).eval()
    z = np.load(os.environ.get("E2E_DS", f"{DS}/clips_robot_L24.npz"))
    f32 = lambda k, nan=0.0: np.nan_to_num(z[k].astype(np.float32), nan=nan)
    R = {"fr": [z["frames"], z["frames_low"]], "tr": [f32("tracks"), f32("tracks_low", 0.5)],
         "ef": [f32("eef"), f32("eef_low", 0.5)], "vs": [z["vis"].astype(np.float32), z["vis_low"].astype(np.float32)]}
    joint = z["joint"].astype(np.float32)
    perm = np.random.default_rng(0).permutation(len(z["low_valid"]))
    ho = np.array([i for i in perm[:HELDOUT] if z["low_valid"][i]])
    mot = np.array([np.linalg.norm(np.diff(R["tr"][0][si, DIT.K:DIT.K + H].mean(1), axis=0), axis=-1).sum() for si in ho])
    chosen = ho[np.argsort(-mot)[:NSEQ]]
    trD = np.concatenate([R["tr"][0][chosen], R["tr"][1][chosen]], 2)
    arms = {}
    for tag, env in [("A", "WM2A"), ("B", "WM2B")]:
        wm2 = torch.load(os.environ[env], map_location=device, weights_only=False).eval()
        pr = W.rollout_dual(wm2, torch.from_numpy(trD).float().to(device),
                            torch.from_numpy(R["ef"][0][chosen]).float().to(device),
                            torch.from_numpy(R["ef"][1][chosen]).float().to(device), H).cpu().numpy()
        arms[tag] = (os.path.basename(os.environ[env]).replace(".pt", ""), pr)
        print(f"② {tag}={arms[tag][0]}", flush=True)
    from interactive_world_sim.algorithms.common.metrics.lpips import LearnedPerceptualImagePatchSimilarity
    lp = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device).eval()
    os.makedirs(f"{OUT}/gifs", exist_ok=True)
    vname = {0: "high", 1: "low"}
    agg = {}
    for n, si in enumerate(chosen):
        si = int(si)
        r_gtf = render_g(m3, R, joint, si)
        rA = render_g_pred(m3, R, joint, si, arms["A"][1][n])
        rB = render_g_pred(m3, R, joint, si, arms["B"][1][n])
        for v in range(2):
            gtf = R["fr"][v][si, DIT.K:DIT.K + H].astype(np.float32) / 255.0
            objm = np.stack([fp_vis(R["tr"][v][si, DIT.K + h], R["vs"][v][si, DIT.K + h]) for h in range(H)])
            for cname, rr in [("gtf", r_gtf), ("A", rA), ("B", rB)]:
                agg.setdefault(f"v{v}_{cname}_olp", []).append(obj_lpips(lp, rr[v].astype(np.float32), gtf, objm))
                agg.setdefault(f"v{v}_{cname}_flp", []).append(full_lpips(lp, rr[v].astype(np.float32), gtf))
                agg.setdefault(f"v{v}_{cname}_ps", []).append(np.mean([psnr(rr[v, h], gtf[h]) for h in range(H)]))
            gt = R["fr"][v][si, DIT.K:DIT.K + H].astype(np.uint8)
            cols = np.stack([gt, u8(r_gtf[v]), u8(rA[v]), u8(rB[v])])
            gt_obj = R["tr"][v][si, DIT.K:DIT.K + H]
            P = R["tr"][0].shape[2]
            fc = build_flow_cols(gt, gt_obj, [None, gt_obj, arms["A"][1][n][:, v*P:(v+1)*P], arms["B"][1][n][:, v*P:(v+1)*P]],
                                 R["ef"][v][si, DIT.K:DIT.K + H])
            save_combined_gif(f"{OUT}/gifs/seq{si}_cam{vname[v]}.gif", cols, fc,
                              ["1 REAL", "2 CEILING", "3 PRED-A", "4 PRED-B"], [None] * 4, DIT.K,
                              caption=f"2=render w/ TRUE motion | 3=WM align_wm pred | 4=WM dummy5 pred | cam_{vname[v]}")
        print(f"  seq{si} done", flush=True)
    mn = lambda k: float(np.nanmean(agg[k]))
    lines = [f"E2E sharp-III | III={os.environ['E2E_CKPT']} | (2)A={arms['A'][0]} B={arms['B'][0]} | n={len(chosen)}"]
    for v in range(2):
        lines.append(f"cam_{vname[v]}: " + " | ".join(
            f"{c} PSNR {mn(f'v{v}_{c}_ps'):.2f} objLPIPS {mn(f'v{v}_{c}_olp'):.3f} fullLPIPS {mn(f'v{v}_{c}_flp'):.3f}"
            for c in ["gtf", "A", "B"]))
    import json; json.dump({k: float(np.nanmean(vv)) for k, vv in agg.items()}, open(f"{OUT}/metrics.json", "w"), indent=2)
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines) + "\n=== DONE ===", flush=True)


if __name__ == "__main__":
    if os.environ.get("COMPARE"):
        compare_runs(*os.environ["COMPARE"].split(","))
    elif os.environ.get("E2E_CKPT"):
        e2e_g()
    else:
        main()
