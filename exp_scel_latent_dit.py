"""detmem-DiT: ③ architecture modernization + flow-injection ablation (spec 2026-07-04-detmem-dit-design.md).
Answers user critique "detmem == IWS stage2 with pred-flow condition" on 3 axes:
  ARCH   : conv stack -> DETERMINISTIC spatio-temporal DiT (DexWM CDiT-style direct regression, no denoising)
           + dual-timescale memory (WEAVER: near prev + sparse far memory tokens). DELTA stays OFF (verdict).
  INJECT : how flow enters becomes an ablated design axis (env INJECT):
             concat = cond feats channel-concat into z0 tokens (detmem mechanism, parity baseline)
             add    = cond feats spatially ADDED to z0 tokens (OSCAR skeleton-latent injection, their winner)
             adaln  = cond global-pooled -> AdaLN-zero modulation (OSCAR's latent-action row, expected floor)
             vaeimg = flow rendered as RGB imgs -> SAME frozen VAE -> latents added (OSCAR/MaskWAM faithful)
  AUX    : cube-center heatmap head from predicted tokens (DexWM HC loss: forces the predicted latent to
           ground object position; flow does MORE than condition). loss = MSE(z) + LAM*LPIPS + LAM_HM*MSE(hm).
Protocol == exp_scel_latent_detmem (same data/seeds/heldout/BS/prev sampling/PREV_DF); eval renders
GT | detmem-conv (pinned detmem.pt) | DiT on the same seqs, GT-flow + pred-flow gifs (hard layout reqs).
Output: outputs/cross_embodiment_wm/latent_dit/<INJECT>_m<MEM>/"""
import os
os.environ["USE_GMASK"] = "1"; os.environ["FOOTPRINT"] = "1"; os.environ["DROP_VIS"] = "1"; os.environ["USE_PREV"] = "0"
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from exp_v3_human_helps_pixels import make_flow, sel_flow, gmask_imgs, load_gmask, cbr, cube_pos_err, K, IMG, device
from exp_scel_latent_renderer import enc, dec, cond_np, latent_ch
from exp_scel_latent_lpips import _decode_grad
from exp_scel_latent_detmem import DetMemRenderer, render_detmem          # unpickle detmem.pt + baseline column
from amplify_wm import rollout_lwc
from viz_combined import build_flow_cols, save_combined_gif

SMOKE = os.environ.get("SMOKE", "0") == "1"
DS = os.environ.get("DS", "outputs/flow_render_dataset_v3_grip")
WM = os.environ.get("WM", "outputs/cross_embodiment_wm/grip_wm/wm_base.pt")
DETMEM_PT = os.environ.get("DETMEM_PT", "outputs/cross_embodiment_wm/latent_detmem/detmem.pt")
H = int(os.environ.get("HORIZON", "40")); HELDOUT = 150; NSEQ = int(os.environ.get("NSEQ", "8"))
EPOCHS = 3 if SMOKE else int(os.environ.get("EPOCHS", "100")); BS = 16; LR = 2e-4
LAM = float(os.environ.get("LAM_LPIPS", "1.0")); LAM_HM = float(os.environ.get("LAM_HM", "10.0"))
PREV_DF = 0.3; CC = 4
INJECT = os.environ.get("INJECT", "add")                                  # concat | add | adaln | vaeimg
MEM = int(os.environ.get("MEM", "4"))                                     # #far-memory frames (0 = z0+prev only)
DIM = int(os.environ.get("DIM", "384")); DEPTH = int(os.environ.get("DEPTH", "8")); HEADS = int(os.environ.get("HEADS", "6"))
GRID = 16                                                                 # latent spatial (128/8)
OUT = os.environ.get("OUT_DIR", f"outputs/cross_embodiment_wm/latent_dit/{INJECT}_m{MEM}")
os.makedirs(f"{OUT}/gifs", exist_ok=True)
u8 = lambda x: (np.clip(x, 0, 1) * 255).astype(np.uint8)


# ----------------------------- flow -> RGB imgs (INJECT=vaeimg: OSCAR/MaskWAM-faithful) -----------------------------
def flow_to_imgs(cond):
    """(B,4,128,128) [dx,dy,footprint,gmask] -> two RGB imgs in [0,1]: flow-viz (hue=dir, val=mag) + mask img."""
    import cv2
    B = cond.shape[0]; f1 = np.zeros((B, 3, IMG, IMG), np.float32); f2 = np.zeros((B, 3, IMG, IMG), np.float32)
    for i in range(B):
        dx, dy, fp, gm = cond[i]
        mag = np.sqrt(dx ** 2 + dy ** 2); ang = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi)
        hsv = np.stack([ang * 179, np.full_like(ang, 255), np.clip(mag * 2, 0, 1) * 255], -1).astype(np.uint8)
        f1[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
        f2[i, 0] = fp; f2[i, 1] = gm                                       # footprint=red, gmask=green
    return f1, f2


class Block(nn.Module):
    """pre-LN transformer block; ADA=True adds AdaLN-zero modulation from vector c (DiT-style)."""
    def __init__(s, D, Hn, ada):
        super().__init__()
        s.ada = ada
        s.n1 = nn.LayerNorm(D, elementwise_affine=not ada); s.n2 = nn.LayerNorm(D, elementwise_affine=not ada)
        s.att = nn.MultiheadAttention(D, Hn, batch_first=True)
        s.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
        if ada:
            s.mod = nn.Linear(D, 6 * D); nn.init.zeros_(s.mod.weight); nn.init.zeros_(s.mod.bias)

    def forward(s, x, c=None, attn_mask=None):
        if s.ada:
            sh1, sc1, g1, sh2, sc2, g2 = s.mod(c)[:, None].chunk(6, -1)
            h = s.n1(x) * (1 + sc1) + sh1; x = x + g1 * s.att(h, h, h, need_weights=False, attn_mask=attn_mask)[0]
            h = s.n2(x) * (1 + sc2) + sh2; x = x + g2 * s.mlp(h)
        else:
            h = s.n1(x); x = x + s.att(h, h, h, need_weights=False, attn_mask=attn_mask)[0]
            x = x + s.mlp(s.n2(x))
        return x


class DetMemDiT(nn.Module):
    """tokens=[z0(256); prev(256); mem(M*16 pooled)] -> z_t (absolute, deterministic) + cube heatmap."""
    def __init__(s, zdim=None, cond_ch=CC, inject=INJECT, mem=MEM, D=DIM, depth=DEPTH, heads=HEADS):
        super().__init__()
        if zdim is None: zdim = latent_ch()
        s.inject, s.mem, s.zdim, s.D = inject, mem, zdim, D
        s.ce = nn.Sequential(cbr(cond_ch, 32), nn.MaxPool2d(2), cbr(32, 64), nn.MaxPool2d(2), cbr(64, 128), nn.MaxPool2d(2))
        z0_in = zdim + (128 if inject == "concat" else 0)
        s.emb_z0 = nn.Linear(z0_in, D); s.emb_prev = nn.Linear(zdim, D); s.emb_mem = nn.Linear(zdim, D)
        if inject == "add": s.emb_cond = nn.Linear(128, D)
        if inject == "vaeimg": s.emb_cond = nn.Linear(2 * zdim, D)
        if inject == "adaln": s.emb_c = nn.Sequential(nn.Linear(128, D), nn.SiLU(), nn.Linear(D, D))
        s.pos = nn.Parameter(torch.randn(GRID * GRID, D) * 0.02)          # shared 16x16 grid pos (z0/prev)
        s.pos_mem = nn.Parameter(torch.randn(16, D) * 0.02)               # pooled 4x4 grid pos
        s.frame_mem = nn.Parameter(torch.randn(max(mem, 1), D) * 0.02)    # mem frame-slot emb
        s.typ = nn.Parameter(torch.randn(3, D) * 0.02)                    # z0 / prev / mem
        s.blocks = nn.ModuleList([Block(D, heads, ada=(inject == "adaln")) for _ in range(depth)])
        s.norm = nn.LayerNorm(D); s.out = nn.Linear(D, zdim)
        s.hm = nn.Sequential(nn.ConvTranspose2d(D, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(), nn.Conv2d(64, 1, 3, 1, 1))

    def _tok(s, z): return z.flatten(2).transpose(1, 2)                    # (B,C,16,16)->(B,256,C)

    def forward(s, z0, cond, prev_z, mem_z=None, cond_lat=None):
        """z0/prev_z (B,Cz,16,16); cond (B,4,128,128); mem_z (B,M,Cz,16,16) or None; cond_lat (B,2Cz,16,16) for vaeimg."""
        B = z0.shape[0]; cf = s.ce(cond)                                   # (B,128,16,16)
        if s.inject == "concat":
            t0 = s.emb_z0(s._tok(torch.cat([z0, cf], 1)))
        else:
            t0 = s.emb_z0(s._tok(z0))
        if s.inject == "add":    t0 = t0 + s.emb_cond(s._tok(cf))
        if s.inject == "vaeimg": t0 = t0 + s.emb_cond(s._tok(cond_lat))
        t0 = t0 + s.pos + s.typ[0]
        tp = s.emb_prev(s._tok(prev_z)) + s.pos + s.typ[1]
        toks = [t0, tp]
        if s.mem > 0 and mem_z is not None:
            mp = Fn.adaptive_avg_pool2d(mem_z.reshape(B * s.mem, s.zdim, GRID, GRID), 4)   # (B*M,Cz,4,4)
            tm = s.emb_mem(mp.flatten(2).transpose(1, 2)).reshape(B, s.mem, 16, s.D)
            tm = tm + s.pos_mem[None, None] + s.frame_mem[None, :s.mem, None] + s.typ[2]
            toks.append(tm.reshape(B, s.mem * 16, s.D))
        x = torch.cat(toks, 1)
        c = s.emb_c(cf.mean((2, 3))) if s.inject == "adaln" else None
        for b in s.blocks: x = b(x, c)
        f = s.norm(x[:, :GRID * GRID])                                     # features at z0 positions
        z = s.out(f).transpose(1, 2).reshape(B, s.zdim, GRID, GRID)        # ABSOLUTE latent (DELTA verdict: off)
        hm = s.hm(f.transpose(1, 2).reshape(B, s.D, GRID, GRID))           # (B,1,32,32) cube-center heatmap
        return z, hm


def hm_gt(centers):
    """(B,2) cube centroid in [0,1] -> (B,1,32,32) unnormalized gaussian (sigma=1 grid cell; DexWM sigma=2px)."""
    B = len(centers); g = np.zeros((B, 1, 32, 32), np.float32)
    yy, xx = np.mgrid[:32, :32].astype(np.float32)
    for i, (cx, cy) in enumerate(centers):
        g[i, 0] = np.exp(-((xx - cx * 32) ** 2 + (yy - cy * 32) ** 2) / 2.0)
    return g


def mem_idx(t, m):
    """m far-memory frame indices uniformly over [0, t-1] (WEAVER sparse memory, train-time = real frames)."""
    return np.unique(np.linspace(0, max(t - 1, 0), m).round().astype(int))


# ----------------------------- latent cache (all frames, one-time) -----------------------------
def latent_cache(fr):
    tag = os.environ.get("VAE_NAME", "sdvae").split("/")[-1]
    p = f"outputs/cross_embodiment_wm/latent_dit/latcache_{os.path.basename(DS)}_{tag}.npy"
    if os.path.exists(p): return np.load(p, mmap_mode="r")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    N, L = fr.shape[:2]; flat = fr.reshape(N * L, IMG, IMG, 3)
    out = np.zeros((N * L, latent_ch(), GRID, GRID), np.float16)
    for i in range(0, N * L, 192):
        x = torch.from_numpy(flat[i:i + 192].transpose(0, 3, 1, 2).astype(np.float32) / 255.0).to(device)
        out[i:i + 192] = enc(x).cpu().numpy().astype(np.float16)
        if i % 19200 == 0: print(f"  latcache {i}/{N*L}", flush=True)
    out = out.reshape(N, L, latent_ch(), GRID, GRID); np.save(p, out); print(f"saved {p}", flush=True)
    return np.load(p, mmap_mode="r")


def train_dit(fr, tr, ef, vs, jt, idx, lat):
    torch.manual_seed(0); m = DetMemDiT().to(device); opt = torch.optim.AdamW(m.parameters(), lr=LR)
    print(f"DetMemDiT params={sum(p.numel() for p in m.parameters())/1e6:.1f}M inject={INJECT} mem={MEM}", flush=True)
    from interactive_world_sim.algorithms.common.metrics.lpips import LearnedPerceptualImagePatchSimilarity
    lp = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device).eval()
    for p in lp.parameters(): p.requires_grad_(False)
    rng = np.random.default_rng(0); L = fr.shape[1]
    for ep in range(EPOCHS):
        m.train(); pe = idx[rng.permutation(len(idx))]; tot = 0; nb = 0
        for i in range(0, len(pe), BS):
            b = pe[i:i + BS]; ts = rng.integers(9, L, size=len(b))
            conds, tgts, cents = [], [], []
            z0s, z1s, prevs, mems = [], [], [], []
            for n, j in enumerate(b):
                t = int(ts[n])
                conds.append(cond_np(tr, ef, vs, jt, j, t)); tgts.append(fr[j, t].astype(np.float32).transpose(2, 0, 1) / 255.0)
                cents.append(tr[j, t].mean(0))
                pt = 0 if rng.random() < 0.15 else t - 1                   # detmem prev protocol
                z0s.append(lat[j, 0]); z1s.append(lat[j, t]); prevs.append(lat[j, pt])
                if MEM > 0:
                    mi = mem_idx(t, MEM); mi = np.pad(mi, (0, MEM - len(mi)), mode="edge")
                    mems.append(lat[j][mi])
            f32 = lambda a: torch.from_numpy(np.stack(a).astype(np.float32)).to(device)
            cond = f32(conds); GT = f32(tgts)
            z0, z1, prev = f32(z0s), f32(z1s), f32(prevs)
            memz = f32(mems) if MEM > 0 else None
            a = torch.rand(len(b), 1, 1, 1, device=device) * PREV_DF; prev = (1 - a) * prev + a * torch.randn_like(prev)
            cl = None
            if INJECT == "vaeimg":
                i1, i2 = flow_to_imgs(np.stack(conds)); cl = torch.cat([enc(f32(list(i1))), enc(f32(list(i2)))], 1)
            pred, hm = m(z0, cond, prev, memz, cl)
            hgt = f32(hm_gt(np.stack(cents)))
            loss = (((pred - z1) ** 2).mean() + LAM * lp(_decode_grad(pred).clamp(0, 1) * 2 - 1, GT * 2 - 1)
                    + LAM_HM * ((hm - hgt) ** 2).mean())
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
        if ep % 10 == 0 or ep == EPOCHS - 1: print(f"  dit ep{ep} loss={tot/nb:.4f}", flush=True)
    return m.eval()


@torch.no_grad()
def render_dit(m, I0, tr_si, ef_si, vs_si, jt_si, obj_seq):
    """autoregressive: prev = own output; far memory = own history (uniform M frames, pooled inside model)."""
    z0 = enc(torch.from_numpy(I0.transpose(2, 0, 1).astype(np.float32) / 255.0)[None].to(device))
    hist = [z0]; outs = []
    for h in range(len(obj_seq)):
        fl = sel_flow(make_flow(tr_si[0], obj_seq[h], ef_si[0], ef_si[h], vs_si[h])); gm = gmask_imgs(jt_si[h][None], ef_si[h][None])[0]
        cond_a = np.concatenate([fl, gm[None]], 0)[None].astype(np.float32)
        cond = torch.from_numpy(cond_a).to(device)
        memz = None
        if MEM > 0:
            mi = np.unique(np.linspace(0, len(hist) - 1, MEM).round().astype(int))
            mi = np.pad(mi, (0, MEM - len(mi)), mode="edge")
            memz = torch.cat([hist[k] for k in mi], 0)[None]
        cl = None
        if INJECT == "vaeimg":
            i1, i2 = flow_to_imgs(cond_a)
            f32 = lambda a: torch.from_numpy(a.astype(np.float32)).to(device)
            cl = torch.cat([enc(f32(i1)), enc(f32(i2))], 1)
        pred, _ = m(z0, cond, hist[-1], memz, cl)
        hist.append(pred); outs.append(dec(pred)[0].cpu().numpy())
    return np.stack(outs).transpose(0, 2, 3, 1)


def psnr(a, b): return float(10 * np.log10(1.0 / (((a - b) ** 2).mean() + 1e-9)))
def interframe(rn): return float(np.mean([np.abs(rn[h] - rn[h - 1]).mean() for h in range(1, len(rn))]) * 255)


def main():
    load_gmask()
    z = np.load(f"{DS}/clips_robot.npz")
    fr, tr, ef, vs, jt = (z["frames"], z["tracks"].astype(np.float32), z["eef"].astype(np.float32), z["vis"].astype(np.float32), z["joint"].astype(np.float32))
    lat = latent_cache(fr)
    perm = np.random.default_rng(0).permutation(len(tr)); ho, pool = perm[:HELDOUT], perm[HELDOUT:]
    if SMOKE: pool = pool[:200]
    print(f"=== train detmem-DiT (INJECT={INJECT} MEM={MEM} LAM_HM={LAM_HM} EPOCHS={EPOCHS}) ===", flush=True)
    if os.environ.get("EVAL_ONLY", "0") == "1":
        print(f"=== EVAL_ONLY: loading {OUT}/dit.pt ===", flush=True)
        m = torch.load(f"{OUT}/dit.pt", map_location=device, weights_only=False).to(device).eval()
    else:
        m = train_dit(fr, tr, ef, vs, jt, pool, lat); torch.save(m, f"{OUT}/dit.pt")
    dm = None
    if os.path.exists(DETMEM_PT):
        dm = torch.load(DETMEM_PT, map_location=device, weights_only=False).to(device).eval()
    wm = torch.load(WM, map_location=device, weights_only=False).to(device).eval()
    from interactive_world_sim.algorithms.common.metrics.lpips import LearnedPerceptualImagePatchSimilarity
    lp = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device).eval()
    lpf = lambda r, g: float(lp(torch.from_numpy(r).permute(0, 3, 1, 2).float().to(device) * 2 - 1, torch.from_numpy(g).permute(0, 3, 1, 2).float().to(device) * 2 - 1).mean())
    gt_obj = tr[ho][:, K:K + H]; mot = np.linalg.norm(np.diff(gt_obj.mean(2), axis=1), axis=-1).sum(1); chosen = np.argsort(-mot)[:NSEQ]
    tags = (["detmem"] if dm is not None else []) + ["dit"]
    agg = {f"{w}_{k}_{t}": [] for w in ["gt", "pr"] for k in ["ps", "lp", "if", "pe", "det"] for t in tags}
    for s in chosen:
        si = int(ho[s]); I0 = fr[si, 0]; gtf = fr[si, K:K + H].astype(np.float32) / 255.0
        pr = rollout_lwc(wm, torch.from_numpy(tr[si:si + 1]).float().to(device), torch.from_numpy(ef[si:si + 1]).float().to(device), H).cpu().numpy()[0]
        args = (I0, tr[si], ef[si, K:K + H], vs[si, K:K + H], jt[si, K:K + H])
        rend = {}
        if dm is not None:
            rend[("detmem", "gt")] = render_detmem(dm, *args, gt_obj[s]); rend[("detmem", "pr")] = render_detmem(dm, *args, pr)
        rend[("dit", "gt")] = render_dit(m, *args, gt_obj[s]); rend[("dit", "pr")] = render_dit(m, *args, pr)
        for t in tags:
            for w, obj in [("gt", gt_obj[s]), ("pr", gt_obj[s])]:
                r = rend[(t, w)]
                agg[f"{w}_ps_{t}"].append(np.mean([psnr(r[h], gtf[h]) for h in range(H)]))
                agg[f"{w}_lp_{t}"].append(lpf(r.astype(np.float32), gtf)); agg[f"{w}_if_{t}"].append(interframe(r))
                pe = [cube_pos_err(r[h], gt_obj[s, h].mean(0)) for h in range(H)]
                agg[f"{w}_pe_{t}"].append(np.nanmean(pe)); agg[f"{w}_det_{t}"].append(float(np.mean(~np.isnan(pe))))
        for wtag, key, obj_pred in [("GTflow", "gt", None), ("predflow", "pr", pr)]:
            cols = [fr[si, K:K + H].astype(np.uint8)] + [u8(rend[(t, key)]) for t in tags]
            names = ["GT"] + [f"{t}-{wtag}" for t in tags]
            fc = build_flow_cols(fr[si, K:K + H].astype(np.uint8), gt_obj[s], [None] + [obj_pred] * len(tags), ef[si, K:K + H])
            errs = [None] + [agg[f"{key}_pe_{t}"][-1] if wtag == "predflow" else None for t in tags]
            save_combined_gif(f"{OUT}/gifs/seq{si}_{wtag}.gif", np.stack(cols), fc, names, errs, K)
    mn = lambda k: float(np.mean(agg[k])) if agg[k] else float("nan")
    lines = [f"detmem-DiT | INJECT={INJECT} MEM={MEM} LAM_HM={LAM_HM} DIM={DIM}x{DEPTH} | NSEQ={NSEQ} H={H}"]
    for t in tags:
        lines += [f"[{t}] GT-flow  : PSNR {mn(f'gt_ps_{t}'):.2f} | LPIPS {mn(f'gt_lp_{t}'):.3f} | IF {mn(f'gt_if_{t}'):.2f} | cube {mn(f'gt_pe_{t}'):.1f}px (det {mn(f'gt_det_{t}'):.2f})",
                  f"[{t}] pred-flow: PSNR {mn(f'pr_ps_{t}'):.2f} | LPIPS {mn(f'pr_lp_{t}'):.3f} | IF {mn(f'pr_if_{t}'):.2f} | cube {mn(f'pr_pe_{t}'):.1f}px (det {mn(f'pr_det_{t}'):.2f})"]
    lines += ["detmem reference (pinned run): GT-flow LPIPS 0.138 / IF 7.36. EYEBALL gifs (用户验证)."]
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines) + f"\nsaved {OUT}/\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
