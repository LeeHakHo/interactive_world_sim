"""LATENT-RESIDUAL MIXED-TRAINING on can_dual (代号"方向3") — Task#21 port.

Task#21 (RENDERER_GRASP_RESEARCH_LOG.md §8, `exp_scel_latent_mix.py`, v3 data, 2026-06-22) found that a
latent world model over a FROZEN general VAE's latents (not IWS's domain-ViT) transfers human->robot even
though the raw latents themselves are domain-disjoint (582x separable) — because the DYNAMICS (Δz between
frames) are domain-shared while the absolute latent z is not. RESIDUAL (predict Δz) beat DIRECT (predict
absolute z_next) sharply in the human-helps % (N50: +35.8% vs +5.6%), which is the empirical signature of
"the starting point doesn't transfer, the change does". This script re-runs that recipe on the newer
can_dual dual-view dataset to see if the effect replicates on a different rig/embodiment pair.

This is a NEW file (not an extension of exp_scel_dualview_dit*.py or exp_scel_latent_mix.py) because it is
architecturally a different axis from both: exp_scel_latent_mix.py is single-view/v3-only (no reusable
data loader for can_dual), and exp_scel_dualview_dit*.py is the flow-conditioned DiT renderer line (③,
image-space, object-flow-conditioned) — this is ② dynamics-only, latent-space, NO flow/renderer, action
conditioned purely on eef, closest in spirit to Task#21's LatentDyn (conv+FiLM), just extended from 1 view
to 2 views (stack view axis into channels for the encoder/decoder, concat both views' 3-pt eef into one
6-pt action vector per frame — same "stack-by-view" convention `dualview_wm.py` uses for dummy5 x 2 views).

Recipe fidelity to Task#21 (exp_scel_latent_mix.py), what's kept vs adapted:
  - KEPT: frozen VAE latents (16ch, `exp_scel_latent_renderer.enc/dec`, ostris/vae-kl-f8-d16), conv+FiLM
    LatentDyn architecture (encoder conv stack -> FiLM from eef action -> decoder conv stack), scheduled
    sampling (p: 1.0->0.3 over training), RESIDUAL = z_hist[-1] + net_out vs DIRECT = net_out, action =
    the FULL K+F eef window fed statically every rollout step (Task#21's convention: action = the known
    "plan", not a per-step control -- it is real teleop/demo eef data, an input not inferred from future
    frames, so this is not a leak), held-out robot latent-space rollout MSE as the primary metric,
    robot-only vs robot+ALL-human arms, SEEDS-averaged.
  - ADAPTED for can_dual: 2 views (latcache_{robot,human}.npy, view dim=2) -> encoder/decoder stack the
    view axis into channels; eef action = both views' 3-pt sets concatenated into 6 points/frame (was
    single-view 3pt); K=4,F=20,H_ROLL=20 (WIN=K+F=24) chosen so the window fits can_dual's human clip
    length exactly (clips_human_L24.npz, L=24) while robot clips (L=48) are truncated to their first 24
    frames -- this keeps the two domains on an identical time window, same spirit as Task#21's v3 data
    (which was L=24 for both domains).
  - NEW (not in Task#21): decoded obj-region LPIPS on H=20 rollouts via the frozen VAE decoder + can_dual's
    own audited metric (`exp_scel_dualview_dit_formal.obj_lpips_audit`, footprint-centered 64x64 crop with
    detection-rate denominator) -- Task#21 only measured latent MSE, never decoded to pixels, so this adds
    a second, more end-to-end-relevant metric on top of the original recipe.

Data: outputs/flow_render_dataset_can_dual/clips_{robot,human_L24}.npz (loaded via
`exp_scel_dualview_dit.load_dual`), latents PRE-BUILT at outputs/cross_embodiment_wm/dualview_dit/
latcache_{robot,human}.npy (built by exp_scel_dualview_dit.latcache; NOT re-encoded here).
Split: okfirst (filter-then-permute) per FLOW_WARP_REPR_LOG.md/CAN_DATA_AUDIT_2026-07-13.md audit item5 --
`np.where(ok)[0]` THEN `rng(0).permutation`, never permute-then-filter (that mismatch caused ③'s published
eval seqs to leak into ②'s training set in a prior run). Human indices (robot+human arms) use ALL
low_valid human clips, unfiltered by any held-out split (human is never held out, matches Task#21).

Arms (env ARM, see ARM_TABLE): {residual, direct} x {robot-only, robot+human} at NROB=100 (robot-scarce,
the human-helps regime) + {residual, direct} robot-only at the FULL robot pool (ceiling reference, no
human) = 6 runs total.

Env: ARM(resid_ro100|resid_rh100|direct_ro100|direct_rh100|resid_rofull|direct_rofull), SMOKE, EPOCHS,
SEEDS, NSEQ, OUT_DIR.
Output: outputs/cross_embodiment_wm/latent_residual_can/<ARM>/{summary.txt,metrics.json,gifs/*.npy}
(symlinked to /scr/yusenluo/iws_overflow/latent_residual_can/, per per-experiment-output-dir convention).
"""
import json
import os

os.environ.setdefault("VAE_NAME", "ostris/vae-kl-f8-d16")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
import torch.nn as nn

os.chdir("/scr2/yusenluo/interactive_world_sim")
from exp_scel_dualview_dit import load_dual, latcache, _fp
from exp_scel_dualview_dit_formal import obj_lpips_audit
from exp_scel_latent_renderer import dec, latent_ch
from exp_v3_human_helps_pixels import K, IMG, device

SMOKE = os.environ.get("SMOKE", "0") == "1"
F = 20                                              # rollout/plan window length (Task#21 default)
H_ROLL = F                                          # held-out rollout horizon (matches decode-LPIPS req)
WIN = K + F                                         # 24: fits human clips_human_L24 exactly
HELDOUT = 150                                       # robot heldout count, matches ③/②-okfirst convention

ARM_TABLE = {                                       # RESIDUAL, HUMAN, NROB(None=full pool)
    "resid_ro100": (True, False, 100),
    "resid_rh100": (True, True, 100),
    "direct_ro100": (False, False, 100),
    "direct_rh100": (False, True, 100),
    "resid_rofull": (True, False, None),
    "direct_rofull": (False, False, None),
}
ARM = os.environ.get("ARM", "resid_ro100")
assert ARM in ARM_TABLE, f"ARM must be one of {sorted(ARM_TABLE)}"
RESIDUAL, HUMAN, NROB_SPEC = ARM_TABLE[ARM]

EPOCHS = 2 if SMOKE else int(os.environ.get("EPOCHS", "60"))
SEEDS = 1 if SMOKE else int(os.environ.get("SEEDS", "2"))
NSEQ = 4 if SMOKE else int(os.environ.get("NSEQ", "16"))
BS = 32
LR = 3e-4

OUT = os.environ.get("OUT_DIR", f"outputs/cross_embodiment_wm/latent_residual_can/{ARM}")
os.makedirs(f"{OUT}/gifs", exist_ok=True)


def split_okfirst(ok, heldout=HELDOUT):
    """Filter-then-permute (audit item5 leakage fix), verbatim protocol of
    exp_scel_dualview_wm.split_okfirst / exp_scel_dualview_dit_formal main split."""
    okr = np.where(ok)[0]
    perm = np.random.default_rng(0).permutation(okr)
    return perm[:heldout], perm[heldout:]


def stack_ef(D, win):
    """D['ef'] = [view0 (N,L,3,2), view1 (N,L,3,2)] -> (N,win,6,2) both-view 3pt eef concat, f32."""
    return np.concatenate([D["ef"][0][:, :win], D["ef"][1][:, :win]], axis=2).astype(np.float32)


class LatentDynDual(nn.Module):
    """Task#21 LatentDyn (exp_scel_latent_mix.py), extended to 2 views by stacking the view axis into
    channels. z_hist (B,K,2,Cz,16,16) + eef (B,WIN,6,2) -> z_next (B,2,Cz,16,16). conv on 16x16, action
    injected as FiLM (gamma,beta) on the bottleneck, same as the original."""

    def __init__(s, Cz, k=K, win=WIN, h=256):
        super().__init__()
        cbr = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, 1, 1), nn.GroupNorm(8, o), nn.SiLU())
        s.Cz = Cz
        s.enc = nn.Sequential(cbr(k * 2 * Cz, h), cbr(h, h), cbr(h, h))
        s.act = nn.Linear(win * 6 * 2, 2 * h)                       # FiLM (γ,β)
        s.dec = nn.Sequential(cbr(h, h), nn.Conv2d(h, 2 * Cz, 1))

    def forward(s, z_hist, eef, residual):
        """z_hist (B,K,2,Cz,16,16), eef (B,WIN,6,2) full plan window (static every step, Task#21 convention)."""
        B = z_hist.shape[0]
        x = s.enc(z_hist.reshape(B, -1, 16, 16))
        objc = eef.mean((1, 2), keepdim=True)                       # per-clip centroid (domain-centering)
        g, b = s.act((eef - objc).reshape(B, -1)).chunk(2, -1)
        x = x * (1 + g[:, :, None, None]) + b[:, :, None, None]
        out = s.dec(x).reshape(B, 2, s.Cz, 16, 16)
        return (z_hist[:, -1] + out) if residual else out           # residual: Δz + own start vs absolute z


def train_dyn(latR, efR, latH, efH, idx_r, idx_h, residual, seed=0, epochs=EPOCHS):
    torch.manual_seed(seed)
    Cz = latR.shape[3]
    m = LatentDynDual(Cz).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    g = torch.Generator().manual_seed(seed)
    samples = [("r", int(i)) for i in idx_r]
    if idx_h is not None:
        samples += [("h", int(i)) for i in idx_h]
    samples = np.array(samples, dtype=object)
    for ep in range(epochs):
        p = 1.0 + (0.3 - 1.0) * ep / max(epochs - 1, 1)             # scheduled sampling: GT prob 1.0->0.3
        m.train()
        order = samples[torch.randperm(len(samples), generator=g).numpy()]
        for i in range(0, len(order), BS):
            b = order[i:i + BS]
            Zb = np.stack([(latR if d == "r" else latH)[int(j)] for d, j in b])
            Eb = np.stack([(efR if d == "r" else efH)[int(j)] for d, j in b])
            Z = torch.from_numpy(Zb).float().to(device)
            Ef = torch.from_numpy(Eb).float().to(device)
            buf = Z[:, :K].clone()
            losses = []
            for h in range(H_ROLL):
                pred = m(buf[:, -K:], Ef[:, :WIN], residual)
                gt = Z[:, K + h]
                losses.append(((pred - gt) ** 2).mean())
                use_gt = torch.rand(len(b), 1, 1, 1, 1, device=device) < p
                buf = torch.cat([buf, torch.where(use_gt, gt, pred.detach())[:, None]], 1)
            loss = torch.stack(losses).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"  [{ARM} seed{seed}] ep{ep} loss={float(loss):.4f}", flush=True)
    return m.eval()


@torch.no_grad()
def rollout(m, lat, eef, residual, hsteps=H_ROLL):
    """lat (B,K+,2,Cz,16,16) np, eef (B,WIN,6,2) np -> pred z sequence (B,hsteps,2,Cz,16,16) torch (on device)."""
    Z = torch.from_numpy(lat).float().to(device)
    Ef = torch.from_numpy(eef).float().to(device)
    buf = Z[:, :K].clone()
    preds = []
    for h in range(hsteps):
        pred = m(buf[:, -K:], Ef[:, :WIN], residual)
        preds.append(pred)
        buf = torch.cat([buf, pred[:, None]], 1)
    return torch.stack(preds, 1)


def dyn_err(m, lat_ho, eef_ho, residual):
    """held-out robot next-latent MSE, open-loop rollout (Task#21's primary metric)."""
    pr = rollout(m, lat_ho, eef_ho, residual)
    gt = torch.from_numpy(lat_ho[:, K:K + H_ROLL]).float().to(device)
    return float(((pr - gt) ** 2).mean())


def psnr(a, b):
    return float(10 * np.log10(1.0 / (((a - b) ** 2).mean() + 1e-9)))


def decode_eval(m, R, latR, efR, chosen, residual, lp):
    """decode held-out rollouts -> per-view obj-region LPIPS/PSNR/det-rate (new metric, not in Task#21)."""
    pr = rollout(m, latR[chosen], efR[chosen], residual)                 # (S,H,2,Cz,16,16)
    res = {f"v{v}_lp": [] for v in range(2)}
    ps = {0: [], 1: []}
    det = {0: [0, 0], 1: [0, 0]}
    for si_i, si in enumerate(chosen):
        si = int(si)
        for v in range(2):
            imgs = dec(pr[si_i, :, v]).cpu().numpy().transpose(0, 2, 3, 1).astype(np.float32)  # (H,128,128,3)
            gtf = R["fr"][v][si, K:K + H_ROLL].astype(np.float32) / 255.0
            objm = np.stack([_fp(R["tr"][v][si, K + h]) for h in range(H_ROLL)])
            lpv, nv, nt = obj_lpips_audit(lp, imgs, gtf, objm)
            res[f"v{v}_lp"].append(lpv); det[v][0] += nv; det[v][1] += nt
            ps[v].append(psnr(imgs, gtf))
        np.save(f"{OUT}/gifs/seq{si}_render.npy", (np.clip(imgs, 0, 1) * 255).astype(np.uint8))
    out = {f"v{v}_lp": float(np.nanmean(res[f"v{v}_lp"])) for v in range(2)}
    out.update({f"v{v}_det": det[v][0] / max(det[v][1], 1) for v in range(2)})
    out.update({f"v{v}_psnr": float(np.mean(ps[v])) for v in range(2)})
    return out


def main():
    print(f"=== LATENT-RESIDUAL can_dual | ARM={ARM} RESIDUAL={RESIDUAL} HUMAN={HUMAN} "
          f"NROB_SPEC={NROB_SPEC} SMOKE={SMOKE} EPOCHS={EPOCHS} SEEDS={SEEDS} ===", flush=True)
    R, Hh = load_dual()
    okh = np.where(Hh["ok"])[0]                                          # ALL valid human clips (never held out)
    ho, pool = split_okfirst(R["ok"], HELDOUT)
    print(f"robot ok {int(R['ok'].sum())}/{len(R['ok'])} -> ho {len(ho)} pool {len(pool)}; "
          f"human ok {len(okh)}/{len(Hh['ok'])}", flush=True)

    Cz = latent_ch()
    latR = np.asarray(latcache(R, "robot")[:, :WIN]).astype(np.float32)   # (N,WIN,2,Cz,16,16)
    latH = np.asarray(latcache(Hh, "human")[:, :WIN]).astype(np.float32)
    efR = stack_ef(R, WIN)
    efH = stack_ef(Hh, WIN)
    print(f"latR {latR.shape} latH {latH.shape} Cz={Cz}", flush=True)

    nrob = len(pool) if NROB_SPEC is None else NROB_SPEC
    if SMOKE:
        nrob = min(nrob, 50)

    errs = []
    m = None
    for seed in range(SEEDS):
        sub = pool[np.random.default_rng(100 + seed).choice(len(pool), min(nrob, len(pool)), replace=False)]
        idx_h = okh if HUMAN else None
        m = train_dyn(latR, efR, latH, efH, sub, idx_h, RESIDUAL, seed=seed)
        e = dyn_err(m, latR[ho], efR[ho], RESIDUAL)
        errs.append(e)
        print(f"  seed{seed} held-out robot latent-MSE = {e:.4f}", flush=True)
    mse = float(np.mean(errs))

    from interactive_world_sim.algorithms.common.metrics.lpips import LearnedPerceptualImagePatchSimilarity
    lp = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device).eval()
    for p in lp.parameters():
        p.requires_grad_(False)
    mot = np.array([np.linalg.norm(np.diff(R["tr"][0][si, K:K + H_ROLL].mean(1), axis=0), axis=-1).sum()
                     for si in ho])
    chosen = ho[np.argsort(-mot)[:NSEQ]]
    dec_res = decode_eval(m, R, latR, efR, chosen, RESIDUAL, lp)

    res = {"arm": ARM, "residual": RESIDUAL, "human": HUMAN, "nrob": int(min(nrob, len(pool))),
           "n_human": int(len(okh)) if HUMAN else 0, "seeds": SEEDS, "epochs": EPOCHS,
           "latent_mse": mse, "latent_mse_per_seed": errs, **dec_res}
    json.dump(res, open(f"{OUT}/metrics.json", "w"), indent=2)

    lines = [f"LATENT-RESIDUAL can_dual human-helps | ARM={ARM} | RESIDUAL={RESIDUAL} HUMAN={HUMAN} "
             f"N_rob={res['nrob']} (+{res['n_human']} human) | SEEDS={SEEDS} EPOCHS={EPOCHS}",
             f"held-out robot latent-MSE (open-loop, H={H_ROLL}): {mse:.4f} (per-seed {errs})",
             f"decode obj-region LPIPS: v0(cam_high)={dec_res['v0_lp']:.4f} (det {dec_res['v0_det']:.2f}) "
             f"v1(cam_low)={dec_res['v1_lp']:.4f} (det {dec_res['v1_det']:.2f})",
             f"decode PSNR: v0={dec_res['v0_psnr']:.2f} v1={dec_res['v1_psnr']:.2f}",
             "", "compare against the sibling ARM's metrics.json for the same NROB to get human-helps Δ%",
             "(e.g. resid_ro100 vs resid_rh100), and against *_rofull for the ceiling gap."]
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines) + f"\nsaved {OUT}/summary.txt\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
