"""Keyboard-style interactive demo: drive ② one step at a time with simple unit actions
(up/down/left/right), like arrow keys. Mimics the interactive_world_sim end product. Each press =
small fixed eef delta in one direction (gentle, stays in frame). Start from a GRASPED+CENTRAL seq so
'pushing' is meaningful. Render EVERY step. baseline-eef3 ② only (thick proven useless).

Adds a SHAPE-COHESION metric (mean 48-pt spread from centroid, px) so we catch the point-scatter that
travel/follow_cos/det missed last round. Output: outputs/cross_embodiment_wm/scel_keyboard/"""
import os
os.environ["USE_GMASK"] = "1"; os.environ["FOOTPRINT"] = "1"; os.environ["DROP_VIS"] = "1"; os.environ["USE_PREV"] = "0"
import numpy as np, torch
from PIL import Image
from viz_combined import build_flow_cols, save_combined_gif
from exp_v3_human_helps_pixels import render_seq, load_gmask, K, F, device
from amplify_wm import rollout_lwc
import exp_scel_ss_long as SL

SMOKE = os.environ.get("SMOKE", "0") == "1"
DS_LONG = "outputs/flow_render_dataset_v3_long"
H = int(os.environ.get("HORIZON", "24")); HELDOUT = 150; IMG = 128
DELTA = float(os.environ.get("DELTA", "0.013"))            # eef move per keypress (normalized)
HO_IDX = [int(x) for x in os.environ.get("HO_IDX", "20,53").split(",")]
RENDERER = os.environ.get("RENDERER", "outputs/cross_embodiment_wm/renderer_long_fp_novis/renderer.pt")
OUT = "outputs/cross_embodiment_wm/scel_keyboard"; os.makedirs(f"{OUT}/gifs", exist_ok=True)
DIRS = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}   # image coords (x right, y down)


def keyboard_eef(ef_hist_last, dx, dy, n):
    """ef_hist_last (3,2); one unit press per step -> (n,3,2), rigid 3-pt offset kept, clipped in-frame."""
    base_c = ef_hist_last.mean(0); off = ef_hist_last - base_c
    step = np.array([dx, dy], np.float32) * DELTA
    c = base_c[None] + np.arange(1, n + 1)[:, None] * step[None]
    return np.clip(c[:, None] + off[None], 0.04, 0.96).astype(np.float32)


def cohesion(pred):
    """pred (H,P,2) -> per-step mean point spread from centroid in px (rising = cube scattering)."""
    return np.linalg.norm(pred - pred.mean(1, keepdims=True), axis=-1).mean(1) * IMG   # (H,)


def contact_sheet(frames, path, cols=8):
    """frames (H,128,128,3) uint8 -> tiled PNG, every step visible at a glance."""
    h = len(frames); rows = (h + cols - 1) // cols; s = frames.shape[1]
    cv = np.full((rows * s, cols * s, 3), 255, np.uint8)
    for i, f in enumerate(frames):
        r, c = divmod(i, cols); cv[r * s:(r + 1) * s, c * s:(c + 1) * s] = f
    Image.fromarray(cv).save(path)


def main():
    if SMOKE:
        import eval_scheduled_sampling as SSm; SSm.WM_EPOCHS = 2
    load_gmask()
    ren = torch.load(RENDERER, map_location=device, weights_only=False).to(device).eval()
    zl = np.load(f"{DS_LONG}/clips_robot.npz")
    lt, le, lv, lf = (zl["tracks"].astype(np.float32), zl["eef"].astype(np.float32),
                      zl["vis"].astype(np.float32), zl["frames"]); ljt = zl["joint"].astype(np.float32)
    ho = np.random.default_rng(0).permutation(len(lt))[:HELDOUT]; lpool = np.random.default_rng(0).permutation(len(lt))[HELDOUT:]
    if SMOKE: lpool = lpool[:200]
    print("=== train baseline-eef3 ② (long data, SS) ===", flush=True)
    wm = SL.train_long(lt, lv, le, torch.from_numpy(lpool), 8 if SMOKE else 32)

    u8 = lambda x: (np.clip(x, 0, 1) * 255).astype(np.uint8)
    lines = [f"Keyboard demo | unit press DELTA={DELTA}/step | H={H} steps | baseline-eef3 ② | renderer_long",
             "cohesion = 48-pt spread px (start->end; big rise = cube scatters apart)", ""]
    for hi in HO_IDX:
        si = ho[hi]
        lines.append(f"--- ho-idx {hi} (seq{si}) ---")
        I0 = lf[si, 0]; vis_seq = lv[si, K:K + H]; jt_seq = ljt[si, K:K + H]
        cube0 = np.repeat(lt[si, K][None], H, 0)
        for dname, (dx, dy) in DIRS.items():
            ef_fut = keyboard_eef(le[si, K - 1], dx, dy, H + F)               # (H+F,3,2)
            ef_full = np.concatenate([le[si, :K], ef_fut], 0)[None]
            ef_render = ef_full[0, K:K + H]
            pr = rollout_lwc(wm, torch.from_numpy(lt[si:si + 1, :K]).float().to(device),
                             torch.from_numpy(ef_full).float().to(device), H).cpu().numpy()[0]
            rseq = render_seq(ren, I0, lt[si, 0], le[si, 0], pr, ef_render, vis_seq, jt_seq)
            coh = cohesion(pr); cc = pr.mean(1)
            travel = float(np.linalg.norm(cc[-1] - cc[0]) * IMG)
            lines.append(f"  {dname:>5}: travel {travel:5.1f}px | cohesion {coh[0]:4.1f}->{coh[-1]:4.1f}px"); print(lines[-1], flush=True)
            flow_cols = build_flow_cols(lf[si, K:K + H].astype(np.uint8), cube0, [pr], ef_render)
            save_combined_gif(f"{OUT}/gifs/seq{si}_{dname}.gif", u8(rseq)[None], flow_cols, [dname], [None], K)
            contact_sheet(flow_cols[0], f"{OUT}/gifs/seq{si}_{dname}_steps.png")
        lines.append("")
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines) + f"\nsaved {OUT}/\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
