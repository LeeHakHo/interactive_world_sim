"""可视化 regularize 后的 eef→dummy5星座, overlay在human帧上配pred-flow. 眼检有无问题.
2列: RAW dummy5星座 | REG dummy5星座; 均叠 GT物体(绿)+dummy5_rh_reg预测物体(红)+星座(白)+eef(黄)."""
import os, sys, numpy as np, torch, cv2, imageio
os.environ.setdefault("HF_HUB_OFFLINE", "1"); sys.path.insert(0, ".")
import exp_scel_dualview_wm as W
setattr(sys.modules["__main__"], "DualLWC", W.DualLWC)
dev = "cpu"
RAW = np.load("outputs/flow_render_dataset_can_dual_L48/clips_human_L48_retrack_realwrist.npz")
REG = np.load("outputs/flow_render_dataset_can_dual_L48/clips_human_L48_retrack_realwrist_REG.npz")
wm = torch.load("outputs/cross_embodiment_wm/epsplit_L48/dummy5_rh_reg/wm_dual.pt", map_location=dev, weights_only=False).eval()
K, P = W.K, 48
efA_full, efB_full = W.load_action_tokens("dummy5", "h", REG)   # REG eef → dummy5(reg模型训在此)
def constellation(ef3):   # (L,3,2)->(L,5,2) [wrist,c,c+ax,c-ax,c+perp]
    w, t1, t2 = ef3[:, 0], ef3[:, 1], ef3[:, 2]
    c = (t1+t2)/2; ax = (t1-t2)/2; perp = np.stack([-ax[:, 1], ax[:, 0]], -1)
    return np.stack([w, c, c+ax, c-ax, c+perp], 1)
def dot(im, xy, col, r):
    x, y = int(xy[0]*128), int(xy[1]*128)
    if 0 <= x < 128 and 0 <= y < 128: cv2.circle(im, (x, y), r, col, -1)
SEQS = [int(x) for x in os.environ.get("SEQS", "0,400,900,1400").split(",")]
OUT = "outputs/cross_embodiment_wm/reg_constellation_viz"; os.makedirs(f"{OUT}/gifs", exist_ok=True)
for si in SEQS:
    fr = RAW["frames"][si]                                       # (L,128,128,3) cam_high
    trg = [np.nan_to_num(RAW["tracks"][si].astype(np.float32)), np.nan_to_num(RAW["tracks_low"][si].astype(np.float32))]
    trD = np.concatenate([trg[0][:K], trg[1][:K]], 1)[None]
    with torch.no_grad():
        pr = W.rollout_dual(wm, torch.from_numpy(trD).float(), torch.from_numpy(efA_full[si][None]).float(),
                            torch.from_numpy(efB_full[si][None]).float(), 48-K, dom="h")[0].numpy()
    predtr = np.concatenate([trg[0][:K], pr[:, :P]], 0)          # cam_high pred物体 (L,48,2)
    con_raw = constellation(np.nan_to_num(RAW["eef"][si]))       # (L,5,2)
    con_reg = constellation(np.nan_to_num(REG["eef"][si]))
    frames = []
    L = min(len(fr), len(predtr), 44)
    for t in range(L):
        base = fr[t].copy()
        cols_im = []
        for con, lab in [(con_raw, "RAW dummy5"), (con_reg, "REG dummy5")]:
            im = (base.astype(np.float32)*0.6).astype(np.uint8).copy()
            for p in trg[0][t]: dot(im, p, (0, 220, 0), 1)         # GT物体 绿
            for p in predtr[t]: dot(im, p, (60, 60, 240), 1)       # 预测物体 红
            # 星座: 白点+连线(wrist-c)
            cc = con[t]
            for p in cc: dot(im, p, (255, 255, 255), 2)
            cv2.line(im, tuple((cc[0]*128).astype(int)), tuple((cc[1]*128).astype(int)), (255, 255, 0), 1)  # wrist→c 黄线
            im = cv2.resize(im, (256, 256), interpolation=cv2.INTER_NEAREST)
            cv2.putText(im, lab, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cols_im.append(im)
        row = np.concatenate(cols_im, 1)
        cv2.putText(row, "white=dummy5星座 green=GT物体 red=pred物体 yellow=wrist->c", (4, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        frames.append(row)
    imageio.mimsave(f"{OUT}/gifs/reg_const_seq{si}.gif", frames, fps=6)
    print(f"saved {OUT}/gifs/reg_const_seq{si}.gif ({L}帧)", flush=True)
print("DONE")
