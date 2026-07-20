"""所有 action representation 的 ② rollout 综合对比 gif。
5 列 = [dummy5, skel-v2, skelv3, raster, rasterg], 每列: cam_high 帧 + action-rep overlay(青,输入形式)
+ ② rollout 预测物体点(红) vs GT 物体点(绿)。r-only n100(全部干净同条件)。多 seq, H=40。
输出 outputs/cross_embodiment_wm/action_reps_compare/。
"""
import os
os.environ.setdefault("SPLIT", "okfirst")
import numpy as np, torch, cv2
import __main__ as _m, exp_scel_dualview_wm as W
_m.DualLWC = W.DualLWC; _m.RasterAct = W.RasterAct
import imageio.v2 as imageio
from augment_clips_fullchain import rasterize_local

dev = "cuda"; K = W.K; HR = int(os.environ.get("HORIZON", "40"))
DS = "outputs/flow_render_dataset_can_dual"
OUT = "outputs/cross_embodiment_wm/action_reps_compare"; os.makedirs(OUT, exist_ok=True)
base = "outputs/cross_embodiment_wm/dualview_wm_skelact"
z = np.load(f"{DS}/clips_robot.npz")
sk2 = np.load(f"{DS}/skel_sidecar_robot_v2.npz")
ras = np.load(f"{DS}/raster_sidecar_robot.npz")
trA = z["tracks"].astype(np.float32); trB = np.nan_to_num(z["tracks_low"].astype(np.float32), nan=0.5)
trD_all = np.concatenate([trA, trB], 2); P = trA.shape[2]
frames = z["frames"]; eef = z["eef"]                        # (N,L,H,W,3), (N,L,3,2)
ok = np.isfinite(trA).all((1, 2, 3)); ho, _ = W.split_okfirst(ok, W.HELDOUT)
# 运动最大的 held-out seq
mot = np.array([np.linalg.norm(np.diff(trA[si, K:K + HR].mean(1), 0), axis=-1).sum() for si in ho])
NSEQ = int(os.environ.get("NSEQ", "6"))
chosen = [int(x) for x in ho[np.argsort(-mot)[:NSEQ]]]

REPS = [("dummy5", "PILOT_dummy5_r_n100", "dummy5", 7.78),
        ("skel-v2", "skel_r_n100", "skel", 8.66),
        ("skelv3", "skelv3_r_n100", "skelv3", 10.50),
        ("raster", "PILOT_raster_r_n100", "raster", 9.01),
        ("rasterg", "PILOT_rasterg_r_n100", "rasterg", 10.00)]
SKEL_IDX = [5, 6, 7, 4]


def load_act(act, si):
    if act in ("raster", "rasterg"):
        efA, efB = W.load_action_tokens(act, "r", z, ras)
    elif act in ("skel", "skelv3"):
        efA, efB = W.load_action_tokens(act, "r", z, sk2)
    else:
        efA, efB = W.load_action_tokens("dummy5", "r", z)
    return efA[si:si + 1], efB[si:si + 1]


def draw_actrep(img, act, si, t):
    """在帧上 overlay 该 action representation 的输入形式(青色)。"""
    im = img.copy()
    if act in ("dummy5", "skel", "skelv3"):
        if act == "dummy5":
            pts = eef[si, t] * 128                                  # 3 点(wrist,fin1,fin2)
            segs = [(0, 1), (0, 2)]
        else:
            pts = sk2["skel2d_high"][si, t][SKEL_IDX] * 128         # 4 点骨架
            segs = [(3, 0), (0, 1), (0, 2)]
        for a, b in segs:
            if np.isfinite(pts[[a, b]]).all():
                cv2.line(im, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), (0, 255, 255), 1)
        for p in pts:
            if np.isfinite(p).all():
                cv2.circle(im, tuple(p.astype(int)), 3, (0, 255, 255), -1)
    else:                                                          # raster/rasterg: 角落放真实光栅输入
        R = ras["raster_v0"][si, t].astype(np.uint8)               # 模型真实看到的 64px 局部光栅图
        thumb = cv2.cvtColor(cv2.resize(R, (44, 44), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2RGB)
        im[2:46, 2:46] = thumb
        cv2.rectangle(im, (2, 2), (46, 46), (0, 255, 255), 1)
    return im


def main():
    models = {}
    for _, d, act, _ in REPS:
        models[d] = torch.load(f"{base}/{d}/wm_dual.pt", map_location=dev, weights_only=False).eval()
    for si in chosen:
        trD = torch.from_numpy(trD_all[si:si + 1]).float().to(dev)
        preds = {}
        for _, d, act, _ in REPS:
            efA, efB = load_act(act, si)
            ea = torch.from_numpy(efA).float().to(dev); eb = torch.from_numpy(efB).float().to(dev)
            with torch.no_grad():
                pr = W.rollout_dual(models[d], trD, ea, eb, HR).cpu().numpy()[0]   # (HR,2P,2)
            preds[d] = pr[:, :P]                                                    # cam_high 物体点
        fr = []
        for h in range(HR):
            t = K + h
            cols = []
            for name, d, act, dr in REPS:
                im = (frames[si, t]).astype(np.uint8).copy()
                im = draw_actrep(im, act, si, t)
                gt = trA[si, t] * 128; pd = preds[d][h] * 128                       # GT 绿, 预测 红
                for p in gt:
                    if np.isfinite(p).all(): cv2.circle(im, tuple(p.astype(int)), 2, (0, 220, 0), -1)
                for p in pd:
                    if np.isfinite(p).all(): cv2.circle(im, tuple(p.astype(int)), 2, (255, 40, 40), -1)
                im = cv2.resize(im, (200, 200), interpolation=cv2.INTER_NEAREST)
                lab = np.zeros((26, 200, 3), np.uint8)
                pad = "extrap" if h >= 24 else ""
                cv2.putText(im, f"h{h}{('  '+pad) if pad else ''}", (4, 194), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
                cv2.putText(lab, f"{name} r{dr:.1f}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
                cols.append(np.concatenate([lab, im], 0))
            fr.append(np.concatenate(cols, 1))
        imageio.mimsave(f"{OUT}/seq{si}.gif", fr, duration=0.15, loop=0)
        print(f"seq{si} done", flush=True)
    open(f"{OUT}/README.txt", "w").write(
        "5 种 action representation 的 ② rollout 对比 (r-only n100, cam_high, H=40)\n"
        "列: dummy5 / skel-v2 / skelv3 / raster / rasterg (标题含 r-only drift px)\n"
        "青=该表示输入(点/骨架线 or 角落光栅缩略); 绿=GT物体点; 红=②预测物体点\n"
        "★ 判据: 红点跟绿点越紧越好. dummy5(精确坐标)预测最准; 其余引入噪声/失配基线更烂\n"
        "h>=24 标 extrap(clip L48 动作窗口 K+F=24, 之后动作填充, 对所有列一样)\n"
        "结论: 5 种 action 表示 dummy5 完胜, rh 绝对精度 3.03 < raster 3.54 < rasterg 3.72\n")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
