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

dev = "cuda"; K = W.K; HR = int(os.environ.get("HORIZON", "24"))   # 合法上限(clip L48, 动作窗口 24)
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

# 全量 r-only 模型(预测好, 公平对比表示; 5点/4点/光栅各是真实输入)
REPS = [("dummy5(5pt)", "dualview_wm", "dummy5", 2.32),
        ("skel-v2(4pt)", "skel_r_nall", "skel", 2.56),
        ("skelv3(scale-inv)", "skelv3_r_nall", "skelv3", 3.73),
        ("raster(local)", "FULL_raster_r_nall", "raster", 2.13),
        ("rasterg(+grip)", "FULL_rasterg_r_nall", "rasterg", 1.96)]
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
    if act == "dummy5":
        e3 = eef[si, t]                                            # (3,2) wrist,fin1,fin2
        w, t1, t2 = e3[0], e3[1], e3[2]
        c = (t1 + t2) / 2; ax = (t1 - t2) / 2; perp = np.array([-ax[1], ax[0]])
        pts = np.stack([w, c, c + ax, c - ax, c + perp]) * 128     # ★ dummy5 真正的 5 点虚拟星座
        segs = [(0, 1), (1, 2), (1, 3), (1, 4)]
    elif act in ("skel", "skelv3"):
        pts = sk2["skel2d_high"][si, t][SKEL_IDX] * 128            # 4 点骨架(v2/v3 信息来源相同)
        segs = [(3, 0), (0, 1), (0, 2)]
    else:                                                         # raster/rasterg: 角落放真实光栅输入
        R = ras["raster_v0"][si, t].astype(np.uint8)              # 模型真实看到的 64px 局部光栅图(夹爪居中)
        thumb = cv2.cvtColor(cv2.resize(R, (44, 44), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2RGB)
        im[2:46, 2:46] = thumb
        cv2.rectangle(im, (2, 2), (46, 46), (0, 255, 255), 1)
        cv2.putText(im, "local/EE-ctr", (2, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
        return im
    for a, b in segs:
        if np.isfinite(pts[[a, b]]).all():
            cv2.line(im, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), (0, 255, 255), 1)
    for p in pts:
        if np.isfinite(p).all():
            cv2.circle(im, tuple(p.astype(int)), 3, (0, 255, 255), -1)
    return im


def main():
    models = {}
    for _, d, act, _ in REPS:
        p = "outputs/cross_embodiment_wm/dualview_wm/wm_dual.pt" if d == "dualview_wm" else f"{base}/{d}/wm_dual.pt"
        models[d] = torch.load(p, map_location=dev, weights_only=False).eval()
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
        "5 种 action representation 的 ② rollout 对比 (★全量 robot-only, cam_high, H=24)\n"
        "列: dummy5(5pt) / skel-v2(4pt) / skelv3(scale-inv) / raster(local) / rasterg(+grip)\n"
        "  标题 r数字 = 该表示全量 robot-only 的 H=20 drift px\n"
        "青=该表示输入(dummy5 5点星座/skel 4点骨架/光栅角落缩略图, 局部夹爪居中);\n"
        "绿=GT物体点; 红=②预测物体点。红跟绿越紧越好。\n"
        "★★重大发现: 表示优劣依赖数据量, 稀缺与充足完全反转:\n"
        "  稀缺 n100:  dummy5 7.78 < raster 9.01 < rasterg 10.00  (精确坐标赢, 光栅化噪声主导)\n"
        "  全量 nall:  rasterg 1.96 < raster 2.13 < dummy5 2.32 < skel 2.56 < skelv3 3.73\n"
        "              (光栅化反超! 大数据压下噪声, 丰富构型信息发挥作用, 印证 OSCAR 大数据用光栅)\n"
        "→ human-helps 场景是 robot 稀缺 -> dummy5 最优; 若有大规模 robot 数据 -> 光栅化(OSCAR式)更好\n")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
