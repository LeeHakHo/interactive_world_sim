"""M3: FK 骨架(已由 augment_clips_skeleton.py 投影到 REAL calib 的 skel_sidecar_robot.npz)可视化。
产物 outputs/video_arch_wm/m3_skel/:
  - skel_overlay_{view}.png : 多帧 [原帧 | 骨架线画 | 叠加] 竖排, 眼检 FK 骨架压在真臂上(R3 gate)。
  - skel_vs_gmask_{view}.png : 骨架 vs agent gmask 剪影并排(ablation B: agent 用 skeleton 还是 mask)。
iws env 跑。骨架 9 点: link_1..6, carriage_L, carriage_R, ee_gripper; 8 segments。
"""
import os, numpy as np, cv2, torch
os.environ.setdefault("RES", "128"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MASKGEN", "outputs/flow_wm/maskgen_caneef/maskgen_caneef.pt")
import exp_v3_human_helps_pixels as HP
import exp_scel_dualview_gmaskcond as G

DS = "outputs/flow_render_dataset_can_dual/clips_robot.npz"
SK = "outputs/flow_render_dataset_can_dual/skel_sidecar_robot.npz"
OUT = "outputs/video_arch_wm/m3_skel"; os.makedirs(OUT, exist_ok=True)
IMG = 128
COL = [(255, 80, 80), (255, 160, 60), (255, 230, 60), (150, 255, 60), (60, 255, 160),
       (60, 200, 255), (120, 120, 255), (220, 100, 255), (255, 100, 180)]


def draw_skel(canvas, pts2d, segs, grip=None):
    """pts2d (9,2) crop-norm -> 画线画; canvas HxWx3 uint8 (in-place 风格, 返回新图)。"""
    img = canvas.copy(); P = (pts2d * IMG).astype(np.int32)
    for a, b in segs:
        pa, pb = P[a], P[b]
        if np.all(np.abs(pts2d[a]) < 3) and np.all(np.abs(pts2d[b]) < 3):
            cv2.line(img, tuple(pa), tuple(pb), (255, 255, 255), 2, cv2.LINE_AA)
    for i, p in enumerate(P):
        if np.all(np.abs(pts2d[i]) < 3):
            cv2.circle(img, tuple(p), 3, COL[i], -1, cv2.LINE_AA)
    return img


def main():
    z = np.load(DS); zs = np.load(SK)
    segs = zs["segments"]
    if not HP._G: HP.load_gmask()
    G.load_gmask_low()
    seqs = [332, 59, 418]; frames_k = [0, 12, 24, 36, 47]
    for view, vname, skkey, frkey in [(0, "cam_high", "skel2d_high", "frames"),
                                      (1, "cam_low", "skel2d_low", "frames_low")]:
        rows_img = []
        for si in seqs:
            for k in frames_k:
                fr = z[frkey][si, k]                                  # (128,128,3) uint8
                sk = zs[skkey][si, k]                                 # (9,2)
                blk = draw_skel(np.zeros_like(fr), sk, segs)
                ovl = draw_skel(fr, sk, segs)
                rows_img.append(np.concatenate([fr, blk, ovl], 1))
        grid = np.concatenate(rows_img, 0)
        cv2.imwrite(f"{OUT}/skel_overlay_{vname}.png", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        # skeleton vs gmask 并排 (agent 通道两选)
        vg = []
        for si in seqs[:2]:
            for k in [12, 24, 36]:
                fr = z[frkey][si, k]; sk = zs[skkey][si, k]
                jt = z["joint"][si, k].astype(np.float32); ef = (z["eef"] if view == 0 else z["eef_low"])[si, k].astype(np.float32)
                g = (HP.gmask_imgs(jt[None], ef[None])[0] if view == 0 else G.gmask_low_imgs(jt[None], ef[None])[0])
                if g.shape[0] != IMG: g = cv2.resize(g, (IMG, IMG))
                gimg = (np.stack([g] * 3, -1) * 255).astype(np.uint8)
                skimg = draw_skel(np.zeros_like(fr), sk, segs)
                vg.append(np.concatenate([fr, skimg, gimg], 1))
        cv2.imwrite(f"{OUT}/skel_vs_gmask_{vname}.png", cv2.cvtColor(np.concatenate(vg, 0), cv2.COLOR_RGB2BGR))
        print(f"{vname}: wrote skel_overlay + skel_vs_gmask", flush=True)
    open(f"{OUT}/README.txt", "w").write(
        "M3 FK 骨架(skel_sidecar_robot.npz, augment_clips_skeleton 用 REAL calib 投影)。\n"
        "skel_overlay_{view}.png 列: 原帧 | 骨架线画 | 叠加. 眼检骨架压在真臂上(R3 gate)。\n"
        "skel_vs_gmask_{view}.png 列: 原帧 | skeleton | gmask剪影. = ablation B 的两种 agent 条件。\n"
        "9点: link_1..6, carriage_L, carriage_R, ee_gripper; 8 segments。\n")
    print("=== M3 DONE ===", flush=True)


if __name__ == "__main__":
    main()
