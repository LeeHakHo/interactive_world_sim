"""keyboard 3D 中间产物可视化 (无GPU, 读 precompute/ + skel2d/).
每 seq×script 出 gif: 列 = [cam_high 原帧+投影(eef黄/物体红/skel白) | cam_low 同 | grip/z 时间线]。
重点看 lift_high: cam_high 罐子水平位移小(cam非top-down有小透视分量)、cam_low 竖直上移 → 证明纯z一致投影。
跑: .venv_wan/bin/python viz_keyboard_3d_intermediate.py
"""
import os, glob, numpy as np, cv2, imageio
DS = "outputs/flow_render_dataset_can_dual/clips_robot.npz"
OUT = os.environ.get("OUT", "outputs/video_arch_wm/keyboard_3d")
IMG = 128
COL = [(255,80,80),(255,160,60),(255,230,60),(150,255,60),(60,255,160),(60,200,255),(120,120,255),(220,100,255),(255,100,180)]


def draw(fr, obj, ef, sk, segs):
    im = fr.copy(); s = IMG
    for a, b in segs:                                       # skel 白线
        if np.all(np.abs(sk[a]) < 3) and np.all(np.abs(sk[b]) < 3):
            cv2.line(im, tuple((sk[a]*s).astype(int)), tuple((sk[b]*s).astype(int)), (255,255,255), 1, cv2.LINE_AA)
    for i, p in enumerate(sk):
        if np.all(np.abs(p) < 3): cv2.circle(im, tuple((p*s).astype(int)), 2, COL[i % 9], -1)
    for p in obj: cv2.circle(im, tuple((p*s).astype(int)), 1, (255,60,60), -1)   # 物体红
    for p in ef: cv2.circle(im, tuple((p*s).astype(int)), 3, (255,230,0), -1)    # eef黄
    return im


def tline(vals, H, t, label, color):
    w = IMG; im = np.zeros((IMG, w, 3), np.uint8)
    v = np.asarray(vals); v = (v - v.min()) / (v.ptp() + 1e-6)
    for k in range(1, H):
        cv2.line(im, (int(w*(k-1)/H), int(IMG-4-v[k-1]*(IMG-24))), (int(w*k/H), int(IMG-4-v[k]*(IMG-24))), color, 1)
    cv2.line(im, (int(w*t/H), 0), (int(w*t/H), IMG), (80,80,80), 1)               # 当前帧游标
    cv2.putText(im, label, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    return im


def lab(im, t):
    return cv2.putText(cv2.copyMakeBorder(im, 18, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0)),
                       t, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)


def main():
    z = np.load(DS)
    os.makedirs(f"{OUT}/intermediate", exist_ok=True)
    pres = sorted(glob.glob(f"{OUT}/precompute/*.npz"))
    if not pres:
        print("no precompute npz; run --stage precompute first", flush=True); return
    for pre in pres:
        base = os.path.basename(pre)[:-4]; skf = f"{OUT}/skel2d/{base}.npz"
        if not os.path.exists(skf):
            print(f"skip {base} (no skel2d)", flush=True); continue
        d = np.load(pre); sk = np.load(skf); si = int(d["si"]); H = int(d["H"])
        segs = sk["segments"]; skel = [sk["skel2d_high"], sk["skel2d_low"]]
        traj = [d["trajA"], d["trajB"]]; ef = [d["efA"], d["efB"]]
        fr0 = [z["frames"][si, 0], z["frames_low"][si, 0]]
        grip = d["grip"]
        frames = []
        for t in range(H):
            ch = draw(fr0[0], traj[0][t], ef[0][t], skel[0][t], segs)
            cl = draw(fr0[1], traj[1][t], ef[1][t], skel[1][t], segs)
            tl = tline(grip, H, t, "grip", (60,200,255))
            row = np.concatenate([lab(ch, f"cam_high {base}"), lab(cl, "cam_low"), lab(tl, "grip timeline")], 1)
            frames.append(row)
        imageio.mimsave(f"{OUT}/intermediate/{base}.gif", frames, fps=8, loop=0)
        print(f"[interm] {base}: H={H}", flush=True)
    print("=== intermediate DONE ===", flush=True)


if __name__ == "__main__":
    main()
