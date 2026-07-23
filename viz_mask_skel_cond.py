"""mask vs skel agent 条件通道的中间产物可视化(不是渲染输出, 是喂进模型的 agent 条件本身)。
每 seq 出动图: 列 = 原帧 | gmask剪影 | gmask叠加 | skel线画 | skel叠加, 两视角。iws env。
"""
import os, sys, numpy as np, cv2, imageio
os.environ.setdefault("RES", "128"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MASKGEN", "outputs/flow_wm/maskgen_caneef/maskgen_caneef.pt")
os.environ.setdefault("GMASK", "1"); os.environ.setdefault("GMASK_LOW", "1")
sys.path.insert(0, ".")
import exp_v3_human_helps_pixels as HP
import exp_scel_dualview_gmaskcond as G

DS = "outputs/flow_render_dataset_can_dual/clips_robot.npz"
SK = "outputs/flow_render_dataset_can_dual/skel_sidecar_robot.npz"
OUT = "outputs/video_arch_wm/m5_cond_intermediate"; os.makedirs(OUT, exist_ok=True)
SEQS = [int(x) for x in os.environ.get("SEQS", "332,442").split(",")]; IMG = 128
COL = [(255,80,80),(255,160,60),(255,230,60),(150,255,60),(60,255,160),(60,200,255),(120,120,255),(220,100,255),(255,100,180)]


def skel_draw(canvas, pts, segs):
    img = canvas.copy(); P = (pts * IMG).astype(np.int32)
    for a, b in segs:
        if np.all(np.abs(pts[a]) < 3) and np.all(np.abs(pts[b]) < 3): cv2.line(img, tuple(P[a]), tuple(P[b]), (255,255,255), 2, cv2.LINE_AA)
    for i, p in enumerate(P):
        if np.all(np.abs(pts[i]) < 3): cv2.circle(img, tuple(p), 3, COL[i], -1, cv2.LINE_AA)
    return img


def main():
    z = np.load(DS); zs = np.load(SK); segs = zs["segments"]
    if not HP._G: HP.load_gmask()
    G.load_gmask_low()
    lab = lambda im, t: cv2.putText(cv2.copyMakeBorder(im,20,0,0,0,cv2.BORDER_CONSTANT,value=(0,0,0)), t, (3,14), cv2.FONT_HERSHEY_SIMPLEX,0.42,(255,255,255),1)
    for si in SEQS:
        for v, vn, frk, skk in [(0,"high","frames","skel2d_high"),(1,"low","frames_low","skel2d_low")]:
            jt = np.nan_to_num(z["joint"][si].astype(np.float32)); ef = np.nan_to_num((z["eef"] if v==0 else z["eef_low"])[si].astype(np.float32))
            L = z[frk][si].shape[0]; frames = []
            gm_all = (HP.gmask_imgs(jt, ef) if v==0 else G.gmask_low_imgs(jt, ef))   # ★批量一次算所有帧
            for t in range(L):
                fr = z[frk][si, t]
                gm = gm_all[t]
                if gm.shape[0] != IMG: gm = cv2.resize(gm, (IMG,IMG))
                gm3 = (np.stack([gm]*3,-1)*255).astype(np.uint8)
                gmov = fr.copy(); gmov[gm>0.4] = (0.5*gmov[gm>0.4]+0.5*np.array([255,80,80])).astype(np.uint8)
                sk = zs[skk][si,t]; skimg = skel_draw(np.zeros_like(fr), sk, segs); skov = skel_draw(fr, sk, segs)
                row = np.concatenate([lab(fr,"orig"), lab(gm3,"mask(gmask)"), lab(gmov,"mask overlay"), lab(skimg,"skel lines"), lab(skov,"skel overlay")], 1)
                frames.append(row)
            imageio.mimsave(f"{OUT}/seq{si}_cam{vn}.gif", frames, fps=8, loop=0)
            print(f"seq{si} cam{vn}: cond intermediate gif", flush=True)
    open(f"{OUT}/README.txt","w").write("mask vs skel agent条件通道中间产物(喂进模型的agent条件本身)。\n列: 原帧|mask剪影|mask叠加|skel线画|skel叠加。两视角。\n")
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
