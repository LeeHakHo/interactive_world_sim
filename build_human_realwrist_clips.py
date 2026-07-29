"""把真腕接进 human clips 的 eef(2026-07-28, 用户: "我建了真腕却没用")。
clips_human_L24_retrack.npz 的 eef[:,:,0]/eef_low[:,:,0]=退化指尖中点(非真腕, 差真腕26px)。
用 wrist_sidecar_human_low.npz(wrist2d_high_proj/wrist2d_low, 真腕world3D投影)替换腕槽,
真腕无效帧回退指尖中点(不引NaN)。其余字段原样拷贝。→ clips_human_L24_retrack_realwrist.npz。
用 .venv_wan/bin/python。
"""
import os, numpy as np
D = os.environ.get("BASE", "outputs/flow_render_dataset_can_dual")
HCLIP = os.environ.get("HCLIP", "clips_human_L24")   # basename含clips_前缀; L48=clips_human_L48
zh = dict(np.load(f"{D}/{HCLIP}_retrack.npz"))
sc = np.load(f"{D}/wrist_sidecar_human_low.npz")
wr_hi = sc["wrist2d_high_proj"].astype(np.float32)      # (N,L,2) 真腕高视角
wr_lo = sc["wrist2d_low"].astype(np.float32)            # (N,L,2) 真腕低视角
vld = sc["wrist_valid_low"]                             # (N,L) low inbounds

for key, wr in [("eef", wr_hi), ("eef_low", wr_lo)]:
    e = zh[key].astype(np.float32).copy()               # (N,L,3,2)
    c = (e[:, :, 1] + e[:, :, 2]) / 2                    # 指尖中点(真腕无效时回退)
    ok = np.isfinite(wr).all(-1)                         # 真腕有效
    neww = np.where(ok[..., None], wr, c)               # 有效用真腕, 否则中点
    e[:, :, 0] = neww
    zh[key] = e
    print(f"{key}: 真腕有效 {ok.mean()*100:.1f}% 帧替换, 其余回退指尖中点", flush=True)

out = f"{D}/{HCLIP}_retrack_realwrist.npz"
np.savez(out, **zh)
print(f"saved {out}", flush=True)
# 校验
z2 = np.load(out); e2 = z2["eef"].astype(float)
c2 = (e2[:, :, 1] + e2[:, :, 2]) / 2; m = np.isfinite(e2).all((-1, -2))
d = np.linalg.norm((e2[:, :, 0] - c2)[m], axis=-1) * 128
print(f"校验: 新eef 腕-c 距离 median {np.median(d):.1f}px (应~26, 非0)", flush=True)
