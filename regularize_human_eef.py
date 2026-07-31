"""regularize human eef 3点 → robot canonical 几何(2026-07-31)。
human手: 开口2.2×/不对称5.7×/斜22°(vs robot对称竖直), 直接喂agent-trace跨具身破功。
保留跨具身共享信息: 接触位置c(两指中点)+ 朝向(指轴u & 腕→中点)+ 开合程度(标量);
强制几何 → robot canonical: 对称、robot尺度R0、腕在固定canonical方位。
=mp_constellation思想但保3点结构(腕+2指尖)供agent-trace用。前后对比证 human≈robot。
用法: python regularize_human_eef.py
"""
import os, numpy as np, cv2

IMG = 128
OUT = "outputs/cross_embodiment_wm/agent_trace_derisk"; os.makedirs(OUT, exist_ok=True)
RB = "outputs/flow_render_dataset_can_dual/clips_robot_retrack.npz"
HM = "outputs/flow_render_dataset_can_dual_L48/clips_human_L48_retrack_realwrist.npz"


def robot_canon_params(zr):
    """从robot eef估canonical几何: 半开口R0, 腕→中点距Lw (robot自己的对称尺度)。"""
    ef = np.nan_to_num(zr["eef"].astype(np.float64))
    w, f1, f2 = ef[:, :, 0], ef[:, :, 1], ef[:, :, 2]
    R0 = np.nanmedian(np.linalg.norm(f1 - f2, axis=-1)) / 2      # 半开口
    mid = (f1 + f2) / 2
    Lw = np.nanmedian(np.linalg.norm(mid - w, axis=-1))         # 腕→中点距
    return float(R0), float(Lw)


def regularize_eef(ef3, R0, Lw, keep_gap=True):
    """ef3 (...,3,2)[腕,指1,指2] -> canonical 3点. 保留c位置+指轴朝向+腕朝向(+可选真实开口比例),
    但对称竖直、robot尺度。"""
    w, t1, t2 = ef3[..., 0, :], ef3[..., 1, :], ef3[..., 2, :]
    c = (t1 + t2) / 2                                            # 接触位置(共享, 保留)
    d = t1 - t2; dn = np.linalg.norm(d, axis=-1, keepdims=True)
    u = d / np.clip(dn, 1e-4, None)                             # 指轴方向(真实朝向, 保留)
    # 腕方向: 用真实腕→中点方向(保留朝向), 但距离归canonical Lw
    wm = w - c; wmn = np.linalg.norm(wm, axis=-1, keepdims=True)   # ★fix: c→腕方向(原c-w把腕side翻到相反侧, 用户眼检抓出)
    uw = wm / np.clip(wmn, 1e-4, None)
    # ★彻底canonical到robot尺度: 半开口=robot R0; 只保留"开合相对变化"(可选, 归一到robot开口范围)
    if keep_gap:
        # human开口相对它自己中位的比例 -> 映射到robot开口(小幅波动, 不带human的2×绝对大开口)
        hmed = np.nanmedian(dn)
        r = R0 * np.clip(dn / (hmed + 1e-6), 0.6, 1.4)         # 围绕robot R0 ±40%, 反映开合但robot尺度
    else:
        r = R0                                                 # 全固定robot尺度
    t1c = c + r * u                                            # 对称: 两指到c等距(asym=0构造保证)
    t2c = c - r * u
    # ★腕: 强制垂直于开口轴(robot几何: 腕在开口轴的垂线上), 保留腕在垂线哪一侧(朝向的符号)
    perp = np.stack([-u[..., 1], u[..., 0]], -1)              # 开口轴的垂线方向
    side = np.sign(np.sum(uw * perp, axis=-1, keepdims=True)); side[side == 0] = 1
    wc = c + Lw * perp * side                                 # 腕: canonical距离 + 垂直 + 真实那一侧
    return np.stack([wc, t1c, t2c], -2)


def draw_eef(ef, color_w=(0, 200, 255), color_f=(255, 80, 80)):
    """画 eef 3点(腕蓝, 指红)+连线 到 (128,128,3)。"""
    im = np.full((IMG, IMG, 3), 30, np.uint8)
    P = (ef * IMG).astype(int)
    cv2.line(im, tuple(P[1]), tuple(P[2]), (120, 120, 120), 1)   # 指-指
    cv2.line(im, tuple(P[0]), tuple(((P[1] + P[2]) // 2)), (80, 80, 80), 1)  # 腕-中点
    cv2.circle(im, tuple(P[0]), 4, color_w, -1)                  # 腕
    cv2.circle(im, tuple(P[1]), 4, color_f, -1); cv2.circle(im, tuple(P[2]), 4, color_f, -1)
    return im


def main():
    zr = np.load(RB); zh = np.load(HM)
    R0, Lw = robot_canon_params(zr)
    print(f"[robot canon] R0(半开口)={R0:.4f} Lw(腕→中点)={Lw:.4f}")
    hef = np.nan_to_num(zh["eef"].astype(np.float64))
    # 前后几何统计
    def stats(ef, tag):
        w, f1, f2 = ef[..., 0, :], ef[..., 1, :], ef[..., 2, :]
        gap = np.linalg.norm(f1 - f2, axis=-1)
        c = (f1 + f2) / 2
        # asym: 两指到中点c是否等距(canonical判据, 应=0); 另报开口轴⊥腕轴角
        s1 = np.linalg.norm(f1 - c, axis=-1); s2 = np.linalg.norm(f2 - c, axis=-1)
        asym = np.abs(s1 - s2) / (np.maximum(s1, s2) + 1e-6)
        ao = f1 - f2; aw = c - w
        cos = np.sum(ao * aw, -1) / (np.linalg.norm(ao, axis=-1) * np.linalg.norm(aw, axis=-1) + 1e-6)
        ang = np.degrees(np.arccos(np.clip(np.abs(cos), 0, 1)))
        v = np.isfinite(gap) & np.isfinite(asym) & np.isfinite(ang)
        print(f"  [{tag}] gap {np.nanmean(gap):.4f}  asym(指对c) {np.nanmean(asym[v]):.3f}  开口⊥腕角 {np.nanmean(ang[v]):.1f}")
    stats(np.nan_to_num(zr["eef"].astype(np.float64)), "robot")
    stats(hef, "human raw")
    hreg = regularize_eef(hef, R0, Lw)
    stats(hreg, "human canon")
    # 眼检: human 3帧(raw|canon) + robot 参考3帧, 局部放大(eef附近crop)看几何
    ref = np.nan_to_num(zr["eef"].astype(np.float64))
    rows = []
    for hi, ht in [(0, 20), (400, 25), (900, 15)]:
        raw = draw_eef(hef[hi, ht]); can = draw_eef(hreg[hi, ht])
        rb = draw_eef(ref[hi, ht])                              # 同idx robot 参考(展示robot几何)
        rows.append(np.concatenate([raw, can, rb], 1))
    grid = np.concatenate(rows, 0)
    grid = cv2.resize(grid, (grid.shape[1] * 2, grid.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
    cv2.putText(grid, "human RAW | human CANON | robot ref   (wrist=blue, fingers=red)", (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
    cv2.imwrite(f"{OUT}/regularize_compare.png", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"[fig] {OUT}/regularize_compare.png (human raw斜/不对称 → canon对称竖直 ≈ robot)", flush=True)
    np.save(f"{OUT}/human_eef_canon.npy", hreg.astype(np.float32))
    print(f"[save] {OUT}/human_eef_canon.npy  (regularized human eef 3点, 供agent-trace用)", flush=True)


if __name__ == "__main__":
    main()
