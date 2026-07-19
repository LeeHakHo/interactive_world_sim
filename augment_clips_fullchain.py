"""完整运动链 sidecar:human 21 个手部关节 + robot 8 个 link,各自完整,不做任何点配对。

动机(2026-07-19):skel-v2 的"同构"把 robot 9 关节和 human 21 关节强行削成 4 点配对,
已量化证明失败 —— 只有腕点真正对齐,两指对称比 robot 1.04 vs human 1.27(分布不重叠)、
槽位3 human 是写死 19.2px 的常数占位符、整体量纲差 1.7×。

OSCAR(2606.04463v2 §3.2)的做法解除了这个约束:两域**各自用完整运动链**,
光栅化成同一种"黑底 + 线段 + 圆点"的 2D 线画图像,格式统一在**像素空间**而非向量槽位上,
因此点数不必相同。原文:"It only depends on the kinematic chain, and changing the embodiment
only updates the kinematic specification. A single representation can therefore capture
different robots, humans, or any mixture of them."

本脚本只做数据:把两域的完整关节投影到两个视角的 crop-norm 坐标,连同各自的拓扑边一起存盘。
光栅化在训练侧做(见 rasterize_chain)。

无泄漏:human 用 `observation.eef.kpts21_3d_world`(当前帧手部标注),
robot 用 URDF 前向运动学(当前帧 joint),都不碰未来帧。

输出 outputs/flow_render_dataset_can_dual/chain_sidecar_{human,robot}.npz
"""
import numpy as np
import pandas as pd

import augment_clips_skeleton as S

DS = "outputs/flow_render_dataset_can_dual"

# HaMeR / MediaPipe 21-keypoint 手部拓扑(标准):
# 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
HAND_SEGMENTS = np.array(
    [(0, 1), (1, 2), (2, 3), (3, 4),           # 拇指
     (0, 5), (5, 6), (6, 7), (7, 8),           # 食指
     (0, 9), (9, 10), (10, 11), (11, 12),      # 中指
     (0, 13), (13, 14), (14, 15), (15, 16),    # 无名指
     (0, 17), (17, 18), (18, 19), (19, 20)],   # 小指
    np.int32)


def build_human(clips_path=f"{DS}/clips_human_L24.npz",
                out_path=f"{DS}/chain_sidecar_human.npz"):
    """human 21 关节 -> 两视角 crop-norm 2D。"""
    z = np.load(clips_path, mmap_mode="r")
    VID, FIDX = np.asarray(z["vid"]), np.asarray(z["fidx"])
    N, L = FIDX.shape
    kp3d = np.full((N, L, 21, 3), np.nan)
    for vid in np.unique(VID):
        d = S.HUMAN_DIRS[int(vid)]
        df = pd.read_parquet(f"{d}/data/chunk-000/episode_000000.parquet",
                             columns=["observation.eef.kpts21_3d_world"])
        kp = np.stack(df["observation.eef.kpts21_3d_world"].values).reshape(-1, 21, 3)
        rows = np.where(VID == vid)[0]
        kp3d[rows] = S.gather(kp, FIDX[rows])
        print(f"  human vid {vid}: {len(rows)} clips", flush=True)

    flat = kp3d.reshape(-1, 3)
    high = S.to_norm_batch(S.project_batch(flat, S.T_HIGH, S.K_HIGH), S.CROP_HIGH).reshape(N, L, 21, 2)
    low = S.to_norm_batch(S.project_batch(flat, S.T_LOW, S.K_LOW), S.CROP_LOW).reshape(N, L, 21, 2)
    np.savez_compressed(out_path, chain2d_high=high.astype(np.float32),
                        chain2d_low=low.astype(np.float32), segments=HAND_SEGMENTS,
                        joint_names=np.array([f"hand_kpt_{i}" for i in range(21)]))
    fin = np.isfinite(high).all((2, 3)).mean()
    print(f"human: {high.shape} 有效帧 {fin:.1%} -> {out_path}")


def build_robot(src=f"{DS}/skel_sidecar_robot_v2.npz",
                out_path=f"{DS}/chain_sidecar_robot.npz"):
    """robot 直接复用 v2 的 8 个 link + 7 条边(它本来就是完整运动链,只是被下游削成 4 点用)。"""
    z = np.load(src)
    np.savez_compressed(out_path, chain2d_high=z["skel2d_high"].astype(np.float32),
                        chain2d_low=z["skel2d_low"].astype(np.float32),
                        segments=z["segments"], joint_names=z["joint_names"], grip=z["grip"])
    print(f"robot: {z['skel2d_high'].shape} -> {out_path}")


# --- agent-centric 局部光栅化(2026-07-19 眼检后改的设计)---
# 直接全图光栅化会翻车, 已量化: robot 链包围盒 1.12x0.46 幅(基座 link_1/2/3 画内率 0%),
# human 手只有 0.23x0.13 幅 —— 差 5 倍, 两域光栅图统计完全不同, 和 skel-v2 是同一类失败。
# 根因: OSCAR 的相机看得到整个机器人, 我们的 crop 只框工作区, robot 大半条臂在框外。
# 改为以末端为中心、固定物理半径的窗口, 只画窗口内的链:
#   半径 0.30 时 robot 保留 link_4/5/6+两指尖(5 个, 基座自动排除), human 保留全部 21 个。
#   两域点数仍不同(5 vs 21) —— 这正是光栅化允许的, 不需要配对。
# 绝对定位由单独的向量通道提供(见 ② 的 action 组装), 不靠这张图。
CHAIN_CENTER = {"r": 5, "h": 0}          # robot=link_6, human=wrist
LOCAL_R = 0.30                            # 窗口半径, crop-norm


def rasterize_local(pts, segments, dom, size=32, r=LOCAL_R, lw=1, dot=1):
    """(...,J,2) crop-norm 关节 -> (...,size,size) 以末端为中心的局部线画图。
    窗口 = 末端 ± r, 映射到 [0,size)。窗口外的关节被裁掉(robot 基座即由此排除)。"""
    import cv2
    ctr = CHAIN_CENTER[dom]
    flat = pts.reshape(-1, pts.shape[-2], 2)
    out = np.zeros((len(flat), size, size), np.float32)
    for i, P in enumerate(flat):
        c = P[ctr]
        if not np.isfinite(c).all():
            continue
        uv = (P - c + r) / (2 * r) * size                      # 局部归一化 -> 像素
        keep = np.isfinite(uv).all(-1) & (uv > -size).all(-1) & (uv < 2 * size).all(-1)
        img = out[i]
        for a, b in segments:
            if keep[a] and keep[b]:
                cv2.line(img, tuple(np.int32(uv[a])), tuple(np.int32(uv[b])), 1.0, lw)
        for j, p in enumerate(uv):
            if keep[j]:
                cv2.circle(img, tuple(np.int32(p)), dot, 1.0, -1)
    return out.reshape(*pts.shape[:-2], size, size)


def rasterize_chain(pts, segments, size=32, lw=1, dot=1):
    """(...,J,2) crop-norm 关节 + (E,2) 边 -> (...,size,size) float32 线画图, 值域 [0,1].

    这是两域共享的唯一格式:关节数 J 可以不同(human 21 / robot 8),
    光栅化后都是同样大小的一张图。纯 numpy(小图, 无需 GPU)。
    """
    import cv2
    flat = pts.reshape(-1, pts.shape[-2], 2)
    out = np.zeros((len(flat), size, size), np.float32)
    for i, P in enumerate(flat):
        uv = np.clip(P * size, -size, 2 * size)
        img = out[i]
        for a, b in segments:
            if np.isfinite(uv[a]).all() and np.isfinite(uv[b]).all():
                cv2.line(img, tuple(np.int32(uv[a])), tuple(np.int32(uv[b])), 1.0, lw)
        for p in uv:
            if np.isfinite(p).all():
                cv2.circle(img, tuple(np.int32(p)), dot, 1.0, -1)
    return out.reshape(*pts.shape[:-2], size, size)


if __name__ == "__main__":
    build_robot()
    build_human()
