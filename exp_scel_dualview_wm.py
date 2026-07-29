"""② DUAL-VIEW joint flow prediction (the missing middle of the interface axis).

2026-07-05 verdict: per-view 2D beats 3D (whose weakness = depth-lift input
noise). Dual-view 2D predicts BOTH views' 48-pt flow jointly: two projections
determine 3D implicitly, no depth consumed. Eval adds TRIANGULATED 3D error
(predicted view pair -> ray intersection -> mm vs depth-lifted GT): if it
approaches the 6.9mm depth-3D quality at 2D-native px accuracy, dual-view is
the interface sweet spot.

Model: FlowWM_LWC trunk, 2P point tokens (learned view embedding), per-token
2D velocity classes (unchanged head); action = per-view dummy5 constellation
(the adopted ② action rep, project_detmem_dit) concatenated.
Data: aligned dual-view clips (gen_dualview_aligned.py).
Env: DS(npz), SMOKE, OUT_DIR, SPLIT(legacy|okfirst; okfirst=③-style filter-then-permute,
audit item5 leakage fix). iws env, GPU.
-> outputs/cross_embodiment_wm/dualview_wm/ (wm_dual.pt + summary.txt)

--- SKELETON ACTION REPRESENTATION experiment (2026-07-15) ---
Attacks the diagnosed human-helps bottleneck (project_action_dflow_separability): dummy5's action
INPUT is the absolute eef constellation, which is domain-disjoint between human/robot (probe 1.0);
an ISOMORPHIC skeleton action rep (same 4-joint geometric chain, isomorphic v2 design from
FLOW_WARP_REPR_LOG.md §1 "同构骨架"/exp_scel_dualview_dit_formal.py's flowskelv2 render-condition
precedent -- SAME sidecar files, but consumed here as the WM's action INPUT, not a render condition)
should let human data help the robot more, especially in the robot-scarce regime.
Env additions (all default to byte-identical old behavior):
  ACTION(dummy5|skel): skel = 4 tokens/view, robot [link_6,fingertipL',fingertipR',link_5] from
    skel_sidecar_robot_v2.npz (idx [5,6,7,4]; link_5->link_6 = approach dir, mirrors human
    forearm_stub->wrist), human [wrist,fin1,fin2,forearm_stub] from skel_sidecar_human.npz verbatim.
    Fed the same way dummy5 feeds its 5 points (object-centroid-relative, concat both views into
    s.act); no-leakage: both sidecars are built from joints/eef/hand-kpts only, never future frames.
  MIX(r|rh): rh = co-train on clips_human_L24.npz (L=24 vs robot's L=48) alongside the robot pool,
    same vis-weighted CE loss, SS rollout steps capped to the human clip length (R_SS_h<=L_h-K).
  NROB(""|int): robot-scarce protocol -- subsample the (okfirst) robot TRAINING pool to NROB clips
    (rng(0), deterministic/shared across arms); held-out eval is always the full robot okfirst
    heldout set (never subsampled).
  OUT_DIR default routes to outputs/cross_embodiment_wm/dualview_wm_skelact/{ACTION}_{MIX}_n{NROB}/
    whenever any of ACTION/MIX/NROB is non-default; the all-default run keeps the original path.
"""
import json
import os
import copy

import numpy as np
import torch
import torch.nn as nn

os.chdir("/scr2/yusenluo/interactive_world_sim")
import eval_scheduled_sampling as SSm
from amplify_wm import FlowWM_LWC, K, F, vel_to_class
from scipy.spatial.transform import Rotation

SMOKE = os.environ.get("SMOKE", "0") == "1"
DS = os.environ.get("DS", "outputs/flow_render_dataset_can_dual/clips_robot.npz")
SPLIT = os.environ.get("SPLIT", "legacy")                     # legacy | okfirst (audit item5 leakage fix)
ACTION = os.environ.get("ACTION", "dummy5")                   # dummy5 (default, unchanged) | skel (isomorphic v2)
MIX = os.environ.get("MIX", "r")                               # r (default, robot-only) | rh (human co-train)
HEAD_MODE = os.environ.get("HEAD_MODE", "single")             # single(默认,逐字节不变) | two(域头) | film(deferred)
NROB = os.environ.get("NROB", "")                               # "" = full robot pool | int = robot-scarce N
_NROB_TAG = NROB if NROB else "all"
SEED = int(os.environ.get("SEED", "0"))                         # 训练种子; seed=0 保持原路径(向后兼容)
_SEED_TAG = "" if SEED == 0 else f"_s{SEED}"
_HEAD_TAG = "" if HEAD_MODE == "single" else f"_{HEAD_MODE}"
_DEFAULT_OUT = ("outputs/cross_embodiment_wm/dualview_wm"
                if (ACTION == "dummy5" and MIX == "r" and not NROB and SEED == 0 and HEAD_MODE == "single")
                else f"outputs/cross_embodiment_wm/dualview_wm_skelact/{ACTION}_{MIX}_n{_NROB_TAG}{_SEED_TAG}{_HEAD_TAG}")
OUT = os.environ.get("OUT_DIR", _DEFAULT_OUT)
os.makedirs(OUT, exist_ok=True)
H = int(os.environ.get("H", "40")); HELDOUT = int(os.environ.get("HELDOUT_N", "150")); IMG = 128; VEL_HALF = 0.06
R_SS_ENV = int(os.environ.get("R_SS", "32"))          # human-only(L=24)须 R_SS<=L-K=20
GRIP_MAX = 0.04          # obs_right_gripper 物理开口上限, 用于归一化 grip -> [0,1] (dummy5g 显式 grip 通道)
device = "cuda" if torch.cuda.is_available() else "cpu"

CROPS = {0: (60, 60, 390, 390), 1: (0, 0, 640, 480)}          # view0=cam_high, view1=cam_low
_CAL = {0: json.load(open("calib/rgb_cam_calib_can_REAL.json")),
        1: json.load(open("calib/rgb_cam_calib_can_low_REAL.json"))}
_RT = {v: (Rotation.from_quat(c["quat_xyzw"]).as_matrix(), np.asarray(c["t_world"]))
       for v, c in _CAL.items()}


def dummy5(eef3):
    """(B,Lw,3,2) -> (B,Lw,5,2) DexWM virtual constellation (per view)."""
    wrist, t1, t2 = eef3[:, :, 0], eef3[:, :, 1], eef3[:, :, 2]
    c = (t1 + t2) / 2; ax = (t1 - t2) / 2
    perp = torch.stack([-ax[..., 1], ax[..., 0]], -1)
    return torch.stack([wrist, c, c + ax, c - ax, c + perp], 2)


# --- ACTION=dummy5rt: DexWM 式 retarget (2026-07-28) ---
# 动机: dummy5 星座形状/尺度域特有(眼检: robot 两指并排腕左侧 / human 两指从腕下张开,
#   半开口 1.96x; 腕-接触点距离 robot 0.19 vs human 0.06 = 6x 拓扑不等价) -> 模型看星座形状就能分域.
# 朴素相似变换桥不过(腕距/开口比值差 6x). DexWM 做法: 不 retarget 原始点, 而是两域用【同一构造函数】
#   从功能参数(接触点 c + 指轴方向 + 腕方向)重建 canonical 星座, 腕距/半开口用固定常数归一化.
#   -> 星座形状域一致(构造相同); 但 c(绝对接触位置)+朝向真实 -> 位置不丢(避开 velcontact/skelv3 陷阱).
# 常数从 robot 域拟合(scratchpad/fit_retarget.py): (腕-c 距离 W0, 半开口 R0) 每视角.
DUMMY5RT_CONST = {0: (0.194, 0.029), 1: (0.190, 0.024)}       # {view: (W0, R0)}


def dummy5rt(eef3, view):
    """(B,Lw,3,2) [wrist,t1,t2] -> (B,Lw,5,2) canonical-shape 星座 (retarget human->robot 几何).
    两域同构造: c 绝对保留; 腕方向真实但距离归 W0; 指轴方向真实但半开口归 R0. 去形状/尺度域泄漏.
    ★实证判死(diag_dummy5rt_separability.py probe 0.9994): 几何retarget没用, 保留当反例."""
    W0, R0 = DUMMY5RT_CONST[view]
    wrist, t1, t2 = eef3[:, :, 0], eef3[:, :, 1], eef3[:, :, 2]
    c = (t1 + t2) / 2
    d = t1 - t2; u = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-4)          # 指轴单位向量
    perp = torch.stack([-u[..., 1], u[..., 0]], -1)
    wd = wrist - c; wu = wd / wd.norm(dim=-1, keepdim=True).clamp_min(1e-4)    # 腕方向单位向量
    wrist_c = c + W0 * wu                                                      # 腕: 方向真实, 距离归一
    return torch.stack([wrist_c, c, c + R0 * u, c - R0 * u, c + R0 * perp], 2)


# --- ACTION=mp: Point-Policy 换锚 (2026-07-28, related-work grounded) ---
# 3-agent 深读结论(AGENT_RETARGET_RELWORK): wrist 难题正解 = 丢解剖腕/link_6, 用【两指中点=接触功能点】
#   当参考(Point Policy 2502.20391): 两域参考点都落接触区, 语义构造上一致 -> 消 6× 腕-指比值失配。
#   + canonical 半尺度 R0 去 1.96× 开口泄漏; 指轴方向真实(carry 朝向); grip 走显式标量(dummy5g 管线)。
# 与 dummy5rt 差别: 完全丢 wrist(dummy5rt 还保留一个 canonical 腕点仍泄漏方向)。判据=下游human-helps(非probe)。
MP_R0 = {0: 0.029, 1: 0.024}       # canonical 半开口 per view (从 robot 拟合, 同 dummy5rt R0)


def mp_constellation(eef3, view):
    """(B,Lw,3,2) [wrist,t1,t2] -> (B,Lw,5,2) 中点参考星座(丢腕). c=两指中点(对象相对carry接触位置),
    指轴方向真实, 半尺度归 canonical R0(去开口域泄漏). grip 另走 act_grip 标量, 不入星座尺度."""
    R0 = MP_R0[view]
    t1, t2 = eef3[:, :, 1], eef3[:, :, 2]                                      # 只用两指, 丢 wrist(eef3[:,:,0])
    c = (t1 + t2) / 2
    d = t1 - t2; u = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-4)
    perp = torch.stack([-u[..., 1], u[..., 0]], -1)
    return torch.stack([c, c + R0 * u, c - R0 * u, c + R0 * perp, c - R0 * perp], 2)


# --- ACTION=dummy5rs: 几何 retarget "别冻死"修正 (2026-07-29) ---
# dummy5rt 把开口冻成常数 R0 -> 抹掉抓取信号(可视化坐实, 下游最差)。mp 把开口塞进 grip 标量。
# dummy5rs = 让星座跟真实开口张合(R=R0·ap/域中位数): 按域中位数归一 -> 域尺度对齐(去 2× 开口泄漏)
#   + 开合变化保留在【几何】里。测"开口留几何(breathing)"是否 ≥ mp"开口留标量"。grip 仍走标量保 keyboard。
DUMMY5RS_MED = {0: {"r": 0.0637, "h": 0.1229}, 1: {"r": 0.0498, "h": 0.0993}}   # per-view 每域开口中位数(fit)


def dummy5rs(eef3, view, dom):
    """(B,Lw,3,2)[wrist,t1,t2] -> (B,Lw,5,2) breathing 中点星座(丢腕)。R=R0·(真实开口/域中位数):
    域尺度对齐 + 开合变化保留在几何。dom∈{r,h} 选域中位数。"""
    R0 = MP_R0[view]; med = DUMMY5RS_MED[view].get(dom, DUMMY5RS_MED[view]["r"])
    t1, t2 = eef3[:, :, 1], eef3[:, :, 2]
    c = (t1 + t2) / 2
    d = t1 - t2; ap = d.norm(dim=-1, keepdim=True).clamp_min(1e-4); u = d / ap
    R = R0 * (ap / med)                                                        # breathing 半尺度(域归一)
    perp = torch.stack([-u[..., 1], u[..., 0]], -1)
    return torch.stack([c, c + R * u, c - R * u, c + R * perp, c - R * perp], 2)


# --- ACTION=skel: isomorphic v2 skeleton action rep (module docstring §skeleton experiment) ---
# skel_sidecar_robot_v2.npz joint order: [link_1,link_2,link_3,link_4,link_5,link_6,fingertipL',fingertipR']
# (idx 0..7); we take [link_6, fingertipL', fingertipR', link_5] = idx [5,6,7,4], isomorphic to
# skel_sidecar_human.npz's verbatim [wrist, fingertip1, fingertip2, forearm_stub] (idx 0..3, used as-is).
SKEL_ROBOT_IDX = [5, 6, 7, 4]

# --- skel-v3: 尺度不变的结构表示 (2026-07-19) ---
# 动机: v2 的"同构"只对齐了拓扑(槽位配对), 几何上三项失配, 已量化确证:
#   (a) 两指对称比  robot 1.04 [1.00,1.11] vs human 1.27 [1.18,1.40]  分布几乎不重叠
#       -> 模型光靠"两指是否等长"就能区分域, 与"两域落同一空间"的目标相反
#   (b) 槽位3        human 腕->前臂根长度恒为 19.2px (stub_2d 写死 FOREARM_STUB_D),
#       robot link_5 是真实连杆 8px 且随姿态 6-10.6px 变化 -> 长度维度无跨域信息
#   (c) 整体量纲     指尖间距中位 robot 9.3px vs human 15.8px (~1.7x)
# v3 的做法: 只保留跨域可比的量, 丢掉本来就不可比的量. token 数与维度不变 (4x2),
# 模型结构无需改动.
#   t0 = 腕绝对位置        (保留! 定位精度必需 —— 丢绝对定位会输, 见 velocity_action 教训)
#   t1 = 手轴单位向量      腕->指尖中点, 归一化 -> 消 (c)
#   t2 = 前臂单位向量      只留方向丢长度       -> 消 (b)
#   t3 = (归一化开合度, 归一化手尺度)          -> 消 (a), 不再暴露两指是否等长
# 无泄漏: 全部由当前帧的 joints/eef/手部关键点导出, 不碰未来帧.
# 常数 = 各域各视角的训练数据分位数(scratchpad/calc_const.py 算出, 域内动作统计, 不涉未来帧).
SKELV3_GAP = {("r", 0): (0.00006, 0.10143), ("r", 1): (0.00005, 0.09592),      # 指尖间距 (p05,p95)
              ("h", 0): (0.06055, 0.19641), ("h", 1): (0.03752, 0.21329)}
SKELV3_SCALE = {("r", 0): 0.22421, ("r", 1): 0.23576,                          # 手尺度 p95
                ("h", 0): 0.28508, ("h", 1): 0.23231}


def skelv3_tokens(P, dom, view):
    """(N,L,4,2) 原始骨架点 [腕,指尖A,指尖B,前臂根] -> (N,L,4,2) 尺度不变结构 token."""
    w, f1, f2, fa = P[..., 0, :], P[..., 1, :], P[..., 2, :], P[..., 3, :]
    axis = (f1 + f2) / 2.0 - w                                        # 腕 -> 指尖中点
    an = np.linalg.norm(axis, axis=-1, keepdims=True)
    axis_u = axis / np.maximum(an, 1e-6)
    fv = fa - w
    fore_u = fv / np.maximum(np.linalg.norm(fv, axis=-1, keepdims=True), 1e-6)
    gap = np.linalg.norm(f1 - f2, axis=-1)
    g0, g1 = SKELV3_GAP[(dom, view)]
    grip = np.clip((gap - g0) / max(g1 - g0, 1e-6), 0.0, 1.0)          # 开合度 [0,1], 两域同尺度
    scale = np.clip(an[..., 0] / SKELV3_SCALE[(dom, view)], 0.0, 2.0)  # 手尺度, 各域按自身 p95 归一
    t3 = np.stack([grip, scale], -1)
    return np.stack([w, axis_u, fore_u, t3], 2).astype(np.float32)


# --- ACTION=raster: OSCAR 式局部光栅图 + 绝对位置向量 (2026-07-19) ---
# 设计依据: OSCAR(2606.04463v2 §3.2) 把两域各自的完整运动链光栅化成同格式线画图,
# 格式统一在**像素空间**而非向量槽位, 因此**点数不必相同** —— 解除了 skel-v2 被迫
# "robot 9 关节 / human 21 关节削成 4 点强行配对"的约束(该配对已量化证明失败)。
# 我们与 OSCAR 的关键差异: 他们的相机看得到整个机器人, 我们的 crop 只框工作区,
# robot 的 link_1/2/3 画内率 0%、链包围盒 1.12 幅 vs human 手 0.23 幅(差 5x)。
# 故改为 **agent-centric 局部光栅化**(末端为中心, 半径 0.30): robot 基座自动出窗被排除,
# 窗口内 robot 5 关节 / human 21 关节, 尺度可比(白像素占比 0.048 vs 0.068, 原为 5x 差)。
# 绝对定位不靠图, 由单独的末端位置向量提供(OSCAR 证明绝对坐标不能丢: 他们的
# frame-to-frame delta baseline 大幅落后 PSNR 19.22 vs 23.48)。
# 打包: (N,L,513,2) = [末端位置(1,2)] + [32x32 光栅图 reshape(512,2)],
# 这样完全复用现有的 [:, h:h+K+F] 窗口索引与末帧填充逻辑, 主训练循环零改动。
RASTER_SIZE = 64


def load_raster_action(dom, z, ras, with_grip=False):
    """-> (N,L, 1[+1]+S*S/2, 2) per view.
    [:,:,0]=末端绝对位置; with_grip 时 [:,:,1]=(归一化抓握程度,0); 其余=局部光栅图。

    with_grip(ACTION=rasterg, 2026-07-20): 诊断证明 grip 在光栅图里弱不是 2D 投影的锅
    (两域到相机深度 0.68 vs 0.66m 几乎相同, 投影比 0.27px/mm 相同), 而是 robot 夹爪
    3D 开合(28mm)本就只有 human(60mm≈罐径)的一半 —— 取点几何(URDF fingertip link vs
    MANO 指尖)+ 机械尺度决定。两域"抓握程度"(各自从张到合)是相近动作, 差的是绝对尺度。
    故显式补一个**归一化抓握程度**标量: 2D 指尖间距按各域各视角的 p05-p95 映射到 [0,1]
    (复用 SKELV3_GAP 常量), 抓住罐子时两域 ≈ 同一值。不动位置(skelv3 归一化位置丢精度已判死)。
    """
    out = []
    for view, key in ((0, "raster_v0"), (1, "raster_v1")):
        R = ras[key].astype(np.float32) / 255.0                    # (N,L,S,S)
        N, L = R.shape[:2]
        e = z["eef"] if view == 0 else z["eef_low"]
        e = np.nan_to_num(e.astype(np.float32), nan=0.5)
        pos = e[:, :, 0:1]                                          # (N,L,1,2) 末端绝对位置
        parts = [pos]
        if with_grip:
            gap = np.linalg.norm(e[:, :, 1] - e[:, :, 2], axis=-1)  # (N,L) 2D 指尖间距
            g0, g1 = SKELV3_GAP[(dom, view)]
            grip = np.clip((gap - g0) / max(g1 - g0, 1e-6), 0.0, 1.0)
            gtok = np.stack([grip, np.zeros_like(grip)], -1)[:, :, None]   # (N,L,1,2)
            parts.append(gtok)
        parts.append(R.reshape(N, L, -1, 2))
        out.append(np.concatenate(parts, 2))
    return out[0], out[1]


class RasterAct(nn.Module):
    """(B, 2*Lw, S, S) 双视角光栅图栈 -> (B, Dm). 时间维当通道, 与原 Linear 把 Lw 展平同构."""

    def __init__(s, in_ch, Dm):
        super().__init__()
        s.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),      # 32->16
            nn.Conv2d(64, 128, 3, 2, 1), nn.GroupNorm(8, 128), nn.SiLU(),       # 16->8
            nn.Conv2d(128, 128, 3, 2, 1), nn.GroupNorm(8, 128), nn.SiLU(),      # 8->4
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, Dm))

    def forward(s, x):
        return s.net(x)


def load_action_tokens(mode, dom, z, sk=None):
    """Per-view raw action-point arrays (N,L,n_raw,2), windowed downstream exactly like the original
    dummy5 eef arrays (train_dual/rollout_dual index [:, h:h+K+F] and pad).
    mode='dummy5' (default): n_raw=3 eef triplet (wrist,fin1,fin2) -- IDENTICAL to the pre-existing
      main() computation (efA has no nan_to_num, matching the original: z['eef'] is 0%-NaN in this
      dataset for both domains, verified 2026-07-15).
    mode='skel': n_raw=4, the isomorphic v2 skeleton tokens (dom='r' selects SKEL_ROBOT_IDX from the
      8-pt robot_v2 sidecar; dom='h' uses the human sidecar's 4 points verbatim); nan_to_num(0.5) on
      both views (matches wm convention for track/action arrays with occasional NaN, e.g. undetected
      human hand frames or a view-1 fall-off)."""
    if mode in ("dummy5", "dummy5rt", "cpt"):                   # cpt: 纯接触点c(单点), 构造在 _act_pts
        efA = z["eef"].astype(np.float32) if mode == "dummy5" else np.nan_to_num(z["eef"].astype(np.float32), nan=0.5)
        efB = np.nan_to_num(z["eef_low"].astype(np.float32), nan=0.5)
        return efA, efB
    if mode in ("mp", "dhc", "dummy5rs"):
        # mp/dhc/dummy5rs = eef 原始三点(_act_pts 内构造星座) + grip 第4槽(归一化指距, 两域[0,1]口径一致).
        # grip = 归一化指距(两域各自[0,1], 口径一致, 非 robot physical grip 避免域差)。
        efA = np.nan_to_num(z["eef"].astype(np.float32), nan=0.5)
        efB = np.nan_to_num(z["eef_low"].astype(np.float32), nan=0.5)

        def gripslot(ef):
            g = np.linalg.norm(ef[:, :, 1] - ef[:, :, 2], axis=-1)
            g = g / (float(np.nanmax(g)) + 1e-6)
            return np.stack([g, np.zeros_like(g)], -1)[:, :, None, :]        # (N,L,1,2)
        return np.concatenate([efA, gripslot(efA)], 2), np.concatenate([efB, gripslot(efB)], 2)
    if mode == "dummy5g":
        # dummy5g = dummy5 三点 + 显式归一化物理 grip 作第4槽 (grip_norm, 0).
        # grip 顺 eef 数组的加窗/pad 管线走 -> train_dual/rollout_dual/fwd_dual 签名零改动.
        # fwd_dual 只读 [...,3,0] 当 grip, y 槽恒 0 (占位, 不被读). 星座只用前3点 (见 _act_pts).
        efA = z["eef"].astype(np.float32)                              # (N,L,3,2)
        efB = np.nan_to_num(z["eef_low"].astype(np.float32), nan=0.5)
        if "grip" in z.files:                                          # robot: obs_right_gripper 物理开口
            g = z["grip"].astype(np.float32) / GRIP_MAX                # (N,L) -> [0,1]
        else:                                                         # human: 无夹爪, 用2D指尖开口当 grip 代理
            ef = z["eef"].astype(np.float32)
            g = np.linalg.norm(ef[:, :, 1] - ef[:, :, 2], axis=-1)     # (N,L) 指尖间距
            g = g / (float(np.nanmax(g)) + 1e-6)
        gpt = np.stack([g, np.zeros_like(g)], -1)[:, :, None, :]       # (N,L,1,2) grip 占位点
        return np.concatenate([efA, gpt], 2), np.concatenate([efB, gpt], 2)   # (N,L,4,2) 两视角同 grip
    if mode in ("raster", "rasterg"):                       # OSCAR 式局部光栅图 (rasterg 加归一化 grip 标量)
        return load_raster_action(dom, z, sk, with_grip=(mode == "rasterg"))
    assert sk is not None, "ACTION=skel/skelv3 requires the skeleton sidecar npz"
    a, b = sk["skel2d_high"], sk["skel2d_low"]
    if dom == "r":
        a, b = a[:, :, SKEL_ROBOT_IDX], b[:, :, SKEL_ROBOT_IDX]
    if mode == "skelv3":                                    # 尺度不变结构表示, 见上方注释
        a = skelv3_tokens(np.nan_to_num(a.astype(np.float32), nan=0.5), dom, 0)
        b = skelv3_tokens(np.nan_to_num(b.astype(np.float32), nan=0.5), dom, 1)
        return a, b
    efA = np.nan_to_num(a.astype(np.float32), nan=0.5)
    efB = np.nan_to_num(b.astype(np.float32), nan=0.5)
    return efA, efB


class DualLWC(FlowWM_LWC):
    def __init__(s, P, action="dummy5", head_mode="single", **kw):
        super().__init__(P, **kw)
        s.P1 = P
        s.action = action
        s.head_mode = head_mode
        s.n_tok = 1 if action == "cpt" else (5 if action in ("dummy5", "dummy5g", "dummy5rt", "mp", "dhc", "dummy5rs") else 4)   # cpt:1 | dummy5(g/rt/rs)/mp/dhc:5 | skel:4
        s.view_emb = nn.Parameter(torch.zeros(2, s.Dm))
        s.act = nn.Linear((K + F) * s.n_tok * 2 * 2, s.Dm)      # action tokens x 2 views
        if action in ("dummy5g", "mp", "dhc", "dummy5rs"):      # 显式 grip 专属嵌入通道 (加到动作 embedding)
            s.act_grip = nn.Linear(K + F, s.Dm)                 # (B,Lw) 归一化 grip 序列 -> Dm
        if action == "dhc":                                     # ★DexWM Δ+HC: 输入喂Δ(同域), HC头从trunk预测绝对c(逼积分Δ→定位)
            s.hc = nn.Sequential(nn.Linear(s.Dm, s.Dm // 2), nn.ReLU(), nn.Linear(s.Dm // 2, 4))  # ->c(2视角×2)
            s._hc_aux = None
        if action in ("raster", "rasterg"):                      # OSCAR 式: 光栅图(CNN) + 末端位置(+grip)(Linear)
            s.act_ras = RasterAct(2 * (K + F), s.Dm)             # 双视角 x Lw 帧当通道
            _posdim = (K + F) * 2 * (3 if action == "rasterg" else 2)   # rasterg: 每视角每帧 (x,y,grip)
            s.act_pos = nn.Linear(_posdim, s.Dm)                 # 双视角末端位置(+归一化 grip)
        if head_mode == "two":
            s.head_h = copy.deepcopy(s.head)          # human 头, init = robot 头(暖启)

    def _readout(s, x, dom):
        """trunk 特征 x (B,P,Dm) -> logits (B,P,F*W*W), 按域路由读出头。
        getattr 默认 'single' 兼容改动前 pickle 的老 ckpt。"""
        hm = getattr(s, "head_mode", "single")
        if hm == "two":
            return (s.head_h if dom == "h" else s.head)(x)
        if hm == "film":
            raise NotImplementedError("HEAD_MODE=film is Phase-2 (deferred)")
        return s.head(x)                              # single

    def _act_pts(s, eef, view=0, dom="r"):
        """eef (B,Lw,n_raw,2) -> (B,Lw,n_tok,2) action tokens. dummy5: DexWM virtual constellation.
        dummy5rt: retarget human->robot canonical 星座 (per-view 常数, 需 view). skel: identity.
        dummy5rs: breathing 星座(R=R0·ap/域中位数, 需 view+dom)。
        getattr default covers unpickling checkpoints saved before this attribute existed."""
        a = getattr(s, "action", "dummy5")
        if a == "dummy5rt":
            return dummy5rt(eef[:, :, :3], view)
        if a == "dummy5rs":
            return dummy5rs(eef[:, :, :3], view, dom)                         # breathing 星座(域尺度对齐)
        if a == "mp":
            return mp_constellation(eef[:, :, :3], view)                      # 中点星座丢腕, 第4槽是grip(不入星座)
        if a == "cpt":                                                        # 纯接触点c(单点), 测"mp是否只是单点"
            return ((eef[:, :, 1] + eef[:, :, 2]) / 2)[:, :, None]            # (B,Lw,1,2)
        return dummy5(eef[:, :, :3]) if a in ("dummy5", "dummy5g", "dhc") else eef   # dummy5g/dhc: 前3点建星座, 第4槽grip

    def fwd_dual(s, hist, eef_a, eef_b, dom="r"):
        """hist (B,2P,K,2) [view0 pts | view1 pts]; eef_a/b (B,K+F,n_raw,2) per view."""
        B, P2 = hist.shape[:2]
        anchor = hist[:, :, -1, :]
        obj = s.inp(torch.cat([(hist - anchor[:, :, None]).reshape(B, P2, 2 * K), anchor], -1))
        obj = obj + torch.cat([s.view_emb[0].expand(B, s.P1, s.Dm),
                               s.view_emb[1].expand(B, s.P1, s.Dm)], 1)
        oc_a = anchor[:, :s.P1].mean(1, keepdim=True)          # per-view object centroid
        oc_b = anchor[:, s.P1:].mean(1, keepdim=True)
        # skelv3: 只有 t0(腕绝对位置) 是位置量, 该减物体质心变成相对坐标;
        # t1/t2 是单位向量、t3 是标量对(开合度,尺度) —— 减质心会破坏它们的语义, 保持原样.
        if getattr(s, "action", "dummy5") in ("raster", "rasterg"):
            S_ = RASTER_SIZE
            g = (s.action == "rasterg"); off = 2 if g else 1               # rasterg 多一个 grip token
            pa = eef_a[:, :, 0] - oc_a; pb = eef_b[:, :, 0] - oc_b          # (B,Lw,2) 末端相对物体
            ra = eef_a[:, :, off:].reshape(B, -1, S_, S_)                   # (B,Lw,S,S) 光栅图
            rb = eef_b[:, :, off:].reshape(B, -1, S_, S_)
            if g:                                                          # 末端位置 + 归一化 grip 标量
                ga = eef_a[:, :, 1, 0:1]; gb = eef_b[:, :, 1, 0:1]          # (B,Lw,1)
                pos_in = torch.cat([pa.reshape(B, -1), ga.reshape(B, -1),
                                    pb.reshape(B, -1), gb.reshape(B, -1)], -1)
            else:
                pos_in = torch.cat([pa.reshape(B, -1), pb.reshape(B, -1)], -1)
            act = (s.act_pos(pos_in) + s.act_ras(torch.cat([ra, rb], 1)))[:, None]
            x = s.tf(torch.cat([obj, act], 1))[:, :P2]
            logits = s._readout(x, dom).reshape(B, P2, F, s.W * s.W)
            return logits, anchor
        if getattr(s, "action", "dummy5") == "dhc":            # ★DexWM Δ+HC: 喂Δ星座(同域)+grip, HC头预测绝对c
            a5 = s._act_pts(eef_a, 0) - oc_a[:, :, None]        # (B,Lw,5,2) 对象相对绝对星座
            b5 = s._act_pts(eef_b, 1) - oc_b[:, :, None]
            da = torch.cat([torch.zeros_like(a5[:, :1]), a5[:, 1:] - a5[:, :-1]], 1)   # Δ (velocity, 同域)
            db = torch.cat([torch.zeros_like(b5[:, :1]), b5[:, 1:] - b5[:, :-1]], 1)
            act = s.act(torch.cat([da.reshape(B, -1), db.reshape(B, -1)], -1))[:, None]
            act = act + s.act_grip(eef_a[:, :, 3, 0])[:, None]                          # grip 标量(保keyboard可控)
            x = s.tf(torch.cat([obj, act], 1))[:, :P2]
            hc_pred = s.hc(x.mean(1))                           # (B,4) 从trunk预测绝对c
            gt_c = torch.cat([a5[:, K - 1, 1], b5[:, K - 1, 1]], -1)   # 对象相对c(dummy5 idx1=中点), anchor帧
            s._hc_aux = ((hc_pred - gt_c) ** 2).mean()          # HC辅助loss(_ss_loss读取)
            logits = s._readout(x, dom).reshape(B, P2, F, s.W * s.W)
            return logits, anchor
        if getattr(s, "action", "dummy5") == "skelv3":
            d5a = s._act_pts(eef_a, 0).clone(); d5b = s._act_pts(eef_b, 1).clone()
            d5a[:, :, 0] = d5a[:, :, 0] - oc_a
            d5b[:, :, 0] = d5b[:, :, 0] - oc_b
        else:
            d5a = s._act_pts(eef_a, 0, dom) - oc_a[:, :, None]   # dom -> dummy5rs 选域中位数(其余 action 忽略)
            d5b = s._act_pts(eef_b, 1, dom) - oc_b[:, :, None]
        act = s.act(torch.cat([d5a.reshape(B, -1), d5b.reshape(B, -1)], -1))[:, None]
        if getattr(s, "action", "dummy5") in ("dummy5g", "mp", "dummy5rs"):  # 显式 grip: 读第4槽x, 加专属嵌入
            gr = eef_a[:, :, 3, 0]                              # (B,Lw) 归一化 grip (取 view0)
            act = act + s.act_grip(gr)[:, None]
        x = s.tf(torch.cat([obj, act], 1))[:, :P2]
        logits = s._readout(x, dom).reshape(B, P2, F, s.W * s.W)
        return logits, anchor


def _ss_loss(m, G, Vv, Ea, Eb, R_SS, pteach, dom="r"):
    """Scheduled-sampling rollout CE loss for one batch (NO optimizer step). Domain-agnostic."""
    buf = G[:, :K].clone(); losses = []
    for h in range(R_SS):
        wa = Ea[:, h:h + K + F]; wb = Eb[:, h:h + K + F]
        if wa.shape[1] < K + F:
            pad = K + F - wa.shape[1]
            wa = torch.cat([wa, wa[:, -1:].repeat(1, pad, 1, 1)], 1)
            wb = torch.cat([wb, wb[:, -1:].repeat(1, pad, 1, 1)], 1)
        logits, _ = m.fwd_dual(buf[:, -K:].permute(0, 2, 1, 3), wa, wb, dom=dom)
        lg0 = logits[:, :, 0, :]
        gt_vel = G[:, K + h] - buf[:, -1]
        cls = vel_to_class(gt_vel, m.W, m.vel_half)
        w = (Vv[:, K + h] * Vv[:, K - 1])
        ce = nn.functional.cross_entropy(lg0.reshape(-1, m.W * m.W), cls.reshape(-1),
                                         reduction="none")
        ce = (ce * w.reshape(-1)).sum() / (w.sum() + 1e-6)
        if getattr(m, "action", "dummy5") == "dhc" and getattr(m, "_hc_aux", None) is not None:
            ce = ce + 100.0 * m._hc_aux                        # DexWM HC头 λ=100(逼trunk积分Δ→定位)
        losses.append(ce)
        nxt = buf[:, -1] + m.expected_vel(lg0)
        use_gt = (torch.rand(len(G), 1, 1, device=device) < pteach)
        buf = torch.cat([buf, torch.where(use_gt, G[:, K + h], nxt.detach())[:, None]], 1)
    return torch.stack(losses).mean()


def _ss_batch_step(m, opt, G, Vv, Ea, Eb, R_SS, pteach):
    """One optimizer step for one batch (robot-only path, byte-identical to original)."""
    loss = _ss_loss(m, G, Vv, Ea, Eb, R_SS, pteach)
    opt.zero_grad(); loss.backward(); opt.step()
    return float(loss)


def train_dual(trD, vsD, efA, efB, idx, seed=0, R_SS=32, action="dummy5",
               trDh=None, vsDh=None, efAh=None, efBh=None, idxh=None, head_mode="single"):
    """SS training, robot pool (trD/.../idx). MIX=rh (trDh et al not None): after each robot epoch's
    batches, also SS-train on the human co-train pool with the SAME vis-weighted CE loss (no extra
    reweighting) -- rollout steps capped to the human clip length (R_SS_h = min(R_SS, L_human-K),
    since clips_human_L24.npz is L=24 vs the robot's L=48). Default args (action='dummy5', no *h)
    reproduce the original robot-only loop exactly (same rng call order/shapes -> byte-identical)."""
    torch.manual_seed(seed); P2 = trD.shape[2]
    m = DualLWC(P2 // 2, action=action, head_mode=head_mode, Dm=384, layers=3, W=15, vel_half=VEL_HALF).to(device)
    # base __init__ sized inp for P; token count doesn't affect Linear dims -> fine
    opt = torch.optim.AdamW(m.parameters(), lr=SSm.WM_LR)
    g = torch.Generator().manual_seed(seed)
    trT = torch.from_numpy(trD).float(); vsT = torch.from_numpy(vsD).float()
    eAT = torch.from_numpy(efA).float(); eBT = torch.from_numpy(efB).float()
    mix_human = trDh is not None
    if mix_human:
        trTh = torch.from_numpy(trDh).float(); vsTh = torch.from_numpy(vsDh).float()
        eATh = torch.from_numpy(efAh).float(); eBTh = torch.from_numpy(efBh).float()
        R_SS_h = min(R_SS, trDh.shape[1] - K)
        gh = torch.Generator().manual_seed(seed + 1000)
    epochs = 2 if SMOKE else SSm.WM_EPOCHS
    for ep in range(epochs):
        pteach = 1.0 + (0.3 - 1.0) * ep / max(epochs - 1, 1)
        m.train(); pe = idx[torch.randperm(len(idx), generator=g)]
        loss = 0.0
        if not mix_human:                                    # robot-only: 原路径, byte-identical
            for i in range(0, len(pe), SSm.WM_BS):
                b = pe[i:i + SSm.WM_BS]
                G = trT[b].to(device); Vv = vsT[b].to(device)
                Ea = eAT[b].to(device); Eb = eBT[b].to(device)
                loss = _ss_batch_step(m, opt, G, Vv, Ea, Eb, R_SS, pteach)
        else:
            # 修复(2026-07-20): 原实现先训完所有 robot batch 再单独训所有 human batch,
            # 稀缺时 human batch 数是 robot 的 ~6x, 全堆在 epoch 结尾 -> 梯度被 human 主导
            # -> robot heldout drift 变差 -> 假的"human 有害"(推翻 curves.json 已验证的 human-helps)。
            # 修成: 每个 optimizer step 同时含 1 个 robot batch + 1 个 human batch 的梯度(一次 backward),
            # robot batch 少则 cycle 复用 -> 稀缺 robot 信号不被淹没, 两域梯度每步均衡。
            rb = [pe[i:i + SSm.WM_BS] for i in range(0, len(pe), SSm.WM_BS)]
            peh = idxh[torch.randperm(len(idxh), generator=gh)]
            hb = [peh[i:i + SSm.WM_BS] for i in range(0, len(peh), SSm.WM_BS)]
            for s in range(max(len(rb), len(hb))):
                br = rb[s % len(rb)]; bh = hb[s % len(hb)]
                lr = _ss_loss(m, trT[br].to(device), vsT[br].to(device),
                              eAT[br].to(device), eBT[br].to(device), R_SS, pteach)
                lh = _ss_loss(m, trTh[bh].to(device), vsTh[bh].to(device),
                              eATh[bh].to(device), eBTh[bh].to(device), R_SS_h, pteach, dom="h")
                loss = lr + lh
                opt.zero_grad(); loss.backward(); opt.step()
                loss = float(loss)
        print(f"ep{ep} loss {loss:.4f}", flush=True)
    return m.eval()


@torch.no_grad()
def rollout_dual(m, trD, efA, efB, Hn, dom="r"):
    buf = trD[:, :K].clone(); Lw = K + F; preds = []
    for h in range(Hn):
        wa = efA[:, h:h + Lw]; wb = efB[:, h:h + Lw]
        if wa.shape[1] < Lw:
            pad = Lw - wa.shape[1]
            wa = torch.cat([wa, wa[:, -1:].repeat(1, pad, 1, 1)], 1)
            wb = torch.cat([wb, wb[:, -1:].repeat(1, pad, 1, 1)], 1)
        logits, _ = m.fwd_dual(buf[:, -K:].permute(0, 2, 1, 3), wa, wb, dom=dom)
        nxt = buf[:, -1] + m.expected_vel(logits[:, :, 0, :])
        preds.append(nxt); buf = torch.cat([buf, nxt[:, None]], 1)
    return torch.stack(preds, 1)                               # (B,H,2P,2)


def rays_from_norm(uv_norm, view):
    """crop-norm (…,2) -> (origin (3,), dirs (…,3)) in world."""
    x, y, w, h = CROPS[view]
    c = _CAL[view]; R, t = _RT[view]
    u = uv_norm[..., 0] * w + x; v = uv_norm[..., 1] * h + y
    d_cam = np.stack([(u - c["cx"]) / c["f"], (v - c["cy"]) / c["f"], np.ones_like(u)], -1)
    return t, d_cam @ R.T


def triangulate(uv0, uv1):
    """two crop-norm point sets (…,2) -> world (…,3) least-squares midpoint."""
    o0, d0 = rays_from_norm(uv0, 0); o1, d1 = rays_from_norm(uv1, 1)
    d0 = d0 / np.linalg.norm(d0, axis=-1, keepdims=True)
    d1 = d1 / np.linalg.norm(d1, axis=-1, keepdims=True)
    b = o1 - o0
    d0d1 = (d0 * d1).sum(-1)
    denom = 1 - d0d1 ** 2
    t0 = ((b * d0).sum(-1) - (b * d1).sum(-1) * d0d1) / (denom + 1e-9)
    t1 = ((b * d0).sum(-1) * d0d1 - (b * d1).sum(-1)) / (denom + 1e-9)
    p0 = o0 + t0[..., None] * d0
    p1 = o1 + t1[..., None] * d1
    return (p0 + p1) / 2


def subsample_robot_pool(pool, nrob):
    """Robot-scarce protocol: deterministic rng(0).choice subset of the (okfirst) robot TRAINING pool,
    SAME subset for a given (pool, nrob) pair regardless of ACTION/MIX (all 2x2 arms at a given N train
    on the identical robot clips) -- rng(0) depends only on pool/nrob, not on ACTION/MIX/SPLIT choices
    upstream. No-op (returns pool unchanged) if nrob is falsy or >= len(pool)."""
    if not nrob:
        return pool
    n = int(nrob)
    if n >= len(pool):
        return pool
    # rng(SEED): 同一 seed 下所有 arm 共享同一子集(控制变量), 换 seed 才换"挑哪 N 条"
    # (稀缺协议下子集选择本身是主要方差来源). SEED=0 时与原 rng(0) 行为完全一致.
    return np.random.default_rng(SEED).choice(pool, size=n, replace=False)


def split_okfirst(ok, heldout=HELDOUT):
    """③-style split (filter-then-permute), verbatim: exp_scel_dualview_dit_formal.py
    `okr = np.where(R["ok"])[0]; perm = rng(0).permutation(okr); ho, pool = perm[:150], perm[150:]`.
    Fixes audit item5: legacy ②-split permutes ALL indices then filters `ok` post-hoc, which
    draws a *different* Fisher-Yates sequence than ③'s filter-first split whenever `ok` has
    holes -> ②'s held-out set is not the same as ③'s, so ② ends up training on most of ③'s
    published eval clips. `ho`/`pool` are returned already ok-filtered, same as legacy."""
    okr = np.where(ok)[0]
    perm = np.random.default_rng(0).permutation(okr)
    return perm[:heldout], perm[heldout:]


def split_by_episode(ok, vid, heldout_vids):
    """IWS 式 episode(vid)分组 split (照 play_eef_dataset.py:378): heldout_vids 里整段 episode
    作 heldout, 其余作训练池, 都 ok-过滤。heldout episode 与训练 vid 完全不相交 -> 无跨-episode
    帧泄漏(修 split_okfirst 按 clip-idx 切致 85% heldout 帧与训练重复, 见 project_clip_heldout_leakage)。"""
    hv = set(int(v) for v in heldout_vids)
    okr = np.where(ok)[0]
    ho = np.array([i for i in okr if int(vid[i]) in hv], dtype=int)
    pool = np.array([i for i in okr if int(vid[i]) not in hv], dtype=int)
    return ho, pool


def main():
    z = np.load(DS)
    ok = z["low_valid"]
    trA = z["tracks"].astype(np.float32); trB = z["tracks_low"].astype(np.float32)
    vsA = z["vis"].astype(np.float32); vsB = z["vis_low"].astype(np.float32)
    t3 = z["tracks3d"].astype(np.float64)
    trD = np.concatenate([trA, np.nan_to_num(trB, nan=0.5)], 2)   # (N,L,2P,2)
    vsD = np.concatenate([vsA, vsB], 2)

    ds_dir = os.path.dirname(DS)
    _PRIM_HUMAN = "human" in os.path.basename(DS)           # ★ho场景 DS=human → 加载human sidecar+dom=h(修skel/raster ho用错robot sidecar的bug)
    _prim_dom = "h" if _PRIM_HUMAN else "r"
    if ACTION in ("raster", "rasterg"):
        sk_r = np.load(f"{ds_dir}/raster_sidecar_{'human' if _PRIM_HUMAN else 'robot'}.npz")
    elif ACTION in ("skel", "skelv3"):
        sk_r = np.load(f"{ds_dir}/skel_sidecar_{'human' if _PRIM_HUMAN else 'robot_v2'}.npz")
    else:
        sk_r = None
    efA, efB = load_action_tokens(ACTION, _prim_dom, z, sk_r)

    print(f"split={SPLIT} action={ACTION} mix={MIX} nrob={NROB or 'all'}", flush=True)
    heldout_vids = [int(x) for x in os.environ.get("HELDOUT_VIDS", "").split(",") if x]
    if heldout_vids:                                           # ★episode(vid)分组 split, 修帧泄漏(见 project_clip_heldout_leakage)
        ho, pool = split_by_episode(ok, z["vid"], heldout_vids)
        print(f"heldout vids {heldout_vids}: ho {len(ho)} pool {len(pool)} (episode-split, no frame leak)", flush=True)
    elif SPLIT == "okfirst":
        ho, pool = split_okfirst(ok, HELDOUT)
    else:
        perm = np.random.default_rng(0).permutation(len(trD))
        ho = np.array([i for i in perm[:HELDOUT] if ok[i]])
        pool = np.array([i for i in perm[HELDOUT:] if ok[i]])
    if SMOKE: pool = pool[:200]; ho = ho[:24]
    pool = subsample_robot_pool(pool, NROB)
    force = [int(x) for x in os.environ.get("HELDOUT_IDS", "").split(",") if x]   # clip-idx 强制留出(遗留); episode-split 下 demo seq 已随 vid 进 ho, 忽略
    if force and not heldout_vids:
        fset = set(force); ho = np.array(sorted(set(ho.tolist()) | fset))
        pool = np.array([i for i in pool if int(i) not in fset])
        print(f"forced heldout ids {force}: pool now {len(pool)}", flush=True)
    print(f"aligned clips: {int(ok.sum())}/{len(ok)}; train {len(pool)} ho {len(ho)}", flush=True)

    trDh = vsDh = efAh = efBh = idxh = None
    n_human = 0
    if MIX == "rh":
        zh = np.load(os.environ.get("DS_H", f"{ds_dir}/clips_human_L24.npz"))   # DS_H: retrack human clips
        okh = zh["low_valid"]
        trAh = zh["tracks"].astype(np.float32); trBh = zh["tracks_low"].astype(np.float32)
        vsAh = zh["vis"].astype(np.float32); vsBh = zh["vis_low"].astype(np.float32)
        trDh = np.concatenate([trAh, np.nan_to_num(trBh, nan=0.5)], 2)
        vsDh = np.concatenate([vsAh, vsBh], 2)
        sk_h = (np.load(f"{ds_dir}/raster_sidecar_human.npz") if ACTION in ("raster", "rasterg")
                else np.load(f"{ds_dir}/skel_sidecar_human.npz") if ACTION in ("skel", "skelv3") else None)
        efAh, efBh = load_action_tokens(ACTION, "h", zh, sk_h)
        idxh_np = np.where(okh)[0]
        hvh = [int(x) for x in os.environ.get("HELDOUT_VIDS_H", "").split(",") if x]   # ★episode 排除 human cotrain(→human-in-domain eval)
        if hvh:
            hvset = set(hvh); vh = zh["vid"]
            idxh_np = np.array([i for i in idxh_np if int(vh[i]) not in hvset])
            print(f"HELDOUT_VIDS_H {hvh}: human cotrain now {len(idxh_np)} (episode-excluded)", flush=True)
        hho = [int(x) for x in os.environ.get("HELDOUT_H", "").split(",") if x]   # clip-idx 排除(遗留); episode-split 下用 HELDOUT_VIDS_H
        if hho and not hvh:
            idxh_np = np.array([i for i in idxh_np if int(i) not in set(hho)])
            print(f"HELDOUT_H {hho}: human cotrain now {len(idxh_np)}", flush=True)
        if SMOKE: idxh_np = idxh_np[:200]
        hn = int(os.environ.get("HUMAN_N", "0"))                                  # >0: 下采样human cotrain到N(稀缺sweep)
        if hn and hn < len(idxh_np):
            idxh_np = np.random.default_rng(SEED).permutation(idxh_np)[:hn]
            print(f"HUMAN_N={hn}: human cotrain subsampled to {len(idxh_np)}", flush=True)
        idxh = torch.from_numpy(idxh_np)
        n_human = len(idxh)
        print(f"human co-train clips: {n_human}/{len(okh)} (L={trDh.shape[1]})", flush=True)

    m = train_dual(trD, vsD, efA, efB, torch.from_numpy(pool), action=ACTION, seed=SEED, R_SS=R_SS_ENV,
                   trDh=trDh, vsDh=vsDh, efAh=efAh, efBh=efBh, idxh=idxh, head_mode=HEAD_MODE)
    torch.save(m, f"{OUT}/wm_dual.pt")

    pr = rollout_dual(m, torch.from_numpy(trD[ho]).float().to(device),
                      torch.from_numpy(efA[ho]).float().to(device),
                      torch.from_numpy(efB[ho]).float().to(device), H).cpu().numpy()
    P = trA.shape[2]
    prA, prB = pr[:, :, :P], pr[:, :, P:]
    gtA = trA[ho][:, K:K + H]; gtB = trB[ho][:, K:K + H]
    eA = np.linalg.norm(prA - gtA, axis=-1) * IMG
    eB = np.linalg.norm(prB - np.nan_to_num(gtB, nan=0.5), axis=-1) * IMG
    # triangulated 3D error vs depth-lifted GT
    tri = triangulate(prA, prB)
    gt3 = t3[ho][:, K:K + H]
    vmask = np.isfinite(gt3[..., 0])
    e3 = np.linalg.norm(tri - np.nan_to_num(gt3), axis=-1) * 1000
    e3 = float(e3[vmask].mean())
    ez = np.abs(tri[..., 2] - np.nan_to_num(gt3[..., 2])) * 1000
    ez = float(ez[vmask].mean())
    zr = np.nanmax(t3[ho][..., 2], axis=(1, 2)) - np.nanmin(t3[ho][..., 2], axis=(1, 2))
    lift = zr > 0.08
    metrics = {
        "action": ACTION, "mix": MIX, "head_mode": HEAD_MODE, "nrob": int(NROB) if NROB else None,
        "heldout_vids": heldout_vids or None,
        "n_train_robot": int(len(pool)), "n_human_cotrain": int(n_human), "n_heldout": int(len(ho)),
        "drift_px_cam_high": float(eA.mean()), "drift_px_cam_low": float(eB.mean()),
        "drift_px_cam_high_lift": float(eA[lift].mean()), "drift_px_cam_low_lift": float(eB[lift].mean()),
        "n_lift": int(lift.sum()), "tri3d_err_mm": e3, "tri3d_z_err_mm": ez,
    }
    json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=2)
    lines = [
        f"DUAL-VIEW ② (joint 2-view flow, action={ACTION}, mix={MIX}, nrob={NROB or 'all'}) "
        f"H={H} held-out n={len(ho)} train_robot={len(pool)} human_cotrain={n_human}",
        f"drift px cam_high: dual {metrics['drift_px_cam_high']:.2f}   (single-view 2D baseline 2.96, 3D 3.11)",
        f"drift px cam_low : dual {metrics['drift_px_cam_low']:.2f}   (single-view 2D baseline 3.10, 3D 7.78)",
        f"lift subset (n={metrics['n_lift']}): high {metrics['drift_px_cam_high_lift']:.2f} "
        f"| low {metrics['drift_px_cam_low_lift']:.2f}",
        f"TRIANGULATED 3D err: {e3:.1f} mm (z {ez:.1f} mm)  [3D-② z-err was 6.9mm clean / 12.5mm camlow-lifted]",
    ]
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines) + "\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
