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
NROB = os.environ.get("NROB", "")                               # "" = full robot pool | int = robot-scarce N
_NROB_TAG = NROB if NROB else "all"
SEED = int(os.environ.get("SEED", "0"))                         # 训练种子; seed=0 保持原路径(向后兼容)
_SEED_TAG = "" if SEED == 0 else f"_s{SEED}"
_DEFAULT_OUT = ("outputs/cross_embodiment_wm/dualview_wm" if (ACTION == "dummy5" and MIX == "r" and not NROB and SEED == 0)
                else f"outputs/cross_embodiment_wm/dualview_wm_skelact/{ACTION}_{MIX}_n{_NROB_TAG}{_SEED_TAG}")
OUT = os.environ.get("OUT_DIR", _DEFAULT_OUT)
os.makedirs(OUT, exist_ok=True)
H = 40; HELDOUT = 150; IMG = 128; VEL_HALF = 0.06
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
    if mode == "dummy5":
        efA = z["eef"].astype(np.float32)
        efB = np.nan_to_num(z["eef_low"].astype(np.float32), nan=0.5)
        return efA, efB
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
    def __init__(s, P, action="dummy5", **kw):
        super().__init__(P, **kw)
        s.P1 = P
        s.action = action
        s.n_tok = 5 if action == "dummy5" else 4                # dummy5: 3->5 virtual pts | skel: 4 tokens as-is
        s.view_emb = nn.Parameter(torch.zeros(2, s.Dm))
        s.act = nn.Linear((K + F) * s.n_tok * 2 * 2, s.Dm)      # action tokens x 2 views
        if action in ("raster", "rasterg"):                      # OSCAR 式: 光栅图(CNN) + 末端位置(+grip)(Linear)
            s.act_ras = RasterAct(2 * (K + F), s.Dm)             # 双视角 x Lw 帧当通道
            _posdim = (K + F) * 2 * (3 if action == "rasterg" else 2)   # rasterg: 每视角每帧 (x,y,grip)
            s.act_pos = nn.Linear(_posdim, s.Dm)                 # 双视角末端位置(+归一化 grip)

    def _act_pts(s, eef):
        """eef (B,Lw,n_raw,2) -> (B,Lw,n_tok,2) action tokens. dummy5: DexWM virtual constellation.
        skel: tokens already ARE the isomorphic geometric skeleton points -> identity (no expansion).
        getattr default covers unpickling checkpoints saved before this attribute existed."""
        return dummy5(eef) if getattr(s, "action", "dummy5") == "dummy5" else eef

    def fwd_dual(s, hist, eef_a, eef_b):
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
            logits = s.head(x).reshape(B, P2, F, s.W * s.W)
            return logits, anchor
        if getattr(s, "action", "dummy5") == "skelv3":
            d5a = s._act_pts(eef_a).clone(); d5b = s._act_pts(eef_b).clone()
            d5a[:, :, 0] = d5a[:, :, 0] - oc_a
            d5b[:, :, 0] = d5b[:, :, 0] - oc_b
        else:
            d5a = s._act_pts(eef_a) - oc_a[:, :, None]
            d5b = s._act_pts(eef_b) - oc_b[:, :, None]
        act = s.act(torch.cat([d5a.reshape(B, -1), d5b.reshape(B, -1)], -1))[:, None]
        x = s.tf(torch.cat([obj, act], 1))[:, :P2]
        logits = s.head(x).reshape(B, P2, F, s.W * s.W)
        return logits, anchor


def _ss_batch_step(m, opt, G, Vv, Ea, Eb, R_SS, pteach):
    """One optimizer step of scheduled-sampling rollout over R_SS steps for one batch. Domain-agnostic
    (robot or human -- caller picks the matching R_SS/tensors); verbatim body of the original
    train_dual inner loop, factored out so MIX=rh can reuse it unchanged for the human pass."""
    buf = G[:, :K].clone(); losses = []
    for h in range(R_SS):
        wa = Ea[:, h:h + K + F]; wb = Eb[:, h:h + K + F]
        if wa.shape[1] < K + F:
            pad = K + F - wa.shape[1]
            wa = torch.cat([wa, wa[:, -1:].repeat(1, pad, 1, 1)], 1)
            wb = torch.cat([wb, wb[:, -1:].repeat(1, pad, 1, 1)], 1)
        logits, _ = m.fwd_dual(buf[:, -K:].permute(0, 2, 1, 3), wa, wb)
        lg0 = logits[:, :, 0, :]
        gt_vel = G[:, K + h] - buf[:, -1]
        cls = vel_to_class(gt_vel, m.W, m.vel_half)
        w = (Vv[:, K + h] * Vv[:, K - 1])
        ce = nn.functional.cross_entropy(lg0.reshape(-1, m.W * m.W), cls.reshape(-1),
                                         reduction="none")
        ce = (ce * w.reshape(-1)).sum() / (w.sum() + 1e-6)
        losses.append(ce)
        nxt = buf[:, -1] + m.expected_vel(lg0)
        use_gt = (torch.rand(len(G), 1, 1, device=device) < pteach)
        buf = torch.cat([buf, torch.where(use_gt, G[:, K + h], nxt.detach())[:, None]], 1)
    loss = torch.stack(losses).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    return float(loss)


def train_dual(trD, vsD, efA, efB, idx, seed=0, R_SS=32, action="dummy5",
               trDh=None, vsDh=None, efAh=None, efBh=None, idxh=None):
    """SS training, robot pool (trD/.../idx). MIX=rh (trDh et al not None): after each robot epoch's
    batches, also SS-train on the human co-train pool with the SAME vis-weighted CE loss (no extra
    reweighting) -- rollout steps capped to the human clip length (R_SS_h = min(R_SS, L_human-K),
    since clips_human_L24.npz is L=24 vs the robot's L=48). Default args (action='dummy5', no *h)
    reproduce the original robot-only loop exactly (same rng call order/shapes -> byte-identical)."""
    torch.manual_seed(seed); P2 = trD.shape[2]
    m = DualLWC(P2 // 2, action=action, Dm=384, layers=3, W=15, vel_half=VEL_HALF).to(device)
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
        for i in range(0, len(pe), SSm.WM_BS):
            b = pe[i:i + SSm.WM_BS]
            G = trT[b].to(device); Vv = vsT[b].to(device)
            Ea = eAT[b].to(device); Eb = eBT[b].to(device)
            loss = _ss_batch_step(m, opt, G, Vv, Ea, Eb, R_SS, pteach)
        if mix_human:
            peh = idxh[torch.randperm(len(idxh), generator=gh)]
            for i in range(0, len(peh), SSm.WM_BS):
                b = peh[i:i + SSm.WM_BS]
                G = trTh[b].to(device); Vv = vsTh[b].to(device)
                Ea = eATh[b].to(device); Eb = eBTh[b].to(device)
                loss = _ss_batch_step(m, opt, G, Vv, Ea, Eb, R_SS_h, pteach)
        print(f"ep{ep} loss {loss:.4f}", flush=True)
    return m.eval()


@torch.no_grad()
def rollout_dual(m, trD, efA, efB, Hn):
    buf = trD[:, :K].clone(); Lw = K + F; preds = []
    for h in range(Hn):
        wa = efA[:, h:h + Lw]; wb = efB[:, h:h + Lw]
        if wa.shape[1] < Lw:
            pad = Lw - wa.shape[1]
            wa = torch.cat([wa, wa[:, -1:].repeat(1, pad, 1, 1)], 1)
            wb = torch.cat([wb, wb[:, -1:].repeat(1, pad, 1, 1)], 1)
        logits, _ = m.fwd_dual(buf[:, -K:].permute(0, 2, 1, 3), wa, wb)
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


def main():
    z = np.load(DS)
    ok = z["low_valid"]
    trA = z["tracks"].astype(np.float32); trB = z["tracks_low"].astype(np.float32)
    vsA = z["vis"].astype(np.float32); vsB = z["vis_low"].astype(np.float32)
    t3 = z["tracks3d"].astype(np.float64)
    trD = np.concatenate([trA, np.nan_to_num(trB, nan=0.5)], 2)   # (N,L,2P,2)
    vsD = np.concatenate([vsA, vsB], 2)

    ds_dir = os.path.dirname(DS)
    sk_r = (np.load(f"{ds_dir}/raster_sidecar_robot.npz") if ACTION in ("raster", "rasterg")
            else np.load(f"{ds_dir}/skel_sidecar_robot_v2.npz") if ACTION in ("skel", "skelv3") else None)
    efA, efB = load_action_tokens(ACTION, "r", z, sk_r)

    print(f"split={SPLIT} action={ACTION} mix={MIX} nrob={NROB or 'all'}", flush=True)
    if SPLIT == "okfirst":
        ho, pool = split_okfirst(ok, HELDOUT)
    else:
        perm = np.random.default_rng(0).permutation(len(trD))
        ho = np.array([i for i in perm[:HELDOUT] if ok[i]])
        pool = np.array([i for i in perm[HELDOUT:] if ok[i]])
    if SMOKE: pool = pool[:200]; ho = ho[:24]
    pool = subsample_robot_pool(pool, NROB)
    print(f"aligned clips: {int(ok.sum())}/{len(ok)}; train {len(pool)} ho {len(ho)}", flush=True)

    trDh = vsDh = efAh = efBh = idxh = None
    n_human = 0
    if MIX == "rh":
        zh = np.load(f"{ds_dir}/clips_human_L24.npz")
        okh = zh["low_valid"]
        trAh = zh["tracks"].astype(np.float32); trBh = zh["tracks_low"].astype(np.float32)
        vsAh = zh["vis"].astype(np.float32); vsBh = zh["vis_low"].astype(np.float32)
        trDh = np.concatenate([trAh, np.nan_to_num(trBh, nan=0.5)], 2)
        vsDh = np.concatenate([vsAh, vsBh], 2)
        sk_h = (np.load(f"{ds_dir}/raster_sidecar_human.npz") if ACTION in ("raster", "rasterg")
                else np.load(f"{ds_dir}/skel_sidecar_human.npz") if ACTION in ("skel", "skelv3") else None)
        efAh, efBh = load_action_tokens(ACTION, "h", zh, sk_h)
        idxh_np = np.where(okh)[0]
        if SMOKE: idxh_np = idxh_np[:200]
        idxh = torch.from_numpy(idxh_np)
        n_human = len(idxh)
        print(f"human co-train clips: {n_human}/{len(okh)} (L={trDh.shape[1]})", flush=True)

    m = train_dual(trD, vsD, efA, efB, torch.from_numpy(pool), action=ACTION, seed=SEED,
                   trDh=trDh, vsDh=vsDh, efAh=efAh, efBh=efBh, idxh=idxh)
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
        "action": ACTION, "mix": MIX, "nrob": int(NROB) if NROB else None,
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
