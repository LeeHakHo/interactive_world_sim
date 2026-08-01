"""IDM 数据: 从 robot clips 建 (窗口输入, Δjoint+grip) 张量 + episode-split。输入集可 config 切(消融)。"""
import os
import numpy as np, sys
sys.path.insert(0, ".")
import torch
from exp_scel_dualview_wm import mp_constellation

# ★接触=物体在动(flow 带 action 信息的帧)。grip 是 0-0.04 物理尺度(非0-1)且闭合≠碰物体, 故不用 grip 判接触。
MOVE_THRESH = 0.02          # 物体质心相邻帧位移>0.02(crop-norm≈2.6px@128)=接触/操作帧; <=此=自由/静止(实测得~20%接触/80%自由)
HELDOUT_VIDS = (100, 102)
YMODE = os.environ.get("YMODE", "delta")   # delta: 预测Δjoint(积分, 会漂移); abs: 预测下一帧绝对joint(无漂移)

INPUT_SPECS = {
    "A0": {"flow": True,  "trace": False, "grip": False, "future": True},
    "A1": {"flow": True,  "trace": True,  "grip": False, "future": True},
    "A2": {"flow": True,  "trace": True,  "grip": True,  "future": True},
    "A3": {"flow": True,  "trace": True,  "grip": False, "future": False},  # causal 窗口
    # B1: trace 用【原始 eef 3 点】(含腕+真实尺度), 不做 mp 星座归一 → 量"不为跨具身丢信息"的精度天花板
    "B1": {"flow": True,  "trace": True,  "grip": False, "future": True, "traceraw": True},
    # C1: 3D 物体流 + 3D eef 轨迹(depth+标定反投影, 天生域不变, 不丢腕/尺度)。测"精度限制是不是2D丢深度"
    "C1": {"flow3d": True, "trace3d": True, "future": True},
}


def _win(KP, FF, spec):
    return KP + (FF if spec["future"] else 0) + 1


def input_dim(spec, KP=4, FF=4):
    W = _win(KP, FF, spec); d = 0
    if spec.get("flow"):   d += W * 96 * 2        # 双视角 48 track x 2 coord
    if spec.get("trace"):                          # 双视角 x 2 coord; mp 星座 5 点 或 原始 eef 3 点
        d += W * (3 if spec.get("traceraw") else 5) * 2 * 2
    if spec.get("grip"):   d += W * 1
    if spec.get("flow3d"): d += W * 48 * 3        # 3D 48 track x 3 coord(视角无关)
    if spec.get("trace3d"):d += W * 3 * 3         # 3D eef 3 点 x 3 coord
    return d


def episode_split(z):
    vid = z["vid"]; ok = z["low_valid"]
    idx = np.where(ok)[0]
    ho = np.array([i for i in idx if vid[i] in HELDOUT_VIDS])
    tr = np.array([i for i in idx if vid[i] not in HELDOUT_VIDS])
    return tr, ho


def _mp_trace(eef3, view):   # (L,3,2) -> (L,5,2) mp 星座(view 决定 canonical 半开口尺度)
    t = torch.from_numpy(np.nan_to_num(eef3.astype(np.float32), nan=0.5))[None]  # (1,L,3,2)
    return mp_constellation(t, view)[0].numpy()


def CONTACT_FN(z, ci, t):
    """接触帧 = 物体质心在 t→t+1 移动 > MOVE_THRESH(物体在被操作)。"""
    c = z["tracks"][ci].astype(np.float32).mean(1)   # (L,2) 物体质心
    return bool(t + 1 < len(c) and np.linalg.norm(c[t + 1] - c[t]) > MOVE_THRESH)


def build_windows(z, spec, KP=4, FF=4, clips=None):
    """向量化(每 clip 一次性 gather 全窗口)。中心帧 t∈[0,L-2](需 t+1 有 Δjoint)。"""
    tracks = z["tracks"].astype(np.float32); tracks_lo = z["tracks_low"].astype(np.float32)
    eef = z["eef"].astype(np.float32); eef_lo = z["eef_low"].astype(np.float32)
    has_y = "joint" in z.files                                       # 人类数据无 robot joint/grip → 推理only, Y置零
    joint = z["joint"].astype(np.float32) if has_y else np.zeros((len(tracks), tracks.shape[1], 7), np.float32)
    grip = z["grip"].astype(np.float32) if "grip" in z.files else np.zeros((len(tracks), tracks.shape[1]), np.float32)
    need3d = spec.get("flow3d") or spec.get("trace3d")
    if need3d:
        tr3 = np.nan_to_num(z["tracks3d"].astype(np.float32), nan=0.0)   # (N,L,48,3)
        ef3 = np.nan_to_num(z["eef3d"].astype(np.float32), nan=0.0)      # (N,L,3,3)
    L = tracks.shape[1]; ff = FF if spec["future"] else 0
    clips = range(len(tracks)) if clips is None else clips
    T = L - 1                                                        # 中心帧数
    offs = np.arange(-KP, ff + 1)                                    # (W,)
    widx = np.clip(np.arange(T)[:, None] + offs[None, :], 0, L - 1)  # (T,W)
    Xs, Ys, ci_l, t_l, con_l = [], [], [], [], []
    for ci in clips:
        oc0 = tracks[ci]; oc1 = tracks_lo[ci]                        # (L,48,2)
        feats = []
        if spec.get("flow"):
            cen0 = oc0[:T].mean(1); cen1 = oc1[:T].mean(1)          # (T,2)
            f0 = oc0[widx] - cen0[:, None, None]; f1 = oc1[widx] - cen1[:, None, None]  # (T,W,48,2)
            feats.append(np.concatenate([f0, f1], 2).reshape(T, -1))
        if spec.get("trace"):
            if spec.get("traceraw"):                                    # 原始 eef 3 点(含腕+真实尺度)
                tr0 = np.nan_to_num(eef[ci], nan=0.5); tr1 = np.nan_to_num(eef_lo[ci], nan=0.5)  # (L,3,2)
            else:
                tr0 = _mp_trace(eef[ci], 0); tr1 = _mp_trace(eef_lo[ci], 1)  # (L,5,2) mp 星座
            feats.append(np.concatenate([tr0[widx], tr1[widx]], 2).reshape(T, -1))
        if spec.get("grip"):
            feats.append(grip[ci][widx])                            # (T,W)
        if spec.get("flow3d"):                                       # 3D 物体流(相对3D质心, 视角无关)
            o3 = tr3[ci]; cen3 = o3[:T].mean(1)                     # (T,3)
            f3 = o3[widx] - cen3[:, None, None]                     # (T,W,48,3)
            feats.append(f3.reshape(T, -1))
        if spec.get("trace3d"):                                      # 3D eef 3 点(腕/两指)
            feats.append(ef3[ci][widx].reshape(T, -1))             # (T,W,3,3)
        X = np.concatenate(feats, 1).astype(np.float32)             # (T, Din)
        ycmd = joint[ci, 1:L, :6] if YMODE == "abs" else joint[ci, 1:L, :6] - joint[ci, :T, :6]   # abs:下一帧绝对joint | delta:Δjoint
        Y = np.concatenate([ycmd, grip[ci, :T, None]], 1).astype(np.float32)
        cen = oc0.mean(1)                                           # (L,2) 质心
        con = np.linalg.norm(cen[1:] - cen[:T], axis=-1) > MOVE_THRESH  # (T,) 接触=物体在动
        Xs.append(X); Ys.append(Y); ci_l.append(np.full(T, ci)); t_l.append(np.arange(T)); con_l.append(con)
    meta = {"clip_idx": np.concatenate(ci_l), "t": np.concatenate(t_l), "contact": np.concatenate(con_l).astype(bool)}
    return np.concatenate(Xs), np.concatenate(Ys), meta


def build_windows_tokens(z, spec, KP=4, FF=4, clips=None):
    """结构化 token 版(给 transformer): 返回 Xtok (M,W,N,C), ptype (N,), Y (M,7), meta。
    每 token = (帧 offset w, 点 n) 的坐标; ptype: 0=物体流, 1=eef/trace。
    2D 双视角: C=4([v0x,v0y,v1x,v1y]); 3D: C=3。N=物体点(48)+trace点。"""
    tracks = z["tracks"].astype(np.float32); tracks_lo = z["tracks_low"].astype(np.float32)
    eef = z["eef"].astype(np.float32); eef_lo = z["eef_low"].astype(np.float32)
    has_y = "joint" in z.files
    joint = z["joint"].astype(np.float32) if has_y else np.zeros((len(tracks), tracks.shape[1], 7), np.float32)
    grip = z["grip"].astype(np.float32) if "grip" in z.files else np.zeros((len(tracks), tracks.shape[1]), np.float32)
    is3d = spec.get("flow3d") or spec.get("trace3d")
    if is3d:
        tr3 = np.nan_to_num(z["tracks3d"].astype(np.float32), nan=0.0); ef3 = np.nan_to_num(z["eef3d"].astype(np.float32), nan=0.0)
    L = tracks.shape[1]; ff = FF if spec["future"] else 0; T = L - 1
    offs = np.arange(-KP, ff + 1); W = len(offs)
    widx = np.clip(np.arange(T)[:, None] + offs[None, :], 0, L - 1)   # (T,W)
    clips = range(len(tracks)) if clips is None else clips
    Xs, Ys, ci_l, t_l, con_l = [], [], [], [], []
    for ci in clips:
        objtok = tratok = None
        if is3d:                                                      # 3D: C=3
            o3 = tr3[ci]; cen3 = o3[:T].mean(1)                       # (T,3)
            objtok = o3[widx] - cen3[:, None, None]                   # (T,W,48,3)
            tratok = ef3[ci][widx]                                    # (T,W,3,3) eef3d 绝对
        else:                                                         # 2D 双视角: C=4
            oc0, oc1 = tracks[ci], tracks_lo[ci]
            cen0 = oc0[:T].mean(1); cen1 = oc1[:T].mean(1)
            f0 = oc0[widx] - cen0[:, None, None]; f1 = oc1[widx] - cen1[:, None, None]   # (T,W,48,2)
            objtok = np.concatenate([f0, f1], -1)                     # (T,W,48,4)
            if spec.get("traceraw"):
                t0 = np.nan_to_num(eef[ci], nan=0.5)[widx]; t1 = np.nan_to_num(eef_lo[ci], nan=0.5)[widx]  # (T,W,3,2)
            else:
                t0 = _mp_trace(eef[ci], 0)[widx]; t1 = _mp_trace(eef_lo[ci], 1)[widx]     # (T,W,5,2)
            tratok = np.concatenate([t0, t1], -1)                     # (T,W,n,4)
        Xtok = np.concatenate([objtok, tratok], 2).astype(np.float32) # (T,W,N,C)
        ycmd = joint[ci, 1:L, :6] if YMODE == "abs" else joint[ci, 1:L, :6] - joint[ci, :T, :6]
        Y = np.concatenate([ycmd, grip[ci, :T, None]], 1).astype(np.float32)
        cen = tracks[ci].mean(1); con = np.linalg.norm(cen[1:] - cen[:T], axis=-1) > MOVE_THRESH
        Xs.append(Xtok); Ys.append(Y); ci_l.append(np.full(T, ci)); t_l.append(np.arange(T)); con_l.append(con)
    nobj = 48; ntra = Xs[0].shape[2] - nobj
    ptype = np.concatenate([np.zeros(nobj, int), np.ones(ntra, int)])
    meta = {"clip_idx": np.concatenate(ci_l), "t": np.concatenate(t_l), "contact": np.concatenate(con_l).astype(bool)}
    return np.concatenate(Xs), ptype, np.concatenate(Ys), meta
