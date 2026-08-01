"""IDM 数据: 从 robot clips 建 (窗口输入, Δjoint+grip) 张量 + episode-split。输入集可 config 切(消融)。"""
import numpy as np, sys
sys.path.insert(0, ".")
import torch
from exp_scel_dualview_wm import mp_constellation

# ★接触=物体在动(flow 带 action 信息的帧)。grip 是 0-0.04 物理尺度(非0-1)且闭合≠碰物体, 故不用 grip 判接触。
MOVE_THRESH = 0.02          # 物体质心相邻帧位移>0.02(crop-norm≈2.6px@128)=接触/操作帧; <=此=自由/静止(实测得~20%接触/80%自由)
HELDOUT_VIDS = (100, 102)

INPUT_SPECS = {
    "A0": {"flow": True,  "trace": False, "grip": False, "future": True},
    "A1": {"flow": True,  "trace": True,  "grip": False, "future": True},
    "A2": {"flow": True,  "trace": True,  "grip": True,  "future": True},
    "A3": {"flow": True,  "trace": True,  "grip": False, "future": False},  # causal 窗口
}


def _win(KP, FF, spec):
    return KP + (FF if spec["future"] else 0) + 1


def input_dim(spec, KP=4, FF=4):
    W = _win(KP, FF, spec); d = 0
    if spec["flow"]:  d += W * 96 * 2         # 双视角 48 track x 2 coord
    if spec["trace"]: d += W * 10 * 2         # 双视角 mp 星座 5 点 x 2 coord
    if spec["grip"]:  d += W * 1
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
    joint = z["joint"].astype(np.float32); grip = z["grip"].astype(np.float32)
    L = tracks.shape[1]; ff = FF if spec["future"] else 0
    clips = range(len(tracks)) if clips is None else clips
    T = L - 1                                                        # 中心帧数
    offs = np.arange(-KP, ff + 1)                                    # (W,)
    widx = np.clip(np.arange(T)[:, None] + offs[None, :], 0, L - 1)  # (T,W)
    Xs, Ys, ci_l, t_l, con_l = [], [], [], [], []
    for ci in clips:
        oc0 = tracks[ci]; oc1 = tracks_lo[ci]                        # (L,48,2)
        feats = []
        if spec["flow"]:
            cen0 = oc0[:T].mean(1); cen1 = oc1[:T].mean(1)          # (T,2)
            f0 = oc0[widx] - cen0[:, None, None]; f1 = oc1[widx] - cen1[:, None, None]  # (T,W,48,2)
            feats.append(np.concatenate([f0, f1], 2).reshape(T, -1))
        if spec["trace"]:
            tr0 = _mp_trace(eef[ci], 0); tr1 = _mp_trace(eef_lo[ci], 1)  # (L,5,2)
            feats.append(np.concatenate([tr0[widx], tr1[widx]], 2).reshape(T, -1))  # (T,W,10,2)
        if spec["grip"]:
            feats.append(grip[ci][widx])                            # (T,W)
        X = np.concatenate(feats, 1).astype(np.float32)             # (T, Din)
        Y = np.concatenate([joint[ci, 1:L, :6] - joint[ci, :T, :6], grip[ci, :T, None]], 1).astype(np.float32)
        cen = oc0.mean(1)                                           # (L,2) 质心
        con = np.linalg.norm(cen[1:] - cen[:T], axis=-1) > MOVE_THRESH  # (T,) 接触=物体在动
        Xs.append(X); Ys.append(Y); ci_l.append(np.full(T, ci)); t_l.append(np.arange(T)); con_l.append(con)
    meta = {"clip_idx": np.concatenate(ci_l), "t": np.concatenate(t_l), "contact": np.concatenate(con_l).astype(bool)}
    return np.concatenate(Xs), np.concatenate(Ys), meta
