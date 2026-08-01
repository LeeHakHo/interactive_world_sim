"""IDM 数据: 从 robot clips 建 (窗口输入, Δjoint+grip) 张量 + episode-split。输入集可 config 切(消融)。"""
import numpy as np, sys
sys.path.insert(0, ".")
import torch
from exp_scel_dualview_wm import mp_constellation

GRIP_OPEN_THRESH = 0.6      # grip<0.6 视为夹爪在闭合/接触(grip 0-1, robot 张开≈1)
MOVE_THRESH = 0.01          # 物体质心相邻帧位移>0.01(crop-norm)视为在动=接触相关帧
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
    if z["grip"][ci, t] < GRIP_OPEN_THRESH:
        return True
    c = z["tracks"][ci].astype(np.float32).mean(1)   # (L,2) 物体质心
    if t + 1 < len(c) and np.linalg.norm(c[t + 1] - c[t]) > MOVE_THRESH:
        return True
    return False


def build_windows(z, spec, KP=4, FF=4, clips=None):
    tracks = z["tracks"].astype(np.float32); tracks_lo = z["tracks_low"].astype(np.float32)
    eef = z["eef"].astype(np.float32); eef_lo = z["eef_low"].astype(np.float32)
    joint = z["joint"].astype(np.float32); grip = z["grip"].astype(np.float32)
    L = tracks.shape[1]; ff = FF if spec["future"] else 0
    clips = range(len(tracks)) if clips is None else clips
    Xs, Ys, ci_l, t_l, con_l = [], [], [], [], []
    for ci in clips:
        tr0 = _mp_trace(eef[ci], 0); tr1 = _mp_trace(eef_lo[ci], 1)   # (L,5,2) each view
        oc0 = tracks[ci]; oc1 = tracks_lo[ci]
        for t in range(L - 1):                                       # 需 t+1 有 Δjoint
            widx = np.clip(np.arange(t - KP, t + ff + 1), 0, L - 1)   # edge-repeat pad
            feats = []
            if spec["flow"]:
                f0 = oc0[widx] - oc0[t].mean(0); f1 = oc1[widx] - oc1[t].mean(0)
                feats.append(np.concatenate([f0, f1], 1).reshape(-1))
            if spec["trace"]:
                feats.append(np.concatenate([tr0[widx], tr1[widx]], 1).reshape(-1))
            if spec["grip"]:
                feats.append(grip[ci][widx].reshape(-1))
            Xs.append(np.concatenate(feats).astype(np.float32))
            Ys.append(np.concatenate([joint[ci, t + 1, :6] - joint[ci, t, :6], [grip[ci, t]]]).astype(np.float32))
            ci_l.append(ci); t_l.append(t); con_l.append(CONTACT_FN(z, ci, t))
    meta = {"clip_idx": np.array(ci_l), "t": np.array(t_l), "contact": np.array(con_l, bool)}
    return np.stack(Xs), np.stack(Ys), meta
