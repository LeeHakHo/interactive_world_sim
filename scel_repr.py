"""SCEL structured representation helpers (pure functions, no IO/side-effects).

object node  = [cx, cy, scale]            质心 + RMS 散布半径（无朝向：2D 近方形 cube 旋转不可靠）
grasp frame  = [gx, gy, cosφ, sinφ, w]    base 位置 + base→指尖中点朝向 + 两指开合
agent-frame  = object 在 grasp frame 坐标系下的可逆变换（rollout 时 grasp 已知 -> 逆变换回世界）
"""
import numpy as np

EPS = 1e-8


def object_node(tracks):
    """tracks (..., P, 2) -> (..., 3) = [cx, cy, scale]. scale = RMS radius about centroid."""
    c = tracks.mean(axis=-2)
    d = tracks - c[..., None, :]
    scale = np.sqrt((d ** 2).sum(-1).mean(-1) + EPS)
    return np.concatenate([c, scale[..., None]], axis=-1)


def grasp_frame(eef):
    """eef (...,3,2) = [base, +finger, -finger] -> (...,5) = [gx,gy,cosφ,sinφ,w].
    朝向 = base -> 指尖中点方向（退化时 fallback +x）；w = 两指距离。"""
    base = eef[..., 0, :]
    f1, f2 = eef[..., 1, :], eef[..., 2, :]
    mid = 0.5 * (f1 + f2)
    d = mid - base
    n = np.sqrt((d ** 2).sum(-1) + EPS)[..., None]
    dirv = np.where(n > 1e-4, d / n, np.broadcast_to(np.array([1.0, 0.0], np.float32), d.shape))
    w = np.sqrt(((f1 - f2) ** 2).sum(-1) + EPS)[..., None]
    return np.concatenate([base, dirv, w], axis=-1)


def _rot(gf):
    """gf (...,>=4) -> cos,sin."""
    return gf[..., 2], gf[..., 3]


def to_agent_frame(p, gf):
    """p (...,P,2) world points, gf (...,5) grasp frame -> p_rel (...,P,2) = R(φ)^T (p-g)."""
    g = gf[..., None, :2]
    c, s = _rot(gf); c = c[..., None, None]; s = s[..., None, None]
    d = p - g
    x, y = d[..., 0:1], d[..., 1:2]
    xr = c * x + s * y
    yr = -s * x + c * y
    return np.concatenate([xr, yr], axis=-1)


def from_agent_frame(p_rel, gf):
    """inverse of to_agent_frame: p = R(φ) p_rel + g."""
    g = gf[..., None, :2]
    c, s = _rot(gf); c = c[..., None, None]; s = s[..., None, None]
    x, y = p_rel[..., 0:1], p_rel[..., 1:2]
    xw = c * x - s * y
    yw = s * x + c * y
    return np.concatenate([xw, yw], axis=-1) + g


def fit_vel_stats(vel, dom):
    """vel (N,...,D); dom (N,) in {0,1}. -> {d: (mean(D,), std(D,))} per domain."""
    vel = np.asarray(vel); dom = np.asarray(dom)
    D = vel.shape[-1]
    flat = vel.reshape(-1, D)
    assert flat.shape[0] % len(dom) == 0, f"vel leading dim {flat.shape[0]} not divisible by dom len {len(dom)}"
    dflat = np.repeat(dom, flat.shape[0] // len(dom)) if flat.shape[0] != len(dom) else dom
    stats = {}
    for d in (0, 1):
        sel = flat[dflat == d]
        if len(sel) == 0:
            stats[d] = (np.zeros(D, np.float32), np.ones(D, np.float32))
        else:
            stats[d] = (sel.mean(0).astype(np.float32), (sel.std(0) + 1e-6).astype(np.float32))
    return stats


def _apply(vel, dom, stats, fwd):
    vel = np.asarray(vel).copy(); dom = np.asarray(dom)
    out = vel.reshape(len(dom), -1, vel.shape[-1])
    for d in (0, 1):
        m, s = stats[d]
        idx = dom == d
        out[idx] = (out[idx] - m) / s if fwd else out[idx] * s + m
    return out.reshape(vel.shape)


def normalize_vel(vel, dom, stats):   return _apply(vel, dom, stats, True)
def denormalize_vel(vel, dom, stats): return _apply(vel, dom, stats, False)
