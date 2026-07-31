"""几何式接触检测(2026-07-31 spec §5)。attachment=夹爪闭合∧夹爪-物近∧速度相关; contact_pt=夹爪-物中点。
学习式(自举标签)后续另加,同接口。"""
import numpy as np

GRIP_MAX = 0.04
D_NEAR = 0.05          # 3D 夹爪-物体"近"的尺度(米)
EPS = 1e-6


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def detect_contact_geometric(tracks3d, eef3d, grip):
    tracks3d = np.asarray(tracks3d, np.float64); eef3d = np.asarray(eef3d, np.float64)
    grip = np.asarray(grip, np.float64)
    obj = np.nanmean(tracks3d, axis=1)                        # (T,3) 物体质心
    gmid = (eef3d[:, 1] + eef3d[:, 2]) / 2                    # (T,3) 夹爪中点
    closed = 1.0 - np.clip(grip / GRIP_MAX, 0, 1)            # 夹爪闭合: grip越小越闭
    dist = np.linalg.norm(gmid - obj, axis=1)               # 夹爪-物体近
    near = _sig((D_NEAR - dist) / (0.5 * D_NEAR))
    vo = np.diff(obj, axis=0, prepend=obj[:1]); vg = np.diff(gmid, axis=0, prepend=gmid[:1])
    mo = np.linalg.norm(vo, axis=1); mg = np.linalg.norm(vg, axis=1)
    cos = np.clip((vo * vg).sum(1) / (mo * mg + EPS), 0, 1)
    moving = (mo > 1e-4) & (mg > 1e-4)                        # ★按各自速度幅度门控(非乘积, 慢速物体乘积会假性<阈值)
    corr = np.where(moving, cos, 0.5)                        # 速度相关(两者都基本不动时中性0.5)
    attachment = closed * near * corr
    contact_pt3d = (gmid + obj) / 2
    return attachment.astype(np.float32), contact_pt3d.astype(np.float32)


D_NEAR_2D = 0.12       # crop-norm 2D "近"尺度(帧宽的~12%)


def detect_contact_2d(tracks2d, eef2d, grip):
    """2D per-view 版(human 无 3D 物体轨迹)。tracks2d (T,K,2) crop-norm, eef2d (T,3,2), grip (T,)
    -> (attachment (T,), contact_pt2d (T,2))。逻辑同 3D 版, 在图像 2D 里算。"""
    tracks2d = np.asarray(tracks2d, np.float64); eef2d = np.asarray(eef2d, np.float64)
    grip = np.asarray(grip, np.float64)
    obj = np.nanmean(tracks2d, axis=1)                       # (T,2) 物体 2D 质心
    gmid = (eef2d[:, 1] + eef2d[:, 2]) / 2                   # (T,2) 夹爪 2D 中点
    closed = 1.0 - np.clip(grip / GRIP_MAX, 0, 1)
    dist = np.linalg.norm(gmid - obj, axis=1)
    near = _sig((D_NEAR_2D - dist) / (0.5 * D_NEAR_2D))
    vo = np.diff(obj, axis=0, prepend=obj[:1]); vg = np.diff(gmid, axis=0, prepend=gmid[:1])
    mo = np.linalg.norm(vo, axis=1); mg = np.linalg.norm(vg, axis=1)
    cos = np.clip((vo * vg).sum(1) / (mo * mg + EPS), 0, 1)
    moving = (mo > 5e-4) & (mg > 5e-4)                       # crop-norm 速度门控
    corr = np.where(moving, cos, 0.5)
    attachment = closed * near * corr
    contact_pt2d = (gmid + obj) / 2
    return attachment.astype(np.float32), contact_pt2d.astype(np.float32)
