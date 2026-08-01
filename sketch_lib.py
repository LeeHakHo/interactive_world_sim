"""交互草图纯函数库(2026-07-31 spec)。3D世界系投影 + 逐通道构建。复用已有标定/flow_cond。"""
import numpy as np, cv2, torch
from keyboard_3d_control import CROP, can_KT, project_norm

IMG = 128; POOL = 8


def project_world_to_view(pts3d, view):
    """(...,3) 世界系 -> (...,2) crop-norm 2D(cam view)。复用 Task1-验证过的标定链。"""
    T, K = can_KT(view)
    flat = np.asarray(pts3d, np.float64).reshape(-1, 3)
    out = np.stack([project_norm(p, T, K, CROP[view]) for p in flat])
    return out.reshape(np.asarray(pts3d).shape[:-1] + (2,)).astype(np.float32)


def object_flow_channels(tr3d0, tr3dt, ef0_2d, eft_2d, vis_t, view):
    """3D物体track投影到view 2D,再走现有 flow_cond -> (3,128,128)[dx,dy,footprint]。
    DIT 延迟导入(而非模块级):exp_scel_dualview_dit -> exp_scel_latent_renderer -> diffusers
    在当前 phantom env 里因 torch 2.1.0+cu121 缺 xpu/device_mesh 而 diffusers==0.37.1 无法导入
    (与本任务无关的预先存在的env版本不兼容,test_dualview_formal.py 等既有测试同样受影响)。
    延迟导入让不依赖 DIT 的另外两个函数不被这个环境问题拖累。"""
    import exp_scel_dualview_dit as DIT
    p0 = project_world_to_view(tr3d0, view); pt = project_world_to_view(tr3dt, view)
    return DIT.flow_cond(p0, pt, ef0_2d, eft_2d, vis_t)


def pool16(x):
    return torch.nn.functional.avg_pool2d(torch.from_numpy(x[None]).float(), POOL)[0].numpy()


def agent_trace_channel(eef3d_hist, view, trail=8):
    """eef3d_hist (H,3,3) 历史(含当前) -> (1,128,128) 中点投影渐亮拖尾。"""
    img = np.zeros((IMG, IMG), np.float32)
    hist = np.asarray(eef3d_hist, np.float64)[-trail:]
    mids = (hist[:, 1] + hist[:, 2]) / 2                      # (h,3) 两指尖中点
    p = project_world_to_view(mids, view)                     # (h,2)
    n = len(p)
    for i, (x, y) in enumerate(p):
        w = (i + 1) / n                                       # 越近越亮
        xi, yi = int(np.clip(x * IMG, 0, IMG - 1)), int(np.clip(y * IMG, 0, IMG - 1))
        cv2.circle(img, (xi, yi), 2, float(w), -1)
    return img[None]


GRIP_MAX = 0.04


def grip_channel(grip_scalar):
    """grip 标量 -> (1,128,128) 整层广播([0,1]=grip/GRIP_MAX)。"""
    return np.full((1, IMG, IMG), np.clip(grip_scalar / GRIP_MAX, 0, 1), np.float32)


def contact_channels(attachment, contact_pt3d, view, sigma=4.0):
    """attachment 标量广播(第0层) + 接触点3D投影2D高斯splat(第1层) -> (2,128,128)。"""
    attach = np.full((1, IMG, IMG), float(np.clip(attachment, 0, 1)), np.float32)
    splat = np.zeros((IMG, IMG), np.float32)
    if np.all(np.isfinite(contact_pt3d)):
        x, y = project_world_to_view(np.asarray(contact_pt3d)[None], view)[0]
        cx, cy = int(np.clip(x * IMG, 0, IMG - 1)), int(np.clip(y * IMG, 0, IMG - 1))
        yy, xx = np.ogrid[:IMG, :IMG]
        splat = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)
    return np.concatenate([attach, splat[None]], 0)


def object_flow_2d(tr2d0, tr2dt, ef0_2d, eft_2d, vis_t):
    """2D per-view object flow(human 无 tracks3d, 走两域都有的 2D tracks)-> (3,128,128)[dx,dy,footprint]。"""
    import exp_scel_dualview_dit as DIT
    return DIT.flow_cond(tr2d0, tr2dt, ef0_2d, eft_2d, vis_t)


def object_dz_scalar(tr3d0, tr3dt, valid):
    """物体世界z位移均值(米, 刚性→单标量)。valid (K,)两帧都有效。无有效点->0。"""
    tr3d0 = np.asarray(tr3d0, np.float64); tr3dt = np.asarray(tr3dt, np.float64)
    dz = (tr3dt[:, 2] - tr3d0[:, 2])[np.asarray(valid, bool)]
    dz = dz[np.isfinite(dz)]
    return float(np.mean(dz)) if len(dz) else 0.0


def world_dz_channel(tr2d_t, dz_scalar, vis_t=None, scale=5.0):
    """世界系 z 位移通道(把"抬起"从图像2D flow分离)。dz_scalar=物体世界z位移(米),
    splat 到当前帧每个可见物体2D点 tr2d_t (K,2) -> (1,128,128) signed。"""
    img = np.zeros((IMG, IMG), np.float32)
    tr2d_t = np.asarray(tr2d_t, np.float64)
    val = float(np.clip(dz_scalar * scale, -1, 1))
    for i in range(len(tr2d_t)):
        if vis_t is not None and vis_t[i] < 0.5: continue
        x, y = int(np.clip(tr2d_t[i, 0] * IMG, 0, IMG - 1)), int(np.clip(tr2d_t[i, 1] * IMG, 0, IMG - 1))
        cv2.circle(img, (x, y), 3, val, -1)
    return img[None]


def contact_channels_2d(attachment, contact_pt2d, sigma=4.0):
    """attachment 广播 + 2D接触点(crop-norm, 已在视角内)高斯splat -> (2,128,128)。"""
    attach = np.full((1, IMG, IMG), float(np.clip(attachment, 0, 1)), np.float32)
    splat = np.zeros((IMG, IMG), np.float32)
    if np.all(np.isfinite(contact_pt2d)):
        cx, cy = int(np.clip(contact_pt2d[0] * IMG, 0, IMG - 1)), int(np.clip(contact_pt2d[1] * IMG, 0, IMG - 1))
        yy, xx = np.ogrid[:IMG, :IMG]
        splat = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)
    return np.concatenate([attach, splat[None]], 0)
