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
