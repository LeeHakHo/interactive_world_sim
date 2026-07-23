import numpy as np, pytest
from keyboard_3d_control import can_KT, project_norm, project_eef_dual, script_eef3d, WORLD_DIRS
CROP = {"high": (60,60,390,390), "low": (0,0,640,480)}

def _clip():
    z = np.load("outputs/flow_render_dataset_can_dual/clips_robot.npz")
    return z

def test_reproj_matches_stored():
    z = _clip(); si = 332
    e3 = z["eef3d"][si].astype(np.float64)      # (48,3,3)
    for view, ek in [("high","eef"), ("low","eef_low")]:
        T, K = can_KT(view); e2 = z[ek][si].astype(np.float32)
        errs = []
        for t in range(48):
            for k in range(3):
                if np.all(np.isfinite(e2[t,k])):
                    uv = project_norm(e3[t,k], T, K, CROP[view])
                    errs.append(np.linalg.norm(uv - e2[t,k]))
        assert np.mean(errs)*128 < 0.2, f"{view} reproj {np.mean(errs)*128:.3f}px"

def _triangulate(uvA, uvB):
    """两视角 crop-norm 2D -> 世界 3D (pixel-space DLT, float32 下稳). 验'纯z投影三角化回来纯竖直'。"""
    from keyboard_3d_control import can_KT, CROP
    rows = []
    for uv, view in [(uvA,"high"), (uvB,"low")]:
        T,K = can_KT(view); P = K @ T[:3]            # 3x4 world->pixel
        x,y,w,h = CROP[view]; px = np.array([uv[0]*w+x, uv[1]*h+y])  # norm->pixel
        rows.append(px[0]*P[2]-P[0]); rows.append(px[1]*P[2]-P[1])
    _,_,Vt = np.linalg.svd(np.stack(rows).astype(np.float64)); X = Vt[-1]; return X[:3]/X[3]

def test_pure_z_triangulates_to_vertical():
    z = _clip(); si = 332
    e30 = z["eef3d"][si,0].astype(np.float64)   # (3,3)
    # 现实幅度 lift (~0.22m): up 6键*REPEAT3*DELTA0.012
    z0 = e30[:,2].mean()
    traj, grip, H = script_eef3d(e30, 0.029, [("up",6)], DELTA=0.012, GRIP_DELTA=0.005, REPEAT=3, F=0,
                                 BOX3D=(-1,1,-1,1,0.0,z0+0.28))
    efA, efB = project_eef_dual(traj)           # (H,3,2)
    # 三角化 eef 质心 首帧 vs 末帧 -> 世界位移应为纯 +z
    w0 = _triangulate(efA[0].mean(0), efB[0].mean(0))
    wT = _triangulate(efA[-1].mean(0), efB[-1].mean(0))
    d = wT - w0
    assert d[2] > 0.15, f"世界z位移应显著为正 {d[2]:.3f}"
    assert abs(d[0]) < 0.02 and abs(d[1]) < 0.02, f"世界水平位移应≈0 (dx{d[0]:.3f} dy{d[1]:.3f})"
    # 且两视角图像运动以竖直为主(透视水平分量小)
    dyA = abs(efA[-1,:,1].mean()-efA[0,:,1].mean()); dxA = abs(efA[-1,:,0].mean()-efA[0,:,0].mean())
    assert dyA > 2*dxA, f"cam_high 应竖直为主 dy{dyA*128:.1f} dx{dxA*128:.1f}px"

def test_grasp_and_object_follow():
    from keyboard_3d_control import object_track_dual, script_eef3d
    z = _clip(); si = 332
    e30 = z["eef3d"][si,0].astype(np.float64)
    obj3d0 = z["tracks3d"][si,0].astype(np.float64)      # (48,3)
    valid0 = z["tracks3d_valid"][si,0]
    # 抓紧(grip 低)后抬起
    traj, grip, H = script_eef3d(e30, 0.029, [("up",6)], DELTA=0.03, GRIP_DELTA=0.01, REPEAT=4, F=0, BOX3D=(-1,1,-1,1,0,1))
    objA, objB, grasp = object_track_dual(traj, grip, obj3d0, valid0)
    assert grasp.mean() > 0.8, "grip低+近物体应判定抓取"
    # 抓取抬起 -> cam_low 物体质心竖直上移(跟随 eef)
    dy_low = objB[-1,valid0,1].mean() - objB[0,valid0,1].mean()
    assert abs(dy_low) > 0.03, f"抓取抬起物体应在cam_low竖直动 {dy_low:.3f}"

def test_fallback_when_under_follow():
    from exp_scel_keyboard_3d import blend_predtr
    H = 24
    # ② 预测物体几乎不动(under-follow), eef 抬起明显
    pA = np.zeros((H,48,2), np.float32); pB = np.zeros((H,48,2), np.float32)
    objA = np.zeros((H,48,2), np.float32)
    objB = np.tile(np.linspace(0.6,0.4,H)[:,None,None],(1,48,2)).astype(np.float32)  # 刚体跟随:cam_low 上移
    grasp = np.ones(H, bool)
    efA = np.zeros((H,3,2),np.float32); efB = np.tile(np.linspace(0.6,0.4,H)[:,None,None],(1,3,2)).astype(np.float32)
    tA,tB,fb = blend_predtr(pA,pB,objA,objB,grasp,efA,efB)
    assert fb.mean() > 0.5, "抓取相②不跟随应触发兜底"
    assert abs(tB[-1,:,1].mean() - 0.4) < 0.05, "兜底应采用刚体投影(cam_low上移)"
