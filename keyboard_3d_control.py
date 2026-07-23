"""世界系 3D keyboard 控制原语:脚本→eef3d轨迹+grip, REAL标定投影两视角。纯numpy,无GPU,可单测。"""
import numpy as np, json
from scipy.spatial.transform import Rotation

_CAL = {"high": "calib/rgb_cam_calib_can_REAL.json", "low": "calib/rgb_cam_calib_can_low_REAL.json"}
CROP = {"high": (60,60,390,390), "low": (0,0,640,480)}
# 世界系单位方向: coord0=x(桌面), coord1=y(桌面), coord2=z(高度). up=+z.
WORLD_DIRS = {"left":(-1,0,0),"right":(1,0,0),"fwd":(0,-1,0),"back":(0,1,0),"up":(0,0,1),"down":(0,0,-1)}
_AR = {"left":"<","right":">","fwd":"^f","back":"vb","up":"Uz","down":"Dz","gopen":"O","gclose":"C"}

def can_KT(view):
    r = json.load(open(_CAL[view])); R = Rotation.from_quat(r["quat_xyzw"]).as_matrix()
    t = np.asarray(r["t_world"]); T = np.eye(4); T[:3,:3] = R.T; T[:3,3] = -R.T @ t
    K = np.array([[r["f"],0,r["cx"]],[0,r["f"],r["cy"]],[0,0,1.0]])
    return T, K

def project_norm(p3, T, K, crop):
    pc = (T @ np.append(p3,1.0))[:3]; uv = K @ pc; uv = uv[:2]/uv[2]
    x,y,w,h = crop; return np.array([(uv[0]-x)/w, (uv[1]-y)/h], np.float32)

def script_eef3d(eef3d0, grip0, script, DELTA, GRIP_DELTA, REPEAT, F, BOX3D):
    base = eef3d0.mean(0); off = eef3d0 - base       # 刚体星座内偏移锁死
    c = base.astype(np.float64).copy(); g = float(grip0); cs=[]; gs=[]
    for cmd, n in script:
        if cmd in ("gopen","gclose"):
            step = GRIP_DELTA if cmd=="gopen" else -GRIP_DELTA
            for _ in range(n*REPEAT):
                g = float(np.clip(g+step, 0.0, 0.04)); cs.append(c.copy()); gs.append(g)
        else:
            d = np.asarray(WORLD_DIRS[cmd], np.float64) * DELTA
            for _ in range(n*REPEAT):
                c = c + d; c[0]=np.clip(c[0],BOX3D[0],BOX3D[1]); c[1]=np.clip(c[1],BOX3D[2],BOX3D[3]); c[2]=np.clip(c[2],BOX3D[4],BOX3D[5])
                cs.append(c.copy()); gs.append(g)
    cs = np.stack(cs); gs = np.asarray(gs, np.float32)
    traj = (cs[:,None] + off[None]).astype(np.float64)      # (S,3,3)
    H = len(traj)
    traj = np.concatenate([traj, np.repeat(traj[-1:],F,0)],0); gs = np.concatenate([gs, np.repeat(gs[-1:],F)])
    return traj, gs, H
# 注:渲染阶段(Task4 precompute)把 eef/grip 轨迹 hold-pad 到恰好 RENDER_H=48(运动 H≤48,末尾保持最后位姿);
#     script_eef3d 只产运动帧(+F 滑窗未来 pad),不负责补到 48。

def project_eef_dual(eef3d_traj):
    Th,Kh = can_KT("high"); Tl,Kl = can_KT("low"); T = len(eef3d_traj)
    efA = np.stack([[project_norm(eef3d_traj[t,k],Th,Kh,CROP["high"]) for k in range(3)] for t in range(T)]).astype(np.float32)
    efB = np.stack([[project_norm(eef3d_traj[t,k],Tl,Kl,CROP["low"]) for k in range(3)] for t in range(T)]).astype(np.float32)
    return efA, efB

GRASP_TH = 0.033
NEAR_TH = 0.174

def grasp_timeline(eef3d_traj, grip_traj, obj3d0):
    ec = eef3d_traj.mean(1); oc = obj3d0.mean(0)      # 物体初始质心
    near = np.linalg.norm(ec - oc[None], axis=1) < NEAR_TH
    grip_low = grip_traj < GRASP_TH
    # 状态机:enter when near+low, stay while low, exit when grip opens
    grasp = np.zeros(len(eef3d_traj), dtype=bool)
    grasping = False
    for t in range(len(eef3d_traj)):
        if grip_low[t] and near[t]:
            grasping = True
        elif not grip_low[t]:
            grasping = False
        grasp[t] = grasping
    return grasp

def object_track_dual(eef3d_traj, grip_traj, obj3d0, valid0):
    Th,Kh = can_KT("high"); Tl,Kl = can_KT("low"); T = len(eef3d_traj)
    grasp = grasp_timeline(eef3d_traj, grip_traj, obj3d0)
    ec = eef3d_traj.mean(1)
    obj3d = np.repeat(obj3d0[None], T, 0).astype(np.float64)   # (T,48,3)
    g0 = None
    for t in range(1, T):
        if grasp[t]:
            if g0 is None: g0 = t                                # 抓取起点:锁 offset
            obj3d[t] = obj3d[g0] + (ec[t] - ec[g0])[None]        # 平移刚体跟随
        else:
            g0 = None; obj3d[t] = obj3d[t-1]                     # 释放:停在原地
    objA = np.stack([[project_norm(obj3d[t,p],Th,Kh,CROP["high"]) for p in range(48)] for t in range(T)]).astype(np.float32)
    objB = np.stack([[project_norm(obj3d[t,p],Tl,Kl,CROP["low"]) for p in range(48)] for t in range(T)]).astype(np.float32)
    return objA, objB, grasp
