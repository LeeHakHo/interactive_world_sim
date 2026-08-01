"""IDM 闭环评估: IDM出Δjoint→从GT joint_0积分→FK→3点eef→mp tokens→前向②(mp_r_all)→object-flow,
对GT比(自由/接触分开)+ eef-recon + 标准protocol overlay gif(列: GT | GT-action天花板 | IDM)。
env: phantom(torch+pinocchio)。"""
import os, sys, numpy as np, torch, imageio
sys.path.insert(0, "."); os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ["ACTION"] = "mp"
import exp_scel_dualview_wm as W
import idm_data as D, idm_model as M, idm_fk as FK
from viz_combined import build_flow_cols, compose_frame
setattr(sys.modules["__main__"], "DualLWC", W.DualLWC)

ARM = os.environ.get("ARM", "A1")
OUT = os.environ.get("OUT_DIR", f"outputs/idm_derisk/{ARM}")
K, P = W.K, 48
z = np.load("outputs/flow_render_dataset_can_dual/clips_robot_retrack.npz")
ck = torch.load(f"{OUT}/idm.pt", map_location="cpu", weights_only=False)
XF = ck.get("model") == "xf"
if XF:
    m = M.IDMTransformer(C=ck["C"], N=ck["N"], W=ck["W"], ptype=ck["ptype"])
else:
    m = M.IDM(din=ck["din"], hidden=ck.get("hid", 512), n_layers=ck.get("nlayer", 3))
m.load_state_dict(ck["state"]); m.eval()
spec, KP, FF = ck["spec"], ck["KP"], ck["FF"]
xm, xs, ym, ys = ck["x_mean"], ck["x_std"], ck["y_mean"], ck["y_std"]
wm2 = torch.load("outputs/cross_embodiment_wm/epsplit_L48/mp_r_all/wm_dual.pt", map_location="cpu", weights_only=False).eval()
tr, ho = D.episode_split(z)
SEQS = [int(x) for x in os.environ.get("SEQS", ",".join(map(str, ho[:4]))).split(",")]

# 数据集全局 grip max(对齐 load_action_tokens("mp") 的 gripslot 归一化口径)
_gmaxA = float(np.nanmax(np.linalg.norm(z["eef"][:, :, 1] - z["eef"][:, :, 2], axis=-1)))
_gmaxB = float(np.nanmax(np.linalg.norm(z["eef_low"][:, :, 1] - z["eef_low"][:, :, 2], axis=-1)))


def mp_tokens(eef_hi, eef_lo):   # (T,3,2)x2 -> (T,4,2)x2, 口径同训练
    def slot(ef, gmax):
        g = np.linalg.norm(ef[:, 1] - ef[:, 2], axis=-1) / (gmax + 1e-6)
        return np.stack([g, np.zeros_like(g)], -1)[:, None, :]
    a = np.concatenate([np.nan_to_num(eef_hi.astype(np.float32), nan=0.5), slot(eef_hi, _gmaxA)], 1)
    b = np.concatenate([np.nan_to_num(eef_lo.astype(np.float32), nan=0.5), slot(eef_lo, _gmaxB)], 1)
    return a.astype(np.float32), b.astype(np.float32)


def idm_joint_traj(ci):
    if XF:
        Xt, _, _, _ = D.build_windows_tokens(z, spec, KP, FF, clips=[ci])
        with torch.no_grad():
            Yp = m(torch.tensor((Xt - xm) / xs).float()).numpy() * ys + ym
    else:
        X, _, _ = D.build_windows(z, spec, KP, FF, clips=[ci])
        with torch.no_grad():
            Yp = m(torch.tensor((X - xm) / xs).float()).numpy() * ys + ym   # (L-1,7)
    dj, gp = Yp[:, :6], Yp[:, 6]
    jt = np.zeros((len(dj) + 1, 6)); jt[0] = z["joint"][ci, 0, :6]       # 帧0锚GT
    if D.YMODE == "abs":
        jt[1:] = dj                                                      # 直接取预测的绝对joint(无积分漂移)
    else:
        for t in range(len(dj)):
            jt[t + 1] = jt[t] + dj[t]                                    # Δ积分
    grip = np.concatenate([gp, gp[-1:]])
    return jt, grip


def rollout(ci, efA, efB):   # 前向② 预测物体(cam_high, 48,48,2)
    trg = [np.nan_to_num(z["tracks"][ci].astype(np.float32)), np.nan_to_num(z["tracks_low"][ci].astype(np.float32))]
    trD = np.concatenate([trg[0][:K], trg[1][:K]], 1)[None]
    with torch.no_grad():
        pr = W.rollout_dual(wm2, torch.from_numpy(trD).float(), torch.from_numpy(efA[None]).float(),
                            torch.from_numpy(efB[None]).float(), 48 - K)[0].numpy()
    return np.concatenate([trg[0][:K], pr[:, :P]], 0)


summ = [f"IDM 闭环 ARM={ARM} (eef-recon px@128 / object-flow-recon px, 自由/接触分开; ceil=GT-action)\n"]
gif_cols = {"render": [], "flow": [], "titles": ["GT", "GT-action(ceil)", "IDM"], "errs": [None, [], []]}
for ci in SEQS:
    jt, grip = idm_joint_traj(ci)
    eh, el = FK.fk_eef2d(jt, grip)                              # 预测 eef (48,3,2)x2
    gh = np.nan_to_num(z["eef"][ci].astype(np.float32)); gl = np.nan_to_num(z["eef_low"][ci].astype(np.float32))
    _, _, meta = D.build_windows(z, spec, KP, FF, clips=[ci])
    con = np.append(meta["contact"], meta["contact"][-1])       # (48,)
    okp = np.all(np.abs(gh) < 3, axis=(-1, -2))
    eef_err = np.linalg.norm(eh - gh, axis=-1).mean(-1) * 128   # (48,)
    fe = np.nanmean(eef_err[okp & ~con]); ce = np.nanmean(eef_err[okp & con])
    # object-flow: IDM action vs GT-action(ceil)
    efA_i, efB_i = mp_tokens(eh, el); efA_g, efB_g = mp_tokens(gh, gl)
    gt_obj = np.nan_to_num(z["tracks"][ci].astype(np.float32))
    pred_i = rollout(ci, efA_i, efB_i); pred_c = rollout(ci, efA_g, efB_g)
    def fsplit(pred):
        e = np.linalg.norm(pred - gt_obj, axis=-1).mean(-1) * 128
        return np.nanmean(e[K:][~con[K:]]), np.nanmean(e[K:][con[K:]])
    fi_free, fi_con = fsplit(pred_i); fc_free, fc_con = fsplit(pred_c)
    summ.append(f"seq{ci}: eef-recon 自由{fe:.1f} 接触{ce:.1f} | flow-recon IDM 自由{fi_free:.1f} 接触{fi_con:.1f} | ceil 自由{fc_free:.1f} 接触{fc_con:.1f}")
    # gif: 3列, Rendered行=GT帧(参考), Flow行=绿GT红pred黄eef
    Tp = min(48, len(z["frames"][ci]))
    gt128 = z["frames"][ci][:Tp].astype(np.uint8)
    flow3 = build_flow_cols(gt128, gt_obj[:Tp], [None, pred_c[:Tp], pred_i[:Tp]], gh[:Tp])   # (3,Tp,128,128,3)
    render3 = np.stack([gt128, gt128, gt128])
    gif_cols["render"].append(render3); gif_cols["flow"].append(flow3)
    gif_cols["errs"][1].append(fi_con); gif_cols["errs"][2].append(fc_con)

open(f"{OUT}/eval_summary.txt", "w").write("\n".join(summ) + "\n")
print("\n".join(summ), flush=True)

# 拼所有 seq(纵向堆)成一个 overlay gif
if gif_cols["render"]:
    Tp = min(r.shape[1] for r in gif_cols["render"])
    frames = []
    for t in range(Tp):
        rows = []
        for s in range(len(gif_cols["render"])):
            titles = gif_cols["titles"]
            errs = [None, gif_cols["errs"][1][s], gif_cols["errs"][2][s]]
            fr = compose_frame([gif_cols["render"][s][c][t] for c in range(3)],
                               [gif_cols["flow"][s][c][t] for c in range(3)],
                               titles, errs, t, caption=f"IDM {ARM} seq{SEQS[s]} | 绿GT红pred黄eef | sub=接触段flow-recon px")
            rows.append(np.array(fr))
        frames.append(np.concatenate(rows, 0))
    gp = f"{OUT}/eval_overlay.gif"; imageio.mimsave(gp, frames, fps=6)
    print(f"[gif] {gp}", flush=True)
print("DONE", flush=True)
