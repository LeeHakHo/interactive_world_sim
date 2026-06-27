"""Keyboard 三列对比 (GT | IWS-naive co-train | ours object-flow).

IWS-naive = hyeonhoo 的 H+R co-train stage-2 ckpt (checkpoints/epoch=3-step=250000.ckpt):
  2 view (camera_0=cam_high, camera_1=cam_right_wrist) + latent 32ch + 7D eef action.
  load 配方见 memory project_stage1_hr_checkpoint / spec 2026-06-27.

本文件先实现 Task1+2:load IWS WM + clip<->parquet world-action 对齐 (图像 MSE) + 2-view replay,
眼检 IWS replay 是否跟 GT 一致。ours 列与 keyboard 合成在后续 task 接入。
"""
import os, glob, importlib.util
import numpy as np, torch, cv2
from omegaconf import OmegaConf
from einops import rearrange

OmegaConf.register_new_resolver("eval", lambda e: eval(e, {"np": np}), replace=True)
OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x), replace=True)

from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import LatentWorldModel
from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm

DEVICE = "cuda"
IMG = 128
CKPT = "checkpoints/epoch=3-step=250000.ckpt"
NPZ = "outputs/flow_render_dataset_v3_grip/clips_robot.npz"
V3_CROP = (190, 225, 210, 205)            # x,y,w,h  cam_high workspace (ours 同款)
OUT = "outputs/cross_embodiment_wm/keyboard_3way_iws"
# action normalizer 区间 (sanity 用,校验构造的 action 落在区间内)
NORM_MIN = np.array([-0.274, -0.120, 0.099, -0.086, 0.102, 1.549, -0.001])
NORM_MAX = np.array([0.199, 0.305, 0.101, 0.037, 0.281, 1.658, 0.042])

# ── 借 eval_stage1_hr_alignment 的 build_cfg / normalizer helper ─────────────
_spec = importlib.util.spec_from_file_location("_hr", "scripts/eval/eval_stage1_hr_alignment.py")
_hr = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_hr)


def load_iws_wm(ckpt=CKPT, device=DEVICE):
    """唯一能 load 这个裸 ckpt 的配方 (见 memory project_stage1_hr_checkpoint)."""
    cfg = _hr.build_cfg()
    cfg.training_stage = 2
    cfg.dynamo_ssl.encoder_backbone = "conv2d"     # ckpt 是 plain conv encoder.0..8
    model = LatentWorldModel(cfg)
    for k in ["camera_0_color", "camera_1_color"]:
        model.normalizer[k] = _hr.get_image_range_normalizer()
        model.normalizer[f"{k}_mask"] = _hr.get_image_range_normalizer()
    model.normalizer["action"] = _hr.SingleFieldLinearNormalizer.create_manual(
        scale=np.ones(7, np.float32), offset=np.zeros(7, np.float32),
        input_stats_dict={s: np.zeros(7, np.float32) if s in ("min", "mean")
                          else np.ones(7, np.float32) for s in ("min", "max", "mean", "std")})
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    res = model.load_state_dict(sd, strict=False)
    bad = [k for k in res.missing_keys if "validation_fvd" not in k] + \
          [k for k in res.unexpected_keys if "validation_fvd" not in k]
    assert not bad, f"load not clean: {bad[:6]}"
    return model.eval().to(device)


# ── clip <-> parquet world-action 对齐 (图像 MSE) ───────────────────────────
def _crop128(img):
    x, y, w, h = V3_CROP
    return cv2.resize(img[y:y + h, x:x + w], (IMG, IMG))


def decode_cam(robot_num, cam="cam_high"):
    """解码 v3 robot 某相机视频 (robot_num=1..18). cam_high 用 V3_CROP→128; wrist 整图 resize→128."""
    import av
    d = f"human_play_data/play_robot_v3_{robot_num}_eef"
    key = "cam_high" if cam == "cam_high" else "cam_right_wrist"
    path = f"{d}/videos/chunk-000/observation.images.{key}/episode_000000.mp4"
    cont = av.open(path); out = []
    for fr in cont.decode(video=0):
        rgb = fr.to_ndarray(format="rgb24")
        out.append(_crop128(rgb) if cam == "cam_high" else cv2.resize(rgb, (IMG, IMG)))
    return np.stack(out)


def locate_clip_start(clip_frame0_u8, vid_video_u8):
    diff = vid_video_u8.astype(np.int32) - clip_frame0_u8.astype(np.int32)[None]
    mse = (diff * diff).reshape(len(vid_video_u8), -1).mean(1)
    return int(mse.argmin())


def clip_world_action(npz, i, cache):
    """npz clip i -> 同帧 (L,7) world action (pos3+euler3+grip1) + wrist clip + start."""
    import pandas as pd
    vid = int(npz["vid"][i])
    rn = vid - 100 + 1            # vid 100..117 -> play_robot_v3_1..18
    if vid not in cache:
        d = f"human_play_data/play_robot_v3_{rn}_eef"
        df = pd.read_parquet(sorted(glob.glob(f"{d}/data/**/*.parquet", recursive=True))[0])
        # IWS 训练用 action_right_* 列(z=0.1 匹配 normalizer;obs_right z=0.08 不匹配)
        pos = np.stack(df["action_right_ee_position"]).astype(np.float32)
        eul = np.stack(df["action_right_ee_euler_xyz"]).astype(np.float32)
        grp = np.asarray(df["action_right_gripper"]).astype(np.float32)[:, None]
        act_all = np.concatenate([pos, eul, grp], 1)
        cache[vid] = (decode_cam(rn, "cam_high"), decode_cam(rn, "wrist"), act_all)
    vid_high, vid_wrist, act_all = cache[vid]
    s = locate_clip_start(npz["frames"][i][0], vid_high)
    L = npz["frames"].shape[1]
    return act_all[s:s + L], vid_wrist[s:s + L], s


# ── IWS 2-view rollout ──────────────────────────────────────────────────────
@torch.no_grad()
def iws_rollout(model, cam_high0_u8, wrist0_u8, actions_raw, device=DEVICE):
    """init = frame0 (cam_high + wrist), actions_raw (T,7) world. -> cam_high pred (T,128,128,3)."""
    def prep(u8):
        return torch.from_numpy(u8).float().div(255).permute(2, 0, 1)[None, None].to(device)
    cam0 = model.normalizer["camera_0_color"].normalize(prep(cam_high0_u8))     # (1,1,3,H,W)
    cam1 = model.normalizer["camera_1_color"].normalize(prep(wrist0_u8))
    obs = torch.cat([cam0, cam1], dim=2)                                        # (1,1,6,H,W)
    z0 = model.encoder_forward(rearrange(obs, "b t c h w -> (b t) c h w"))
    z0 = rearrange(z0, "(b t) c h w -> b t c h w", b=1)                         # (1,1,32,Hl,Wl)
    act = model.normalizer["action"].normalize(
        torch.from_numpy(actions_raw).float()[None].to(device))                # (1,T,7)
    z_pred = model.dynamics_forward(z0, act)                                    # (1,T-1,32,Hl,Wl)
    z_all = torch.cat([z0, z_pred], dim=1)[0]                                   # (T,32,Hl,Wl)
    px = render_img_cm(model, z_all, IMG, model.normalizer, num_views=2, batch_size=50)  # (T,6,H,W)
    cam_high = px[:, :3].clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    return (cam_high * 255).astype(np.uint8)


def _sanity(seq_idx=0, wrist_zero=False):
    z = np.load(NPZ)
    gt = z["frames"][seq_idx]                                  # (L,128,128,3) cam_high GT
    act, wrist_clip, s = clip_world_action(z, seq_idx, {})
    print(f"[seq{seq_idx}] vid={int(z['vid'][seq_idx])} start={s} L={len(gt)}")
    print(f"  action min/max per dim:\n   min {act.min(0).round(3)}\n   max {act.max(0).round(3)}")
    inside = (act.min(0) >= NORM_MIN - 1e-2).all() and (act.max(0) <= NORM_MAX + 1e-2).all()
    print(f"  action 落在 normalizer 区间内: {inside}")
    w0 = np.zeros_like(gt[0]) if wrist_zero else wrist_clip[0]
    model = load_iws_wm()
    pred = iws_rollout(model, gt[0], w0, act)
    os.makedirs(f"{OUT}/diag", exist_ok=True)
    K = min(8, len(pred))
    idxs = np.linspace(0, len(pred) - 1, K).astype(int)
    rows = [np.concatenate([gt[t] for t in idxs], 1),
            np.concatenate([pred[t] for t in idxs], 1)]
    img = cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR)
    p = f"{OUT}/diag/task1_replay_seq{seq_idx}{'_wz' if wrist_zero else ''}.png"
    cv2.imwrite(p, img); print(f"  saved {p}  (上 GT | 下 IWS replay, cam_high)")


if __name__ == "__main__":
    _sanity(int(os.environ.get("SEQ", 0)), wrist_zero=bool(int(os.environ.get("WZ", "0"))))
