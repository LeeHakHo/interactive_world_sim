"""组装新 clips(2026-07-27): 把 GDINO re-track 的干净 tracks 合进 clips, 排除 retrack_ok=False。
产 clips_{human_L24,robot}_retrack.npz = 原 clips 全字段, 但 tracks/tracks_low/vis/vis_low 换成 re-track,
+ retrack_ok(N,)=两视角都成功才 True(下游训练/诊断用它排除失败 clip)。
env: 无。iws。
"""
import os, numpy as np
BASE = os.environ.get("BASE", "outputs/flow_render_dataset_can_dual")
RT = f"{BASE}/retrack"
HCLIP = os.environ.get("HCLIP", "clips_human_L24")   # human clip basename(含clips_前缀); L48重建=clips_human_L48
DOMAINS = os.environ.get("DOMAINS", "human,robot").split(",")   # 只重建human时=human


def assemble(clip_name, out_name):
    z = dict(np.load(f"{BASE}/{clip_name}.npz"))
    rh = np.load(f"{RT}/{out_name}_high.npz"); rl = np.load(f"{RT}/{out_name}_low.npz")
    z["tracks"] = rh["tracks"].astype(np.float32); z["vis"] = rh["vis"].astype(np.float32)
    z["tracks_low"] = rl["tracks"].astype(np.float32); z["vis_low"] = rl["vis"].astype(np.float32)
    ok = rh["retrack_ok"] & rl["retrack_ok"]                     # 两视角都成功
    z["retrack_ok"] = ok
    op = f"{BASE}/{clip_name}_retrack.npz"
    np.savez(op, **z)
    print(f"{clip_name}: retrack_ok {int(ok.sum())}/{len(ok)} ({100*(~ok).mean():.2f}% 排除) -> {op}", flush=True)


if __name__ == "__main__":
    if "human" in DOMAINS:
        assemble(HCLIP, "human")
    if "robot" in DOMAINS:
        assemble("clips_robot", "robot")
    print("=== 组装完成 ===", flush=True)
