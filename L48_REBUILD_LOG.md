# L48 human rebuild + clean episode-split retrain (2026-07-29)

目标: human clips L24→L48 重建, 在干净 L48 + episode-split 上重训 ②(ro/rh/rh_two) 和 ③(ro/rh), 复核双头判决。
robot 已 L48 不动。新数据全进独立目录 `outputs/flow_render_dataset_can_dual_L48/`(不覆盖 L24)。

## 参数化改动 (向后兼容, env 默认保持旧 L24 行为)
- `retrack_sam2_full.py`: 加 env `BASE`(默认 .../can_dual), `HCLIP`(默认 human_L24); OUT/clip load/wsc 走 BASE; wsc 缺失时跳过(dead code, 本就未用)。
- `assemble_retrack_clips.py`: 加 env `BASE`, `HCLIP`(默认 clips_human_L24), `DOMAINS`(默认 human,robot)。
- `build_human_realwrist_clips.py`: 加 env `BASE`, `HCLIP`(默认 clips_human_L24)。
- `build_human_wrist_low.py`: 加 env `BASE`, `HCLIP`; 老 sidecar 校验缺失时跳过。

## 阶段状态
| 阶段 | 脚本 | job id | 状态 |
|---|---|---|---|
| 1 base human L48 (smoke) | sbatch_gen_human_L48.sbatch | 54843 | PENDING |
| 1 base human L48 (full) | | | - |
| 2 dualview cam_low | gen_dualview_aligned.py | | - |
| 3 retrack GDINO+SAM2 (high/low) | retrack_sam2_full.py | | - |
| 4 assemble | assemble_retrack_clips.py | | - |
| 5 realwrist + wrist sidecar L48 | build_human_wrist_low.py + build_human_realwrist_clips.py | | - |
| 6 wan latents L48 | build_can256_latents.py | | - |
| 7 cond agent-skel L48 | build_can_cond.py | | - |
| 8a 重训 ② {ro,rh,rh_two} | exp_scel_dualview_wm.py | | - |
| 8b 重训 ③ {ro,rh} | train_multihead_wm.py | | - |

## 数据流 (L48, 全在 ..._L48/ 目录)
- 1 → clips_human.npz (high view)
- 2 → clips_human_L48.npz (+tracks_low/eef_low/vis_low)
- 3 → retrack/human_{high,low}.npz
- 4 → clips_human_L48_retrack.npz
- 5 → wrist_sidecar_human_low.npz + clips_human_L48_retrack_realwrist.npz
- 6 → wan latents
- 7 → cond
- 8 → 重训

## 双头判决 (待填, 干净 L48)
- ro (MIX=r single):
- rh (MIX=rh single):
- rh_two (MIX=rh two):
(L24 旧: rh_two 2.314 ≈ rh_single 2.318 ≫ ro 2.083 = 两头无效)
