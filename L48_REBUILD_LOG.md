# L48 human rebuild + clean episode-split retrain (2026-07-29)

目标: human clips L24→L48 重建, 在干净 L48 + episode-split 上重训 ②(ro/rh/rh_two) 和 ③(ro/rh), 复核双头判决。
robot 已 L48 不动。新数据全进独立目录 `outputs/flow_render_dataset_can_dual_L48/`(不覆盖 L24)。

## 参数化改动 (向后兼容, env 默认保持旧 L24 行为)
- `retrack_sam2_full.py`: 加 env `BASE`(默认 .../can_dual), `HCLIP`(默认 human_L24); OUT/clip load/wsc 走 BASE; wsc 缺失时跳过(dead code, 本就未用)。
- `assemble_retrack_clips.py`: 加 env `BASE`, `HCLIP`(默认 clips_human_L24), `DOMAINS`(默认 human,robot)。
- `build_human_realwrist_clips.py`: 加 env `BASE`, `HCLIP`(默认 clips_human_L24)。
- `build_human_wrist_low.py`: 加 env `BASE`, `HCLIP`; 老 sidecar 校验缺失时跳过。

## 阶段状态 — 全链 afterok 依赖已提交 (2026-07-29)
阶段1冒烟 job 54843 COMPLETED 53s: clips_human.npz N=8 frames=(8,48,128,128,3) → **L=48确认**。
全量链一次性提交(submit_L48_chain.sh), afterok 串联, 上游失败下游自动不跑。

| 阶段 | 脚本/sbatch | job id | dep |
|---|---|---|---|
| 1 base human L48 (full) | sbatch_gen_human_L48.sbatch | **54860** | - |
| 2 dualview cam_low | sbatch_gen_dualview_L48.sbatch | **54861** | afterok:54860 |
| 3a retrack high | sbatch_retrack.sbatch (BASE/HCLIP=human_L48) | **54862** | afterok:54861 |
| 3b retrack low | sbatch_retrack.sbatch | **54863** | afterok:54861 |
| 4 assemble | sbatch_assemble_L48.sbatch | **54864** | afterok:54862,54863 |
| 5 realwrist + wrist sidecar | sbatch_realwrist_L48.sbatch | **54865** | afterok:54864 |
| 6 wan latents L48 (.venv_wan) | sbatch_human_latents.sbatch (OUTDIR=..._L48) | **54866** | afterok:54865 |
| 7 cond eef3 L48 | sbatch_cond_human_L48.sbatch | **54867** | afterok:54866 |
| 8a ② ro full | sbatch_wm2_shard.sbatch (MIX=r single) | **54868** | afterok:54865 |
| 8a ② rh full | sbatch_wm2_shard.sbatch (MIX=rh single) | **54869** | afterok:54865 |
| 8a ② rh_two full ★判决 | sbatch_wm2_shard.sbatch (MIX=rh HEAD_MODE=two) | **54870** | afterok:54865 |
| 8a ② ro n100 | | **54871** | afterok:54865 |
| 8a ② rh n100 | | **54872** | afterok:54865 |
| 8a ② ro n300 | | **54873** | afterok:54865 |
| 8a ② rh n300 | | **54874** | afterok:54865 |
| 8b ③ ro (robot-only) | sbatch_mh_train_L48.sbatch (MIX="") | **54875** | afterok:54867 |
| 8b ③ rh (cotrain) | sbatch_mh_train_L48.sbatch (MIX=human) | **54876** | afterok:54867 |

② 全量 OUT: outputs/cross_embodiment_wm/epsplit_L48/{mp_r_all,mp_rh_all,mp_rh_all_two,mp_r_n{100,300},mp_rh_n{100,300}}
③ OUT: outputs/video_arch_wm/epsplit_L48_mh/{ro,rh}; summary=各 OUT/summary*.txt / mh eval log
② env: DS=clips_robot_retrack DS_H=clips_human_L48_retrack_realwrist ACTION=mp HELDOUT_VIDS=100,102 HELDOUT_VIDS_H=0,1
③ env: LAT/COND=robot旧, LAT_H/COND_H=L48新, HELDOUT_VIDS=100,102 STEPS=40000 CCOND=4 PRED=abs
★注: ③ cond=eef3 7ch(匹配现有 cond_human_eef3_*), 非skel; human latents tL 6→12 修L不齐; 全在新目录不覆盖L24。

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

## ★recovery (2026-07-30): jid5 realwrist 秒挂(build_human_wrist_low viz UnboundLocalError ohi, L48 old=None)
- 修 build_human_wrist_low.py: viz_clips 仅 old!=None 时跑(commit 14b40f4)。sidecar 本已存好, 只崩非必要viz。
- 撤死链 54866-876, 从阶段5重提: realwrist=55028(无dep) → latents=55029 → cond=55030 → ②{ro=55031,rh=55032,rh_two=55033,+稀缺} → ③{ro=55038,rh=55039}。
