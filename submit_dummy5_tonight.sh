#!/bin/bash
# 今晚自主: dummy5 ②(ro/rh) + grip-aware+warp ③(ro/rh), 全量, retrack+L48+episode-split(不用老数据源)
# 目的: (1)dummy5 ②③ 全量 human-helps 结论  (2)重出 keyboard/replay 测"grip+warp修render"假设
set -e
cd /scr2/yusenluo/interactive_world_sim

# --- ② dummy5 (retrack, episode-split, HELDOUT_VIDS内部export) ---
J_2ro=$(sbatch --parsable --job-name=d5wm2ro --export=ALL,MIX=r,OUT_DIR=outputs/cross_embodiment_wm/epsplit_L48/dummy5_r_all sbatch_dummy5_wm2_clean.sbatch)
J_2rh=$(sbatch --parsable --job-name=d5wm2rh --export=ALL,MIX=rh,OUT_DIR=outputs/cross_embodiment_wm/epsplit_L48/dummy5_rh_all sbatch_dummy5_wm2_clean.sbatch)
echo "② dummy5: ro=$J_2ro rh=$J_2rh"

# --- cond-build (retrack grip) → ③ grip ro/rh afterok ---
J_cond=$(sbatch --parsable sbatch_build_grip_cond.sbatch)
echo "cond-build=$J_cond"
J_3ro=$(sbatch --parsable --job-name=grip3ro --dependency=afterok:$J_cond \
  --export=ALL,OUT=outputs/video_arch_wm/m4_grip_retrack_L48_ro,STEPS=60000 sbatch_grip3_retrack.sbatch)
J_3rh=$(sbatch --parsable --job-name=grip3rh --dependency=afterok:$J_cond \
  --export=ALL,MIX=human,OUT=outputs/video_arch_wm/m4_grip_retrack_L48_rh,STEPS=60000 sbatch_grip3_retrack.sbatch)
echo "③ grip: ro=$J_3ro rh=$J_3rh (afterok cond=$J_cond)"

echo "JOBS: 2ro=$J_2ro 2rh=$J_2rh cond=$J_cond 3ro=$J_3ro 3rh=$J_3rh"
echo "② human-helps → dummy5_{r,rh}_all/summary.txt (drift, 自动)"
echo "③ human-helps + keyboard/replay → 训完后配 ckpt 跑(2ro=keyboard②, 3ro=keyboard③)"
