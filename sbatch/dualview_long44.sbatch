#!/bin/bash
#SBATCH --job-name=dv_long44
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/cross_embodiment_wm/dualview_gmask/long44.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=64G

# long-horizon (44-step, full L48 clips) e2e with the definitive dual-silhouette
# 100ep renderer; (2) rolls 44 steps (trained 16 -> honest compounding stress test).
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
GMASK=1 GMASK_LOW=1 HORIZON=44 \
E2E_CKPT=outputs/cross_embodiment_wm/dualview_gmask/g1low_100ep/dvdit_g.pt \
E2E_DS=outputs/flow_render_dataset_can_dual/clips_robot.npz \
WM2A=outputs/cross_embodiment_wm/dualview_comb/extra400_s2/wm_align_wm_rh_N400.pt \
WM2B=outputs/cross_embodiment_wm/dualview_comb/extra400_s2/wm_dummy5_rh_N400.pt \
OUT=outputs/cross_embodiment_wm/dualview_gmask/e2e_long44 \
python exp_scel_dualview_gmaskcond.py
