#!/bin/bash
#SBATCH --job-name=dv_e2e_def
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/cross_embodiment_wm/dualview_gmask/e2e_def.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --mem=64G

# definitive e2e: warp+dual-silhouette 100ep III, paired (2) ckpts, H=20 + H=44.
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
CK=outputs/cross_embodiment_wm/dualview_gmask/final_warp_100ep/dvdit_g.pt
WA=outputs/cross_embodiment_wm/dualview_comb/extra400_s2/wm_align_wm_rh_N400.pt
WB=outputs/cross_embodiment_wm/dualview_comb/extra400_s2/wm_dummy5_rh_N400.pt
GMASK=1 GMASK_LOW=1 WARP=1 E2E_CKPT=$CK WM2A=$WA WM2B=$WB \
  OUT=outputs/cross_embodiment_wm/dualview_gmask/e2e_definitive python exp_scel_dualview_gmaskcond.py
GMASK=1 GMASK_LOW=1 WARP=1 HORIZON=44 E2E_CKPT=$CK WM2A=$WA WM2B=$WB \
  E2E_DS=outputs/flow_render_dataset_can_dual/clips_robot.npz \
  OUT=outputs/cross_embodiment_wm/dualview_gmask/e2e_definitive_long44 python exp_scel_dualview_gmaskcond.py
