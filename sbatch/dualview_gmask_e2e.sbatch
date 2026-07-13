#!/bin/bash
#SBATCH --job-name=dv_gm_e2e
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/cross_embodiment_wm/dualview_gmask/e2e_sharp.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --mem=64G
#SBATCH --dependency=afterok:53277

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
GMASK=1 E2E_CKPT=outputs/cross_embodiment_wm/dualview_gmask/g1_100ep/dvdit_g.pt WM2A=outputs/cross_embodiment_wm/dualview_comb/extra400_s2/wm_align_wm_rh_N400.pt WM2B=outputs/cross_embodiment_wm/dualview_comb/extra400_s2/wm_dummy5_rh_N400.pt OUT=outputs/cross_embodiment_wm/dualview_gmask/e2e_sharp python exp_scel_dualview_gmaskcond.py
