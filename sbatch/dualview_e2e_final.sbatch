#!/bin/bash
#SBATCH --job-name=dcomb_e2e
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/cross_embodiment_wm/dualview_e2e/final_slurm.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --dependency=afterok:53213_2

# e2e final: paired L24-split ckpts (align_wm vs dummy5, N=400 seed2) -> flow-DiT.
# Replaces the plumbing run whose (2) had L48-window leakage.

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
CK=outputs/cross_embodiment_wm/dualview_comb/extra400_s2
WM2=$CK/wm_align_wm_rh_N400.pt OUT=outputs/cross_embodiment_wm/dualview_e2e/final_alignwm python exp_scel_dualview_e2e.py
WM2=$CK/wm_dummy5_rh_N400.pt   OUT=outputs/cross_embodiment_wm/dualview_e2e/final_dummy5  python exp_scel_dualview_e2e.py
