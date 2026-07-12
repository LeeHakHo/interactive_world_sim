#!/bin/bash
#SBATCH --job-name=comb_dual
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/cross_embodiment_wm/dualview_comb/full_s%a/slurm.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --array=0-2

# dual-view (can_dual L24) combine action featurization sweep, split by seed.
# 4 methods (dummy5/skel anchors + a1/align_wm), N in {50,100,400}; co-train
# robot+human, held-out robot per-view drift px. Base arch = DualLWC joint
# 2-view prediction (exp_scel_dualview_wm). User 2026-07-12: formal method on
# dual-view + DiT base architecture.

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
mkdir -p outputs/cross_embodiment_wm/dualview_comb/full_s${SLURM_ARRAY_TASK_ID}

SEEDS=${SLURM_ARRAY_TASK_ID} \
OUT=outputs/cross_embodiment_wm/dualview_comb/full_s${SLURM_ARRAY_TASK_ID} \
python exp_scel_dualview_comb.py
