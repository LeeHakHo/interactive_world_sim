#!/bin/bash
#SBATCH --job-name=comb_v3
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/cross_embodiment_wm/combine_action/full_s%a/slurm.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --array=0-2

# v3 combine action featurization full sweep, split by seed (array idx = seed).
# 6 methods (world/skel anchors + a1/align/align_wm/warm), N in {20,50,100,400}.
# align_wm = LaST-HD (2606.23685) faithful: frozen mixed skel-WM teacher + cosine
# latent align. See docs/superpowers/specs/2026-07-12-combine-skel-world-action-design.md

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
mkdir -p outputs/cross_embodiment_wm/combine_action/full_s${SLURM_ARRAY_TASK_ID}

SEEDS=${SLURM_ARRAY_TASK_ID} \
COMBINES=a1,align,align_wm,warm \
OUT=outputs/cross_embodiment_wm/combine_action/full_s${SLURM_ARRAY_TASK_ID} \
python exp_scel_combine_action.py
