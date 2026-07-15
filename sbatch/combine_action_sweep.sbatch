#!/bin/bash
#SBATCH --job-name=combine_action_sweep
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/combine_action_sweep.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=64G

# Combine skel+world action featurization full sweep (Task 3 Step4 + Task 4 Step1).
# world/skel (Pareto 锚, via V.train_feat) + a1/align/warm (结合机制, via train_comb),
# each robot-only (ro) vs robot+human (rh), held-out robot ADE px@224, N∈{20,50,100,400}×3 seeds.
# spec docs/superpowers/specs/2026-07-12-combine-skel-world-action-design.md
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMBINES=a1,align,warm \
OUT=outputs/cross_embodiment_wm/combine_action/full \
  python exp_scel_combine_action.py
