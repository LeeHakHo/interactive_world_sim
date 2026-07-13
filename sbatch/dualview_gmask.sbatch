#!/bin/bash
#SBATCH --job-name=dv_gmask
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/cross_embodiment_wm/dualview_gmask/g%a.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --array=0-1

# agent-conditioning ablation: GMASK = array idx. Fixes the blur complaint:
# can single-view detmem (LPIPS 0.147) had gmask agent cond; dual-view DiT
# (0.25-0.31) doesn't -> arm is hallucinated -> smear. Single variable ablation.

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
GMASK=${SLURM_ARRAY_TASK_ID} EPOCHS=60 \
OUT=outputs/cross_embodiment_wm/dualview_gmask/g${SLURM_ARRAY_TASK_ID} \
python exp_scel_dualview_gmaskcond.py
