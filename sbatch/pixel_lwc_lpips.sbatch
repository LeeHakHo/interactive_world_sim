#!/bin/bash
#SBATCH --job-name=pixel_lpips
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/pixel_lpips_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=64G
# Path A: pixel-③ (LWC, no latent). Push down L2 blur with stronger LPIPS. env: LAMBDA_LPIPS USE_PREV USE_LWC.
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export HF_HOME=/scr2/yusenluo/interactive_world_sim/.hf_cache
echo "=== node $(hostname) job $SLURM_JOB_ID | pixel LWC LPIPS=$LAMBDA_LPIPS PREV=$USE_PREV (path A) ==="
python exp_v3_full_vs_scarce_human.py
echo "=== DONE ==="
