#!/bin/bash
#SBATCH --job-name=iws_latent
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/iws_latent_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=64G
# Path B: latent->latent transition (stage2-like). z0+flow+gmask -> DDPM predict z_t -> FROZEN stage1 decode.
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export HF_HOME=/scr2/yusenluo/interactive_world_sim/.hf_cache
echo "=== node $(hostname) job $SLURM_JOB_ID | iws latent predict (path B) ==="
python exp_v3_iws_latent_predict.py
echo "=== DONE ==="
