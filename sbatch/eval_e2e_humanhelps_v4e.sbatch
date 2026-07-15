#!/bin/bash
#SBATCH --job-name=e2e_hh_v4e
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/e2e_hh_v4e_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --mem=96G

# Proper pixel-level human-helps evidence: densev4e (+g-mask), N=1 render, N_rob=50.
# Trains ② ro@50 vs rh@50, renders held-out robot frames, ro-vs-rh object-region MSE.
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export HF_HOME=/scr2/yusenluo/interactive_world_sim/.hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== node $(hostname) gpu $CUDA_VISIBLE_DEVICES job $SLURM_JOB_ID ==="
python eval_e2e_humanhelps_v4e.py
echo "=== DONE ==="
