#!/bin/bash
#SBATCH --job-name=maskgen_j2m_v2
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/maskgen_j2m_v2_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:30:00
#SBATCH --mem=64G

# densev4d step1: learn g(joint action -> agent mask), supervised by robot_v3, held-out=video3.
# Deployable + non-leaking silhouette source; bypasses URDF camera calibration.
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export HF_HOME=/scr2/yusenluo/interactive_world_sim/.hf_cache
echo "=== node $(hostname) gpu $CUDA_VISIBLE_DEVICES job $SLURM_JOB_ID ==="
python train_maskgen_joint2mask_v2.py
echo "=== DONE ==="
