#!/bin/bash
#SBATCH --job-name=trossen_joint_qc
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/trossen_joint_qc_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --mem=32G

# QC: joint action[:6] -> URDF FK -> MuJoCo arm mask on real play_robot frame.
# Validates joint order + camera alignment before committing to densev4c.
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export HF_HOME=/scr2/yusenluo/interactive_world_sim/.hf_cache
export MUJOCO_GL=egl
echo "=== node $(hostname) gpu $CUDA_VISIBLE_DEVICES job $SLURM_JOB_ID ==="
python trossen_joint_mask_qc.py
echo "=== DONE ==="
