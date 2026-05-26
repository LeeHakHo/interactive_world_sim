#!/bin/bash
#SBATCH --job-name=inpaint_gap_test
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/inpaint_gap_test.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=64G

# Correct agent-removal test (user's reference approach): human seeds SAM2 from
# phantom masks_arm (true hand) -> whole arm; robot dark-gripper seed. cv2.inpaint
# bg-fill. Then DINOv2 H-vs-R probe/MMD + color-match + grayscale controls.
set -e
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python inpaint_gap_test.py
