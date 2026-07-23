#!/bin/bash
#SBATCH --job-name=kbd3d
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/video_arch_wm/logs/kbd3d_%x_%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --mem=48G
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export HF_HUB_OFFLINE=1
set -e
echo "=== 阶段A: precompute (.venv_wan, ②+IK+物体) ==="
.venv_wan/bin/python exp_scel_keyboard_3d.py --stage precompute
echo "=== 阶段B: FK skel2d (conda phantom, pinocchio) ==="
conda run -n phantom python fk_skel2d.py
echo "=== 阶段C: render (.venv_wan, skel③ VideoDiT) ==="
.venv_wan/bin/python exp_scel_keyboard_3d.py --stage render
echo "=== 中间产物可视化 ==="
.venv_wan/bin/python viz_keyboard_3d_intermediate.py || echo "(intermediate viz 脚本待建, 跳过)"
echo "=== ALL DONE ==="
