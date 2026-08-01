#!/bin/bash
#SBATCH --job-name=build_sketch
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/video_arch_wm/sketch_can_dual/build_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
cd /scr2/yusenluo/interactive_world_sim
export HF_HUB_OFFLINE=1
WARP=1 TL=12 SRC=robot .venv_wan/bin/python build_sketch.py
WARP=1 TL=12 SRC=human .venv_wan/bin/python build_sketch.py
echo "=== DONE ==="
