#!/bin/bash
#SBATCH --job-name=video_e2e
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/video_arch_wm/logs/e2e_%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=48G
# M5 e2e: ②pred-flow端到端 vs GT-flow天花板 (标准protocol). ③=base video DiT ckpt(实时最新), ②=dummy5.
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export HF_HUB_OFFLINE=1
# Wan 需独立 venv python (iws 的 diffusers 无 AutoencoderKLWan)
.venv_wan/bin/python eval_video_e2e.py
