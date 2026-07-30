#!/bin/bash
#SBATCH --job-name=scarce_e2e
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/video_arch_wm/logs/scarce_e2e_%x_%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --mem=48G
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export HF_HUB_OFFLINE=1
mkdir -p outputs/video_arch_wm/logs
set -e
# 真 e2e: 匹配的 scarce ②(mp)预测flow → 匹配的 scarce ③ 渲染; gif带flow叠加(绿GT/红②pred/黄eef)
# 每对由外部传 CKPT(③)/WM_CKPT(②)/OUT/SEQS; CCOND=4(multihead ③)
echo "=== scarce e2e: ③=$CKPT ②=$WM_CKPT SEQS=$SEQS ==="
CCOND=4 ACTION=${ACTION:-mp} CKPT=$CKPT WM_CKPT=$WM_CKPT OUT=$OUT SEQS=${SEQS:-332,59,418,442} \
  .venv_wan/bin/python eval_video_e2e.py
echo "=== DONE: $OUT ==="
