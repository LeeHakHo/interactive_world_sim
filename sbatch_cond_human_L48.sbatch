#!/bin/bash
#SBATCH --job-name=condhL48
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/outputs/video_arch_wm/logs/condhL48_%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=shard:8
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --mem=64G
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
mkdir -p outputs/video_arch_wm/logs
export HF_HUB_OFFLINE=1
export SRC=human
export DS=${DS:-outputs/flow_render_dataset_can_dual_L48/clips_human_L48_retrack_realwrist.npz}
export AGENT=${AGENT:-eef3}                     # human agent 通道 = eef3 (匹配现有 cond_human_eef3_*, 7ch flow+agent+warp)
export COND_NAME=${COND_NAME:-cond_human_eef3_L48}
echo "=== build human cond L48 SRC=$SRC AGENT=$AGENT DS=$DS COND_NAME=$COND_NAME ==="
python build_can_cond.py
echo "=== DONE $(ls -la outputs/video_arch_wm/cond_can_dual/${COND_NAME}.npz 2>/dev/null) ==="
