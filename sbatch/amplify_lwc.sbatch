#!/bin/bash
#SBATCH --job-name=amplify_lwc
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/amplify_lwc.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=128G

# AMPLIFY local-window-classification (LWC) ② vs regression on the dense 400-grid.
# 2 GPUs: shard the (config,seed) jobs across both, then merge.
# See exp_amplify_lwc_eval.py ; results -> outputs/cross_embodiment_wm/09_amplify_lwc_eval/

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT=outputs/cross_embodiment_wm/09_amplify_lwc_eval
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0 SHARD=0 NSHARD=2 python exp_amplify_lwc_eval.py > "$OUT/shard0.log" 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 SHARD=1 NSHARD=2 python exp_amplify_lwc_eval.py > "$OUT/shard1.log" 2>&1 &
P1=$!
wait $P0; wait $P1
echo "both shards done; merging"
python exp_amplify_lwc_eval.py --merge > "$OUT/merge.log" 2>&1
echo "DONE"
