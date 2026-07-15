#!/bin/bash
#SBATCH --job-name=masks_sam2
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/masks_sam2_%a.log
#SBATCH --array=0-14%5
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=64G

# SAM2 whole-arm agent masks for the EEF play data (the approved set:
# human=256-crop SAM2, robot=128 dil3). Overwrites the per-episode cache
#   agent_mask_res128_crop195-195-256-256.npy
# that MixedPlayEEFDataset reads. Replaces the deterministic dark-CC masks.
#
# Array maps 1 task -> 1 episode (per-episode np.save => preempt-safe, [skip]-resumable):
#   index 0-11  -> robot episodes 0..11  (12 eps, ~214k frames total)
#   index 12-14 -> human episodes 0..2   (3 eps,  ~107k frames total)

set -euo pipefail

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate phantom

cd /scr2/yusenluo/interactive_world_sim

TID=${SLURM_ARRAY_TASK_ID}
if [ "$TID" -lt 12 ]; then
  DOMAIN=robot; IDX=$TID
else
  DOMAIN=human; IDX=$((TID - 12))
fi

echo "[task $TID] domain=$DOMAIN only-index=$IDX  $(date)"
python precompute_agent_masks_sam2.py \
    --domain "$DOMAIN" \
    --only-index "$IDX" \
    --overwrite
echo "[task $TID] done $(date)"
