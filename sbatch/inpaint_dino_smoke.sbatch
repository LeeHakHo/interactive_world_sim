#!/bin/bash
#SBATCH --job-name=inpaint_dino_smoke
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/inpaint_dino_smoke.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=64G

# Smoke run for the inpaint-then-DINO subspace diagnostic.
# Stage A (iws): dump val clips.  Stage B (phantom): SAM2 + E2FGVI + DINO.
# Small per-domain count to produce QC montages for visual inpaint inspection
# BEFORE scaling up. See docs/superpowers/specs/2026-05-26-inpaint-dino-subspace-design.md
set -e
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
cd /scr2/yusenluo/interactive_world_sim
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1

OUT=outputs/inpaint_dino_subspace_smoke
CLIPS=$OUT/clips
N=96

echo "===== STAGE A (iws): dump val clips ====="
conda activate iws
python dump_val_clips.py --per-domain $N --out $CLIPS

echo "===== STAGE B (phantom): SAM2 + E2FGVI inpaint + DINO ====="
conda deactivate
conda activate phantom
python inpaint_dino_subspace.py --clips $CLIPS --per-domain $N --n-qc 6 --out-dir $OUT
