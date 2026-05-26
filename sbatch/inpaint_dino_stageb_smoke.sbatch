#!/bin/bash
#SBATCH --job-name=inpaint_dino_stageb
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/inpaint_dino_stageb.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=64G

# Stage-B-only smoke: reuse already-dumped clips, re-run SAM2+E2FGVI+DINO.
# For fast iteration on mask/inpaint quality. See spec 2026-05-26.
set -e
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate phantom
cd /scr2/yusenluo/interactive_world_sim
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT=outputs/inpaint_dino_subspace_smoke
python inpaint_dino_subspace.py --clips $OUT/clips --per-domain 96 --n-qc 6 --out-dir $OUT
