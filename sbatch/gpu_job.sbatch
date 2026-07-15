#!/bin/bash
#SBATCH --job-name=iws_gpu
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/%x_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --mem=64G

# Generic single-GPU launcher — SLURM allocates a HEALTHY gpu (avoids the broken GPU2 node-fault).
# Usage: sbatch --job-name=<name> --export=ALL,CMD="ENV=v python script.py",OUTDIR=... sbatch/gpu_job.sbatch
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 VAE_NAME=ostris/vae-kl-f8-d16
echo "SLURM gpu allocation: $CUDA_VISIBLE_DEVICES  on $(hostname)"
eval "$CMD"
echo "=== SBATCH JOB DONE (exit $?) ==="
