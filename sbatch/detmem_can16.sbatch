#!/bin/bash
#SBATCH --job-name=detmem_can16
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/detmem_can_%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --mem=200G
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
export HF_HUB_OFFLINE=1 VAE_NAME=ostris/vae-kl-f8-d16
cd /scr2/yusenluo/interactive_world_sim
DS=outputs/flow_render_dataset_can_grip \
WM=outputs/cross_embodiment_wm/grip_wm_can/wm_base.pt \
MASKGEN=outputs/flow_wm/maskgen_caneef/maskgen_caneef.pt \
OUT_DIR=outputs/cross_embodiment_wm/latent_detmem_can16 \
  python exp_scel_latent_detmem.py
