#!/bin/bash
#SBATCH --job-name=dit_can
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/dit_can_%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --mem=200G
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
HF_HUB_OFFLINE=1 VAE_NAME=ostris/vae-kl-f8-d16 \
DS=outputs/flow_render_dataset_can_grip \
WM=outputs/cross_embodiment_wm/grip_wm_can/wm_base.pt \
MASKGEN=outputs/flow_wm/maskgen_caneef/maskgen_caneef.pt \
INJECT=add MEM=4 LAM_HM=0.0 \
OUT_DIR=outputs/cross_embodiment_wm/latent_dit_can/add_m4_hm0.0 \
  python exp_scel_latent_dit.py
