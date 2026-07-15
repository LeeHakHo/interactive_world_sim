#!/bin/bash
#SBATCH --job-name=detmem_can_low
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/detmem_can_low_%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --mem=200G
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws
cd /scr2/yusenluo/interactive_world_sim
DS=outputs/flow_render_dataset_can_low_grip \
WM=outputs/cross_embodiment_wm/grip_wm_can_low/wm_base.pt \
MASKGEN=outputs/flow_wm/maskgen_caneef_low/maskgen_caneef.pt \
OUT_DIR=outputs/cross_embodiment_wm/latent_detmem_can_low \
  python exp_scel_latent_detmem.py
