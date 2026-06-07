#!/bin/bash
#SBATCH --job-name=phantom_v3_bbox_gate
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/v3_bbox_gate_%a.txt
#SBATCH --array=0-1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=256G

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
cd /scr2/yusenluo/interactive_world_sim/phantom/phantom
conda activate phantom

# Pilot chunks: ep1 chunk0/chunk1 -> demo_num 1000,1001
DEMOS=(1000 1001)
DEMO_NUM=${DEMOS[${SLURM_ARRAY_TASK_ID:-0}]}
RAW=/scr2/yusenluo/interactive_world_sim/phantom/data/raw
PROCESSED=/scr2/yusenluo/interactive_world_sim/phantom/data/processed_play_human_v3_chunks

SRC=${RAW}/play_human_v3_chunks/${DEMO_NUM}/video_L.mp4
if [ ! -f "${SRC}" ]; then echo "missing ${SRC}"; exit 0; fi

HYDRA_FULL_ERROR=1 python process_data.py \
    demo_name=play_human_v3_chunks \
    data_root_dir=${RAW} \
    processed_data_root_dir=${PROCESSED} \
    'mode=[bbox]' \
    square=false input_resolution=480 \
    bimanual_setup=single_arm target_hand=right constrained_hand=false \
    camera_intrinsics=/scr2/yusenluo/interactive_world_sim/phantom_human_play/intrinsics_cam_high.json \
    camera_extrinsics=/scr2/yusenluo/interactive_world_sim/phantom_human_play/identity_extrinsics.json \
    demo_num=${DEMO_NUM}
