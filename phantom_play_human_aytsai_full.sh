#!/bin/bash
#SBATCH --job-name=phantom_play_human_v3_full
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/play_human_v3_full_%a.txt
#SBATCH --array=0-23
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --mem=256G

# v3: 12 eps x 2 chunks = array 0-23. Runs bbox(crop-detect)+hand2d+action.
# Split to parallelize:  sbatch --array=0-11 ... ;  sbatch --array=12-23 ...

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
cd /scr2/yusenluo/interactive_world_sim/phantom/phantom
conda activate phantom

ARRAY_ID=${SLURM_ARRAY_TASK_ID:-${1:-0}}
CHUNKS_PER_EP=2
EP=$(( ARRAY_ID / CHUNKS_PER_EP + 1 ))
CHUNK=$(( ARRAY_ID % CHUNKS_PER_EP ))
DEMO_NUM=$(( EP * 1000 + CHUNK ))

RAW=/scr2/yusenluo/interactive_world_sim/phantom/data/raw
PROCESSED=/scr2/yusenluo/interactive_world_sim/phantom/data/processed_play_human_v3_chunks

SRC=${RAW}/play_human_v3_chunks/${DEMO_NUM}/video_L.mp4
if [ ! -f "${SRC}" ]; then echo "Skipping ${ARRAY_ID}: ${SRC} not found"; exit 0; fi

HYDRA_FULL_ERROR=1 python process_data.py \
    demo_name=play_human_v3_chunks \
    data_root_dir=${RAW} \
    processed_data_root_dir=${PROCESSED} \
    'mode=[bbox,hand2d,action]' \
    square=false input_resolution=480 \
    bimanual_setup=single_arm target_hand=right constrained_hand=false \
    camera_intrinsics=/scr2/yusenluo/interactive_world_sim/phantom_human_play/intrinsics_cam_high.json \
    camera_extrinsics=/scr2/yusenluo/interactive_world_sim/phantom_human_play/identity_extrinsics.json \
    demo_num=${DEMO_NUM}
