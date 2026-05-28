#!/bin/bash
#SBATCH --job-name=ceve_build
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/ceve_build.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=128G

# Cross-embodiment video-edit dataset build (Plan 1/4).
# 把 human (play_human_aytsai chunks) 和 robot (play_robot_eef) 切成 25 帧 clip，
# 每 clip 产出 {frames, agent_mask, eef(M_s,相机系), obj_traj(O_s), embodiment_image(C), domain}
# 写到 data/ceve/ + index.json。
#   - human agent mask: phantom masks_arm + 暗袖连通域 - 红块
#   - robot agent mask: crop 到桌面后在 crop 图上 SAM2 种最暗点(=黑夹爪)
#   - robot EEF: world -> camera (robot_world_to_cam, position-only)
#   - O_s 目前全 NaN（VGGT 深度未接 + 红块/肤色分离待修，Plan 2 前置）
# 单 GPU、单进程顺序跑：append_index 不是并行安全的，顺序写避免 index.json 竞争。
# 瓶颈是 robot 逐帧 SAM2（~12 episode），human 用现成 masks_arm 很快。

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws

cd /scr2/yusenluo/interactive_world_sim

export HYDRA_FULL_ERROR=1

# 全新构建：清掉上一次（含交互式小规模试跑留下的 stale clip + index），避免 index.json 重复
rm -rf data/ceve

python build_ceve_dataset.py --domain all

echo "=== build done, index summary ==="
python -c "import json; from collections import Counter; idx=json.load(open('data/ceve/index.json')); print('total clips:', len(idx)); print(Counter(e['domain'] for e in idx))"
