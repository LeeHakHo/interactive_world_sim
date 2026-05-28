#!/bin/bash
#SBATCH --job-name=ceve_build
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/ceve_build.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=128G

# Cross-embodiment video-edit dataset build (Plan 1/4). 纯 CPU（dark-CC mask，无 SAM2/GPU）。
# 把 human (play_human_aytsai chunks) 和 robot (play_robot_eef) 切成 25 帧 clip，
# 每 clip 产出 {frames, agent_mask, eef(M_s,相机系), obj_traj(O_s), embodiment_image(C), domain}
# 写到 data/ceve/ + index.json。
#   - human agent mask: 裁桌面后 phantom masks_arm + 暗袖连通域 - 红块（crop-first，避免连通域外溢背景）
#   - robot agent mask: 裁桌面后暗连通域，用 EEF 投影像素的圆盘重叠选对夹爪域（不再用 SAM2，SAM2 点种会爆炸）
#   - robot EEF: world -> camera (robot_world_to_cam fallback, position-only)
#   - O_s 目前全 NaN（VGGT 深度未接 + 红块/肤色分离待修，Plan 2 前置）
# 单进程顺序跑：append_index 不是并行安全的，顺序写避免 index.json 竞争。
# 一个坏源不会拖垮全部：build_human_chunk 跳过缺文件源 + main 对每源 try/except（保证 robot 必跑）。

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate iws

cd /scr2/yusenluo/interactive_world_sim

export HYDRA_FULL_ERROR=1

# 全新构建：清掉上一次（含交互式小规模试跑留下的 stale clip + index），避免 index.json 重复
rm -rf data/ceve

python build_ceve_dataset.py --domain all

echo "=== build done, index summary ==="
python -c "import json; from collections import Counter; idx=json.load(open('data/ceve/index.json')); print('total clips:', len(idx)); print(Counter(e['domain'] for e in idx))"
