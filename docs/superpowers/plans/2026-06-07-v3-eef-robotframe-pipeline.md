# v3 EEF Pipeline (human→robot-world frame) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 12 个 `play_human_v3_{1..12}` 用现成 phantom 流水线抽出 human hand EEF，并用 URDF 变换把 human eef 钉到 robot 锁定的桌面平面（robot-world 系，z=全局常数 ≈0.100m），打包成 LeRobot 数据集上传 HF；bbox 先核验后铺开；产 bbox+eef 可视化给用户。

**Architecture:** 复用旧 `play_human_aytsai` 流水线（download→chunk→Pass1[bbox,hand2d,action]→concat+smooth），新增 (1) 集中版本配置避免 sprawl、(2) `pin_eef_to_plane.py` 把 cam 系 eef 经 ray∩plane 转到 robot-world 系、(3) v3 版 build/push。跳过 backplant+inpaint。关键 gate：先只跑 bbox 看 `video_bboxes.mkv`，对了再跑全量 24 个 Pass1 job。

**Tech Stack:** Python 3 (env `iws`: numpy 1.26 / pandas / huggingface_hub；phantom 用 env `phantom`)、SLURM sbatch、ffmpeg、pytest。

**Spec:** `docs/superpowers/specs/2026-06-07-v3-eef-robotframe-pipeline-design.md`

---

## File Structure

| 文件 | 职责 | 动作 |
|---|---|---|
| `aytsai_pipeline_config.py` | 版本化配置（v1/v3：repo 模式、ep 列表、chunks/ep、磁盘子目录、HF 模式） | Create |
| `pin_eef_to_plane.py` | 纯几何：cam 系 eef → robot-world 平面（ray∩plane） | Create |
| `compute_z_plane.py` | 从 18 个 robot 数据集算 z_plane 常数，存 `z_plane_v3.json` | Create |
| `tests/test_pin_eef_to_plane.py` | pin 几何单测 | Create |
| `download_aytsaiusc_play_human.py` | 下载 human v3_{1..12} | Modify |
| `download_aytsaiusc_play_robot.py` | 下载 robot v3_{1..18}_eef | Modify |
| `chunk_play_human_aytsai.py` | 切 chunk（2/ep，v3 目录） | Modify |
| `phantom_play_human_aytsai_full.sh` | Pass1 sbatch（版本化，array 0-23） | Modify |
| `phantom_play_human_v3_bbox_gate.sh` | bbox-only pilot sbatch | Create |
| `concat_smooth_play_human_aytsai.py` | concat + 全 ep smooth（版本化） | Modify |
| `build_eef_dataset.py` | v3：调 pin，写 robot-world eef，检测率实算 | Modify |
| `push_eef_dataset.py` | 上传 `yusenluo9z/play_human_v3_{N}_eef` | Modify |
| `viz_eef_v3.py` | 对 v3_1/5/10 渲 eef 叠加视频（基于 visualize_eef） | Create |

---

## Task 1: 集中版本配置 `aytsai_pipeline_config.py`

**Files:**
- Create: `aytsai_pipeline_config.py`

- [ ] **Step 1: 写配置模块**

```python
"""Version-keyed config for the aytsai play_human phantom EEF pipeline.

Select version via env AYTSAI_VERSION (default 'v3'). Keeps the old 3-episode
'v1' batch (play_human_1/2/3) reproducible while pointing new drivers at the
12-episode 'v3' batch with its own on-disk dirs so outputs never collide.
"""
import os

_REPO_ROOT = "/scr2/yusenluo/interactive_world_sim"

CONFIGS = {
    "v1": dict(
        human_repo_pattern="aytsaiusc/play_human_{n}",
        human_eps=list(range(1, 4)),                 # 1..3
        robot_repo_pattern="aytsaiusc/play_robot_{n}_eef",
        robot_eps=list(range(1, 7)),                 # legacy, unused by new run
        chunks_per_ep=4,
        raw_subdir="play_human_aytsai_chunks",
        processed_chunks_subdir="processed_play_human_aytsai_chunks",
        processed_full_subdir="processed_play_human_aytsai_full",
        eef_local_subdir="play_human_eef_{n}",
        hf_repo_pattern="yusenluo9z/play_human_eef_{n}",
    ),
    "v3": dict(
        human_repo_pattern="aytsaiusc/play_human_v3_{n}",
        human_eps=list(range(1, 13)),                # 1..12
        robot_repo_pattern="aytsaiusc/play_robot_v3_{n}_eef",
        robot_eps=list(range(1, 19)),                # 1..18 (download + z_plane)
        chunks_per_ep=2,                             # 17930 frames / 9000 -> 2
        raw_subdir="play_human_v3_chunks",
        processed_chunks_subdir="processed_play_human_v3_chunks",
        processed_full_subdir="processed_play_human_v3_full",
        eef_local_subdir="play_human_v3_eef_{n}",
        hf_repo_pattern="yusenluo9z/play_human_v3_{n}_eef",
    ),
}

# phantom's demo_name = the raw_subdir leaf (BaseProcessor scans that dir).
def get(version: str | None = None) -> dict:
    v = version or os.environ.get("AYTSAI_VERSION", "v3")
    if v not in CONFIGS:
        raise ValueError(f"unknown AYTSAI_VERSION={v!r}, have {list(CONFIGS)}")
    cfg = dict(CONFIGS[v])
    cfg["version"] = v
    cfg["repo_root"] = _REPO_ROOT
    return cfg


def demo_num(ep: int, chunk: int) -> int:
    """Pure-integer chunk dir name phantom can int()-parse."""
    return ep * 1000 + chunk


if __name__ == "__main__":
    import json, sys
    print(json.dumps(get(sys.argv[1] if len(sys.argv) > 1 else None), indent=2))
```

- [ ] **Step 2: 冒烟验证 v1/v3 都能取**

Run:
```bash
cd /scr2/yusenluo/interactive_world_sim
python aytsai_pipeline_config.py v3 | python -c "import json,sys; d=json.load(sys.stdin); assert d['chunks_per_ep']==2 and d['human_eps'][-1]==12 and d['robot_eps'][-1]==18; print('v3 ok')"
python aytsai_pipeline_config.py v1 | python -c "import json,sys; d=json.load(sys.stdin); assert d['chunks_per_ep']==4 and d['human_eps']==[1,2,3]; print('v1 ok')"
```
Expected: `v3 ok` then `v1 ok`

- [ ] **Step 3: Commit**

```bash
git add aytsai_pipeline_config.py
git commit -m "feat(v3-eef): central version config (v1/v3) for aytsai phantom pipeline"
```

---

## Task 2: pin 几何 — 失败测试 `tests/test_pin_eef_to_plane.py`

**Files:**
- Create: `tests/test_pin_eef_to_plane.py`

- [ ] **Step 1: 写失败测试**

```python
import numpy as np
import pytest

from pin_eef_to_plane import intrinsics_matrix, pin_points_to_plane


def test_pinned_points_lie_on_plane_and_reproject_to_same_pixel():
    K = intrinsics_matrix(381.092, 381.092, 310.085, 245.318)
    # Identity cam<-world so robot-world == cam frame: plane {z=z_plane} is flat in cam.
    T_cam_world = np.eye(4, dtype=np.float64)
    z_plane = 0.10
    # Three arbitrary cam-frame eef points at various depths.
    p_cam = np.array([[0.05, -0.02, 0.9],
                      [-0.10, 0.03, 1.3],
                      [0.0, 0.0, 0.7]], dtype=np.float64)
    p_world, p_cam_pin = pin_points_to_plane(p_cam, K, T_cam_world, z_plane)

    # 1) every pinned world point sits exactly on the robot plane z=z_plane
    np.testing.assert_allclose(p_world[:, 2], z_plane, atol=1e-9)

    # 2) pinned point reprojects to the SAME pixel as the original eef point
    def proj(p):
        uvw = (K @ p.T).T
        return uvw[:, :2] / uvw[:, 2:3]
    np.testing.assert_allclose(proj(p_cam), proj(p_cam_pin), atol=1e-6)


def test_nonidentity_extrinsic_keeps_points_on_world_plane():
    K = intrinsics_matrix(381.092, 381.092, 310.085, 245.318)
    T_cam_world = np.array([[0.0, -1.0, 0.0, 0.0090],
                            [-0.9063, 0.0, -0.4226, 0.1486],
                            [0.4226, 0.0, -0.9063, 1.0868],
                            [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    z_plane = 0.10
    p_cam = np.array([[0.0, 0.05, 0.95], [0.08, -0.03, 1.1]], dtype=np.float64)
    p_world, _ = pin_points_to_plane(p_cam, K, T_cam_world, z_plane)
    np.testing.assert_allclose(p_world[:, 2], z_plane, atol=1e-6)
```

- [ ] **Step 2: 运行确认失败**

Run:
```bash
cd /scr2/yusenluo/interactive_world_sim
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh && conda activate iws
python -m pytest tests/test_pin_eef_to_plane.py -v
```
Expected: FAIL（`ModuleNotFoundError: No module named 'pin_eef_to_plane'`）

---

## Task 3: pin 几何实现 `pin_eef_to_plane.py`

**Files:**
- Create: `pin_eef_to_plane.py`

- [ ] **Step 1: 实现**

```python
"""Pin phantom human EEF (cam_high optical frame) onto the robot's locked
tabletop plane, expressed in the robot-world frame.

The hand's 2D image position is reliable; the monocular depth z is not. We keep
the eef pixel (ray direction) and replace depth by intersecting that camera ray
with the robot's operating plane {z_world = z_plane}. Output is in robot-world
frame, so human and robot EEF live in ONE coordinate system on the same plane.
"""
import numpy as np


def intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def pin_points_to_plane(p_cam, K, T_cam_world, z_plane):
    """Args:
        p_cam: (N,3) eef points in cam_high optical frame.
        K: (3,3) intrinsics.
        T_cam_world: (4,4) robot-world -> cam transform.
        z_plane: scalar robot-world plane height.
    Returns:
        p_world: (N,3) eef in robot-world frame, p_world[:,2] == z_plane.
        p_cam_pin: (N,3) the same points in cam frame (reproject == original pixel).
    """
    p_cam = np.asarray(p_cam, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T_cam_world = np.asarray(T_cam_world, dtype=np.float64)

    # Pixel of each eef point, then the camera ray direction through that pixel.
    uvw = (K @ p_cam.T).T                       # (N,3)
    uv1 = uvw / uvw[:, 2:3]                      # (N,3) homogeneous pixel [u,v,1]
    d = (np.linalg.inv(K) @ uv1.T).T            # (N,3) ray dir in cam frame

    R = T_cam_world[:3, :3]
    t = T_cam_world[:3, 3]
    x0_cam = R @ np.array([0.0, 0.0, z_plane]) + t   # a point on the plane, cam frame
    n_cam = R @ np.array([0.0, 0.0, 1.0])            # plane normal, cam frame

    # Ray from camera origin s*d intersects plane: n.(s d) = n.x0  ->  s = n.x0 / n.d
    s = (n_cam @ x0_cam) / (d @ n_cam)               # (N,)
    p_cam_pin = d * s[:, None]                       # (N,3)

    T_world_cam = np.linalg.inv(T_cam_world)
    Rw = T_world_cam[:3, :3]
    tw = T_world_cam[:3, 3]
    p_world = (Rw @ p_cam_pin.T).T + tw              # (N,3); z == z_plane by construction
    return p_world, p_cam_pin


def rotate_oris_to_world(ee_oris, T_cam_world):
    """Rotate (N,3,3) cam-frame orientation matrices into robot-world frame."""
    ee_oris = np.asarray(ee_oris, dtype=np.float64)
    R_world_cam = np.linalg.inv(np.asarray(T_cam_world, dtype=np.float64))[:3, :3]
    return np.einsum("ij,njk->nik", R_world_cam, ee_oris)
```

- [ ] **Step 2: 运行测试确认通过**

Run:
```bash
cd /scr2/yusenluo/interactive_world_sim
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh && conda activate iws
python -m pytest tests/test_pin_eef_to_plane.py -v
```
Expected: 2 passed

- [ ] **Step 3: Commit**

```bash
git add pin_eef_to_plane.py tests/test_pin_eef_to_plane.py
git commit -m "feat(v3-eef): pin_eef_to_plane (ray-plane) + geometry tests"
```

---

## Task 4: z_plane 常数 `compute_z_plane.py`

**Files:**
- Create: `compute_z_plane.py`

- [ ] **Step 1: 实现（从 18 个 robot 数据集算 action z 均值）**

```python
"""Compute the global locked-plane height z_plane from the robot v3 datasets.

Uses action_right_ee_position[:,2] (the COMMANDED, truly-locked z; std~3e-4)
averaged over all robot eps. Caches to z_plane_v3.json so build is deterministic.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

import aytsai_pipeline_config as cfg


def compute(version: str = "v3") -> dict:
    c = cfg.get(version)
    zs = []
    per_ep = {}
    for n in c["robot_eps"]:
        repo = c["robot_repo_pattern"].format(n=n)
        p = hf_hub_download(repo, "data/chunk-000/episode_000000.parquet", repo_type="dataset")
        df = pd.read_parquet(p)
        z = np.stack(df["action_right_ee_position"].to_numpy())[:, 2]
        per_ep[repo] = dict(mean=float(z.mean()), std=float(z.std()), n=int(z.size))
        zs.append(z)
    allz = np.concatenate(zs)
    out = dict(
        version=version,
        z_plane=float(allz.mean()),
        z_plane_std=float(allz.std()),
        n_frames=int(allz.size),
        source_column="action_right_ee_position[:,2]",
        per_ep=per_ep,
    )
    Path(f"z_plane_{version}.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    o = compute("v3")
    print(f"z_plane = {o['z_plane']:.5f} m  (std {o['z_plane_std']:.5f}, n={o['n_frames']})")
```

- [ ] **Step 2: 运行并核对**

Run:
```bash
cd /scr2/yusenluo/interactive_world_sim
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh && conda activate iws
python compute_z_plane.py
python -c "import json; d=json.load(open('z_plane_v3.json')); assert 0.095 < d['z_plane'] < 0.105, d['z_plane']; assert d['n_frames'] > 18*15000; print('z_plane', round(d['z_plane'],5), 'ok')"
```
Expected: `z_plane = 0.100xx m ...` 然后 `z_plane 0.100xx ok`

- [ ] **Step 3: Commit**

```bash
git add compute_z_plane.py z_plane_v3.json
git commit -m "feat(v3-eef): compute global locked-plane z_plane from robot action z (~0.100m)"
```

---

## Task 5: 下载脚本（human v3_{1..12} + robot v3_{1..18}_eef）

**Files:**
- Modify: `download_aytsaiusc_play_human.py`
- Modify: `download_aytsaiusc_play_robot.py`

- [ ] **Step 1: 改 human 下载脚本走配置**

替换 `download_aytsaiusc_play_human.py` 全文：
```python
"""Download the human play datasets for the active version into human_play_data/."""
import os

from huggingface_hub import snapshot_download

import aytsai_pipeline_config as cfg

c = cfg.get()
OUTPUT_DIR = os.path.join(c["repo_root"], "human_play_data")

for n in c["human_eps"]:
    repo_id = c["human_repo_pattern"].format(n=n)
    local_dir = os.path.join(OUTPUT_DIR, repo_id.split("/")[-1])
    print(f"Downloading {repo_id} -> {local_dir}")
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir)
    print(f"  Done: {repo_id}")

print("\nAll human downloads complete!")
```

- [ ] **Step 2: 改 robot 下载脚本走配置**

替换 `download_aytsaiusc_play_robot.py` 全文：
```python
"""Download the robot eef datasets for the active version into human_play_data/."""
import os

from huggingface_hub import snapshot_download

import aytsai_pipeline_config as cfg

c = cfg.get()
OUTPUT_DIR = os.path.join(c["repo_root"], "human_play_data")

for n in c["robot_eps"]:
    repo_id = c["robot_repo_pattern"].format(n=n)
    local_dir = os.path.join(OUTPUT_DIR, repo_id.split("/")[-1])
    print(f"Downloading {repo_id} -> {local_dir}")
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir)
    print(f"  Done: {repo_id}")

print("\nAll robot downloads complete!")
```

- [ ] **Step 3: 运行下载（前台，约几十 GB，可能数分钟到十几分钟）**

Run:
```bash
cd /scr2/yusenluo/interactive_world_sim
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh && conda activate iws
python download_aytsaiusc_play_human.py
python download_aytsaiusc_play_robot.py
```
Expected: 末尾 `All human downloads complete!` / `All robot downloads complete!`

- [ ] **Step 4: 核对落盘（12 human + 18 robot）**

Run:
```bash
ls -d human_play_data/play_human_v3_* | wc -l       # expect 12
ls -d human_play_data/play_robot_v3_*_eef | wc -l    # expect 18
ls human_play_data/play_human_v3_1/videos/chunk-000/observation.images.cam_high/episode_000000.mp4
```
Expected: `12`、`18`、文件存在

- [ ] **Step 5: Commit**

```bash
git add download_aytsaiusc_play_human.py download_aytsaiusc_play_robot.py
git commit -m "feat(v3-eef): version-aware HF download (human v3_1..12, robot v3_1..18_eef)"
```

---

## Task 6: 切 chunk（2/ep，v3 目录）

**Files:**
- Modify: `chunk_play_human_aytsai.py`

- [ ] **Step 1: 改成走配置**

替换 `chunk_play_human_aytsai.py` 中的常量块与 `main`：把顶部 `SRC_BASE/RAW_BASE/EPS/CHUNK_*` 改为从配置取，源视频路径用 v3 数据集名，chunk 根目录用 `c["raw_subdir"]`，符号链接目录改 `play_human_v3`。新全文：
```python
"""Pre-slice each full play_human (active version) cam_high video into 5-min
sub-clips for the phantom pipeline. Chunk dir names are pure integers
ep*1000+chunk_idx (phantom int()-parses them)."""
import os
import subprocess
from pathlib import Path

import aytsai_pipeline_config as cfg

c = cfg.get()
SRC_BASE = Path(c["repo_root"]) / "human_play_data"
RAW_BASE = Path(c["repo_root"]) / "phantom" / "data" / "raw"
EPS = c["human_eps"]
FPS = 30
CHUNK_SEC = 300
CHUNK_FRAMES = FPS * CHUNK_SEC  # 9000


def total_frames(video: Path) -> int:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_packets", "-show_entries", "stream=nb_read_packets",
         "-of", "csv=p=0", str(video)], text=True).strip()
    return int(out)


def chunk_video(src: Path, ep: int, dst_root: Path, n_chunks: int) -> None:
    dst_root.mkdir(parents=True, exist_ok=True)
    for i in range(n_chunks):
        demo_num = cfg.demo_num(ep, i)
        dst = dst_root / str(demo_num) / "video_L.mp4"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            print(f"[skip] {dst} exists")
            continue
        start = i * CHUNK_SEC
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-i", str(src),
             "-t", str(CHUNK_SEC), "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-preset", "fast", str(dst)], check=True)
        print(f"[ok] chunk {i:03d} -> {dst}")


def main() -> None:
    for ep in EPS:
        repo_leaf = c["human_repo_pattern"].format(n=ep).split("/")[-1]
        src = SRC_BASE / repo_leaf / "videos/chunk-000/observation.images.cam_high/episode_000000.mp4"
        nframes = total_frames(src)
        n_chunks = (nframes + CHUNK_FRAMES - 1) // CHUNK_FRAMES
        print(f"ep {ep} ({repo_leaf}): {nframes} frames -> {n_chunks} chunks")
        chunk_video(src, ep, RAW_BASE / c["raw_subdir"], n_chunks)

        full_dir = RAW_BASE / f"play_human_{c['version']}" / str(ep)
        full_dir.mkdir(parents=True, exist_ok=True)
        link = full_dir / "video_L.mp4"
        if not link.exists():
            link.symlink_to(src)
            print(f"[ok] symlink full ep{ep} -> {link}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行 + 核对（12ep × 2chunk = 24 个整数目录）**

Run:
```bash
cd /scr2/yusenluo/interactive_world_sim
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh && conda activate iws
python chunk_play_human_aytsai.py
ls phantom/data/raw/play_human_v3_chunks | sort -n | tr '\n' ' '; echo
ls phantom/data/raw/play_human_v3_chunks | wc -l   # expect 24
```
Expected: 打印 `1000 1001 2000 2001 ... 12000 12001`，计数 `24`

- [ ] **Step 3: Commit**

```bash
git add chunk_play_human_aytsai.py
git commit -m "feat(v3-eef): version-aware chunking (2 chunks/ep into play_human_v3_chunks)"
```

---

## Task 7: BBOX GATE — pilot 只跑 bbox，肉眼核 `video_bboxes.mkv`（关卡）

> 用户强调：bbox 不对，eef 白跑。先在少量 chunk 上只跑 DINO bbox（不跑 HaMeR），核验后才铺开全量。

**Files:**
- Create: `phantom_play_human_v3_bbox_gate.sh`

- [ ] **Step 1: 写 bbox-only sbatch（pilot：v3_1 的 chunk 0/1 = demo 1000,1001）**

```bash
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
```

- [ ] **Step 2: 提交并等完成**

Run:
```bash
cd /scr2/yusenluo/interactive_world_sim
mkdir -p phantom_logs
sbatch phantom_play_human_v3_bbox_gate.sh
# 轮询直到两个 job 结束
while squeue -u "$USER" -n phantom_v3_bbox_gate -h | grep -q .; do sleep 30; done
echo "bbox gate done"
tail -5 phantom_logs/v3_bbox_gate_0.txt
```
Expected: job 完成，log 无 traceback

- [ ] **Step 3: 转码 bbox overlay 给用户看（ffv1 .mkv → mp4），打印绝对路径**

Run:
```bash
cd /scr2/yusenluo/interactive_world_sim
mkdir -p outputs/v3_eef_qc
for d in 1000 1001; do
  IN=phantom/data/processed_play_human_v3_chunks/play_human_v3_chunks/$d/bbox_processor/video_bboxes.mkv
  OUT=outputs/v3_eef_qc/bbox_$d.mp4
  ffmpeg -y -loglevel error -i "$IN" -c:v libx264 -pix_fmt yuv420p "$OUT"
  echo "BBOX QC: $(readlink -f $OUT)"
done
```
Expected: 打印 2 个绝对路径

- [ ] **Step 4: 人工核验关卡（STOP — 给用户看）**

把两个 `bbox_*.mp4` 绝对路径贴给用户，请确认：右手**绿框**稳定贴手、无左右混淆/漏检/误检；并顺带确认 human 视频**机位与 robot 一致**（plane-pin 前提）。
- 通过 → 继续 Task 8。
- 不通过 → 调 `phantom/phantom/processors/bbox_processor.py` 的 `DINO_CONFIDENCE_THRESH` 或 "a hand" prompt，重跑本 Task，直至通过。**未通过不得进入全量 Pass1。**

- [ ] **Step 5: Commit**

```bash
git add phantom_play_human_v3_bbox_gate.sh
git commit -m "feat(v3-eef): bbox-gate sbatch (DINO-only pilot) + QC transcode"
```

---

## Task 8: 全量 Pass1 sbatch（版本化，array 0-23）

**Files:**
- Modify: `phantom_play_human_aytsai_full.sh`

- [ ] **Step 1: 改 Pass1 sbatch 走 v3（array 0-23、CHUNKS_PER_EP=2、v3 路径）**

替换 `phantom_play_human_aytsai_full.sh` 全文：
```bash
#!/bin/bash
#SBATCH --job-name=phantom_play_human_v3_full
#SBATCH --output=/scr2/yusenluo/interactive_world_sim/phantom_logs/play_human_v3_full_%a.txt
#SBATCH --array=0-23
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --mem=256G

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
```

- [ ] **Step 2: 提交（拆两批并行抢卡）并等完成**

Run:
```bash
cd /scr2/yusenluo/interactive_world_sim
sbatch --array=0-11 phantom_play_human_aytsai_full.sh
sbatch --array=12-23 phantom_play_human_aytsai_full.sh
while squeue -u "$USER" -n phantom_play_human_v3_full -h | grep -q .; do sleep 60; done
echo "Pass1 done"
```
Expected: `Pass1 done`

- [ ] **Step 3: 核对 24 个 chunk 都产出 action npz**

Run:
```bash
cd /scr2/yusenluo/interactive_world_sim
N=0
for ep in $(seq 1 12); do for ch in 0 1; do
  d=$((ep*1000+ch))
  f=phantom/data/processed_play_human_v3_chunks/play_human_v3_chunks/$d/action_processor/actions_right_single_arm.npz
  [ -f "$f" ] && N=$((N+1)) || echo "MISSING $f"
done; done
echo "action npz present: $N / 24"
```
Expected: `action npz present: 24 / 24`（若有缺，查对应 `phantom_logs/play_human_v3_full_*.txt` 重跑该 array id）

- [ ] **Step 4: Commit**

```bash
git add phantom_play_human_aytsai_full.sh
git commit -m "feat(v3-eef): version Pass1 sbatch (12ep x 2chunk = array 0-23)"
```

---

## Task 9: concat + 全 ep smooth（版本化）

**Files:**
- Modify: `concat_smooth_play_human_aytsai.py`

- [ ] **Step 1: 改成走配置（路径/eps/chunks 从 cfg）**

替换 `concat_smooth_play_human_aytsai.py` 顶部常量：
```python
import argparse
import sys
from pathlib import Path
import numpy as np

import aytsai_pipeline_config as cfg

c = cfg.get()
sys.path.insert(0, str(Path(c["repo_root"]) / "phantom"))
from phantom.processors.smoothing_processor import SmoothingProcessor  # type: ignore

CHUNKS_BASE = Path(c["repo_root"]) / "phantom" / "data" / c["processed_chunks_subdir"] / c["raw_subdir"]
OUT_BASE = Path(c["repo_root"]) / "phantom" / "data" / c["processed_full_subdir"] / f"play_human_{c['version']}"
EPS = c["human_eps"]
CHUNKS_PER_EP = c["chunks_per_ep"]
CHUNK_FRAMES = 9000
```
其余 `concat_ep` / `main` 逻辑不变（已用 `ep*1000+i`、`CHUNKS_PER_EP`、`CHUNK_FRAMES`）。把 `main()` 的 `default=EPS` 保持。

- [ ] **Step 2: 运行 + 核对全 ep smoothed 产出**

Run:
```bash
cd /scr2/yusenluo/interactive_world_sim
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh && conda activate iws
python concat_smooth_play_human_aytsai.py
ls phantom/data/processed_play_human_v3_full/play_human_v3/*/smoothing_processor/smoothed_actions_right_single_arm.npz | wc -l  # expect 12
python -c "
import numpy as np
s=np.load('phantom/data/processed_play_human_v3_full/play_human_v3/1/smoothing_processor/smoothed_actions_right_single_arm.npz')
a=np.load('phantom/data/processed_play_human_v3_full/play_human_v3/1/action_processor/actions_right_single_arm.npz')
print('ep1 detections', s['ee_pts'].shape, 'union', a['union_indices'].shape)
assert s['ee_pts'].shape[0]==a['union_indices'].shape[0]
print('ok')
"
```
Expected: `12`、`ep1 detections (M,3) union (M,)`、`ok`

- [ ] **Step 3: Commit**

```bash
git add concat_smooth_play_human_aytsai.py
git commit -m "feat(v3-eef): version-aware concat + full-ep smoothing"
```

---

## Task 10: build v3 EEF 数据集（调 pin，robot-world 系，检测率实算）

**Files:**
- Modify: `build_eef_dataset.py`

- [ ] **Step 1: 改 build —— 路径走 cfg，eef 经 pin 转 robot-world，z=z_plane**

替换 `build_eef_dataset.py` 顶部常量与 `build_eef_arrays`、`process_episode`、`main`、README：

顶部：
```python
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import aytsai_pipeline_config as cfg
from pin_eef_to_plane import intrinsics_matrix, pin_points_to_plane, rotate_oris_to_world
import sys
sys.path.insert(0, str(Path(cfg.get()["repo_root"]) / "interactive_world_sim" / "algorithms" / "latent_decompose"))
from ot_align import robot_world_to_cam  # type: ignore

c = cfg.get()
SRC_LEROBOT_ROOT = Path(c["repo_root"]) / "human_play_data"
SRC_PHANTOM_ROOT = Path(c["repo_root"]) / "phantom" / "data" / c["processed_full_subdir"] / f"play_human_{c['version']}"
DST_ROOT = Path(c["repo_root"]) / "human_play_eef_data"

_K_JSON = json.load(open(Path(c["repo_root"]) / "phantom_human_play" / "intrinsics_cam_high.json"))["left"]
K = intrinsics_matrix(_K_JSON["fx"], _K_JSON["fy"], _K_JSON["cx"], _K_JSON["cy"])
T_CAM_WORLD = robot_world_to_cam()
Z_PLANE = json.load(open(Path(c["repo_root"]) / f"z_plane_{c['version']}.json"))["z_plane"]
```
（`EEF_FEATURES` 字典保持不变；`copy_tree_concrete` 不变。）

`build_eef_arrays` 改为 pin 后写 robot-world：
```python
def build_eef_arrays(n_frames: int, phantom_ep_dir: Path) -> dict:
    raw = np.load(phantom_ep_dir / "action_processor" / "actions_right_single_arm.npz")
    sm = np.load(phantom_ep_dir / "smoothing_processor" / "smoothed_actions_right_single_arm.npz")
    idx = raw["union_indices"].astype(np.int64)
    ee_pts = sm["ee_pts"].astype(np.float64)            # (M,3) cam frame
    ee_oris = sm["ee_oris"].astype(np.float64)          # (M,3,3) cam frame
    ee_widths = sm["ee_widths"].astype(np.float32)      # (M,)

    assert idx.max() < n_frames, f"union_indices.max={idx.max()} >= n_frames={n_frames}"
    assert len(ee_pts) == len(idx)

    # Pin cam-frame eef onto robot's locked plane -> robot-world frame.
    p_world, _ = pin_points_to_plane(ee_pts, K, T_CAM_WORLD, Z_PLANE)   # (M,3), z==Z_PLANE
    oris_world = rotate_oris_to_world(ee_oris, T_CAM_WORLD)             # (M,3,3)

    pos = np.full((n_frames, 3), np.nan, dtype=np.float32)
    rot = np.full((n_frames, 9), np.nan, dtype=np.float32)
    width = np.full((n_frames, 1), np.nan, dtype=np.float32)
    det = np.zeros((n_frames, 1), dtype=bool)
    pos[idx] = p_world.astype(np.float32)
    rot[idx] = oris_world.reshape(-1, 9).astype(np.float32)
    width[idx, 0] = ee_widths
    det[idx, 0] = True
    return {
        "observation.eef.position_right": pos,
        "observation.eef.rotation_matrix_right": rot,
        "observation.eef.width_right": width,
        "observation.eef.detected_right": det,
    }
```

`process_episode` 路径走 cfg、README 实算检测率：
```python
def process_episode(ep: int) -> None:
    print(f"=== ep {ep} ===")
    src = SRC_LEROBOT_ROOT / c["human_repo_pattern"].format(n=ep).split("/")[-1]
    dst = DST_ROOT / c["eef_local_subdir"].format(n=ep)
    print(f"  copy {src} -> {dst}")
    copy_tree_concrete(src, dst)

    info_path = dst / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    n_frames = int(info["total_frames"])

    phantom_ep_dir = SRC_PHANTOM_ROOT / str(ep)
    eef_arrays = build_eef_arrays(n_frames, phantom_ep_dir)
    n_det = int(eef_arrays["observation.eef.detected_right"].sum())
    rate = 100.0 * n_det / n_frames
    print(f"  detected {n_det}/{n_frames} ({rate:.1f}%)")
    if rate < 70.0:
        print(f"  WARNING: low detection rate {rate:.1f}% for ep {ep}")

    parquet_path = dst / "data" / "chunk-000" / "episode_000000.parquet"
    rewrite_parquet(parquet_path, eef_arrays)
    update_info_json(info_path)
    update_readme(dst / "README.md", ep, rate)
    print(f"  done ep {ep}")
```

`update_readme(readme_path, ep, rate)` 重写说明坐标系语义：
```python
def update_readme(readme_path: Path, ep: int, rate: float) -> None:
    repo = c["human_repo_pattern"].format(n=ep)
    extra = f"""# {c['eef_local_subdir'].format(n=ep)}

LeRobot re-release of `{repo}` with right-hand end-effector annotations from
phantom (HaMeR) + smoothing. **EEF is in the ROBOT-WORLD frame** (one shared
coordinate system with the robot datasets), pinned onto the robot's locked
tabletop plane: x/y from the reliable 2D hand detection, z fixed to the constant
plane height z_plane = {Z_PLANE:.5f} m (mean of robot v3 action z).

| column | dtype | shape | meaning |
|---|---|---|---|
| `observation.eef.position_right` | float32 | (3,) | EE position in robot-world frame (m); z==z_plane; NaN if no detection |
| `observation.eef.rotation_matrix_right` | float32 | (9,) | EE orientation 3x3 (row-major) rotated into robot-world frame; NaN if none |
| `observation.eef.width_right` | float32 | (1,) | thumb-index distance (m), gripper-opening proxy; NaN if none |
| `observation.eef.detected_right` | bool | (1,) | True iff HaMeR detected the right hand |

Detection rate: {rate:.1f}% of frames. Original videos and parquet columns are
unchanged. Pinning uses cam_high intrinsics + URDF-derived robot-world<-cam
extrinsic (verified).
"""
    readme_path.write_text(extra)
```

`main()`：
```python
def main() -> None:
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    for ep in c["human_eps"]:
        process_episode(ep)
    print(f"\nAll {len(c['human_eps'])} datasets built under {DST_ROOT}")
```

- [ ] **Step 2: 运行 build + 核对（z 恒为 z_plane、列齐、检测 bool）**

Run:
```bash
cd /scr2/yusenluo/interactive_world_sim
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh && conda activate iws
python build_eef_dataset.py
python -c "
import pandas as pd, numpy as np, json
zp=json.load(open('z_plane_v3.json'))['z_plane']
df=pd.read_parquet('human_play_eef_data/play_human_v3_eef_1/data/chunk-000/episode_000000.parquet')
pos=np.stack(df['observation.eef.position_right'].to_numpy())
det=np.stack(df['observation.eef.detected_right'].to_numpy()).reshape(-1)
z=pos[det,2]
print('cols ok:', all(k in df.columns for k in ['observation.eef.position_right','observation.eef.width_right','observation.eef.detected_right']))
print('detected z min/max:', round(float(z.min()),5), round(float(z.max()),5), 'z_plane', round(zp,5))
assert np.allclose(z, zp, atol=1e-4), 'z not pinned to plane'
print('ok')
"
```
Expected: `cols ok: True`、detected z min==max==z_plane、`ok`

- [ ] **Step 3: Commit**

```bash
git add build_eef_dataset.py
git commit -m "feat(v3-eef): build v3 eef datasets in robot-world frame (plane-pinned z, computed detect rate)"
```

---

## Task 11: eef 可视化（v3_1/5/10）给用户终检

**Files:**
- Create: `viz_eef_v3.py`

- [ ] **Step 1: 写 viz —— 把 pin 后 eef 投影回 cam 系画在 human 视频上（短 clip）**

```python
"""Render eef overlay on a short clip for QC. Projects the plane-pinned eef
(robot-world -> cam via T_cam_world -> K) back onto the original human video;
the green dot should sit on the hand and the z label should be constant z_plane.
"""
import json
from pathlib import Path

import cv2
import numpy as np

import aytsai_pipeline_config as cfg
from pin_eef_to_plane import intrinsics_matrix, pin_points_to_plane
import sys
c = cfg.get()
sys.path.insert(0, str(Path(c["repo_root"]) / "interactive_world_sim" / "algorithms" / "latent_decompose"))
from ot_align import robot_world_to_cam  # type: ignore

KJ = json.load(open(Path(c["repo_root"]) / "phantom_human_play" / "intrinsics_cam_high.json"))["left"]
K = intrinsics_matrix(KJ["fx"], KJ["fy"], KJ["cx"], KJ["cy"])
T_CAM_WORLD = robot_world_to_cam()
Z_PLANE = json.load(open(Path(c["repo_root"]) / f"z_plane_{c['version']}.json"))["z_plane"]
N_FRAMES = 600  # ~20s @30fps


def render_ep(ep: int, out_dir: Path) -> Path:
    full = Path(c["repo_root"]) / "phantom" / "data" / c["processed_full_subdir"] / f"play_human_{c['version']}" / str(ep)
    raw = np.load(full / "action_processor" / "actions_right_single_arm.npz")
    sm = np.load(full / "smoothing_processor" / "smoothed_actions_right_single_arm.npz")
    idx = raw["union_indices"].astype(int)
    _, p_cam_pin = pin_points_to_plane(sm["ee_pts"].astype(np.float64), K, T_CAM_WORLD, Z_PLANE)
    by_frame = {int(f): p_cam_pin[i] for i, f in enumerate(idx)}

    src = SRC = Path(c["repo_root"]) / "human_play_data" / c["human_repo_pattern"].format(n=ep).split("/")[-1] \
        / "videos/chunk-000/observation.images.cam_high/episode_000000.mp4"
    cap = cv2.VideoCapture(str(src))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"eef_v3_ep{ep}.mp4"
    wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    fi = 0
    while fi < N_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        if fi in by_frame:
            p = by_frame[fi]
            uvw = K @ p
            u, v = int(uvw[0] / uvw[2]), int(uvw[1] / uvw[2])
            cv2.circle(frame, (u, v), 7, (0, 255, 0), -1)
            cv2.putText(frame, f"z={Z_PLANE:.3f}m", (u + 8, v),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"ep{ep} f{fi}", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        wr.write(frame)
        fi += 1
    cap.release(); wr.release()
    return out


if __name__ == "__main__":
    out_dir = Path(c["repo_root"]) / "outputs" / "v3_eef_qc"
    for ep in (1, 5, 10):
        p = render_ep(ep, out_dir)
        print("EEF QC:", p.resolve())
```

- [ ] **Step 2: 运行 + 打印绝对路径**

Run:
```bash
cd /scr2/yusenluo/interactive_world_sim
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh && conda activate iws
python viz_eef_v3.py
```
Expected: 打印 3 个 `EEF QC: /scr2/.../outputs/v3_eef_qc/eef_v3_ep{1,5,10}.mp4`

- [ ] **Step 3: 人工核验关卡（STOP — 给用户看）**

把 3 个 `eef_v3_ep*.mp4` 绝对路径贴给用户：绿点应落在右手 eef 处、z 标注恒为 z_plane。通过 → Task 12；不通过 → 回查 bbox/检测或 pin。

- [ ] **Step 4: Commit**

```bash
git add viz_eef_v3.py
git commit -m "feat(v3-eef): eef overlay QC viz (v3_1/5/10, plane-pinned reprojection)"
```

---

## Task 12: 上传到 HF（用户放行后）

**Files:**
- Modify: `push_eef_dataset.py`

- [ ] **Step 1: 改 push 走配置（命名 `play_human_v3_{N}_eef`）**

替换 `push_eef_dataset.py` 全文：
```python
"""Push locally-built v3 EEF datasets to HF. Run ONLY after the user approves
the QC videos under outputs/v3_eef_qc/."""
from pathlib import Path

from huggingface_hub import HfApi

import aytsai_pipeline_config as cfg

c = cfg.get()
LOCAL_ROOT = Path(c["repo_root"]) / "human_play_eef_data"


def push_one(api: HfApi, ep: int) -> None:
    repo_id = c["hf_repo_pattern"].format(n=ep)
    folder = LOCAL_ROOT / c["eef_local_subdir"].format(n=ep)
    assert folder.exists(), f"missing {folder}"
    print(f"\n=== {repo_id}  <-  {folder} ===")
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=False)
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(folder),
                      commit_message=f"v3 EEF (robot-world frame, plane-pinned z): {c['human_repo_pattern'].format(n=ep)}")
    print(f"   pushed {repo_id}")


def main() -> None:
    api = HfApi()
    for ep in c["human_eps"]:
        push_one(api, ep)
    print(f"\nAll {len(c['human_eps'])} datasets pushed.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 上传（需用户明确放行 + HF 登录态）**

Run:
```bash
cd /scr2/yusenluo/interactive_world_sim
source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh && conda activate iws
python push_eef_dataset.py
```
Expected: 12 行 `pushed yusenluo9z/play_human_v3_{n}_eef`，末尾 `All 12 datasets pushed.`

- [ ] **Step 3: 核对 HF 上线**

Run:
```bash
python -c "
from huggingface_hub import HfApi
api=HfApi()
for n in range(1,13):
    rid=f'yusenluo9z/play_human_v3_{n}_eef'
    try: api.dataset_info(rid); print('OK', rid)
    except Exception as e: print('NO', rid, e)
"
```
Expected: 12 行 `OK ...`

- [ ] **Step 4: Commit**

```bash
git add push_eef_dataset.py
git commit -m "feat(v3-eef): push v3 eef datasets to yusenluo9z/play_human_v3_{n}_eef"
```

---

## Self-Review 备注（写计划时已核对）

- **Spec 覆盖**：§1 数据范围→T2/T5；§2 变换→T3（robot_world_to_cam 复用）；§3 plane-pin→T2/T3/T4/T10；§4 流水线→T5/T7/T8/T9/T10；§4 bbox-gate→T7；§5 viz→T7(bbox)/T11(eef)；§6 代码组织→T1 配置+各脚本版本化；§7 成本→T8 拆批；§9 风险 R1→T7 gate / R2→T7 Step4 机位确认 / R3→T10 检测率告警 / R5→T10 README 说明。
- **命名一致**：`pin_points_to_plane` / `rotate_oris_to_world` / `intrinsics_matrix` / `robot_world_to_cam` / `aytsai_pipeline_config.get()` / `cfg.demo_num()` 全计划统一。
- **依赖**：build(T10) 依赖 z_plane_v3.json(T4)、pin(T3)、smooth(T9)；viz(T11) 依赖同上；执行顺序即任务顺序。
- **环境**：phantom 步骤(T7/T8) 用 `conda activate phantom`；其余用 `iws`。
