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
