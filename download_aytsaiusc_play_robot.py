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
