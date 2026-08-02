"""Push locally-built EEF datasets to HF (version-aware). Run ONLY after the user
approves the QC videos. Creates PUBLIC repos hf_repo_pattern (default v3 ->
yusenluo9z/play_human_v3_{n}_eef)."""
import os
from pathlib import Path

from huggingface_hub import HfApi

import aytsai_pipeline_config as cfg

c = cfg.get()
VER = os.environ.get("AYTSAI_VERSION", "v3")           # ★commit message 随版本(v3/can), 别硬编码v3
LOCAL_ROOT = Path(c["repo_root"]) / "human_play_eef_data"


def push_one(api: HfApi, ep: int) -> None:
    repo_id = c["hf_repo_pattern"].format(n=ep)
    folder = LOCAL_ROOT / c["eef_local_subdir"].format(n=ep)
    assert folder.exists(), f"missing {folder}"
    print(f"\n=== {repo_id}  <-  {folder} ===")
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=False)
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(folder),
                      commit_message=f"{VER} human EEF (HaMeR+depth 3D, robot-world frame, 21kpt + 3pt crop128 2D): "
                                     f"{c['human_repo_pattern'].format(n=ep)}")
    print(f"   pushed {repo_id}")


def main() -> None:
    api = HfApi()
    for ep in c["human_eps"]:
        push_one(api, ep)
    print(f"\nAll {len(c['human_eps'])} datasets pushed.")


if __name__ == "__main__":
    main()
