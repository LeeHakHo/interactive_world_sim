import os, subprocess


def test_train_smoke(tmp_path):
    out = str(tmp_path / "A1")
    env = {**os.environ, "ARM": "A1", "EPOCHS": "3", "SMOKE": "1", "OUT_DIR": out, "OMP_NUM_THREADS": "4"}
    r = subprocess.run(["python", "idm_train.py"], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    import torch
    ck = torch.load(f"{out}/idm.pt", map_location="cpu", weights_only=False)
    assert ck["spec"]["trace"] is True and "state" in ck and ck["din"] > 0
