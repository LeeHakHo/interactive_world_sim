import os, subprocess
import pytest


def test_eval_smoke():
    ck = "outputs/idm_derisk/A1/idm.pt"
    if not os.path.exists(ck):
        pytest.skip("需先训 A1(idm_train.py ARM=A1)")
    env = {**os.environ, "ARM": "A1", "SEQS": "0", "OUT_DIR": "outputs/idm_derisk/A1", "OMP_NUM_THREADS": "4"}
    r = subprocess.run(["python", "idm_eval.py"], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert os.path.exists("outputs/idm_derisk/A1/eval_summary.txt")
