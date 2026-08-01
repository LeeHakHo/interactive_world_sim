import numpy as np, sys
sys.path.insert(0, ".")
import idm_fk as FK
Z = np.load("outputs/flow_render_dataset_can_dual/clips_robot_retrack.npz")


def test_fk_gt_joint_matches_gt_eef():
    ci = 0
    eh, el = FK.fk_eef2d(Z["joint"][ci], Z["grip"][ci])   # (48,3,2) each view
    gt = np.nan_to_num(Z["eef"][ci].astype(np.float32))    # GT eef high (48,3,2)
    ok = np.all(np.abs(gt) < 3, axis=(-1, -2))
    err = np.linalg.norm(eh[ok] - gt[ok], axis=-1).mean() * 128
    assert err < 8.0, f"FK-eef vs GT-eef 误差 {err:.1f}px 太大 -> EEF_SKEL_IDX 可能不对"
