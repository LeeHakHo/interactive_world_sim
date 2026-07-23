import numpy as np


def test_fk_skel2d_shape():
    from fk_skel2d import fk_skel2d_dual
    joint = np.zeros((3, 7), np.float32)
    skel2d_high, skel2d_low, segments = fk_skel2d_dual(joint)
    assert skel2d_high.shape == (3, 9, 2)
    assert skel2d_low.shape == (3, 9, 2)
    assert segments.shape == (8, 2)


def test_fk_matches_sidecar():
    from fk_skel2d import fk_skel2d_dual
    clips = np.load("outputs/flow_render_dataset_can_dual/clips_robot.npz", mmap_mode="r")
    sidecar = np.load("outputs/flow_render_dataset_can_dual/skel_sidecar_robot.npz")
    si = 332
    joint = np.asarray(clips["joint"][si]).astype(np.float32)  # (48,7)
    skel2d_high, skel2d_low, segments = fk_skel2d_dual(joint)

    gt_high = np.asarray(sidecar["skel2d_high"][si])  # (48,9,2)
    gt_low = np.asarray(sidecar["skel2d_low"][si])

    err_high = np.abs(skel2d_high - gt_high)
    err_low = np.abs(skel2d_low - gt_low)

    med_high = np.median(err_high) * 128
    med_low = np.median(err_low) * 128
    assert med_high < 2.0, f"cam_high median px err {med_high:.3f} >= 2px"
    assert med_low < 2.0, f"cam_low median px err {med_low:.3f} >= 2px"

    np.testing.assert_array_equal(segments, sidecar["segments"])
