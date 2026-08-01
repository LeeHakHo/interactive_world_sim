import numpy as np
from build_sketch import build_clip_sketch, load_robot_arrays, load_human_arrays


def test_robot_sketch_shape_and_channels():
    A = load_robot_arrays()
    out = build_clip_sketch(A, 356, tL=6, L=48, src="robot")
    assert out.shape == (2, 9, 6, 16, 16)
    assert np.abs(out[0, :4]).sum() > 0        # flow+skel high 有内容
    assert all(np.allclose(out[0, 5, k], out[0, 5, k].flat[0]) for k in range(6))  # grip(ch5)每帧空间恒定(广播)


def test_human_sketch_shape():
    A = load_human_arrays()
    n = int(A["_valid_idx"][0])
    out = build_clip_sketch(A, n, tL=6, L=48, src="human")
    assert out.shape == (2, 9, 6, 16, 16)
    assert np.abs(out[0, 4]).sum() > 0        # human skel(ch4) high 非空
