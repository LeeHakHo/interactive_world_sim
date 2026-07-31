import numpy as np
from build_object_heatmap import heatmap_at, build_clip_heatmap


def test_heatmap_peak_at_center():
    h = heatmap_at(0.5, 0.5)
    assert h.shape == (16, 16)
    peak = np.unravel_index(h.argmax(), h.shape)
    assert abs(peak[0] - 8) <= 1 and abs(peak[1] - 8) <= 1   # 峰在中心附近
    assert h.max() > 0.9


def test_build_clip_heatmap_shape_and_tracks_object():
    L = 48
    tr = np.full((L, 6, 2), 0.3, np.float32)                 # 物体在 (0.3,0.3)
    trl = np.full((L, 6, 2), 0.7, np.float32)
    out = build_clip_heatmap(tr, trl, tL=6, L=L)
    assert out.shape == (2, 1, 6, 16, 16)
    # high 视角峰应在 0.3*16≈4.8
    peak_hi = np.unravel_index(out[0, 0, 0].argmax(), (16, 16))
    assert abs(peak_hi[1] - 4.8) <= 1.5
