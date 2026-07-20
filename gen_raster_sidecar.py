"""预计算 agent-centric 局部光栅图 sidecar(训练时直接读, 免去 on-the-fly cv2 循环).
存 uint8(0/255) 省空间. 见 augment_clips_fullchain.rasterize_local 的设计说明.
"""
import numpy as np
from augment_clips_fullchain import rasterize_local

DS = "outputs/flow_render_dataset_can_dual"
SIZE = 64

for dom, src, out in (("r", "chain_sidecar_robot", "raster_sidecar_robot"),
                      ("h", "chain_sidecar_human", "raster_sidecar_human")):
    z = np.load(f"{DS}/{src}.npz")
    segs = z["segments"]
    d = {}
    for view, key in ((0, "chain2d_high"), (1, "chain2d_low")):
        P = z[key]
        N, L, J, _ = P.shape
        R = rasterize_local(P.reshape(-1, J, 2), segs, dom, size=SIZE).reshape(N, L, SIZE, SIZE)
        d[f"raster_v{view}"] = (R * 255).astype(np.uint8)
        print(f"  {dom} view{view}: {R.shape}  白像素占比 {R.mean():.4f}", flush=True)
    np.savez_compressed(f"{DS}/{out}.npz", size=SIZE, **d)
    print(f"{dom} -> {DS}/{out}.npz")
