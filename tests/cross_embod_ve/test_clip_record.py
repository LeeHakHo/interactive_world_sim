import numpy as np
from interactive_world_sim.cross_embod_ve.data.embodiment_image import crop_around
from interactive_world_sim.cross_embod_ve.data.clip_record import assemble_clip


def test_crop_around_clamps_to_bounds():
    img = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)
    c = crop_around(img, center_uv=(5, 5), size=128)   # 角落，需 clamp
    assert c.shape == (128, 128, 3)
    c2 = crop_around(img, center_uv=(320, 240), size=128)
    assert c2.shape == (128, 128, 3)


def test_assemble_clip_shapes():
    T, RES = 25, 256
    rec = assemble_clip(
        frames=np.zeros((T, RES, RES, 3), np.uint8),
        agent_mask=np.zeros((T, RES, RES), bool),
        eef=np.zeros((T, 8), np.float32),
        obj_traj=np.full((T, 2, 3), np.nan, np.float32),
        embodiment_image=np.zeros((128, 128, 3), np.uint8),
        domain="human",
        meta={"source_id": "chunk_1000", "frame_start": 0, "fps": 30, "crop": [195, 195, 256, 256]},
    )
    assert rec["frames"].shape == (T, RES, RES, 3)
    assert rec["eef"].shape == (T, 8)
    assert rec["obj_traj"].shape == (T, 2, 3)
    assert str(rec["domain"]) == "human"
    assert rec["meta"].item()["source_id"] == "chunk_1000"
