import os
import numpy as np
os.environ.setdefault("DS", "outputs/flow_render_dataset_can_dual/clips_robot_retrack.npz")
import exp_scel_dualview_wm as W


def test_synthetic_disjoint_and_okfiltered():
    vid = np.array([0, 0, 1, 1, 2, 2])
    ok = np.array([True, True, True, False, True, True])
    ho, pool = W.split_by_episode(ok, vid, [0])
    assert set(ho.tolist()) == {0, 1}                 # vid0 的 clip
    assert set(pool.tolist()) == {2, 4, 5}            # vid1(idx2)+vid2(idx4,5); idx3 被 ok 滤掉
    assert set(vid[ho].tolist()).isdisjoint(set(vid[pool].tolist()))   # vid 集合不相交
    assert ok[ho].all() and ok[pool].all()            # 都 ok-过滤


def test_heldout_vids_str_or_int():
    vid = np.array([100, 100, 102, 103])
    ok = np.ones(4, bool)
    ho, pool = W.split_by_episode(ok, vid, ["100", "102"])   # 字符串也行
    assert set(vid[ho].tolist()) == {100, 102}
    assert set(vid[pool].tolist()) == {103}


def test_real_robot_no_leak_and_demo_in_heldout():
    z = np.load(os.environ["DS"], mmap_mode="r")
    vid = np.asarray(z["vid"]); ok = np.asarray(z["low_valid"])
    ho, pool = W.split_by_episode(ok, vid, [100, 102])
    # 1) demo seq 全在 heldout
    for s in (59, 332, 418, 442):
        assert s in set(ho.tolist()), f"demo clip{s} 不在 heldout"
    # 2) heldout 只含 vid100/102, 训练池完全不含 → episode 不相交(无跨-episode 帧泄漏)
    assert set(vid[ho].tolist()) <= {100, 102}
    assert set(vid[pool].tolist()).isdisjoint({100, 102})
    # 3) heldout 与训练池的 vid 集合零交集(泄漏根因清除)
    assert set(vid[ho].tolist()).isdisjoint(set(vid[pool].tolist()))
    # 4) 规模合理(2 episode heldout ~279, 16 episode pool ~2311)
    assert 200 < len(ho) < 320 and 2000 < len(pool) < 2400
