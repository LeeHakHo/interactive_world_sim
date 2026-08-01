import numpy as np, sys
sys.path.insert(0, ".")
import idm_data as D
Z = np.load("outputs/flow_render_dataset_can_dual/clips_robot_retrack.npz")


def test_split_no_leak():
    tr, ho = D.episode_split(Z)
    vids = Z["vid"]
    assert set(vids[ho].tolist()) <= set(D.HELDOUT_VIDS)
    assert set(vids[tr].tolist()).isdisjoint(D.HELDOUT_VIDS)
    assert len(tr) > 0 and len(ho) > 0


def test_action_is_delta_joint():
    X, Y, meta = D.build_windows(Z, D.INPUT_SPECS["A1"], clips=[0, 1, 2])
    i = np.where((meta["t"] > 5) & (meta["t"] < 40))[0][0]
    ci, t = meta["clip_idx"][i], meta["t"][i]
    exp = Z["joint"][ci, t + 1, :6] - Z["joint"][ci, t, :6]
    assert np.allclose(Y[i, :6], exp, atol=1e-4)
    assert np.isclose(Y[i, 6], Z["grip"][ci, t], atol=1e-4)


def test_input_dim_matches_spec():
    for arm in ["A0", "A1", "A2"]:
        X, Y, meta = D.build_windows(Z, D.INPUT_SPECS[arm], clips=[0, 1])
        assert X.ndim == 2 and X.shape[0] == Y.shape[0]
        assert X.shape[1] == D.input_dim(D.INPUT_SPECS[arm])
    d0 = D.input_dim(D.INPUT_SPECS["A0"]); d1 = D.input_dim(D.INPUT_SPECS["A1"]); d2 = D.input_dim(D.INPUT_SPECS["A2"])
    assert d0 < d1 < d2


def test_causal_window_smaller():
    assert D.input_dim(D.INPUT_SPECS["A3"]) < D.input_dim(D.INPUT_SPECS["A1"])
