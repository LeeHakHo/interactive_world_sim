"""Diagnostic: WHY does agent-frame kill human-helps (and harm at N=20)?
Probe human-vs-robot separability of object-motion in world frame vs agent-frame, using a
CONVERGED LogReg on a balanced, held-out split (per the 'probe must converge' lesson).
Feature = per-clip centroid velocity sequence ((L-1)*2). Hypothesis: if agent-frame motion is
MORE separable than world, then under agent-frame human transitions are more out-of-domain for
robot -> human injects domain-mismatch noise (explains afΔ<0 at low N)."""
import os, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
import scel_repr as S

DS = "outputs/flow_render_dataset_v3"
OUT = "outputs/cross_embodiment_wm/scel_transition_probe"; os.makedirs(OUT, exist_ok=True)


def feats(tr, ef, agentframe):
    """tr (N,L,48,2), ef (N,L,3,2) -> per-clip centroid velocity sequence (N,(L-1)*2)."""
    if agentframe:
        tr = S.to_agent_frame(tr, S.grasp_frame(ef))
    c = tr.mean(2)                               # (N,L,2) centroid
    return (c[:, 1:] - c[:, :-1]).reshape(len(tr), -1)


def probe(Xr, Xh, seed=0):
    """balanced subsample, 70/30 split, converged LogReg, balanced held-out accuracy. chance=0.5."""
    rng = np.random.default_rng(seed)
    n = min(len(Xr), len(Xh))
    Xr = Xr[rng.choice(len(Xr), n, replace=False)]
    Xh = Xh[rng.choice(len(Xh), n, replace=False)]
    X = np.concatenate([Xr, Xh]); y = np.concatenate([np.ones(n), np.zeros(n)])
    idx = rng.permutation(len(X)); X, y = X[idx], y[idx]
    ntr = int(0.7 * len(X))
    sc = StandardScaler().fit(X[:ntr])
    clf = LogisticRegression(max_iter=5000).fit(sc.transform(X[:ntr]), y[:ntr])
    return balanced_accuracy_score(y[ntr:], clf.predict(sc.transform(X[ntr:])))


def main():
    zr = np.load(f"{DS}/clips_robot.npz"); zh = np.load(f"{DS}/clips_human.npz")
    r_tr, r_ef = zr["tracks"].astype(np.float32), zr["eef"].astype(np.float32)
    h_tr, h_ef = zh["tracks"].astype(np.float32), zh["eef"].astype(np.float32)
    lines = ["SCEL transition separability probe (human vs robot) | converged LogReg, balanced held-out | chance=0.5",
             "feature = per-clip centroid velocity sequence",
             f"{'frame':>12} | {'probe acc':>9}"]
    for agentframe in (False, True):
        Xr = feats(r_tr, r_ef, agentframe); Xh = feats(h_tr, h_ef, agentframe)
        accs = [probe(Xr, Xh, seed=s) for s in range(3)]
        tag = "agent-frame" if agentframe else "world"
        lines.append(f"{tag:>12} | {np.mean(accs):9.3f}  (seeds {np.round(accs,3)})")
        print(lines[-1], flush=True)
    lines += ["",
              "Higher under agent-frame => structure EXPOSES embodiment diff in relative motion",
              "  -> human is more out-of-domain -> explains afΔ<=0 (human stops helping / hurts).",
              "Lower under agent-frame => motion more shared, human SHOULD help -> different cause."]
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines) + f"\nsaved {OUT}/\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
