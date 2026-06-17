"""Cheap de-risk diagnostic BEFORE building contact-aware ②: can a 2D shared contact proxy
(eef-cube distance + motion correlation) predict whether the GT cube actually FOLLOWS the eef?

thick-latent's contact gate degenerated because 2D distance was a weak proxy. This tests, on GT
data (no ② needed), whether the proxy discriminates 'cube follows eef' from 'cube ignores eef'.
If yes -> contact-aware ② has a basis. If no -> 2D contact is fundamentally weak (need wrist-cam
robot-side supervision or another route). Robot + human both computed (signal must be shared).
Output: outputs/cross_embodiment_wm/contact_diag/summary.txt"""
import os, numpy as np
from sklearn.metrics import roc_auc_score
os.makedirs("outputs/cross_embodiment_wm/contact_diag", exist_ok=True)
OUT = "outputs/cross_embodiment_wm/contact_diag"
DS = "outputs/flow_render_dataset_v3"


def analyze(tag, tr, ef, K=4):
    """TEMPORAL (no circularity): use PAST hist [0..K-1] proxies to predict FUTURE [K..L-1] follow.
    tr (N,L,48,2), ef (N,L,3,2)."""
    cc = tr.mean(2); ec = ef.mean(2)                                          # (N,L,2)
    # past (hist) signals
    cvel_h = cc[:, 1:K] - cc[:, :K - 1]; evel_h = ec[:, 1:K] - ec[:, :K - 1]  # (N,K-1,2)
    past_corr = ((cvel_h * evel_h).sum(-1) /
                 (np.linalg.norm(cvel_h, axis=-1) * np.linalg.norm(evel_h, axis=-1) + 1e-6)).mean(1)  # (N,)
    past_dist = np.linalg.norm(cc[:, K - 1] - ec[:, K - 1], axis=-1)          # (N,)
    # future net displacement [K-1 -> L-1]
    cnet = cc[:, -1] - cc[:, K - 1]; enet = ec[:, -1] - ec[:, K - 1]          # (N,2)
    cs = np.linalg.norm(cnet, axis=-1); es = np.linalg.norm(enet, axis=-1)
    fcos = (cnet * enet).sum(-1) / (cs * es + 1e-6)
    m = es > np.percentile(es, 50)                                            # eef moves in future
    follows = ((fcos > 0.5) & (cs > np.percentile(cs[m], 50))).astype(int)
    f, pc, pd = follows[m], past_corr[m], past_dist[m]
    out = [f"[{tag}] clips(eef-moving-future)={m.sum()}  P(future cube follows eef)={f.mean():.3f}"]
    if 0 < f.mean() < 1:
        out.append(f"  AUC PAST-motion-corr -> FUTURE-follow = {roc_auc_score(f, pc):.3f}   (>0.7 usable, ~0.5 useless)")
        out.append(f"  AUC PAST-dist        -> FUTURE-follow = {roc_auc_score(f, -pd):.3f}")
        out.append(f"  AUC PAST-corr+dist   -> FUTURE-follow = {roc_auc_score(f, pc - pd / (pd.std() + 1e-6)):.3f}")
    return "\n".join(out)


def main():
    lines = ["Contact proxy de-risk: can 2D (eef-cube dist + motion corr) predict 'GT cube follows eef'?",
             "AUC>0.7 => proxy usable as contact gate => contact-aware ② has basis. ~0.5 => weak (thick-latent repeat).", ""]
    for tag in ("robot", "human"):
        z = np.load(f"{DS}/clips_{tag}.npz")
        lines.append(analyze(tag, z["tracks"].astype(np.float32), z["eef"].astype(np.float32)))
        lines.append("")
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines) + f"\nsaved {OUT}/\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
