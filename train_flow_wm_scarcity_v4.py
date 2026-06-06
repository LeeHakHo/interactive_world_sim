"""Step 3 v4: thick flow dynamic latent (contact-gated motion + anti-drift training).

Extends train_flow_wm_scarcity_v3.py. `--thin` reproduces v3 exactly (ablation via
flag, not a fork). Reads outputs/flow_dataset/flow_ds_v3.npz. See
docs/superpowers/specs/2026-06-06-thicken-flow-latent-design.md.
"""
import argparse, os, numpy as np, torch, torch.nn as nn

DS = "outputs/flow_dataset/flow_ds_v3.npz"
K, F, L, Dm, P = 4, 12, 16, 128, 48
EPOCHS, BS, LR = 60, 128, 3e-4
TEST_ROBOT_VID = 12
N_LIST = [1400, 400, 200, 100, 50]
SEEDS = 5
LAMBDA_GATE = 1e-2
NOISE_STD = 0.01
LAMBDA_CONSIST = 0.5
STATIC_TAU = 0.02
device = "cuda" if torch.cuda.is_available() else "cpu"


class FlowWMThick(nn.Module):
    def __init__(self, P, thin=False):
        super().__init__()
        self.thin = thin
        self.inp = nn.Linear(2 * K + 2, Dm)
        self.act = nn.Linear(L * 3 * 2, Dm)            # full-window EEF as action
        enc = nn.TransformerEncoderLayer(Dm, 4, Dm * 2, batch_first=True, dropout=0.0)
        self.tf = nn.TransformerEncoder(enc, 3)
        self.head = nn.Linear(Dm, F * 2)
        if not thin:
            self.gctx = nn.Linear(L, Dm)               # grasp-sequence token (thick only)
            self.gate = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, hist, eef3, g):
        # hist (B,P,K,2), eef3 (B,L,3,2), g (B,L) normalized grasp
        B, P = hist.shape[:2]
        anchor = hist[:, :, -1, :]                                       # (B,P,2)
        obj = self.inp(torch.cat([(hist - anchor[:, :, None]).reshape(B, P, 2 * K), anchor], -1))
        objc = anchor.mean(1, keepdim=True)                              # (B,1,2)
        act = self.act((eef3 - objc[:, :, None]).reshape(B, -1))[:, None, :]
        tokens = [obj, act]
        if not self.thin:
            tokens.append(self.gctx(g)[:, None, :])
        x = self.tf(torch.cat(tokens, 1))[:, :P]                         # (B,P,Dm)
        raw = self.head(x).reshape(B, P, F, 2)                           # displacement from anchor
        if self.thin:
            return anchor[:, :, None, :] + raw, None
        feat = contact_gate_features(anchor, eef3, g)                    # (B,P,F,3)
        alpha = torch.sigmoid(self.gate(feat)).squeeze(-1)              # (B,P,F)
        return anchor[:, :, None, :] + alpha[..., None] * raw, alpha


# --- pure helpers (filled in later tasks) ---
def contact_gate_features(anchor, eef3, g):
    # anchor (B,P,2), eef3 (B,L,3,2), g (B,L) -> (B,P,F,3): [dist to 2 future tips, grasp]
    B, Pn = anchor.shape[:2]
    tips = eef3[:, K:, 1:3, :]                                   # (B,F,2,2) future 2 tips
    d = torch.linalg.norm(anchor[:, :, None, None, :] - tips[:, None], dim=-1)  # (B,P,F,2)
    gf = g[:, K:][:, None, :].expand(B, Pn, F)                   # (B,P,F)
    return torch.cat([d, gf[..., None]], -1)                     # (B,P,F,3)


def grasp_openness(eef3):
    # eef3 (...,L,3,2) -> (...,L): distance between the two tips (points 1,2)
    return torch.linalg.norm(eef3[..., 1, :] - eef3[..., 2, :], dim=-1)


def fit_grasp_stats(g_raw, domain):
    # per-domain p5/p95 over flattened grasp values
    g_flat = g_raw.reshape(len(g_raw), -1) if g_raw.dim() > 1 else g_raw[:, None]
    stats = {}
    for d in domain.unique().tolist():
        v = g_flat[domain == d].reshape(-1)
        stats[int(d)] = (torch.quantile(v, 0.05).item(), torch.quantile(v, 0.95).item())
    return stats


def normalize_grasp(g_raw, domain, stats):
    out = torch.zeros_like(g_raw)
    for d, (lo, hi) in stats.items():
        m = domain == d
        out[m] = ((g_raw[m] - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)
    return out


def inject_state_noise(hist, std, generator):
    # additive Gaussian noise on history object points (normalized coords); cheap DAgger
    noise = torch.randn(hist.shape, generator=generator, device=hist.device) * std
    return hist + noise


def multi_step_consistency(model, tr, eef3, g):
    # tr (B,L,P,2) -> (cons_pred, cons_tgt) each (B,P,overlap,2), overlap = L-2K
    B = tr.shape[0]
    hist1 = tr[:, :K].permute(0, 2, 1, 3)                       # (B,P,K,2)
    pred1, _ = model(hist1, eef3, g)                            # (B,P,F,2) frames K..L-1
    hist2 = pred1[:, :, :K, :]                                  # predicted frames K..2K-1
    pred2, _ = model(hist2, eef3, g)                            # frames 2K..2K+F-1 (own-feedback)
    overlap = L - 2 * K                                         # frames 2K..L-1 still have GT
    cons_pred = pred2[:, :, :overlap, :]
    cons_tgt = tr[:, 2 * K:].permute(0, 2, 1, 3)               # (B,P,overlap,2)
    return cons_pred, cons_tgt


def _centroid_path_len(seq):
    # seq (B,T,P,2) -> (B,) total centroid path length in normalized coords
    c = seq.mean(2)                                            # (B,T,2)
    return torch.linalg.norm(torch.diff(c, dim=1), dim=-1).sum(1)


def rollout_drift_static(pred, gt, tau):
    # pred,gt (B,F,P,2); restrict to GT-static clips, report mean predicted centroid
    # path length there, in pixels (x224). Returns (drift_px, n_static).
    gt_len = _centroid_path_len(gt)
    static = gt_len < tau
    n = int(static.sum().item())
    if n == 0:
        return 0.0, 0
    drift = (_centroid_path_len(pred)[static].mean().item()) * 224.0
    return drift, n


def load():
    z = np.load(DS)
    return (torch.from_numpy(z["tracks"]).float(), torch.from_numpy(z["vis"]).float(),
            torch.from_numpy(z["eef3"]).float(), torch.from_numpy(z["domain"]).long(),
            torch.from_numpy(z["vid"]).long())


def cg_descriptor(tracks, eef3, g_norm):
    # tracks (N,L,P,2), eef3 (N,L,3,2), g_norm (N,L) -> (N, L*2 + L)
    c = tracks.mean(2)                                          # (N,L,2) object centroid
    tips = eef3[:, :, 1:3, :]                                   # (N,L,2,2)
    d = torch.linalg.norm(c[:, :, None, :] - tips, dim=-1)      # (N,L,2)
    return torch.cat([d.reshape(len(d), -1), g_norm], -1)


def run_probe_cg(out_dir="outputs/flow_wm_v4/probe_cg"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    os.makedirs(out_dir, exist_ok=True)
    tr, vis, eef3, dom, vid = load()
    gstats = fit_grasp_stats(grasp_openness(eef3), dom)
    g_norm = normalize_grasp(grasp_openness(eef3), dom, gstats)
    X = cg_descriptor(tr, eef3, g_norm).numpy(); y = dom.numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    acc = clf.score(Xte, yte); base = max((yte == 0).mean(), (yte == 1).mean())
    msg = (f"[c,g] admission probe (converged LogReg, held-out)\n"
           f"  test acc = {acc:.3f}  (majority baseline = {base:.3f})\n"
           f"  verdict  = {'PASS ~0.5, admissible' if acc < base + 0.10 else 'FAIL leaks domain -> re-normalize/drop'}\n"
           f"  N={len(X)} dim={X.shape[1]}\n")
    print(msg, flush=True)
    open(os.path.join(out_dir, "summary.txt"), "w").write(msg)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-cg", action="store_true")
    ap.add_argument("--phase", type=int, default=0)
    a = ap.parse_args()
    if a.probe_cg:
        run_probe_cg()
    elif a.phase == 1:
        run_phase1()
    elif a.phase == 2:
        run_phase2()
    else:
        print("specify --probe-cg | --phase 1 | --phase 2", flush=True)


if __name__ == "__main__":
    main()


def _masked_mse(pred, fut, w):
    return ((pred - fut) ** 2 * w).sum() / (w.sum() + 1e-6)


def train_eval(tr, vis, eef3, dom, gstats, train_idx, test_idx, thin, seed, epochs=EPOCHS):
    torch.manual_seed(seed)
    g_all = normalize_grasp(grasp_openness(eef3), dom, gstats)          # (N,L)
    m = FlowWMThick(P, thin=thin).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(seed)
    ngen = torch.Generator(device=device).manual_seed(seed + 777)

    def batches(idx, train):
        idx = idx[torch.randperm(len(idx), generator=gen)] if train else idx
        for i in range(0, len(idx), BS):
            yield idx[i:i + BS]

    for ep in range(epochs):
        m.train()
        for b in batches(train_idx, True):
            h = tr[b, :K].permute(0, 2, 1, 3).to(device)               # (B,P,K,2)
            if not thin:
                h = inject_state_noise(h, NOISE_STD, ngen)
            fut = tr[b, K:].permute(0, 2, 1, 3).to(device)             # (B,P,F,2)
            ef = eef3[b].to(device); gg = g_all[b].to(device)
            hv = vis[b, :K].to(device); fv = vis[b, K:].to(device)
            w = (fv.permute(0, 2, 1) * hv[:, -1:].permute(0, 2, 1))[..., None]
            pred, alpha = m(h, ef, gg)
            loss = _masked_mse(pred, fut, w)
            if not thin:
                loss = loss + LAMBDA_GATE * alpha.abs().mean()
                cp, ct = multi_step_consistency(m, tr[b].to(device), ef, gg)
                wc = fv[:, K:].permute(0, 2, 1)[..., None]             # (B,P,overlap,1)
                loss = loss + LAMBDA_CONSIST * _masked_mse(cp, ct, wc)
            opt.zero_grad(); loss.backward(); opt.step()

    m.eval(); an = ad = fn = fd = 0.0; preds = []; futs = []
    with torch.no_grad():
        for b in batches(test_idx, False):
            h = tr[b, :K].permute(0, 2, 1, 3).to(device)
            fut = tr[b, K:].permute(0, 2, 1, 3).to(device)
            ef = eef3[b].to(device); gg = g_all[b].to(device)
            hv = vis[b, :K].to(device); fv = vis[b, K:].to(device)
            w = fv.permute(0, 2, 1) * hv[:, -1:].permute(0, 2, 1)
            pred, _ = m(h, ef, gg)
            err = torch.linalg.norm(pred - fut, dim=-1) * 224.0
            an += (err * w).sum().item(); ad += w.sum().item()
            fn += (err[..., -1] * w[..., -1]).sum().item(); fd += w[..., -1].sum().item()
            preds.append(pred.permute(0, 2, 1, 3).cpu()); futs.append(fut.permute(0, 2, 1, 3).cpu())
    pred_all = torch.cat(preds); fut_all = torch.cat(futs)              # (Ntest,F,P,2)
    drift, n_static = rollout_drift_static(pred_all, fut_all, STATIC_TAU)
    return {"ade": an / ad, "fde": fn / fd, "drift": drift, "n_static": n_static}
