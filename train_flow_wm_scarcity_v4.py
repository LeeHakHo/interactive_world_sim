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
        self.dist_stats = None
        self.inp = nn.Linear(2 * K + 2, Dm)
        self.act = nn.Linear(L * 3 * 2, Dm)            # full-window EEF as action
        enc = nn.TransformerEncoderLayer(Dm, 4, Dm * 2, batch_first=True, dropout=0.0)
        self.tf = nn.TransformerEncoder(enc, 3)
        self.head = nn.Linear(Dm, F * 2)
        if not thin:
            self.gctx = nn.Linear(L, Dm)               # grasp-sequence token (thick only)
            self.gate = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, hist, eef3, g, dom):
        # hist (B,P,K,2), eef3 (B,L,3,2), g (B,L) normalized grasp, dom (B,)
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
        feat = contact_gate_features(anchor, eef3, g, dom, self.dist_stats)  # (B,P,F,3)
        alpha = torch.sigmoid(self.gate(feat)).squeeze(-1)              # (B,P,F)
        return anchor[:, :, None, :] + alpha[..., None] * raw, alpha


# --- pure helpers (filled in later tasks) ---
def _raw_contact_dist(anchor, eef3):
    # anchor (B,P,2), eef3 (B,L,3,2) -> (B,P,F,2): distance to the 2 future tips
    B, Pn = anchor.shape[:2]
    tips = eef3[:, K:, 1:3, :]                                   # (B,F,2,2)
    return torch.linalg.norm(anchor[:, :, None, None, :] - tips[:, None], dim=-1)


def fit_dist_stats(tracks, eef3, domain):
    # per-domain (mean,std) of contact distances (object pts at frame K-1 -> future tips)
    anchor = tracks[:, K - 1]                                    # (N,P,2)
    feat = _raw_contact_dist(anchor, eef3)                       # (N,P,F,2)
    stats = {}
    for d in domain.unique().tolist():
        v = feat[domain == d].reshape(-1)
        stats[int(d)] = (v.mean().item(), v.std().item() + 1e-6)
    return stats


def contact_gate_features(anchor, eef3, g, dom, dist_stats):
    # anchor (B,P,2), eef3 (B,L,3,2), g (B,L) already z-scored, dom (B,) -> (B,P,F,3)
    B, Pn = anchor.shape[:2]
    d = _raw_contact_dist(anchor, eef3)                          # (B,P,F,2)
    mu = torch.tensor([dist_stats[int(x)][0] for x in dom.tolist()], device=d.device)
    sd = torch.tensor([dist_stats[int(x)][1] for x in dom.tolist()], device=d.device)
    d = (d - mu[:, None, None, None]) / sd[:, None, None, None]  # per-domain z-score
    gf = g[:, K:][:, None, :].expand(B, Pn, F)                   # (B,P,F)
    return torch.cat([d, gf[..., None]], -1)                     # (B,P,F,3)


def grasp_openness(eef3):
    # eef3 (...,L,3,2) -> (...,L): distance between the two tips (points 1,2)
    return torch.linalg.norm(eef3[..., 1, :] - eef3[..., 2, :], dim=-1)


def fit_grasp_stats(g_raw, domain):
    # per-domain (mean, std) scalar over flattened grasp values
    g_flat = g_raw.reshape(len(g_raw), -1) if g_raw.dim() > 1 else g_raw[:, None]
    stats = {}
    for d in domain.unique().tolist():
        v = g_flat[domain == d].reshape(-1)
        stats[int(d)] = (v.mean().item(), v.std().item() + 1e-6)
    return stats


def normalize_grasp(g_raw, domain, stats):
    out = torch.zeros_like(g_raw)
    for d, (mu, sd) in stats.items():
        m = domain == d
        out[m] = (g_raw[m] - mu) / sd
    return out


def inject_state_noise(hist, std, generator):
    # additive Gaussian noise on history object points (normalized coords); cheap DAgger
    noise = torch.randn(hist.shape, generator=generator, device=hist.device) * std
    return hist + noise


def multi_step_consistency(model, tr, eef3, g, dom):
    # tr (B,L,P,2) -> (cons_pred, cons_tgt) each (B,P,overlap,2), overlap = L-2K
    B = tr.shape[0]
    hist1 = tr[:, :K].permute(0, 2, 1, 3)                       # (B,P,K,2)
    pred1, _ = model(hist1, eef3, g, dom)                       # (B,P,F,2) frames K..L-1
    hist2 = pred1[:, :, :K, :]                                  # predicted frames K..2K-1
    pred2, _ = model(hist2, eef3, g, dom)                       # frames 2K..2K+F-1 (own-feedback)
    overlap = L - 2 * K                                         # frames 2K..L-1 still have GT
    cons_pred = pred2[:, :, :overlap, :]
    cons_tgt = tr[:, 2 * K:].permute(0, 2, 1, 3)               # (B,P,overlap,2)
    return cons_pred, cons_tgt


def chained_rollout(model, tr, eef3, g, dom):
    # tr (B,L,P,2) -> (p_open, p_chain, gt) each (B,P,overlap,2), overlap = L-2K.
    # p_open: open-loop prediction of frames 2K..L-1 (teacher-forced history).
    # p_chain: same frames but predicted from the model's OWN frames K..2K-1 (compounding).
    overlap = L - 2 * K
    hist1 = tr[:, :K].permute(0, 2, 1, 3)                       # (B,P,K,2)
    pred0, _ = model(hist1, eef3, g, dom)                       # (B,P,F,2) frames K..L-1
    p_open = pred0[:, :, K:, :]                                 # frames 2K..L-1
    hist2 = pred0[:, :, :K, :]                                  # predicted frames K..2K-1
    pred1, _ = model(hist2, eef3, g, dom)                       # frames 2K..
    p_chain = pred1[:, :, :overlap, :]                          # frames 2K..L-1
    gt = tr[:, 2 * K:].permute(0, 2, 1, 3)                      # (B,P,overlap,2)
    return p_open, p_chain, gt


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


def cg_descriptor(tracks, eef3, g_norm, dom):
    # tracks (N,L,P,2), eef3 (N,L,3,2), g_norm (N,L) already z-scored, dom (N,) -> (N, L*2 + L)
    c = tracks.mean(2)                                          # (N,L,2) object centroid
    tips = eef3[:, :, 1:3, :]                                   # (N,L,2,2)
    d = torch.linalg.norm(c[:, :, None, :] - tips, dim=-1).reshape(len(tracks), -1)  # (N,L*2)
    dz = torch.zeros_like(d)
    for dd in dom.unique().tolist():
        m = dom == dd
        dz[m] = (d[m] - d[m].mean()) / (d[m].std() + 1e-6)
    return torch.cat([dz, g_norm], -1)


def run_probe_cg(out_dir="outputs/flow_wm_v4/probe_cg"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    os.makedirs(out_dir, exist_ok=True)
    tr, vis, eef3, dom, vid = load()
    gstats = fit_grasp_stats(grasp_openness(eef3), dom)
    g_norm = normalize_grasp(grasp_openness(eef3), dom, gstats)
    X = cg_descriptor(tr, eef3, g_norm, dom).numpy(); y = dom.numpy()
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


def _masked_mse(pred, fut, w):
    return ((pred - fut) ** 2 * w).sum() / (w.sum() + 1e-6)


def train_eval(tr, vis, eef3, dom, gstats, train_idx, test_idx, thin, seed, epochs=EPOCHS):
    torch.manual_seed(seed)
    g_all = normalize_grasp(grasp_openness(eef3), dom, gstats)          # (N,L)
    m = FlowWMThick(P, thin=thin).to(device)
    m.dist_stats = fit_dist_stats(tr, eef3, dom)
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
            ef = eef3[b].to(device); gg = g_all[b].to(device); dm = dom[b].to(device)
            hv = vis[b, :K].to(device); fv = vis[b, K:].to(device)
            w = (fv.permute(0, 2, 1) * hv[:, -1:].permute(0, 2, 1))[..., None]
            pred, alpha = m(h, ef, gg, dm)
            loss = _masked_mse(pred, fut, w)
            if not thin:
                loss = loss + LAMBDA_GATE * alpha.abs().mean()
                cp, ct = multi_step_consistency(m, tr[b].to(device), ef, gg, dm)
                wc = fv[:, K:].permute(0, 2, 1)[..., None]             # (B,P,overlap,1)
                loss = loss + LAMBDA_CONSIST * _masked_mse(cp, ct, wc)
            opt.zero_grad(); loss.backward(); opt.step()

    m.eval(); an = ad = fn = fd = 0.0; preds = []; futs = []
    ao_n = ao_d = ac_n = ac_d = 0.0
    OV = L - 2 * K
    with torch.no_grad():
        for b in batches(test_idx, False):
            h = tr[b, :K].permute(0, 2, 1, 3).to(device)
            fut = tr[b, K:].permute(0, 2, 1, 3).to(device)
            ef = eef3[b].to(device); gg = g_all[b].to(device); dm = dom[b].to(device)
            hv = vis[b, :K].to(device); fv = vis[b, K:].to(device)
            w = fv.permute(0, 2, 1) * hv[:, -1:].permute(0, 2, 1)
            pred, _ = m(h, ef, gg, dm)
            err = torch.linalg.norm(pred - fut, dim=-1) * 224.0
            an += (err * w).sum().item(); ad += w.sum().item()
            fn += (err[..., -1] * w[..., -1]).sum().item(); fd += w[..., -1].sum().item()
            preds.append(pred.permute(0, 2, 1, 3).cpu()); futs.append(fut.permute(0, 2, 1, 3).cpu())
            po, pc, gtov = chained_rollout(m, tr[b].to(device), ef, gg, dm)
            wov = fv[:, K:].permute(0, 2, 1)                              # (B,P,OV)
            eo = torch.linalg.norm(po - gtov, dim=-1) * 224.0
            ec = torch.linalg.norm(pc - gtov, dim=-1) * 224.0
            ao_n += (eo * wov).sum().item(); ao_d += wov.sum().item()
            ac_n += (ec * wov).sum().item(); ac_d += wov.sum().item()
    pred_all = torch.cat(preds); fut_all = torch.cat(futs)              # (Ntest,F,P,2)
    drift, n_static = rollout_drift_static(pred_all, fut_all, STATIC_TAU)
    ade_open = ao_n / (ao_d + 1e-6); ade_chain = ac_n / (ac_d + 1e-6)
    return {"ade": an / ad, "fde": fn / fd, "drift": drift, "n_static": n_static,
            "ade_open": ade_open, "ade_chain": ade_chain,
            "compound_ratio": ade_chain / (ade_open + 1e-6)}


def run_phase1(out_dir="outputs/flow_wm_v4/phase1_robot_drift"):
    os.makedirs(out_dir, exist_ok=True)
    tr, vis, eef3, dom, vid = load()
    gstats = fit_grasp_stats(grasp_openness(eef3), dom)
    test = torch.where((dom == 1) & (vid == TEST_ROBOT_VID))[0]
    rob = torch.where((dom == 1) & (vid != TEST_ROBOT_VID))[0]
    lines = [f"Phase-1 robot-only | train={len(rob)} test(held-out vid={TEST_ROBOT_VID})={len(test)}",
             f"component versions: model=FlowWMThick (v4), data=flow_ds_v3.npz",
             f"{'variant':>6} | {'ADE':>7} | {'FDE':>7} | {'drift_px':>8} | {'ADE_open':>9} | {'ADE_chain':>10} | {'cmp_ratio':>9} | {'n_static':>8}"]
    for thin in (True, False):
        accs = [train_eval(tr, vis, eef3, dom, gstats, rob, test, thin=thin, seed=s)
                for s in range(SEEDS)]
        ade = np.mean([a["ade"] for a in accs]); fde = np.mean([a["fde"] for a in accs])
        drift = np.mean([a["drift"] for a in accs]); ns = accs[0]["n_static"]
        aopen = np.mean([a["ade_open"] for a in accs]); achain = np.mean([a["ade_chain"] for a in accs])
        ratio = np.mean([a["compound_ratio"] for a in accs])
        lines.append(f"{'thin' if thin else 'thick':>6} | {ade:7.2f} | {fde:7.2f} | {drift:8.3f} | {aopen:9.2f} | {achain:10.2f} | {ratio:9.3f} | {ns:8d}")
    msg = "\n".join(lines) + "\n"
    print(msg, flush=True)
    open(os.path.join(out_dir, "summary.txt"), "w").write(msg)


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
