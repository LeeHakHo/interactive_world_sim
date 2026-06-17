"""Route 1: longer scheduled-sampling rollout training to cut ② long-horizon drift.
Current SS trains R_SS=16 on short (L=24) clips, then eval H=40 -> drift 6.9px. Here we train LWC
on LONG (L=48) data with a SLIDING eef window so R_SS can go to 30-40 (train distribution = H=40
rollout distribution). Sweep R_SS=[16,24,32,40] on long data, eval H=40 centroid drift on long
held-out. Isolates rollout-training-length. (Long data is robot-only -> ② drift, not human-helps.)
Output: outputs/cross_embodiment_wm/scel_ss_long/summary.txt"""
import os, numpy as np, torch, torch.nn as nn
import amplify_wm as A
import eval_scheduled_sampling as SSm
from amplify_wm import K, F, device, vel_to_class

SMOKE = os.environ.get("SMOKE", "0") == "1"
DS_LONG = "outputs/flow_render_dataset_v3_long"
H = int(os.environ.get("HORIZON", "40")); HELDOUT = 150; IMG = 128; VEL_HALF = 0.06
R_SS_LIST = [int(x) for x in os.environ.get("R_SS_LIST", "16,24,32,40").split(",")]
OUT = "outputs/cross_embodiment_wm/scel_ss_long"; os.makedirs(OUT, exist_ok=True)


class ActThickLWC(A.FlowWM_LWC):
    """action-thickened ②: act input = eef [rel-pos, velocity, acceleration] vs position-only.
    Pure action condition (eef both domains, known input) -> safe to thicken; targets OOD-action."""
    def __init__(s, P, **kw):
        super().__init__(P, **kw)
        s.act = nn.Linear((K + F) * 3 * 2 * 3, s.Dm)

    def trunk(s, hist, eef3):
        B, P = hist.shape[:2]
        anchor = hist[:, :, -1, :]
        obj = s.inp(torch.cat([(hist - anchor[:, :, None]).reshape(B, P, 2 * K), anchor], -1))
        objc = anchor.mean(1, keepdim=True)
        ef_rel = eef3 - objc[:, :, None]
        ev = torch.zeros_like(eef3); ev[:, 1:] = eef3[:, 1:] - eef3[:, :-1]
        ea = torch.zeros_like(eef3); ea[:, 2:] = ev[:, 2:] - ev[:, 1:-1]
        act = s.act(torch.cat([ef_rel, ev, ea], -1).reshape(B, -1))[:, None, :]
        return s.tf(torch.cat([obj, act], 1))[:, :P], anchor


def train_long(tracks, vis, eef, idx, R_SS, seed=0, p_fixed=None, model_cls=None):
    """SS training on long clips with SLIDING eef window (K+F=24), R_SS up to L-K steps.
    p_fixed=1.0 -> teacher-forced (SS OFF); None -> scheduled sampling. model_cls -> custom ② (ActThickLWC)."""
    torch.manual_seed(seed); P = tracks.shape[2]
    m = (model_cls or A.FlowWM_LWC)(P, Dm=384, layers=3, W=15, vel_half=VEL_HALF).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=SSm.WM_LR)
    g = torch.Generator().manual_seed(seed)
    tr = torch.from_numpy(tracks).float(); vs = torch.from_numpy(vis).float(); ef = torch.from_numpy(eef).float()
    for ep in range(SSm.WM_EPOCHS):
        pteach = p_fixed if p_fixed is not None else 1.0 + (0.3 - 1.0) * ep / max(SSm.WM_EPOCHS - 1, 1)
        m.train(); pe = idx[torch.randperm(len(idx), generator=g)]
        for i in range(0, len(pe), SSm.WM_BS):
            b = pe[i:i + SSm.WM_BS]
            G = tr[b].to(device); Vv = vs[b].to(device); Ef = ef[b].to(device)
            buf = G[:, :K].clone(); losses = []
            for h in range(R_SS):
                ew = Ef[:, h:h + K + F]                                  # sliding eef window
                if ew.shape[1] < K + F:
                    ew = torch.cat([ew, ew[:, -1:].repeat(1, K + F - ew.shape[1], 1, 1)], 1)
                logits, _ = m(buf[:, -K:].permute(0, 2, 1, 3), ew)
                lg0 = logits[:, :, 0, :]
                gt_vel = G[:, K + h] - buf[:, -1]
                cls = vel_to_class(gt_vel, m.W, m.vel_half)
                w = (Vv[:, K + h] * Vv[:, K - 1])
                ce = nn.functional.cross_entropy(lg0.reshape(-1, m.W * m.W), cls.reshape(-1), reduction="none")
                losses.append((ce * w.reshape(-1)).sum() / (w.sum() + 1e-6))
                nxt = buf[:, -1] + m.expected_vel(lg0)
                use_gt = (torch.rand(len(b), 1, 1, device=device) < pteach)
                buf = torch.cat([buf, torch.where(use_gt, G[:, K + h], nxt.detach())[:, None]], 1)
            loss = torch.stack(losses).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


@torch.no_grad()
def drift(wm, tr, ef, ho):
    pred = A.rollout_lwc(wm, torch.from_numpy(tr[ho]).float().to(device),
                         torch.from_numpy(ef[ho]).float().to(device), H).cpu().numpy()
    gt = tr[ho][:, K:K + H]
    return np.linalg.norm(pred.mean(2) - gt.mean(2), axis=-1) * IMG          # (HO,H) centroid px@128


@torch.no_grad()
def per_step_gain(wm, tr, ef, ho, n=300, sigma=0.005):
    """error amplification gain = |Δpred_vel| / |δ_hist| (finite diff). <1 => bounded for ANY horizon;
    >1 => blows up. This is the hard criterion for arbitrary-length rollout (gain≤1)."""
    idx = ho[:n]
    hist = torch.from_numpy(tr[idx, :K]).float().to(device)
    ew = torch.from_numpy(ef[idx, :K + F]).float().to(device)
    def vel(h):
        lg, _ = wm(h.permute(0, 2, 1, 3), ew); return wm.expected_vel(lg[:, :, 0, :])
    v0 = vel(hist)
    noise = torch.randn_like(hist[:, -1]) * sigma
    hp = hist.clone(); hp[:, -1] = hp[:, -1] + noise
    v1 = vel(hp)
    g = (v1 - v0).norm(dim=-1) / (noise.norm(dim=-1) + 1e-9)
    return float(g.mean())


def main():
    if SMOKE: SSm.WM_EPOCHS = 2
    zl = np.load(f"{DS_LONG}/clips_robot.npz")
    tr, ef, vs = zl["tracks"].astype(np.float32), zl["eef"].astype(np.float32), zl["vis"].astype(np.float32)
    perm = np.random.default_rng(0).permutation(len(tr)); ho, pool = perm[:HELDOUT], perm[HELDOUT:]
    if SMOKE: pool = pool[:200]
    actthick = os.environ.get("ACTTHICK", "0") == "1"
    configs = ([("baseline-eef3", None), ("action-thick", ActThickLWC)] if actthick
               else [(f"R_SS={r}", r) for r in R_SS_LIST])
    lines = [f"{'ACTTHICK (in-dist drift)' if actthick else 'Route1 longer-SS'} | LONG data robot-only | eval H={H} px@128",
             f"{'config':>14} | {'s0':>6} {'s10':>6} {'s20':>6} {'s39':>6} {'overall':>8} | {'gain':>6}"]
    print("\n".join(lines), flush=True)
    for name, cfg in configs:
        wm = (train_long(tr, vs, ef, torch.from_numpy(pool), 8 if SMOKE else 32, model_cls=cfg) if actthick
              else train_long(tr, vs, ef, torch.from_numpy(pool), cfg))
        e = drift(wm, tr, ef, ho); g = per_step_gain(wm, tr, ef, ho)
        s = lambda k: e[:, min(k, H - 1)].mean()
        lines.append(f"{name:>14} | {s(0):6.1f} {s(10):6.1f} {s(20):6.1f} {s(39):6.1f} {e.mean():8.1f} | {g:6.3f}")
        print(lines[-1], flush=True)
    lines += ["", "in-dist drift (action REPLAY = in-dist). OOD-action (synthetic) eval is separate."]
    open(f"{OUT}/{'summary_actthick' if actthick else 'summary'}.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines) + f"\nsaved {OUT}/\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
