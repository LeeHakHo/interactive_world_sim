"""IDM 训练: supervised regression on robot Δjoint+grip。输入臂由 ARM env 切(消融)。"""
import os, numpy as np, torch
import idm_data as D, idm_model as M

ARM = os.environ.get("ARM", "A1")
EPOCHS = int(os.environ.get("EPOCHS", "200"))
SMOKE = os.environ.get("SMOKE") == "1"
OUT = os.environ.get("OUT_DIR", f"outputs/idm_derisk/{ARM}")
os.makedirs(OUT, exist_ok=True)
KP, FF = 4, 4

z = np.load("outputs/flow_render_dataset_can_dual/clips_robot_retrack.npz")
spec = D.INPUT_SPECS[ARM]
tr, ho = D.episode_split(z)
if SMOKE:
    tr = tr[:20]
X, Y, meta = D.build_windows(z, spec, KP, FF, clips=tr)
xm, xs = X.mean(0), X.std(0) + 1e-6
ym, ys = Y.mean(0), Y.std(0) + 1e-6
Xn = torch.tensor((X - xm) / xs)
Yn = torch.tensor((Y - ym) / ys)
m = M.IDM(din=X.shape[1])
opt = torch.optim.Adam(m.parameters(), 1e-3)
bs = 4096
n = len(Xn)
for ep in range(EPOCHS):
    perm = torch.randperm(n)
    tot = 0.0
    for i in range(0, n, bs):
        idx = perm[i:i + bs]
        opt.zero_grad()
        loss = torch.nn.functional.smooth_l1_loss(m(Xn[idx]), Yn[idx])
        loss.backward()
        opt.step()
        tot += loss.item() * len(idx)
    if ep % 50 == 0 or ep == EPOCHS - 1:
        print(f"ep{ep} loss {tot / n:.4f}", flush=True)
torch.save({"state": m.state_dict(), "spec": spec, "KP": KP, "FF": FF,
            "x_mean": xm, "x_std": xs, "y_mean": ym, "y_std": ys, "din": X.shape[1]}, f"{OUT}/idm.pt")
open(f"{OUT}/train_summary.txt", "w").write(
    f"ARM={ARM} din={X.shape[1]} nwin_train={n} epochs={EPOCHS} smoke={SMOKE}\n")
print(f"[ckpt] {OUT}/idm.pt", flush=True)
