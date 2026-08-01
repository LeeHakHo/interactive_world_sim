"""IDM transformer 训练(结构化 token 输入)。ARM 切输入配方; 标准化 per-(N,C) token 位置。"""
import os, numpy as np, torch
import idm_data as D, idm_model as M

ARM = os.environ.get("ARM", "A1")
EPOCHS = int(os.environ.get("EPOCHS", "300"))
SMOKE = os.environ.get("SMOKE") == "1"
OUT = os.environ.get("OUT_DIR", f"outputs/idm_derisk/{ARM}xf")
os.makedirs(OUT, exist_ok=True)
KP, FF = 4, 4

z = np.load("outputs/flow_render_dataset_can_dual/clips_robot_retrack.npz")
spec = D.INPUT_SPECS[ARM]
tr, ho = D.episode_split(z)
if SMOKE:
    tr = tr[:20]
dev = "cuda" if torch.cuda.is_available() else "cpu"
Xtok, ptype, Y, meta = D.build_windows_tokens(z, spec, KP, FF, clips=tr)   # (M,W,N,C)
M_, W, N, C = Xtok.shape
xm = Xtok.mean((0, 1), keepdims=True); xs = Xtok.std((0, 1), keepdims=True) + 1e-6   # (1,1,N,C)
ym = Y.mean(0); ys = Y.std(0) + 1e-6
Xn = torch.tensor((Xtok - xm) / xs).to(dev)
Yn = torch.tensor((Y - ym) / ys).to(dev)
m = M.IDMTransformer(C=C, N=N, W=W, ptype=ptype).to(dev)
opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=1e-4)
bs = 2048; n = len(Xn)
print(f"[xf] ARM={ARM} M={M_} W={W} N={N} C={C} params={sum(p.numel() for p in m.parameters())/1e6:.2f}M", flush=True)
for ep in range(EPOCHS):
    perm = torch.randperm(n); tot = 0.0
    for i in range(0, n, bs):
        idx = perm[i:i + bs]; opt.zero_grad()
        loss = torch.nn.functional.smooth_l1_loss(m(Xn[idx]), Yn[idx])
        loss.backward(); opt.step(); tot += loss.item() * len(idx)
    if ep % 25 == 0 or ep == EPOCHS - 1:
        print(f"ep{ep} loss {tot / n:.4f}", flush=True)
torch.save({"model": "xf", "state": m.state_dict(), "spec": spec, "KP": KP, "FF": FF,
            "x_mean": xm, "x_std": xs, "y_mean": ym, "y_std": ys, "ptype": ptype,
            "N": N, "W": W, "C": C}, f"{OUT}/idm.pt")
open(f"{OUT}/train_summary.txt", "w").write(f"ARM={ARM} xf N={N} W={W} C={C} nwin={n} epochs={EPOCHS}\n")
print(f"[ckpt] {OUT}/idm.pt", flush=True)
