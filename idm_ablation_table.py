"""汇总各臂 idm_derisk/<ARM>/eval_summary.txt -> 一张表 ABLATION.txt。
读每臂 seq 行, 平均 eef-recon(自由/接触)、flow-recon IDM(自由/接触)、ceil(自由/接触)。"""
import os, re, glob

ROOT = os.environ.get("ROOT", "outputs/idm_derisk")
ARMS = os.environ.get("ARMS", "A0 A1 A2 A3").split()
DESC = {"A0": "flow-only", "A1": "+mp接触trace", "A2": "+grip", "A3": "causal(无未来窗)"}

PAT = re.compile(
    r"eef-recon 自由([\d.nan]+) 接触([\d.nan]+) \| "
    r"flow-recon IDM 自由([\d.nan]+) 接触([\d.nan]+) \| "
    r"ceil 自由([\d.nan]+) 接触([\d.nan]+)")


def fnum(s):
    try:
        return float(s)
    except ValueError:
        return float("nan")


def mean(xs):
    xs = [x for x in xs if x == x]   # 去 nan
    return sum(xs) / len(xs) if xs else float("nan")


rows = []
for arm in ARMS:
    p = f"{ROOT}/{arm}/eval_summary.txt"
    if not os.path.exists(p):
        continue
    cols = [[] for _ in range(6)]
    for line in open(p):
        m = PAT.search(line)
        if not m:
            continue
        for i in range(6):
            cols[i].append(fnum(m.group(i + 1)))
    if not cols[0]:
        continue
    ef_f, ef_c, fi_f, fi_c, ce_f, ce_c = [mean(c) for c in cols]
    rows.append((arm, DESC.get(arm, ""), ef_f, ef_c, fi_f, fi_c, ce_f, ce_c))

lines = ["IDM 输入消融汇总 (px@128, heldout episode 100/102; 自由=物体静止段, 接触=物体运动段)",
         "flow-recon = IDM动作→FK→前向②→object-flow vs GT; ceil = GT动作同管线(②本身误差下界)",
         "",
         f"{'ARM':<4} {'输入':<16} {'eef自由':>8} {'eef接触':>8} {'flow自由':>9} {'flow接触':>9} {'ceil自由':>9} {'ceil接触':>9}"]
for r in rows:
    lines.append(f"{r[0]:<4} {r[1]:<16} {r[2]:>8.1f} {r[3]:>8.1f} {r[4]:>9.1f} {r[5]:>9.1f} {r[6]:>9.1f} {r[7]:>9.1f}")
lines += ["",
          "判据: A1接触段flow-recon接近ceil→动作可从flow+trace解出(PASS); A0(flow-only)自由段eef崩→证需trace;",
          "A3(causal去未来窗)eef崩→逆动力学需双向/未来窗(离线解demo合法)。详见 VERDICT.txt。"]
out = f"{ROOT}/ABLATION.txt"
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\n[table] {out}")
