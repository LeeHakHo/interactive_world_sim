"""agg_dualview_formal.py — 聚合 metrics.json -> mean±std 消融表 + variance-first 门控判定.
门控 (spec M1): 两视角都要 |mean(flow)-mean(eeffilm)| > 2*pooled_std 才可下 'flow 赢' 结论;
否则输出 EXTEND (加 seed 3,4 或 EPOCHS 90). 用法: python agg_dualview_formal.py"""
import glob
import json

import numpy as np

ROOT = "outputs/cross_embodiment_wm/dualview_dit_formal"
runs = {}
for p in sorted(glob.glob(f"{ROOT}/*/metrics.json")):
    r = json.load(open(p))
    if "smoke" in p or "cond" not in r: continue        # skip non-controlled-arm metrics.json (e.g. Task5 e2e/, different schema)
    runs.setdefault((r["cond"], r["crossview"], r.get("mix", "rh")), []).append(r)

lines = ["# dual-view DiT formal — ablation table", "",
         "| cond | crossview | mix | n_seed | v0 obj-LPIPS | v1 obj-LPIPS | v0 PSNR | v1 PSNR | det v0/v1 |",
         "|---|---|---|---|---|---|---|---|---|"]
stat = {}
for (cond, cv, mix), rs in sorted(runs.items()):
    g = lambda k: np.array([x[k] for x in rs])
    stat[(cond, cv, mix)] = {k: (g(k).mean(), g(k).std(ddof=1) if len(rs) > 1 else float("nan"))
                             for k in ["v0_lp", "v1_lp", "v0_ps", "v1_ps"]}
    s = stat[(cond, cv, mix)]
    lines.append(f"| {cond} | {cv} | {mix} | {len(rs)} | {s['v0_lp'][0]:.4f}±{s['v0_lp'][1]:.4f} "
                 f"| {s['v1_lp'][0]:.4f}±{s['v1_lp'][1]:.4f} | {s['v0_ps'][0]:.2f}±{s['v0_ps'][1]:.2f} "
                 f"| {s['v1_ps'][0]:.2f}±{s['v1_ps'][1]:.2f} "
                 f"| {np.mean(g('det_rate_v0')):.2f}/{np.mean(g('det_rate_v1')):.2f} |")

IWS_ROOT = "outputs/cross_embodiment_wm/dualview_iws_stage2"
iws_runs = []
for p in sorted(glob.glob(f"{IWS_ROOT}/s*/metrics.json")):
    if "smoke" in p: continue
    iws_runs.append(json.load(open(p)))
if iws_runs:
    g2 = lambda k: np.array([x[k] for x in iws_runs])
    iws_stat = {k: (g2(k).mean(), g2(k).std(ddof=1) if len(iws_runs) > 1 else float("nan"))
                for k in ["v0_lp", "v1_lp", "v0_ps", "v1_ps"]}
    lines.append(f"| IWS-stage2(DF, external) | - | - | {len(iws_runs)} "
                 f"| {iws_stat['v0_lp'][0]:.4f}±{iws_stat['v0_lp'][1]:.4f} "
                 f"| {iws_stat['v1_lp'][0]:.4f}±{iws_stat['v1_lp'][1]:.4f} "
                 f"| {iws_stat['v0_ps'][0]:.2f}±{iws_stat['v0_ps'][1]:.2f} "
                 f"| {iws_stat['v1_ps'][0]:.2f}±{iws_stat['v1_ps'][1]:.2f} "
                 f"| {np.mean(g2('det_rate_v0')):.2f}/{np.mean(g2('det_rate_v1')):.2f} |")

lines.append("")
if ("flow", 1, "rh") in stat and ("eeffilm", 1, "rh") in stat:
    for v in ["v0_lp", "v1_lp"]:
        mf, sf = stat[("flow", 1, "rh")][v]; me, se = stat[("eeffilm", 1, "rh")][v]
        pooled = float(np.sqrt(np.nanmean([sf ** 2, se ** 2])))
        d = me - mf                                        # LPIPS 低好: d>0 = flow 赢
        verdict = "PASS" if (not np.isnan(pooled) and abs(d) > 2 * pooled) else "EXTEND(加seed/epoch)"
        lines.append(f"- gate {v}: flow {mf:.4f} vs eeffilm {me:.4f} | d={d:+.4f} pooled_std={pooled:.4f} -> **{verdict}**")

lines.append("")
for cond in ["flow", "eeffilm"]:                           # M1c human-helps: Δ = LPIPS(r-only) - LPIPS(r+h), >0 = human 帮
    if (cond, 1, "rh") in stat and (cond, 1, "r") in stat:
        for v in ["v0_lp", "v1_lp"]:
            mrh, srh = stat[(cond, 1, "rh")][v]; mr, sr = stat[(cond, 1, "r")][v]
            pooled = float(np.sqrt(np.nanmean([srh ** 2, sr ** 2])))
            dd = mr - mrh
            sig = "显著" if (not np.isnan(pooled) and abs(dd) > 2 * pooled) else "不显著(如实报)"
            lines.append(f"- human-helps {cond} {v}: r-only {mr:.4f} vs r+h {mrh:.4f} | Δ={dd:+.4f} "
                         f"pooled_std={pooled:.4f} -> {sig}")
out = "\n".join(lines) + "\n"
open(f"{ROOT}/ablation_table.md", "w").write(out); print(out)
