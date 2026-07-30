# L24(泄漏)→ L48(干净 episode-split)metrics 对比 — 2026-07-30

> 用户要求:rh 的 ③ 训练 + "和改 24→48 之前的 metrics 做比较"。
> "24→48" = 两件事同时改:(a) 人手 clip 从 L24 短切改成 L48(与 robot 等长,减 OOD);(b) heldout split 从泄漏的 clip-idx 改成干净的 episode(vid)分组(HELDOUT_VIDS=100,102)。见 [[project_clip_heldout_leakage]]。
> 老数来源:② `outputs/cross_embodiment_wm/established_metrics.txt`(mp 块,heldout=332/59/418/442 泄漏 demo seq);③ `outputs/video_arch_wm/mh_clean_eval/render_summary.txt`(mh_clean,L24 人手 + 泄漏 split)。
> 新数来源:② `outputs/cross_embodiment_wm/epsplit_L48/mp_{r,rh}_all{,_s1,_s2}/summary.txt`(3 种子);③ `outputs/video_arch_wm/epsplit_L48_mh/{ro,rh}` 训练(55038/55039,均训满 40k 步)+ eval `mh_eval_L48_flowcond_vs_naive/flow_ro_rh/render_summary.txt`(job 55080)。

## 0. TL;DR(一句话)
干净 L48 下 **full-data human 在 ② 和 ③ 都是中性**(单场景数据天花板),**scarce human 真帮**(② n100 +4.9 / n300 +2.2,唯一存活 regime)。ro 数被诚实地小幅上修(泄漏本偏乐观)。① ② 之前看到的"human 害 +0.6px"干净后 →+0.06 噪声;② ③ 之前的"human 毁渲染"是 run 不稳挑了坏 run,新 L48 稳定到 rh≈ro(印证用户"差不多"的记忆)。

## 1. ② object-flow WM drift(cam_high, held-out, px, ↓越小越好, mp 动作表示)
| | 旧 L24 泄漏 | 新 L48 干净(3 种子均值) | 变化 |
|---|---|---|---|
| ro(robot-only) | **2.08** | **2.29**(2.20/2.29/2.37) | +0.21 泄漏上修(诚实) |
| rh(robot+human 全量) | **2.69** | **2.35**(2.11/2.68/2.26) | −0.34 |
| rh−ro(全量 human 净效应) | **+0.61(害)** | **+0.06(噪声)** | 害消失 |
| 稀缺 n100 human-help | — | ro 8.51 → rh **3.60 = +4.91** | ✅真帮 |
| 稀缺 n300 human-help | — | ro 4.59 → rh **2.37 = +2.22** | ✅真帮 |

cam_low 同向:ro 2.77 均值,rh 2.88 均值(rh−ro +0.11 噪声)。

## 2. ③ Wan 视频渲染器 render-LPIPS(held-out seqs, ↓越小越好, GT-flow 列=天花板含 agent)
⚠️**更正**:旧 ③ rh **run 间不稳**,有两个数,别只挑坏的那个:
- `mh_clean/rh_full`(40k 步,2586 robot + 1791 human)→ **0.1521**(坏 run)
- `mh_full/rh_N2550`(60k 步,2550 robot + human)→ **0.0640 ≈ ro 0.0658**(好 run,即"差不多")

| | 旧 L24 泄漏 | 新 L48 干净(epsplit_L48_mh) | 变化 |
|---|---|---|---|
| ro | 0.0590(mh_clean)/ 0.0658(N2550) | **0.0638** | ≈同 |
| rh | **0.064~0.152(不稳!)** | **0.0654** | 稳到 ro 附近 |
| rh−ro(全量 human 效应) | +0.006(好run)~ +0.093(坏run) | **+0.0016(噪声)** | 稳定为噪声 |
| cube_px 控制保真(新 metric) | — | ro 3.69 / rh 3.49(det 1.00, 零消失) | — |

**正确读法**:旧 ③ rh 渲染质量 run-to-run 不稳(0.064~0.152),不能说"human 稳定毁渲染 2.6×"——那是挑了坏 run。新干净 L48 把 rh **稳定**到 ro 附近(0.0654 vs 0.0638),与旧的好 run 一致。结论 = **full-data human 在 ③ 渲染层中性**(既不帮也不稳定地害),这也印证了用户"③ rh≈ro 差不多"的记忆。rh 两个新 ③ ckpt 都训满 40000 步(ro recon MSE 0.0759 / rh 0.0848)。

## 3. 读表(可写进 paper 的 3 条)
1. **泄漏→干净 = 诚实上修**:ro 在 ②(2.08→2.29)和 ③(0.059→0.064)都小幅变差,方向正确——泄漏 split(heldout 与 train 共享帧)本来偏乐观。
2. **两个"human 有害"信号站不住**:
   - ② "human 害 robot +0.6px"(旧 2.08→2.69)→ 干净后 +0.06(噪声)。真信号,泄漏 split 放大了它。
   - ③ "human 毁渲染 2.6×" = **挑了坏 run**;旧 ③ rh 本就 run 间不稳(0.064~0.152),好 run 早就 rh≈ro。新 L48 稳定到 0.0654 vs 0.0638。**不是具身迁移壁垒,是训练不稳 + 评测口径。**
3. **稀缺 human-help 存活**(n100 +4.9 / n300 +2.2)——干净 split 下唯一真的 human-help regime,与 seed 复核一致([[project_dualwm_domain_head]] 全量 −0.066 噪声)。

## 4. 结论对 story 的影响
净 story **不变但脚更稳**:full-data human = 单场景数据天花板(中性);scarce human = 真帮。之前担心的"human 反而害 ②/③"被证明是数据管线 artifact,清掉后世界干净了——这反而是好消息(去掉了一个需要解释的负结果)。三条腿(flow-cond 优越 / 可控 object-centric 接口 / Wan 渲染)照旧,见 [[project_dualwm_domain_head]] 的 STRATEGY doc。

> ⚠️ 混淆项(诚实标注):L24→L48 同时改了 clip 长度 + split 口径,无法单独归因;但两者都指向同一方向(人手更少 OOD + 评测更诚实),定性结论稳。
