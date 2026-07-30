# L24(泄漏)→ L48(干净 episode-split)metrics 对比 — 2026-07-30

> 用户要求:rh 的 ③ 训练 + "和改 24→48 之前的 metrics 做比较"。
> "24→48" = 两件事同时改:(a) 人手 clip 从 L24 短切改成 L48(与 robot 等长,减 OOD);(b) heldout split 从泄漏的 clip-idx 改成干净的 episode(vid)分组(HELDOUT_VIDS=100,102)。见 [[project_clip_heldout_leakage]]。
> 老数来源:② `outputs/cross_embodiment_wm/established_metrics.txt`(mp 块,heldout=332/59/418/442 泄漏 demo seq);③ `outputs/video_arch_wm/mh_clean_eval/render_summary.txt`(mh_clean,L24 人手 + 泄漏 split)。
> 新数来源:② `outputs/cross_embodiment_wm/epsplit_L48/mp_{r,rh}_all{,_s1,_s2}/summary.txt`(3 种子);③ `outputs/video_arch_wm/epsplit_L48_mh/{ro,rh}` 训练(55038/55039,均训满 40k 步)+ eval `mh_eval_L48_flowcond_vs_naive/flow_ro_rh/render_summary.txt`(job 55080)。

## 0. TL;DR(一句话)
**两个"human 有害"的吓人信号在干净 L48 下都塌成噪声**——它们是泄漏 split + L24 短人手 OOD 的假象,不是真具身壁垒。full-data human = 中性(单场景天花板);scarce human = 真帮(唯一存活)。ro 数被诚实地小幅上修(泄漏本来偏乐观)。

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
| | 旧 L24 泄漏(mh_clean) | 新 L48 干净(epsplit_L48_mh) | 变化 |
|---|---|---|---|
| ro | **0.0590** | **0.0638** | +0.005 ≈同 |
| rh | **0.1521** | **0.0654** | **−0.087 巨降** |
| rh−ro(全量 human 效应) | **+0.093(渲染烂 2.6×)** | **+0.0016(噪声)** | 害消失 |
| cube_px 控制保真(新 metric) | — | ro 3.69 / rh 3.49(det 1.00, 零消失) | — |

rh 两个 ③ ckpt 都训满 40000 步(ro recon MSE 0.0759 / rh 0.0848),非半途。

## 3. 读表(可写进 paper 的 3 条)
1. **泄漏→干净 = 诚实上修**:ro 在 ②(2.08→2.29)和 ③(0.059→0.064)都小幅变差,方向正确——泄漏 split(heldout 与 train 共享帧)本来偏乐观。
2. **两个"human 有害"信号都是假象**:
   - ② "human 害 robot +0.6px" → 干净后 +0.06(噪声)。
   - ③ "human 毁渲染 2.6×"(0.152 vs 0.059)→ 干净后 rh≈ro(+0.0016 噪声)。
   根因 = 泄漏 clip-idx split + L24 短人手 clip 相对 robot 更 OOD;把人手切到 L48 等长 + episode 干净分组后,两者一起消掉。**不是具身迁移壁垒。**
3. **稀缺 human-help 存活**(n100 +4.9 / n300 +2.2)——干净 split 下唯一真的 human-help regime,与 seed 复核一致([[project_dualwm_domain_head]] 全量 −0.066 噪声)。

## 4. 结论对 story 的影响
净 story **不变但脚更稳**:full-data human = 单场景数据天花板(中性);scarce human = 真帮。之前担心的"human 反而害 ②/③"被证明是数据管线 artifact,清掉后世界干净了——这反而是好消息(去掉了一个需要解释的负结果)。三条腿(flow-cond 优越 / 可控 object-centric 接口 / Wan 渲染)照旧,见 [[project_dualwm_domain_head]] 的 STRATEGY doc。

> ⚠️ 混淆项(诚实标注):L24→L48 同时改了 clip 长度 + split 口径,无法单独归因;但两者都指向同一方向(人手更少 OOD + 评测更诚实),定性结论稳。
