# scarce ③ human-helps + flow-cond vs naive 结果 — 2026-07-30

> 两个正向定论。所有数带源文件,det_rate/vanish 一并报(防 [[feedback_cube_metric_nan_trap]]),单口径已核。
> 相关 [[project_human_helps_renderer_exp]] [[project_multihead_aux_wm]] [[project_dualwm_domain_head]] [[reference_flow_wm_three_papers_insight]]

## 结果 A:scarce ③ —— human 帮到渲染(★填上 §6.3 开放问题)
源 `outputs/video_arch_wm/mh_eval_L48_scarce3/render_summary.txt`(clean L48 episode-split, heldout SEQS 332/59/418/442, 40k 步; scarce 已证 4k 即近收敛)

| run | render-LPIPS ↓ | cube_px ↓ | det_rate | vanish |
|---|---|---|---|---|
| ro_n100(只 100 robot) | 0.1585 | 8.53 | 0.85 | 15% |
| **rh_n100(100 robot + human)** | **0.1369** | **7.22** | 0.91 | 9% |
| ro_n300 | 0.1229 | 5.83 | 0.98 | 2% |
| **rh_n300(300 robot + human)** | **0.1013** | **5.37** | 0.96 | 4% |

**human 增益:** render-LPIPS n100 −13.6% / n300 −17.6%;cube_px n100 −15.4% / n300 −7.9%;且 n100 把 cube 消失率 15%→9%(检测更稳 → 不是 nan 陷阱奖励"cube 消失",是真帮)。

**判读:** 稀缺 robot 渲染数据下,human 像素/latent 数据塑造共享 trunk,**同时**改善外观和物体位置控制,穿过 ③ 到像素。这是**第一次干净证明 human-helps 到渲染层**(此前 full 数据 rh≈ro 是天花板,②稀缺帮但没测过③稀缺)。与 ② 稀缺 human 帮(n100 +4.9 / n300 +2.2)同源 → human-helps 在**稀缺 regime 全栈成立**(②→③→像素)。

## 结果 B:flow-cond vs naive-cond(FlowWAM Fig.4 式,固定 backbone/数据只换 cond)
源 `outputs/video_arch_wm/mh_eval_L48_flowcond_vs_naive/flow_vs_naive/render_summary.txt`(flow ro 与 naive ro_noflow 均 40k 步、同数据/backbone;naive=cond flow 3 通道清零=IWS-stage2 等价)

| cond | render-LPIPS ↓ | cube_px ↓ | det_rate |
|---|---|---|---|
| **flow-cond** | **0.0638** | **3.52** | 1.00 |
| naive(NOFLOW) | 0.0711 | 7.02 | 1.00 |
| flow 优势 | +10.3% | **1.99× (≈2×)** | — |

**判读:** 外观(render-LPIPS)flow 仅好 10%,但**控制保真(cube_px = 物体跟不跟条件, FlowWAM 的 TA 类)flow 好 2×**;两者 det_rate 都 1.00(无消失,2× 是真位置精度)。→ **flow 当 cond 相比 naive/数值动作,在控制保真上碾压**(外观差不多但"物体听不听话"差 2 倍)。这是 paper 第一条腿(flow-cond 优越,object-centric 差异化 FlowWAM 的 agent-flow)。

## 诚实标注
- flow-cond vs naive 在 **40k 对齐**(公平);全量 80k 收敛版(55087/88)出后可复核,预期结论不变(2× 控制差距远大于步数噪声)。
- scarce ③ 40k:scarce 收敛快(ro_n100 step4k recon 0.096),40k 充分。
- flow ro cube_px 本次 3.52 vs 前次 flow_ro_rh eval 3.69 = 扩散采样随机性,同量级(~3.5-3.7),不影响 2× 结论。
