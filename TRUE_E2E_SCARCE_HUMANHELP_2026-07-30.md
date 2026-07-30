# 真 e2e(匹配 scarce ②→③)human-help 结果 — 2026-07-30

> 用户疑点:之前 scarce ③ 的 gif/metric 是不是用了全量 rh 的 ②?→ 核实=**没用任何 ②,喂的是 GT-flow(③天花板/replay)**,不是全量②污染,但也不是真 e2e。本 doc = 补上真 e2e(每个 ③ 由**匹配同 N 的 ②(mp)预测 flow** 驱动),gif 带 flow 叠加。
> 相关 [[project_human_helps_renderer_exp]] [[feedback_cite_source_check_sibling_runs]] [[feedback_cube_metric_nan_trap]] [[project_action_dflow_separability]]

## 0. 一句话
**human-help 在真 e2e 存活,而且比天花板层更大(+20%),因为 human 同时帮 ②(flow 更准)和 ③(渲染更好),两层 compound。** ② 误差穿过 ③ 只掉 0-4%(③ 宽容)。★但这是 render-LPIPS(指标层);scarce 渲染质量差(物体常消失),视觉不明显、cube_px 不可靠——干净的控制数是全量 flow-vs-naive。

## 1. 配对(真 e2e,整链同 N、同 with/without-human)
| ③(渲染) | ← ②(预测flow) |
|---|---|
| epsplit_L48_mh/ro_n{100,300} | epsplit_L48/mp_r_n{100,300} |
| epsplit_L48_mh/rh_n{100,300} | epsplit_L48/mp_rh_n{100,300} |
脚本 `eval_video_e2e.py`(注册 MultiHeadVideoWM + CCOND=4 + ACTION=mp 用 load_action_tokens 建 grip 第4槽);gif 三列 GT | ③GT-flow天花板 | ②→③e2e,flow 行 绿=GT/红=②预测/黄=eef。结果 `outputs/video_arch_wm/scarce_e2e/`。

## 2. render-LPIPS(含 agent, held-out 4seq×2view=8, ↓越小越好)
| run | ③天花板(GT-flow) | 真 e2e(②预测flow) | compound |
|---|---|---|---|
| ro_n100 | 0.1824 | 0.1903 | +4.4% |
| rh_n100 | 0.1500 | 0.1519 | +1.3% |
| ro_n300 | 0.1511 | 0.1541 | +2.0% |
| rh_n300 | 0.1244 | 0.1242 | −0.2% |

**human-help(ro vs rh,真 e2e 列):**
- **n100: 0.1903 → 0.1519 = +20.2%**
- **n300: 0.1541 → 0.1242 = +19.4%**
- 天花板层 human-help = +17.8%/+17.7% → **真 e2e 更大**(② human-help 叠加到 ③ human-help)。

**读法:**
1. **human-help 穿过整条链存活并放大**(+20% > +18%):真 e2e 下 human 帮 ②(scarce ② mp_rh_n300 2.37 < mp_r_n300 4.59,flow 更准)+ 帮 ③(渲染器更好),compound。
2. **② 误差 compound 极小(+0~4%)**:scarce ② 的预测 flow 穿过 ③ 几乎不掉质量(③ 宽容 ② 误差,与 e2e≈GT-flow 一致);human 帮过的 ②(rh)compound 更小(flow 更准)。

## 3. ★诚实 caveat(别重蹈过度包装)
1. **这是 render-LPIPS,不是视觉**。scarce 渲染 0.12-0.19 远差于全量 0.064,黑拖影、物体常消失;+20% 是指标层,**肉眼未必看得出**(smoke 眼检坐实:e2e 列红点跟绿点但渲染烂)。
2. **cube_px(控制保真)在 scarce 不可靠**:物体在退化渲染里 **~40% 消失(不分 ro/rh)**,按 [[feedback_cube_metric_nan_trap]] 规矩 cube 值没意义。**干净的控制保真数只有全量 flow-vs-naive(flow 3.52 vs naive 7.02,2×,det 1.00)**。真 e2e cube det/vanish(留档,弱信号):

| run | gtflow det | e2e det | cube_px(检测帧) |
|---|---|---|---|
| ro_n100 | 0.55 | 0.59 | 25.5 |
| rh_n100 | 0.60 | 0.54 | 20.5 |
| ro_n300 | 0.56 | 0.56 | 15.3 |
| rh_n300 | **0.65** | **0.66** | 12.4 |

→ human 弱帮物体存活(n300 det +9pts)+ 检测帧 cube 更低(12.4<15.3),但 ~40% vanish 下噪声大,**不当结论**。"flow 让物体活/naive 让物体消失"的真测在 scarce flow-vs-naive(pending 55124)。
3. **可分性只是代理**:human 帮不帮由此 A/B 判,不由 probe 判(见 [[project_action_dflow_separability]] 家族教训)。

## 4. 对 story 的意义
- **护城河核心命题(human 帮③)在稀缺 regime 成立且穿到像素**——但只在稀缺(全量天花板),且是指标层非视觉层。诚实定位:human-help 是"稀缺 + 指标"现象,不是 hero demo。
- 控制保真的干净证据在全量 flow-vs-naive(2×),不在 scarce e2e。
- 用户方法学疑点已澄清:那些数从没用全量②污染;真 e2e 确认结论。
