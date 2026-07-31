# 渲染器 ablation + "新模块退化"调查 — 完整总结(2026-07-31)

> 自包含,给接手/paper。起因:用户眼检 keyboard/replay gif 觉得"新模块(mp②+ro_80k③)渲染退化了"。系统排查(systematic-debugging)+ 逐个清 confound 后的干净定论。
> ★核心教训:本调查一堆"dramatic 结论"其实是**数据口径 confound**(非retrack tracks / L24 human)撑起来的,清完 retrack+L48 后只剩 grip+warp 一个真结论。相关 [[project_gripwarp_e2e_robustness]] [[feedback_cite_source_check_sibling_runs]] [[project_clip_heldout_leakage]]

## 0. 组件定义
- **② action→object-flow WM**(DualLWC):mp(midpoint+grip标量)vs dummy5(原始三点);ro=robot-only / rh=robot+human混训。都训在 **retrack GDINO tracks + episode-split(HELDOUT_VIDS=100,102)**。
- **③ Wan 视频渲染器**(MultiHeadVideoWM,AUX空=plain):cond 通道 = flow(3)+agent(1)[+warp(3)]。
  - **ro_80k** = CCOND=4(flow+skel,**无warp无grip**),warm-start resume到80k。
  - **grip_ro/grip_rh** = CCOND=7(flow+**grip-skel**+**warp**),from-scratch 80k,robot-only / robot+human。
- e2e = ② 预测flow驱动③;GT-flow = 喂真值flow(③天花板)。

## 1. ★唯一 dramatic 且干净的发现:grip+warp 让 ③ 对 ② flow 误差鲁棒
| ③ (mp②驱动,retrack 16-seq n=32) | cube_px | LPIPS |
|---|---|---|
| grip③-GTflow(天花板) | 3.28 | 0.1142 |
| **mp→ro_80k**(无grip/warp) | **5.84** | 0.1196 |
| **mp→grip_ro**(grip+warp) | **3.47** | 0.1139 |

- **grip+warp 帮 1.7×**(cube 5.84→3.47)。★注意:GT-flow(完美flow)下 ro_80k≈grip③(测不出);只有 e2e(真②flow有误差)才暴露。**机制**:warp带物体搬运几何+grip-skel带agent结构 → flow略错时③有别的信息把东西放对。
- ⚠️**曾误报 2.8×**:那是用**非retrack**的烂②输入把 ro_80k 抬差了;retrack 修正后 = **1.7×**(仍明确赢)。
- ★★方法论:**渲染器 ablation 必须 e2e + 双视角,不能 GT-flow 天花板**(掩盖鲁棒性)。

## 2. 清 confound 后"软化"的结论(曾 dramatic,实为口径问题)
| 我曾说 | 清 confound 后 | confound 根因 |
|---|---|---|
| grip+warp 帮 2.8× | **1.7×** | 非retrack②输入抬差ro_80k |
| dummy5 human反害 +1.17 | **+0.31**(轻害) | dummy5训在**L24 human**,mp训在L48(L24更OOD) |
| grip_rh 害cube +1.22 | **≈grip_ro中性**(cube3.43≈3.47,LPIPS更好0.1101) | 非retrack②输入 |
| mp 明显赢 dummy5 | **打平** | drift favors mp / render-cube favors dummy5 / 都小 |

## 3. ② human-help(dummy5 L48 对齐 mp,drift cam_high)
| 动作 | ro | rh | human效应 |
|---|---|---|---|
| mp | 2.29 | 2.35 | +0.06 中性 |
| dummy5(L48) | 2.46 | 2.77 | +0.31 轻害 |

全量 human 中性到轻害(单场景数据天花板),各层一致。human 只稀缺帮(见 [[project_human_helps_renderer_exp]])。

## 4. mp vs dummy5(② 动作表示,全轴)
| metric | mp | dummy5 | 谁赢 |
|---|---|---|---|
| ② drift(全heldout) | 2.29 | 2.46 | mp |
| rendered cube(retrack 16-seq) | 3.47 | **2.66** | dummy5 |
| ② human-help | +0.06 | +0.31 | mp(略) |
| keyboard fb(独立) | 0.139 | 0.121 | dummy5(小1.15×) |
→ **打平**,谁也没稳赢(metric/seq 敏感)。

## 5. ★另 session 的 3 个 eval bug(用户"担心eval错"全中,独立修正)
| metric | 他们(用错老GT/flow) | 我独立(retrack) |
|---|---|---|
| mp replay ②ADE | 4.32/4.08 | **2.01/1.95** |
| ro_80k GT-flow render-LPIPS | 0.0723 | **0.0620**(两独立eval一致) |
| ro_80k cube_px / vanish | 10.40 / 11% | **3.54 / 0%** |
| keyboard fb (mp) | 0.228(虚高) | **0.139** |
→ **"新模块退化"绝大部分是这些 eval bug + gif 128缩放显示口径**,新模块本身好。

## 6. 我自己栽的口径问题(用户逐个抓出)
1. render脚本 tracks 用 **clips_robot(非retrack)** → ②输入历史+flow overlay是老flow。修=DS→retrack。
2. dummy5 rh 训在 **L24 human**(mp用L48)→ human-help confound。修=L48重训 dummy5_rh_L48h。
3. GT-flow ablation 静帧误判"grip+warp无用"→ 用户眼检e2e动画抓出。
4. gif 128缩放 vs 老gif 256显示 → 误以为老m4更锐。

## 7. 最终推荐管线
**② = mp（human友好+drift略优,但vs dummy5基本打平）；③ = grip_ro 或 grip_rh（grip+warp,human混训中性可选）。**
- **③ 必带 grip+warp**(1.7× e2e鲁棒)——唯一铁结论。
- ② mp/dummy5 打平,mp 略稳(human友好)。

## 8. Drive gif 路径(全在 mygoogle:iws_evals/)
| 路径 | 内容 |
|---|---|
| `2026-07-31_终版_retrack_L48_干净/` | ★**最终干净版**:16seq双视角,retrack+L48,5列cube_px+LPIPS |
| `2026-07-31_6列对比_含grip_rh/` | 6列(GT/ceiling/ro_80k/grip_ro/grip_rh/dummy5),含grip_rh(★注:非retrack口径,数偏) |
| `2026-07-31_综合对比_grip+warp在e2e帮2.2x/` | 首版综合5列(★2.2x是非retrack,已修为1.7x) |
| `2026-07-31_一图对比_眼见为实/` | 单动图4列(GT/mp→ro_80k/mp→grip③/dummy5→grip③) |
| `2026-07-31_render_ablation_眼见为实/` | ③轴GTflow老m4/ro_80k/grip③ + 各e2e分开gif |
| `2026-07-31_新模型mp_replay_keyboard/` | 另session原始mp keyboard/replay(触发本调查) |
| `2026-07-30_真e2e匹配②③_humanhelp存活+20%/` | scarce真e2e human-help |
| `2026-07-30_scarce③human帮渲染+flowcond碾压naive/` | scarce human帮③ + flow-cond vs naive |

本地结果树:`outputs/video_arch_wm/COMPARE_FINAL/`(终版)、`COMPARE_16seq/`、`COMPARE_6col/`、`render_ablation/`;summary.txt 在各目录。脚本:`render_compare_full.py`(5-6列双视角)、`render_grip_rh.py`(③human-help)。

## 9. 附:dummy5-on-regularized-human(负结果,2026-07-31)
另 session `regularize_human_eef.py` 把人手3点强制robot几何(gap2.2×→1×/腕角60°→90°,canon≈robot)。测 dummy5 用它 human-help 是否改善 → **反而更害**:dummy5 rh human效应 raw +0.31 → regularized **+0.85**。
★机制:regularize改了eef但配对object-flow还是真手产生→action↔flow因果断裂→②学错映射→human更害。**降域差不能强改agent几何(砸因果),要velocity/Δ(域不变+保运动因果)。** mp不害=丢绝对位置+保真实指轴/开合(不砸因果)。→ regularize别喂②(可视化overlay无妨)。见 [[project_regularize_eef_breaks_causality]]。
