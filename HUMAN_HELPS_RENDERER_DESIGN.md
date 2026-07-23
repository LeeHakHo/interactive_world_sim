# 让 human data 帮到 ③(latent/像素预测)—— 设计 + related-work 综述

> 2026-07-23 夜,自主 brainstorm(用户指示)。目标:攻下我们跨具身 WM 的**护城河核心** ——
> 让 human 演示数据不只帮 ②(object-flow trace),也帮 **③(视频渲染器 / latent 预测)**。
> μ₀(最强 related work)直接不渲染绕过了这问题;攻下它 = 真差异化。

## 0. 一句话结论(先看)

**「human 帮不上 ③」很可能不是铁律,而是我们测错了架构 + 没用对配方。** 三条独立证据:
- 我们旧结论(rh≈ro)是在**旧 per-frame ③**(detmem/keyboard_3way)上测的,**新视频 ③(Wan VAE + video DiT)从没测过**。
- **OSCAR(同架构 Wan VAE video DiT)明确证 human 帮渲染**(warm-start 后 PSNR/LPIPS/FVD 全涨)——但靠的是**多样性 + I₀ 锚锁外观**,不是共享外观。
- **EgoWAM 受控消融**证「像素目标下 human 迁移弱(我们症状),换成 DINO/3D-flow 目标就强」——即这是**已知的、可解的**目标选择问题。

→ 最先该做的不是造复杂新机制,而是**在新视频 ③ 上补做 warm-start human-mix 的判决实验**,大概率直接就有正向;residual-Δz(用户假设)和 appearance-invariant target 是加固/novel 化的第二梯队。

**★Δz 诊断已出(§3),精炼了用户假设**:latent 逐帧增量 Δz_consec 跨域**大幅同域(probe 0.6 vs 绝对 z 的 0.97)**,但对锚残差 Δz_anchor 仍可分(0.9)。→ residual 路线**成立且形式确定 = latent 速度(consecutive-delta)预测,非 anchor 残差**。0.6 与 object-flow 0.645 同档(probe 门偏严,下游才是判据),绿灯偏乐观。

## 1. 问题设定

- ③ = Wan VAE(冻结)+ 小 video DiT,rectified-flow + Diffusion Forcing,**I₀ 锚**(首帧 latent 锁 GT),条件 = object-flow + agent(skel/gmask)+ warp。预测未来 latent → decode 像素。
- 观察(旧架构):robot-only 训 ③ 和 robot+human co-train ③ 渲染质量几乎一样(rh≈ro,[[project_keyboard_3way_ablation]]),即 human 帮不上 ③。根因判断:human/robot 像素在 latent 里 582× 线性可分,渲染是外观任务,human 外观帮不到 robot 外观。
- 用户护城河命题:**让 human 也帮到 ③**。μ₀ 不渲染 = 绕过;攻下 = 没人做的硬地。

## 2. Related-work 判决(他们怎么处理「human→渲染/latent」)

| 论文 | 做法 | 对我们的用处 |
|---|---|---|
| **EgoWAM** (2607.08436) | 受控消融:同 backbone,只换 WM 预测目标(Pixel VAE / DINO / 3D-flow)。**Pixel 迁移弱(我们症状);DINO OOD +4×;3D-flow in-domain +20-30%** | ★直接答案:换/加 **appearance-invariant 预测目标**,让 human 走 dynamics 通道而非脆弱的 appearance 通道 |
| **OSCAR** (2606.04463) | 同架构(Cosmos-2B + **Wan VAE**)。human 帮渲染=**warm-start**(robot-only 预训→混 human)+ **I₀ 锚逐clip锁外观** + 骨架条件**零纹理**。human 加的是场景/运动**多样性**,不共享外观 | ★低风险直接可试:warm-start + 严格外观只来自锚;核实我们条件通道没泄漏外观 |
| **Mask WM** (2604.19683) / **MaskWAM** (2606.13515) | 预测**语义 mask**(外观不变)作瓶颈/辅助目标,模型没法靠 human-vs-robot 外观抄近路;部署仍 RGB | 加 mask 预测辅助头当 appearance-invariant 瓶颈 |
| **DexWM** (2512.13644) | 最像我们:latent WM 预测未来 latent→decode 像素,**human 预训→小 robot 微调** + **hand-consistency 辅助损失**,zero-shot 真机 | 训练配方(human-heavy 预训 + robot 微调)+ agent-consistency aux |
| **ceve** (2605.03637) | 冻结 VACE 翻 human→robot 外观 + **CLUB(MI-min)+InfoNCE 解耦**,无配对 | 翻译-再-cotrain 对渲染质量**证据弱**(LPIPS 0.674≈基线);但**解耦 aux loss** 可借来把外观从条件 latent 推出去 |
| **X-Diffusion** (2511.04671) | **Ambient Diffusion**:分类器找 human/robot 不可分的最小噪声级,human 只在该级以上用 | 直接针对「可分性」:human latent 只在高噪声段监督 ③ |
| **UniT** (2604.19734) | 跨重建共享 human/robot token 条件化**视频生成**,提升跨具身可控性 | 共享 token 条件化 ③ 的结构参考 |
| **AMPLIFY/UMA/EgoBridge/LaST-HD/Point Policy/HumanEgo** | 多为 policy 或 ② 层(motion token / OT 对齐 / interaction token),非渲染 | 已在项目记忆;主要佐证 ② 层,非 ③ |

**综合**:没人声称「共享 human/robot 像素外观」能work(ceve 试了证据弱)。**成功路线都绕开外观**:要么锚锁外观 + human 只加多样性(OSCAR),要么换 appearance-invariant 预测目标(EgoWAM/MaskWM),要么 ambient-noise 只用 human 的高层信号(X-Diffusion)。**用户的 residual-Δz 属于「锚锁外观 + 共享 dynamics」这一类**,与 OSCAR 精神一致但更显式。

## 3. Δz 诊断(residual 假设的实证 gate)

**假设**(用户):human/robot 绝对 latent z 不同域,但 **Δz(delta latent,z_{t+1}−z_t 或 z_t−z_0)在类似 flow 下可能同域** → 若成立,③ 预测 Δz + robot 锚 = human 帮 dynamics。
**测法**:Wan latent 上,human vs robot 的收敛 LogReg 可分性(chance 0.5),对 z_abs / Δz_consec / Δz_anchor。

**结果(N=150/域,Wan latent,收敛 LogReg,chance 0.5)**:
| 特征 | pool1 | pool4 |
|---|---|---|
| z_abs(绝对 latent) | 0.972 | 0.996 |
| **Δz_consec(z_{t+1}−z_t 逐帧增量)** | **0.621** | **0.596** |
| Δz_anchor(z_t−z_0 对锚残差) | 0.901 | 0.996 |

**★判读(精炼假设)**:residual 成立**但只对逐帧增量形式**。
- 绝对 z 高度可分(0.97-1.0)= 外观不同域(坐实旧发现)。
- **Δz_consec(≈速度/局部动力学)大幅同域(~0.6)** = 帧间"怎么变"跨具身共享。
- Δz_anchor(对锚累积残差)仍可分(0.9-1.0)= 离锚越远累积的域特异外观越多 → **不能用对锚残差,要用逐帧增量**。
- 0.6 与 object-flow probe 0.645 同档([[project_amplify_grid_probe]]:probe RED 但下游 human-helps 成立)→ probe 门偏严,**这是绿灯偏乐观,真判据在下游 render LPIPS**。
→ R2 绿灯,且明确形式 = **consecutive-delta latent(latent 速度)预测**,非 anchor-residual。

## 4. 提案:路线菜单(按风险/优先级排)

**R1 — warm-start 多样性(最先做,低风险,OSCAR 同架构验证)**
- 在**新视频 ③** 上补判决:robot-only vs robot+human,且用 **warm-start**(robot-only 预训 N 步 → 混 human 继续),严格 I₀ 锚 + appearance-free 条件。
- 度量:robot held-out render LPIPS/PSNR/FVD。看 human 是否经**多样性**帮到新 ③(旧架构没帮 ≠ 新架构没帮)。
- 若正:护城河的最简版本成立,不需要新机制。

**R2 — consecutive-delta latent ③(novel,用户假设,§3 已绿灯,形式确定)**
- ★形式=**逐帧增量** Δz_consec = z_{t+1} − z_t(latent 速度,§3 证 0.6 同域),**不是** anchor 残差 z_t−z_0(§3 证 0.9 仍可分)。
- ③ 预测每步增量并积分:ẑ_{t+1} = z_t + f_θ(z_t, flow, agent);从 robot 锚 z_0 起 integrate。co-train human+robot 的逐帧增量 → human 帮共享的 per-step 动力学,绝对外观仍来自 robot 锚 + 积分。
- 变体(更 surgical,配 object-centric):**只对 object/flow 区共享增量**,agent 区由 embodiment-specific 条件(skel/gmask)单独渲 → 共享 object-Δz + 具身 agent。
- 与 [[project_latent_crossembodiment_residual]](Task#21:② 层 residual 跨域帮 robot,也是增量形式)一脉相承,搬到 ③。
- ⚠️积分漂移风险:逐帧增量积分长 horizon 会累积误差 → 需 scheduled sampling / 周期性锚 re-anchor(或与 I₀ 锚 + RF 联合)。

**R3 — appearance-invariant 预测目标(EgoWAM/MaskWM 验证)**
- 给 ③ 加辅助预测目标:DINO 特征 / 语义 mask / 3D-flow(EgoWAM 证 DINO+4× OOD、flow+20-30%)。human 走该通道迁移。
- 可与 R1/R2 叠加(aux head)。

**R4 — 解耦 / ambient(加固)**
- ceve 的 CLUB+InfoNCE 把外观从条件 latent 推出(aux loss)。
- X-Diffusion:human latent 只在高噪声段监督(分类器定阈值)。

## 5. 建议的最小判决实验(白天开跑)

新视频 ③ 上一个 2×2 sweep(robot held-out render LPIPS 为唯一判据):
- 轴 A:数据 = {robot-only, robot+human warm-start(OSCAR 式:robot 预训→混 human 继续)}
- 轴 B:目标 = {absolute z, **consecutive-delta Δz_consec**(§3 定的形式)}
- 4 run:主信号 = **human 帮不帮 ③(A)× 增量形式是否放大 human-help(B)**。
- 假设(据 §3+OSCAR):A 单独可能就正(warm-start 多样性);B 让 human-help 更大(共享 per-step 动力学)。
- 若 A×B 不够,再加轴 C:辅助 = {+mask/DINO appearance-invariant aux(EgoWAM/MaskWM)}。
- 度量口径:robot 留出集 render LPIPS(含 agent)+ 眼检,别只 latent MSE([[feedback_measure_real_deliverable_metric]])。
- Δz 诊断(§3)已完成 → B 绿灯,值得跑。

## 6. 与护城河/story 的关系

- μ₀/OSCAR/UMA 占「trace 作跨具身接口」——红海。我们楔子 = **可解码可交互视觉 WM**。
- **「让 human 帮到 ③」是这楔子的硬核卖点**:一个能渲染、且 human 数据真能提升其渲染/预测的跨具身 WM,是 trace-only 阵营做不到的(他们不渲染)。
- 诚实边界:OSCAR 已证 human 帮渲染(warm-start 多样性),所以我们的**新颖点不能只是"human 帮 ③"**,得是**机制**(residual-Δz / object-region 共享 / appearance-invariant target 的某个组合)+ **可交互可控**的整体定位。R1 若成是基线,R2/R3 的机制才是 novelty。

## 7. 相关记忆
[[reference_mu0_interaction_trace_wm]] [[project_keyboard_3way_ablation]] [[project_renderer_latent_route]]
[[project_latent_crossembodiment_residual]] [[reference_oscar_tri_crossembody_wm]] [[project_human_helps_end2end]]
[[reference_flow_repr_survey]] [[project_wan_vae_migration]]
