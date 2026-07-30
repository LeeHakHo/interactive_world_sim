# 策略/定位 doc — 2026-07-30(泄漏修复 + L48 + seed-verified 之后)

> 自包含,给用户 + 下 session。把这一大轮(heldout 泄漏、L48 重建、two-head/FiLM、seed 复核、FlowWAM 对标)的定论收成一份可执行策略。
> 相关: [[project_clip_heldout_leakage]] [[project_dualwm_domain_head]] [[project_multihead_aux_wm]] [[project_dualview_dit_formal_done]] [[project_keyboard_3way_ablation]]

## 0. TL;DR
- **★human-helps-at-scale 基本证死**:满量 human 在**任何**地方都是**单场景数据天花板**(②/③/aux 头/two-head/FiLM 大概率),seed 复核确认全量信号 < 训练噪声。**只在稀缺(小 demo)真帮(+4.9 ≫ 噪声)。**
- **活着的三条腿**(paper 该押):
  1. **flow-cond 优越**(FlowWAM 式,但我们 **object-centric** 差异化 vs 他们 agent-flow)。
  2. **可控 object-centric 跨具身接口**(② 预测 object-flow → 驱动 ③;keyboard / human-flow-prompt)。
  3. **会渲染**(Wan VAE 视频 ③);human-helps 老实讲**只在稀缺**。
- **不要**再写"human 帮训出更好的满量 robot WM"(已 debunk)。

## 1. seed-verified 定论(干净 L48 episode-split)
| 结论 | 数(3 种子均值) | 判 |
|---|---|---|
| 全量 human-help(mp) | −0.066(横跳 −0.39~+0.11) | ❌噪声, 数据天花板 |
| two-head robot 域 | −0.21 | ❌无提升/略差 |
| two-head human 域 | −0.28 | ❌各域都不如单头 |
| FiLM(软域) | 跑中(bwrndxr37) | 预期全量噪声, 看稀缺 |
| 稀缺 human-help(n100) | +4.9 | ✅真, ≫噪声 |
**根因**:小单场景数据(18 robot / 12 human ep,同 rig/罐/桌)→ 满量 human 冗余;硬两头分数据喂不饱;共享单头对两域都最优。

## 2. 为什么 human 帮不到渲染(机制,可写进 paper)
**② 已经准到 ③ 懒得计较**:② ADE 2.7/3.7px → e2e render 0.270/0.298 **只比 GT-flow 天花板差 +0.007/+0.016**(DUALVIEW_DIT_REPORT),明显赢 naive eef-FiLM 0.285/0.345。
→ **③ 对 ② flow 误差宽容 → 把 ② 改准(靠 human)对渲染几乎零增益 → rh≈ro@③**。这是 flow 接口的胜利,也是 human-helps 在渲染层的天花板。宽容有边界(稀缺/OOD 误差大时 ③ 才敏感)。

## 3. flow-cond vs IWS-stage2(naive)重做方案 — 照 FlowWAM Fig.4 方法学
FlowWAM 的干净 ablation = **固定 backbone/VAE/数据,只换 condition**;numerical-action(=naive eef=IWS-stage2 等价)是最大 gap(成功率 +20pts,TA 35.92→64.26)。照做:
1. **同当前 ③(Wan L48)backbone,只换 cond**:flow-cond vs eef-only-cond(naive) [vs 可选 mask]。**别再拿"我们③ vs IWS③"**(当年 ours-noflow≫IWS 混了实现差,不纯)。
2. **两套 metric 分开**:(a)**cube_px / object 位置误差 = 我们的 TA**(物体跟不跟条件)、(b)render-LPIPS(外观)。**当年只报 obj-LPIPS,要补控制保真度。**
3. **foreground naive-eef 行 = IWS-stage2 等价物**,量化 gap。
4. **decodability check 已有**:e2e≈GT-flow 就是"flow 可解码"证据;补 flow-error vs 控制成功率相关性(FlowWAM Fig.4c,r=−0.81)。
5. eval 口径不一致先声明(FlowWAM footnote 式)。

## 4. 在 ③ 下功夫 = 两件事,别混
- **③ 质量(渲染清晰/天花板)= renderer capacity**(Wan VAE / 分辨率 / DiT),**和 human 无关**,纯工程/架构。这是提升渲染的主力。
- **③ human-helps = aux 头**([[project_multihead_aux_wm]]:DINO/depth/mask/HC 塑共享 trunk):**满量也数据天花板消失**,只稀缺/OOD 可能帮;几何头(mask/HC)> DINO。**定位=机制故事 + 稀缺,不是质量主力。**
- **warp 已关**(③ CCOND=4 只取 flow+agent,丢 warp;ablation 早证冗余)。cond 文件仍存 7ch,以后 NOWARP=1 省掉即可。

## 5. object-flow ② 定位(还要单独拿出来)
- ② **不靠 human-helps 立足**(human 在 ② 稀缺帮、不传渲染)。
- ② 立足于:**(a) 可控 object-centric 跨具身接口**(差异化 FlowWAM agent-flow / OSCAR 骨架)、**(b) flow-cond 优越 claim 的预测器**(没 ② 就得喂 GT-flow,系统不成立)。→ **独立贡献,保留。**

## 6. 下一步(优先级)
1. **flow-cond vs naive-cond 重做**(当前 Wan ③ + L48,加 cube_px 控制 metric)—— paper 的第 1 条腿,最该做。
2. **FiLM 结果**(bwrndxr37)→ 收口头模式全表(single/two/film × 域 × 全量/稀缺,seed-verified)。
3. **③ 稀缺/OOD human-helps**(aux 头,几何头优先)—— 唯一还有戏的 human-helps,框定稀缺。
4. **③ 渲染质量**(renderer capacity:分辨率/VAE)—— 和 human 无关,提渲染。
5. **learning-based retarget**(spec 已写)/ **③ agent-cond OSCAR 线画**(human MANO vs pinch)—— 稀缺/迁移侧。
6. 多场景 human 数据 = 满量 human-helps 唯一真出路(超出当前数据)。

## 7. 开放问题
- FiLM 稀缺是否 > 单头(软域条件够不够格)?
- flow-cond vs naive 的 cube_px gap 多大(TA 类)?
- ③ aux 头在 OOD(keyboard)上 human 是否帮(唯一没测干净的 human-helps regime)?
- 切片 overlap stride(Phase 2,当前 episode-split 已够诚实)。
