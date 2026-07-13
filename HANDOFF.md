# HANDOFF — cross-embodiment object-flow WM(2026-07-13 更新,给新 session)

> ⚡⚡ 2026-07-13 最新完成:**dual-view DiT ③ 正式化(项目目标)已交付** — 读仓库根
> **`DUALVIEW_DIT_REPORT.md`**(主门控 PASS: flow>eeffilm 双视角>2σ;e2e ADE 2.1px 近天花板;
> ★eefsp(OSCAR-skeleton式)cam_high 反超 flow 的 nuance;human-helps ③ 层三臂全不显著=帮在②;
> IWS stage2 DF external 垫底)。memory=`project_dualview_dit_formal_done`;
> run 实体在 /scr/yusenluo/iws_overflow/(/scr2 满盘迁移,原路径 symlink);
> drive=iws_evals/2026-07-13_dualviewDiT正式化_flow接口判决/。

> ⚡ 2026-06-26 最新待办(优先级最高,见 §6.0):用户批评"乱参考架构、没看好的 related work"。
> 下个 session 第一件事 = 系统读仓库根 14 个未跟踪 PDF(UMA 2606.16917 / WEAVER 2606.13672 /
> 2605.03637 / 2606.13515 / 2511.04671 / 2512.13644 等),产出"我们做法 vs SOTA"对照,把渲染器③
> 和混 human 的选型重新 anchor 到论文,**别再拿 in-house 的 stage2/CMDecoder/自建 SPADER 当参照系**。
> 教训已存 memory `feedback_ground_in_related_work`。


> 新 session 从这里入。读完本文 + `RENDERER_GRASP_RESEARCH_LOG.md` + `MEMORY.md` 索引即可接手。
> 分支 `phantom_dynamo`,conda env **iws**(base 的 numpy2 会崩 cv2)。

## 0. 一句话
跨具身世界模型(human 演示帮 robot)。架构:① CoTracker object-flow(2D点)→ ② FlowWM(action→cube flow,动力学)→ ③ flow→像素/latent 渲染。这几天主线 = 渲染器 ③ 攻坚 + **latent 跨具身判决(重大转折)** + human-helps 分析。

## 1. 怎么快速接手(读这3个)
1. **`RENDERER_GRASP_RESEARCH_LOG.md`** — 主 log,§1-8 全程 option/结果/舍弃/为什么(含 latent 翻盘 §8、human action §7)。
2. **`MEMORY.md`**(在 memory 目录)— 索引;重点读 `project_latent_crossembodiment_residual`、`project_renderer_latent_route`、`project_keyboard_interactive_demo`、`project_grip_grasp_aware_wm`、`reference_uma_unified_motion_action`。
3. **`outputs/cross_embodiment_wm/RESULTS_2026-06-21/INDEX.md`** — 所有可看 gif(渲染对比 + keyboard)。
4. **`RENDERER_IMPL_DETAILS.md`**(06-25 新增)— 渲染器③两赢家(detmem/temporal-SPADE)的实现&data flow 速查:SPADE per-pixel×per-channel γ/β、detmem 纯conv 四件套、temporal loss、cond=flow+gmask 4ch、7-arch ablation 表。memory 索引 `reference_renderer_impl_details`。

## 2. 关键结论(这几天的产出)
- **③ 渲染赢家 = detmem-16ch**(`exp_scel_latent_detmem.py`):frozen 通用 VAE(ostris 16ch)+ 确定性预测**绝对** latent + decode-LPIPS + prev-memory,**不用 flow-matching**。GT-flow LPIPS **0.138 锐度赢 pixel** temporal-SPADE 0.141;帧间 pixel 略稳(7.05 vs 7.36)。无单一全赢:detmem 最锐 / temporal-SPADE 最稳。
  - 四要素缺一不可:16ch VAE / decode-LPIPS / prev-memory / 不用flow-matching。delta-latent 渲染判负(累积误差,③保持绝对)。
- **★latent 跨具身转折(Task#21,`exp_scel_latent_mix.py`)**:latent 混训**确实帮 robot**(scarce),推翻"latent 不跨具身"(旧只推断没验证)。**residual(Δz)>>direct(绝对z)**:起点不跨域、dynamics 变化跨域。关键=**frozen 通用 VAE(非 domain-ViT)+ residual dynamics**。与 object-flow human-helps 帮幅相当(object-flow 天然 residual=anchor+Δ,同机制)。
- **human action 有用(Task#22,`exp_scel_human_actionfree.py`)**:估计的 human eef(HaMeR,有噪声)仍有用(Δhelps +6),action-free 更差(N大有害)。不去掉 human eef。
- **contact gate 判决否定**(06-17/18,memory `project_contact_gate_greenlight`):contact AUC 高(0.95-0.99)但 gate 干预无用——weld sweep 证 baseline② 本就接触敏感(travel 随 eef-cube 距离 39.5→0.4px 降 99%),"无条件焊住"前提推翻,gate 冗余+有害(drift 2.59→4.13)。★可分性≠干预有用。
- **keyboard demo**(`exp_scel_keyboard.py`):组合动作脚本(5左2上1下等)驱动②/③;`REN_KIND`=detmem/temporal/pixel;agent 用 **IK adapter**(eef→joint,`exp_scel_ik_adapter.py`)铰接(根部固定);**BOX clip**(框内,框外OOD agent消失)质心刚体clip;`MAXH` 任意horizon(IK joint+合成vis,H=80验证)。gripper 控制未做(Task#20)。

## 3. paper story / novelty(用户重视,2026-06-22 精炼)
**framing**:cross-embodiment WM **何时/为何能跨具身**。核心原则(比"哪个接口"更本质):**状态表示是否同域决定要不要 residual**。
- **同域**(object-flow 的 cube 2D 位置,human/robot 都在桌面同区)→ **direct 也跨**(`exp_scel_objflow_residual.py`:direct +27~42% ≈甚至> residual);
- **不同域**(latent 整帧起点,agent 外观 582×可分)→ **只有 residual(Δz)才跨**(`exp_scel_latent_mix.py`:direct +5.6% << residual +35.8%)。
- object-flow / latent 是同一原则的两个实例;**frozen 通用 VAE(非 domain-ViT)**是 latent 能用 residual 跨域的前提。
- **"residual" 非万能**:③ 渲染 delta 判负(单域累积误差);residual 只在"状态不同域的跨具身迁移"是杠杆。
完整 story = negative(IWS domain-ViT+绝对latent 不跨)→ diagnosis(状态同域? + encoder选择 + 绝对vs residual)→ positive。
+ human-helps 量化(human 顶 ~150 robot clip,scarce 才帮)+ human action 有用(非 action-free,Task#22)。
UMA(2606.16917)/WEAVER(2606.13672)混了 human 但**没做此机制分析** = 我们的 contribution。**渲染器 ③ 本身 novelty 弱(工程组合),别当卖点**。
⚠️ 教训(用户揪出):别拿不同 metric 的 Δ% 比大小(latent-MSE vs flow-ADE px);对照实验两边都要在 direct/residual 同框架跑。SMOKE 欠训会误导(SMOKE 见 residual>>direct,full 推翻)→ 结论必须 full。

**★flow contribution 综合判决(2026-06-22,三对照全跑完,关键)**:核心问题=若直接 latent co-train 就够好,flow 绕路难立。跑了整帧/cube-region/选项①(latent+decode-LPIPS)三对照(`exp_scel_flow_vs_latent_pixel.py` + `exp_scel_latent_dyn_lpips.py`):
| | human-helps | 渲染质量(眼检) |
|---|---|---|
| latent co-train | ✅ 强(整帧+5.5%/选项①**+12.4%**,真跨具身) | ❌ 糊黑团不可用(0.307,decode-LPIPS 只救一点) |
| flow(②+detmem) | ❌ 卡②没传像素(cube-region **−1.8%**) | ✅ 清晰可用(0.244) |
**两路各一半,无两全**。flow contribution:**不能立**在"latent 不跨/co-train 没用"(推翻,latent+12.4%强)、**不能立**在"flow human-helps 传像素更好"(推翻,卡固定③);**只能立**在"flow 显式物体结构(cube位置+I0锚)是**可用渲染**的关键"(latent dyn 缺→糊)——渲染论点,novelty 弱。机制:detmem 清晰=I0静态锚+flow精确引导+prev mem;latent dyn 糊=无I0锚、z_hist漂移累积。**★最有前途=结合**:latent dyn 的 human-helps + detmem 的渲染清晰(I0锚+物体引导)。**Task#23 下一步=给 latent dynamics 加 I0 静态锚再试**。详见 RESEARCH_LOG §9 综合判决 + `RESULTS_2026-06-21/{06,07}`。⚠️必须眼检 gif(数字骗人:latent +12.4% 看着赢,眼检是糊团)。

## 4. 文件结构
**脚本(仓库根)**:
- 渲染③:`exp_scel_latent_detmem.py`(赢家)、`exp_scel_temporal_renderer.py`、`exp_scel_render_cond.py`(7-arch ablation:concat/spade/...)、`exp_scel_latent_fm.py`(WEAVER flow-matching)、`exp_scel_latent_renderer.py`(det-latent + VAE enc/dec,`VAE_NAME` env)、`exp_scel_latent_lpips.py`、`exp_scel_final_eval.py`(全候选 GT-flow+pred-flow gif)、`exp_scel_render3_compare.py`
- latent跨具身:`exp_scel_latent_mix.py`(Task#21,`RESIDUAL` env)
- human:`exp_scel_human_actionfree.py`(Task#22)、`exp_v3_human_helps_flow.py`(原 human-helps)
- keyboard:`exp_scel_keyboard.py`、`exp_scel_ik_adapter.py`
- 数据生成:`gen_flow_render_dataset_v3eef.py`(v3 + grip,改过:存grip+MOVE_MIN保留开合)
- ②动力学:`amplify_wm.py`(FlowWM_LWC)、`e2e_flow_wm_render.py`(FlowWM)、`eval_scheduled_sampling.py`(train_wm_ss)

**数据**:`outputs/flow_render_dataset_v3/clips_{robot,human}.npz`(L=24,human 1800+robot 2700,有eef无human-joint)、`outputs/flow_render_dataset_v3_grip/clips_robot.npz`(L=48,含grip,robot only)

**关键 ckpt**(`outputs/cross_embodiment_wm/`):`latent_detmem/detmem.pt`(③赢家,16ch)、`temporal_renderer/temporal_spade.pt`、`render_cond/spade.pt`、`ik_adapter/ik_adapter.pt`、`grip_wm/{wm_base.pt,wm_grip.pt}`(②)、`latentmix_cache/{robot,human}.npy`(VAE latents 缓存)

**VAE**:`ostris/vae-kl-f8-d16`(16ch/8×,HF 已 cache);用 `VAE_NAME=ostris/vae-kl-f8-d16` + `HF_HUB_OFFLINE=1`(或 dangerouslyDisableSandbox 走网络)。

**结果**:`outputs/cross_embodiment_wm/RESULTS_2026-06-21/`(gif+INDEX)、各实验 `*/summary.txt`

## 5. 运行须知
- `conda activate iws`;detmem/latent 需 `HF_HUB_OFFLINE=1 VAE_NAME=ostris/vae-kl-f8-d16`。
- GPU 0-3,7 通常可用(4,5,6 常被别人占);跑前 `nvidia-smi` 确认,只用空闲。
- 长任务后台 nohup,用 `until grep DONE` monitor 等。
- 仓库大量代码未提交→**就地工作,只 git add 自己的文件,不加 Co-Authored-By**。

## 6. todo / 下一步(优先级)
0. **★最高优先(2026-06-26 新):读 related work 做对照**。仓库根 14 个未跟踪 arxiv PDF 是用户下的 related work,我(上个 session)没系统读、没把渲染器③/cross-embodiment WM 选型对标它们。任务=逐篇读→产出"我们 vs SOTA"对照→把方案 anchor 到论文(非 in-house 架构)。用户原话:"你的问题就在于乱参考架构,而没有去看看比较好的 related work"。教训 memory `feedback_ground_in_related_work`。若用户点名某几篇优先啃。
1. **latent-mix 稳化 + 双因子对照**(Task#21 延伸,paper 主表):object-flow direct 对照**已做**(`exp_scel_objflow_residual.py`:direct≈residual,object-flow 靠"状态同域"非 residual);**剩**:① SEEDS↑ 稳化(现2,noisy,latent N400/object N400 都有翻负噪声);② **IWS domain-ViT latent + residual** 对照(验证"encoder 选择"那半:domain-ViT 即使 residual 也不跨?→坐实 frozen 通用 VAE 是前提);③ 汇总"状态同域? × direct/residual"总表(object-flow / VAE-latent / ViT-latent)。
2. ~~**#20 keyboard gripper 控制**~~ **✅已做(2026-06-26,`GRIP=1`)**:`exp_scel_keyboard.py GRIP=1 REN_KIND=detmem`(v3_grip+grip②+IK adapter)。script 加 `("open"/"close",n)` 伪动作,指尖随 grip 张合(`_fingertip_offset`,对称不挪质心,进 gmask+flow),新 `grasp_compare` 渲 CLOSED|OPEN 并排。结果 `outputs/cross_embodiment_wm/scel_keyboard_grip/`(INDEX+50gif):可控 demo 成立(pump/open_drag/close_drag 有别),grasp 判决 CLOSED 19.9px>OPEN 13.6px 张爪一律减弱但只 1/10 跨0.5×释放阈(坐实 weld 未破,~5/30)。见 memory `project_keyboard_interactive_demo`。
3. paper 写作:把 §8(latent跨具身)+ human-helps + Task#22 拼成"cross-embodiment WM 的表示/数据形式怎么选"系统研究。

## 7. 工作习惯(用户硬要求,memory 有)
- **★重要可视化 eval 一律上传 Google Drive**(2026-07-05 新):`upload_evals_gdrive.sh "易读名=目录"`,结构 iws_evals/<日期>_<主题>/<序号_易读实验名>/+INDEX.md;**文件名和结构必须易读**(中文含义名,别传 add_m4_hm0.0 这种);gdrive 路径随本地路径一起给用户。rclone 用 ~/.local/bin/rclone(v1.74)。
- **一切对比出 gif**(网格动态,非静态png);③渲染 gif 必须 **GT-flow(flow=GT绿)+ pred-flow(flow=PRED红=②预测)两版**,NSEQ≥8。
- **记录所有 option 含舍弃**(以后可能复活);结论存盘 summary.txt 非口述;给图绝对路径。
- 我(助手)眼检可能不准 → 用 **metric + gif** 让用户验证。
- 声明组件版本;per-experiment 独立输出目录。
