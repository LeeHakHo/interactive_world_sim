# IDM / object-flow→action 线 完整日志与战略反思 (2026-08-01)

> 自包含,写给接手的 Claude / 协作者。目标:把这条"从 object-flow 学逆动力学(IDM)解动作、再迁移到 human 演示"的线,试过什么、结果如何、related work 怎么说、以及为什么最后判断"在当前数据上这条线是边际的、我们一直在换表示上打转",一次说清。
> 相关记忆:[[project_idm_flow_to_action]] · 母题 [[project_policy_in_wm_pipeline]] · [[reference_policy_in_wm_playbook]]

---

## 0. 一句话结论(先看这个)

**在我们这批"同域、共享场景、物体同位置"的 can 数据上,把 object-flow 当作解动作/latent-action 的接口,价值是薄的 —— 因为(a)动作信号其实在 agent(手/eef)里,object-flow 单独放不了夹爪;(b)同域下 agent-trace 直接 IK 就够,所有表示最后都收敛,分不出胜负。表示之争在简单数据上本来就无解。** 真正能让 object-centric 站住的地方是 **OOD 泛化**(物体换位置/机器人够不到的姿势)和 **可 decode 的视觉 WM**,而不是同域的动作 placement。

---

## 1. 这条线要解决什么(初始动机)

我们有 ② object-flow 前向 WM(动作→物体怎么动)+ ③ Wan 渲染器(可 decode 回像素)。想:**学一个逆动力学模型(IDM),从 robot proprio + object-flow 学到预测 action;再从 human 演示解出 action** → real-world 做 BPP / Do-As-I-Do。IDM = 前向 WM 的逆。

---

## 2. ② 层面:IDM de-risk(动作能不能从 flow+trace 解出)

管线:窗口(object-flow + mp接触trace)→ IDM(小MLP)→ Δjoint → 积分 → FK → eef → mp tokens → 前向②(mp_r_all)→ object-flow,对 GT 比。数据 clips_robot_retrack L48 retrack episode-split(heldout vid 100/102)。

**输入消融(px@128,自由=物体静止/接触=运动):**

| ARM | 输入 | eef自由 | eef接触 | flow-recon接触 | ceil接触 | 判 |
|---|---|---|---|---|---|---|
| A0 | 只flow | 18.4💥 | 5.2 | 5.2 | 2.6 | flow-only 自由段臂位崩→证需trace |
| **A1** | flow+mp接触trace(双向) | **2.8** | **1.4** | **3.3** | 2.6 | ✅赢, de-risk PASS |
| A3 | causal(去未来窗) | 35.3💥 | 31.1💥 | 31.2 | 2.6 | 逆动力学需双向/未来窗 |

**结论**:A1 PASS —— 动作可从(flow+trace)双向窗解出,接触段贴天花板。A0 证 object-flow 单独放不了臂(自由段 18px)。A3 证 IDM 必须双向。
**★但埋了个雷**:A1 的 eef-trace 输入**本身就是 2D eef**,而 eef-recon 又拿 FK 的 eef 去比 → **半循环**,1.4px 有水分。

---

## 3. 精度调查(用户觉得 A1 不够精确 —— 对)

实测每步 Δjoint 相对误差 ~26% + 积分漂移(eef 逐帧 1.2→5.4px,关节漂移到 6°)。逐一排查瓶颈:

**(a) 输入表示不是瓶颈**(每步 Δjoint 相对误差,12seq 稳健):

| 模型 | 每步Δjoint误差 | overall eef | 备注 |
|---|---|---|---|
| A1 2D-mp | ~26% | 3.4 | — |
| B1 2D-raw eef | ~26% | 3.9 | ★反直觉:原始eef比mp**更差**(mp结构化去噪) |
| C1 3D(eef3d+tracks3d) | ~27% | 5.7 | 最难第4关节44%→30%改善;eef差因无2D-eef半循环捷径 |

→ 2D-mp / 2D-raw / 3D 三档每步误差**全 ~26% 齐平**。**信息量/表示不是瓶颈**。C1 eef-recon 更差恰说明去掉半循环后真精度齐平(A1 的 1.4px 确有水分)。

**(b) 容量不是瓶颈**:大 MLP(1024×5,600ep)每步误差 ~28%(没降,略过拟合)。

**(c) abs vs delta**:预测**绝对关节**去掉积分漂移,robot 自测 eef 大幅改善(C1abs 1-2px vs C1delta 4-6px),单帧稍抖。

**(d) transformer**(时空分解注意力,照 UMA/Point Policy,idm_train_xf.py / IDMTransformer):训练中(job55550,共享GPU极慢),**pending**。判据:破 26%→架构问题;破不了→任务/数据真极限。**注:即便破了,下面的战略结论不变。**

---

## 4. Thread B:human 演示解 robot 动作(Do-As-I-Do)

idm_solve_human.py:3D IDM(只robot训)吃 human(3D物体流+3D手轨迹eef3d)→ robot关节 → FK 骨架叠 human 画面。

**前提验证**:human eef3d 用**robot相机投影 = human eef2d(0px)**,z范围重叠 → **人机共享同一相机帧+世界系**,叠加合法。

**delta vs abs(夹爪抓取点↔物体 px)**:

| | seq0 | seq1 | seq2 | seq3 | 判 |
|---|---|---|---|---|---|
| delta C1 | 57💥 | 5 | 24💥 | — | 只1/4落对:固定J0锚点+漂移够不到 |
| **abs C1** | **7** | **4** | **7** | **7** | ✅4/4落对(=Human2Any"2D flow→action迁移OOD"实锤被abs绕过) |

abs 去锚点依赖是 human 迁移的关键。**但**:夹爪**尖端(idx8)超出** 13-25px(robot GT 抓取时 idx8 才 9px=最近点)→ 夹爪**朝向(腕关节,最难)**解偏,残差+OOD 显示在尖端。

---

## 5. ★ 用户尖锐质疑:"有 eef 不就直接 IK 出 joint 吗?object-flow 有用吗?"

**消融(human 迁移,夹爪抓取点↔物体 px):**

| 喂给 IDM | 抓取点↔物体 | 判 |
|---|---|---|
| 只 object-flow(Cf) | 33💥 | 放不了夹爪(flow相对质心,按构造无绝对位置) |
| 只 eef(Ct)≈学出来的IK人手 | 7.6 | ≈纯IK人手上限7.9 |
| flow+eef(C1) | 6.5 | 只比只eef好1.1px |

→ **就"把夹爪放到物体上"这件事,object-flow 没有独立贡献,IK 人手就够。用户对了。**

**闭环 decider(solved-action→②→物体运动 vs human演示实际物体运动,接触帧 px):**

| 命令方式 | 复现物体运动误差 |
|---|---|
| A 直接用人手eef命令 | 19.1 💥最差 |
| C 只eef(≈IK) | 13.8 |
| **B IDM(flow+eef)** | **12.6** ✅最好 |

→ 两个反向于 placement 的结论:**(1) 直接用人手 eef 命令反而最差** —— 具身不匹配,经学出来的 IDM 映射成机器人可行动作更能达成物体结果(19→13,IDM 是"具身桥");**(2) object-flow 在 outcome 上加了薄薄一层**(13.8→12.6,好1.2px),和 placement 版加分为0不同。
**★confound**:② 是 robot 训的,直接喂人手 eef(A)对 ② 就是 OOD → A 天生吃亏(但这也正是具身差本身)。

---

## 6. Related work 定位(LAPA / CLAM,latent-action-via-IDM)

- **LAPA**(2410.11758,ICLR25):VQ**离散** latent action(帧对→码),三段式(latent预训练→VLA学预测latent→少量真action标签接地)。**打爆视频预测(62 vs 22)**、**打败监督IDM/VPT(62 vs 44)**、**输给同域真action标签(62 vs 77)**;但跨具身+真机Bridge赢(纯human视频超OpenVLA)。**精细抓取输**(离散码分辨率)。**自承 latent 编进了相机运动(shortcut)**。
- **CLAM**(2505.04999,USC Erdem Bıyık):**连续** latent action(VQ对精细连续控制太钝,MetaWorld 9-20%→76%);**联合训练 action decoder 接地是关键**(连续单独只23%→联合74%);信息瓶颈防作弊;真机WidowX ~50k无标签+30demo,追平BC-with-labels。
- **2506.15691**:LAM ≈ 转移上的PCA,编码方差最大变化=常是**相机/背景非agent动作**。缓解=清洗/增广/**辅助动作预测头**。

**映射到我们**:
1. **我们的 IDM = 监督IDM/VPT 弱支**(LAPA 62 vs 44 打败它)→ 难怪同域边际。
2. **object-flow ≈ latent action,但 object-only 不够** —— LAPA/CLAM 的 latent 编码**整个转移(agent+物体)**,动作信号在 agent 里(我们 Cf=33px 自证)。AMPLIFY=object-flow版LAPA(objflow→forward→inverse head)。
3. **object-flow 的真差异 = 显式抗 shortcut**(2506.15691 的痛点)+ **可 decode 的 WM**(它们没有)。

---

## 7. ★ 战略反思:为什么"越来越绕"

**死循环的根**:动作信号在 **agent(手/eef)** → 但 agent **不跨具身**;**object-flow 跨具身** → 但**不带动作信号**。我们一直想两全,不停加层(3D→abs→transformer→latent action),**每修一个补丁露出下一个缺口**=该质疑架构而非加第4个补丁。

**关键洞察**:**在同域简单数据上,agent-trace 直接 IK 就解决了,object-flow 没有要解决的难题,所有表示收敛 → 怎么调都分不出胜负 → 感觉在绕。表示之争只在 OOD 下才会分开。**

**human 数据的价值本来就在 OOD/scarce**(我们 ② rh 只在 scarce 帮,full 饱和),不在同域刷精度。

---

## 8. 三条前进路(不再是"换表示")+ 待决定

1. **停止提纯 object**:用 agent+物体一起当表示(像 LAPA/CLAM),embodiment gap 用领域标准 retarget/IK(我们 IK 本就 work)。最简单诚实,但不新。
2. **只押 object-centric 唯一真赢处**:可 render WM + **OOD 泛化**(物体换位置 object-centric 该赢 agent-centric)。需造 OOD 测试床,是我们故事唯一站得住的地方。
3. **判 IDM/动作迁移线在此数据探完、边际**,human 数据价值收回 **② co-train(观测同域)+ 渲染**(有据可依),IDM 降级附属。

**决策准则**:相信 object-centric 在 OOD 会赢 → 走2造OOD测试一锤定音;心里没底 → 走3别再绕。**方向待用户拍板。**

---

## 9. 代码 / 产物索引(全提交)

- 数据/模型:idm_data.py(build_windows + build_windows_tokens + INPUT_SPECS:A0-3/B1/C1/Cf/Ct;YMODE delta/abs)、idm_model.py(IDM MLP + IDMTransformer)、idm_fk.py、idm_train.py、idm_train_xf.py、idm_eval.py、idm_solve_human.py、idm_ablation_table.py
- sbatch:sbatch_idm.sbatch(OUT_SUF/YMODE可切)、sbatch_idmabs.sbatch、sbatch_idmxf.sbatch
- spec/plan:docs/superpowers/specs/2026-08-01-idm-flow-to-action-derisk-design.md,plans/同名
- 结果:outputs/idm_derisk/{A0,A1,A3,B1,C1,C1abs,Cfabs,Ctabs,A1big,C1big,A1xf,C1xf}/、VERDICT.txt、ABLATION.txt
- Drive:iws_evals/2026-08-01_evals/{idm_derisk,human_solve,human_solve_abs}
- 天平数据源:② mp_r_all=outputs/cross_embodiment_wm/epsplit_L48/mp_r_all/wm_dual.pt
