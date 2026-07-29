# Session Report 2026-07-29 — 域头判决 · ★★heldout 泄漏 · dataloading 审计 · episode-split 修复

> 自包含,给用户看 + 给下个 session 接手看。今晚讨论极多、牵动之前一大堆结果,这里一次记全。
> 相关 memory: [[project_clip_heldout_leakage]] [[project_dualwm_domain_head]] [[project_bidirectional_transfer_3x2]] [[project_multihead_aux_wm]]

---

## 0. TL;DR(先看这个)

1. **★★头号发现:heldout 帧泄漏(严重)。** clip 是重叠滑窗(stride12帧≪窗口192/96帧)+ `split_okfirst` 按 clip 下标切(非 episode)→ **85% heldout clip 与训练 clip 共享 ≥50% 完全相同帧**,17% 近重复。**前面所有 drift 绝对数偏乐观。** 相对 delta(匹配臂)大概率仍成立。
2. **two-head 域头判决 = 无效。** rh_two 2.314 ≈ rh_single 2.318 ≫ ro 2.083 → 域专属输出头没削掉全量饱和害;害是 **trunk 级**,输出侧治不了(正中 related-work 预测)。
3. **几何 retarget(dummy5rt)判负 + 可视化坐实**:它把指开口冻成 canonical R0 → robot open/closed 星座变一样 → 抹掉抓取信号 → 全场最差 ro2.928。mp 赢因为把开口留成域中性标量。
4. **修复:episode-split 已实现**(照 IWS `play_eef_dataset:378`),决定性 9 臂正在干净重训。
5. **底层真相不变**:human 满量不帮 = 单场景数据天花板(18robot/12human episode,同一 rig/罐/桌);retarget/域头/切片都救不了满量,只在稀缺/迁移有价值。真出路 = 多场景数据。

---

## 1. ★★heldout 泄漏(今晚最重要,并行 agent 实锤)

**根因(代码+数据实证)**
- 切片重叠滑窗:`gen_flow_render_dataset_caneef.py:37` `CLIP_STRIDE=12`;窗口跨 `L×S`=192(robot L48)/96(human L24)帧。stride12 是子采样 S=4 整数倍 → 相邻 clip 共享**逐帧完全相同**源帧。
- split 按 clip 下标:`split_okfirst`(`exp_scel_dualview_wm.py:528`)= `rng(0).permutation(np.where(ok)[0])[:150]`,**无 vid 分组**。③ multihead 用 `HELDOUT={332,59,418,442}`(clip 下标)同泄漏。

**严重度(robot 154 heldout / 2436 pool 实测)**
| 指标 | 值 |
|---|---|
| episode 横跨 train/heldout | 18/18 |
| heldout clip 与训练共享 ≥1 帧 | 147/154 (95%) |
| 共享 ≥50% 相同帧 | 131/154 (85%) |
| 共享 ≥90%(近重复) | 26/154 (17%) |

**影响**:绝对 drift/LPIPS 全部偏乐观;相对 delta(同 N 同 heldout 匹配臂)大概率成立;scarce→full 曲线可能被泄漏放大(训练越多→重叠越多→越乐观)。

**关键:IWS 自己不漏。** `play_eef_dataset.py:378` "最后 val_ratio 个 episode 作 val" = episode 级(干净)。只我们 bolt-on 的 ②/③ 偏离了 house standard。

---

## 2. two-head 域专属读出头(设计→实现→判决)

**动机**:单头 mp 全量 human 害 robot(ro2.083→rh2.318,−0.235)。给共享 trunk 加 robot头/human头,让 human 只塑 trunk 不污染 robot 输出。
**实现**(SDD 全 review clean,`4e86a8b..6795f4d`):`HEAD_MODE=single|two|film`,`_readout(x,dom)` 路由,`head_h=deepcopy(head)` 暖启,dom 穿 `_ss_loss/train_dual/rollout_dual`;single 逐字节不变。终审抓到并修了 `rollout_dual` dom 契约破坏兄弟脚本的集成 bug(`6795f4d`)。

**判决(GPU1 直跑全量,配置全对齐)**:

| (H40, robot flow) | drift |
|---|---|
| ro | **2.083** |
| rh_single | 2.318 |
| **rh_two** | **2.314** |

→ **无效**。害是 trunk 级(robot 头再干净读的还是被 human 拽偏的 trunk)。

**2×2 头诊断**(H20 同数据比头):robot头 robot-flow 1.696 / human-flow 2.946;human头 human-flow 2.067 / robot-flow 2.307。→ **两头确实各自分化了**(对角优于非对角),但分化不改 trunk 级害 = 从假设变实测证据。

---

## 3. related-work grounding(两轮并行精读)

**第一轮(多头):FlowWAM/OSCAR/EgoWAM/WEAVER/MaskWAM**
- **没一篇用硬域输出头**;全共享头。OSCAR 把 "avoids embodiment-specific prediction heads" 当卖点。→ two-head 无先例。
- 场里做法 = **软域嵌入(EgoWAM 默认)+ warm-start(OSCAR/FlowWAM)**。
- **负迁移只在 action 通道,不在 world/dynamics 通道**(EgoWAM 核心);他们 world 通道从没见 human 害 robot → 我们全量见害很可能 setup 特有(他们 human 重/robot 稀缺永远在帮区)。
- MaskWAM 84.9→21.6:条件不配预测监督就被无视 → 辅助预测头是更大杠杆([[project_multihead_aux_wm]])。

**第二轮(dataloading):**
| | clip 长度 | 相邻窗口 | split | fps |
|---|---|---|---|---|
| FlowWAM | 帧桶{17..81} | 桶内随机起点 | 按 task | 15 |
| WEAVER | 8帧 | 随机窗口 | 按轨迹 256val | 5Hz |
| **OSCAR** | 固定81 | **grasp/release 事件加权起点** | **按 episode+语义去重** | 15 |
| **我们** | 48/24不齐 | **密集重叠滑窗** | **clip 下标(泄漏)** | **30(偏高)** |
- 没一篇用重叠滑窗;全按 episode/轨迹 split。我们 30fps 偏高 → human/robot 长度不齐一半是 fps 假象。horizon 都和 clip 长度解耦(自回归开放式 eval)。

---

## 4. action-rep 线(mp/dummy5/cpt + 几何 retarget)

**干净全量表(H40,drift_px_cam_high)**:
| rep | ro | rh(+human) | human 帮 |
|---|---|---|---|
| **mp** | **2.083** | 2.22 | −0.13(害最小) |
| dummy5 | 2.44 | 2.76 | −0.32 |
| dummy5rt(几何retarget) | **2.928** | 3.297 | −0.37(最差) |
| cpt(纯接触点) | 2.505 | 3.199 | — |

**★dummy5rt 判负 + 可视化坐实**(`outputs/cross_embodiment_wm/action_rep_viz/`):它把腕距/指开口冻成固定 canonical(W0/R0),只 c+朝向真实 → **robot OPEN(开口0.103)和 CLOSED(0.001)星座变一样大** → 抹掉夹爪抓取信号 → 最差。**mp 赢**因为:同样冻星座形状,但**把开口留成域中性显式标量**(grip 通道),不塞进几何里再一起抹掉。cpt 更差(3.199)证明纯接触点丢太多(mp≫cpt,mp 不是一个点,是转的小十字+朝向+grip)。

**learning-based retarget(未做)**:MT-π 式学 human→robot 映射 / EgoWAM per-domain 输入 stem。是**输入侧对齐**(域头是输出侧,一枚硬币两面)。比几何 retarget 原则化(几何已判死)。**但同样别指望救满量**(数据天花板);价值在稀缺/迁移 + Do-As-I-Do BC。

---

## 5. 稀缺 sweep(human-helps 曲线,干净版但泄漏未修)

| NROB | 100 | 300 | 900 | 1800 | 2436(full) |
|---|---|---|---|---|---|
| mp human 帮(ro−rh) | **+6.62** | +2.87 | +1.53 | +0.33 | **−0.13(翻负)** |
| dummy5 | +5.97 | +2.07 | +3.02 | +0.17 | −0.32 |

- **稀缺 human 帮巨大**(mp n100 +6.62),随 robot 增多单调衰减到全量翻负。
- mp 稀缺帮 > dummy5 → 对齐/表示在稀缺区让 human 更能帮。
- ⚠️这批**泄漏未修**;干净重训后曲线形状待复核(泄漏可能放大"饱和")。

---

## 6. horizon / eval 哲学(纠正 + 定调)

- **切片长度不限制模型 H**:rollout 自回归,keyboard/推理 H 无限。**clip 长度只限"能和 GT 比多远"**(human 24帧→drift 只能测 H20,robot 48帧→H40)。**H40 vs H20 是评测手头 GT 的差,不是模型能力差。**
- **held-out 有必要吗**:要下量化结论(human 帮多少/mp 赢)就必须有 GT 对照=held-out。**keyboard 单独不够**(无 GT,只能测可控性不能测准确度)。**IWS 本身两条都用**(val episode + 交互 rollout)。FlowWAM 主 eval 是**量化闭环**(任务成功率)——所以 keyboard 值得**从眼检升级成量化成功率**(差异化,[[project_policy_in_wm_pipeline]])。
- **结论**:held-out 必要但单场景先天弱、别过度投入;主力押 human-helps 相对 delta(泄漏稳)+ 量化 keyboard 控制。

---

## 7. 数据事实(核实)

- robot: N=2700 clip(low_valid 2590),L=48 帧,**18 episode(vid100–117,各150)**,~3h play。
- human: N=1800(valid 1795),L=24 帧,**12 episode(vid0–11)**,~2h play。
- **play data = 连续同场景**(同 rig/罐/桌/机位);不同 episode ≠ 不同分布,是同场景不同时间段 → **单场景天花板**。
- clip 是重叠滑窗(见 §1)→ "N clip" 因重叠而虚。
- demo seq 归属:clip59→vid100;clip332/418/442→vid102。

---

## 8. 今晚的修复(已落地)

**episode-split(commit `7868fa8`)**:`exp_scel_dualview_wm.py` 加 `split_by_episode(ok,vid,heldout_vids)` + env `HELDOUT_VIDS`(默认空=旧行为逐字节不变)/`HELDOUT_VIDS_H`;config 记 `heldout_vids`。测试 `tests/test_episode_split.py`(合成+真数据泄漏-已修)3/3 绿,冒烟通(ho279/pool2311,human cotrain 1497)。spec=`docs/superpowers/specs/2026-07-29-episode-split-leakage-fix-design.md`。
- 留出:robot vid{100,102}(279 clip)/ human vid{0,1};干净训练池 robot **2311**(16 episode)。

**决定性 9 臂重训**(② 7 臂 GPU2 串行 job `blfhmgkgk` 跑中;③ 2 臂待):
② `{ro,rh,rh_two}全量 + {ro,rh}@n100 + {ro,rh}@n300`,ACTION=mp,retrack clips,`HELDOUT_VIDS=100,102`。输出 `outputs/cross_embodiment_wm/epsplit_clean/`。

---

## 9. 决策 / TODO / 开放问题

**已决**
- 域头两头无效,不铺 sweep;`HEAD_MODE=two` 代码保留。
- 修 split(episode 级),不改切片(slicing = Phase 2 独立立项,要重建数据)。
- 留出 robot 2/human 2 episode;今晚决定性 9 臂。

**TODO(优先级序)**
1. ② 7 臂重训完 → 读干净 drift,复核方向性判决(two-head 无效/human 稀缺帮/满量害)+ 记诚实 headline。
2. ③ split 同口径修(`train_multihead_wm` HELDOUT→按 vid)+ 重训 {ro,rh} render 一对。
3. 干净数出来后更新所有相关 memory/report 的绝对数(标注旧数泄漏)。
4. (Phase 2)切片 best-practice:episode 内随机起点 + contact 加权采样(顺带治 grip/release 稀缺)——要重建数据(重 track)。
5. (Phase 2)量化 keyboard controllability 成功率。
6. (方向)learning-based retarget(稀缺/迁移 + BC),别指望救满量。
7. (根本)多场景 human 数据才是真解锁 human-helps。

**开放问题**
- 干净 split 后 scarce→full 曲线形状是否变(泄漏是否放大了"饱和")?
- learning-based retarget 在稀缺区能否 > mp?
- fps 降采样 + human/robot clip 长度统一,值不值得(Phase 2)?

---

## 10. 当前 running / 文件索引

**running**:② 7 臂 episode-split 干净重训 = **sbatch job 54816–54822**(shard;用户要求晚上全走 sbatch;轮询器 `poll_epsplit.sh`)。输出 `outputs/cross_embodiment_wm/epsplit_clean/`。③ 2 臂待(train_multihead_wm split 同口径修后提交)。
**关键文件**
- split 修复:`exp_scel_dualview_wm.py:split_by_episode`(commit 7868fa8)+ `tests/test_episode_split.py`
- 域头:`exp_scel_dualview_wm.py` HEAD_MODE(4e86a8b..6795f4d)+ `tests/test_dualwm_head_mode.py`
- 泄漏排查实证:`scratchpad/leak.py,leak2.py`;2×2 头诊断:`scratchpad/eval_twohead.py`
- retarget 可视化:`outputs/cross_embodiment_wm/action_rep_viz/`(已传 Drive `iws_evals/2026-07-29_evals/`)
- 切片 builder:`gen_flow_render_dataset_caneef.py:37,179,184,209`
- spec:`docs/superpowers/specs/2026-07-29-episode-split-leakage-fix-design.md` / `2026-07-29-dualwm-domain-head-design.md`
