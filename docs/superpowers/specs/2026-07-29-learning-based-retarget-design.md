# learning-based human→robot action retarget 设计(B)

日期: 2026-07-29
状态: 设计(A=几何 dummy5rs 已跑;B 今晚只设计不实现)
相关: [[project_action_retarget_deltahc]] [[project_dualwm_domain_head]] AGENT_RETARGET_RELWORK_2026-07-26.md

## 0. 一句话
不再手工规定 human→robot 的动作变换(几何 retarget 已判死:dummy5rt 冻死开口=最差;dummy5rs=按域缩放,在跑),而是**让网络端到端学**这个变换,由下游 ② WM loss 决定保留什么——它**不会过度冻结**(冻掉有用信息 loss 变差)。

## 1. 动机 / 为什么几何不够
- **dummy5rt(冻 canonical)判负**(ro2.928 最差,可视化坐实抹掉抓取信号)。
- **dummy5rs(按域中位数缩放,breathing)在跑**——但仍是**手工公式**(用固定中位数常数)。
- learning-based = 让模型自己学"对齐什么尺度、保留什么变化",不靠人拍常数。

## 2. related-work grounding(已精读,别闭门造车)
- **MT-π**(2502.20391 组内近似):学一个 retarget MLP(human eef → robot 动作空间)。= 我们 setup 命名来源。
- **EgoWAM**(2607.08436):**per-embodiment 薄输入 stem → 共享 trunk**(proprio stem = per-domain 浅 MLP 14→256)。那个"per-domain 输入编码器"就是**学出来的 retarget**,EgoWAM 验证够用;输出头共享。
- **DexWM**(2512.13644):Δ(手 kpt 差分,同域)+ HC 辅助头预测绝对位置。我们有 dhc(判负,但那是几何 Δ)。
- **X-Diffusion**(2511.04671):human 动作 = 加噪 robot 动作,按可迁移度加权。软学习桥。

## 3. 设计(两候选,推荐 A2)

### A1. 显式学习 retarget MLP(MT-π 式)
human eef3(+grip)→ 小 MLP → robot-like 动作 tokens → 喂现有共享 trunk。robot 走 identity(或同 MLP)。端到端 ② loss 训。
- 优点:显式、可解释、retarget 后的 human demo 可直接喂 BC(Do-As-I-Do)。
- 缺点:无配对监督 → 只能靠下游 loss 隐式对齐,可能欠约束。

### A2.(推荐)per-domain 薄输入 stem → 共享 trunk(EgoWAM 式)
`s.act` 换成 **两个薄输入编码器** `act_stem_r` / `act_stem_h`(各一层 Linear/小 MLP,dom 路由,复用 HEAD_MODE=two 的 dom 穿线机制!),都映射到共享动作嵌入维度 Dm,再进共享 trunk。human stem 学 human eef→共享嵌入的映射。
- 优点:**直接复用刚建的 dom 穿线**(`fwd_dual(dom)` 已有);薄 stem 不过拟合;EgoWAM 验证;输出/trunk 全共享保迁移。
- 与域头(two-head,输出侧)对称:这是**输入侧**对齐。可叠加。

### 保 grip
两候选都在 mp 基础上(接触点+grip 标量),grip 通道不动 → keyboard 可控保住。

## 4. 判决
- **稀缺 sweep(clean episode-split)**:learned-retarget 的 human-help(ro−rh)@ n100/n300 是否 > mp / dummy5rs?(那里 human 帮、表示有区别)
- **别期望救满量**(单场景数据天花板,已反复证);价值在稀缺/迁移。
- 附:Do-As-I-Do BC —— retarget 后 human demo 当伪 robot demo 喂 policy([[project_policy_in_wm_pipeline]])。

## 5. 落地(实现时,今晚不做)
- 复用 `HEAD_MODE=two` 的 dom 穿线;加 `ACTION_STEM=per_domain` env + `act_stem_r/h`。
- 薄 stem(1 层)先试;端到端 ② loss。
- 全走 clean episode-split(HELDOUT_VIDS)。
- TDD + smoke + 稀缺 sweep sbatch,对比 mp/dummy5rs。

## 6. 非目标
- 不救满量 human-help(数据天花板)。
- 不改切片(Phase 2)。
- 6DoF IK / 硬 retarget 到机械臂骨架(三方一致判死,见 AGENT_RETARGET_RELWORK)。
