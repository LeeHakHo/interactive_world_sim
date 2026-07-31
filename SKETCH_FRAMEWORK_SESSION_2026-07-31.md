# 交互草图框架 — 夜间自动会话报告(2026-07-31)

> 你睡前令"自动一下"。全程**只新建自己的文件 + 只读现成 sidecar**,没碰你别 session 在用的任何文件(`train_multihead_wm.py` 等一律没动)。GPU/批量走 sbatch。

## TL;DR

1. **定了整个框架 spec**:human/robot 都编码成同一种域无关的"**交互草图**"(看得见的 8 通道简笔画),再 decode 回像素。翻译/数据生成/policy 都是下游。护城河 = 领域都草图→policy,我们 **decode 回像素**。
2. **补了 world-model 闭环**(你问的 rollout gap):§6.5 = 草图是状态空间,agent=控制/object=响应;**我们只有 flow 预测器 ②**,attachment/contact 由 ContactDetector 从 rollout 状态**导出**(不单独预测)。完整 WM=②→③。
3. **Plan A(草图 builder)代码完成、review clean、11 单测过、双域三联眼检对**。全量草图已 build(修了一个 human grip 饱和 bug 后重 build)。
4. **Plan B(decoder+aux)只做了非侵入式准备**(物体 heatmap aux builder),**没自动 launch 训练**——见下"为什么没自动训"。
5. **一个悬而未决的核心问题**(你提的):flow-only 够不够,还是要超大状态预测器?我的结论=**先 flow-only + 导出,用 e2e rollout 实测在哪崩再升级**,不先验押超大预测器。判决标准写在 spec/report。

## 框架定案(spec: `docs/superpowers/specs/2026-07-31-interaction-sketch-crossembodiment-design.md`)

**草图 8 通道**(每视角 2D,pool16):`[flow(3) · agent-skel(1) · grip(1) · agent-trace(1) · attachment(1) · contact-splat(1)]`。
- agent-skel:human 经 **IK** 补成 robot 机械臂(域统一);robot 从 FK。都从现成 sidecar 读。
- **grip 两处**:烘在 skel 手指 + 单独标量通道(你定)。
- **contact**:attachment 标量 + 接触点 splat(你定的形式)。
- **warp 移出草图**,归 decoder(草图=纯抽象动态 / 锚=外观,你那个解耦)。

**decoder(Plan B)**:主 latent 头 **robot-only**(human 不进 recon MSE,避"输入robot输出human"矛盾)+ aux 头 **robot+human** 塑 trunk。★aux 目标 = **域共享的物体位置 heatmap**(不是 disjoint 的 DINO——你抓的:DINO 两域不相干 trunk 不被逼共享)。

## ★关键决策/偏离(醒来请过目)

1. **草图坐标系降回 2D per-view**(spec 原写 3D-canonical)。**原因**:`clips_human_L48` **没有 tracks3d 字段**(human 物体轨迹只有 2D)。所以 object-flow + contact 走两域都有的 **2D tracks**;agent-trace 走 eef3d(两域都有)投影;agent-skel 仍 3D 派生(IK)。2D object-flow 本就是我们验证过的同域接口(probe 0.645),不亏。**若你想要真 3D object-flow,需要先给 human 补 tracks3d(depth 抬升)。**
2. **human grip 饱和 bug(review 抓+已修)**:`skel_sidecar_human_ik` 的 grip 是旧 robot-range 映射,饱和 near-open(p50=0.04 死信号),human attachment 只 7.5% 活。修=build 时用**自身range重算**(mean 0.022≈robot 0.026),attachment 活。★但 sidecar 里 skel 手指画的开合仍是饱和的(cosmetic),要彻底修得重生成 human skel sidecar(带正确 grip,phantom+pinocchio)——留给你定。
3. **contact 不单独预测**:rollout 时由 ContactDetector 从(②预测物体 + 命令 agent)确定性导出。所以"只有 flow 预测器"够用(前提是准,待 e2e 测)。

## Plan A 交付(全 review clean)

| 文件(全新建) | 作用 |
|---|---|
| `sketch_lib.py` | 投影 + object-flow(2D) + trace + grip + contact 通道(纯函数,6 单测) |
| `contact_detector.py` | 几何式 attachment+contact(2D/3D,2 单测;修了速度门控 bug) |
| `build_sketch.py` | 组装 8ch 草图 robot/human(读现成 sidecar)+ VIZ 三联(2 单测) |
| `build_object_heatmap.py` | Plan B: 物体位置 heatmap aux target(2 单测) |
| `sbatch_build_sketch.sbatch` | 全量 build |

产物:`outputs/video_arch_wm/sketch_can_dual/sketch_{robot,human}.npz` (N,2,8,6,16,16)。
**Drive**:`iws_evals/2026-07-31_交互草图builder/01_8通道草图三联_robot+human/`(眼检:两域 8 通道全对,human skel 是 IK 机械臂)。

## 为什么没自动 launch Plan B 训练

Plan B 的 decoder 训练要:(a) cond = 8ch 草图 **+ warp(3)**——warp 要从 z0+flow 现算并 concat 成 11ch;(b) 改 `train_multihead_wm::losses`(主头 robot-only)——但那是**你别 session 在用的文件,不能碰**,得 copy 一份;(c) 物体 heatmap aux 接进去;(d) latent 与草图对齐核验。这几处**集成一旦错,白烧几小时 GPU**。无人值守下我判断**不该赌**,所以只把 heatmap aux 写好+测好,其余等你早上过一眼再 launch。

## 悬而未决:flow 够不够?(你提的)

- **完整性**:flow-only + 导出 contact **足以产出完整草图**(agent=给定/flow=②预测/contact=导出),不缺件。
- **准确性**:只能实测。判决标准(任一崩就升级):① 抓/放时机错位;② 长 horizon 漂移;③ contact 导出与②预测打架。
- **测法**:Plan B 训完 decoder → e2e rollout(② + 导出 contact → ③)vs GT,量物体轨迹准度+抓取时机+眼检长 rollout。**够就定,不够按上面三条对症升级(最小动作:给②加 grasp/contact 头 或 flow 表示做厚),不必先上超大预测器。**

## 早上建议顺序
1. 看 Drive 草图三联(眼检 8 通道)。
2. 定 human skel sidecar 要不要带正确 grip 重生成(cosmetic,不阻塞)。
3. 定 Plan B 的 warp/cond 集成方式 → 我 copy trainer + 接 heatmap aux + launch 训练。
4. 训完做"flow 够不够"e2e rollout 判决。

memory:`project_interaction_sketch_framework`(总)、`project_human2robot_translate_ro3`(翻译+穿模+状态锚)、`project_ik_eef2skel_human_agent`(IK)。ledger:`.superpowers/sdd/progress.md`。
