# Session 维护文档（2026-07-12 夜 ~ 2026-07-14）：LaST-HD 机制移植 + 数据汇率 + 渲染治糊

> **性质**：本 session 的完整工作记录，既给用户读，也给下一个 session 接手。包含每一次讨论、
> 每个尝试（含放弃的候选）、参考论文、实验结果、事故、开放问题、接手指南。
> **配套精简版**：`SESSION_SUMMARY_2026-07-14.md`（无黑话版总结）。主研究日志：`COMBINE_ACTION_LOG.md`。
> **git**：分支 phantom_dynamo，本 session 提交 dd5bdda → 9a93801 及之后（每个提交信息可 grep）。
> **并行 session**：另一个 session 同时在做渲染器正式化（`exp_scel_dualview_dit_formal.py`，
> 三种条件 × 跨视角开关 × 多种子消融）；分工=本 session 做②机制/端到端/数据价值/渲染质量实验，
> 他们做③正式化。共享交流通过 memory 文件。

---

## 0. 快速接手三步（下个 session 先读这里）

1. 读本文件 §1（时间线）+ §8（开放问题/待办）；
2. 环境：conda `iws`（`/scr/yusenluo/anaconda3/envs/iws/bin/python`）；GPU 任务一律 sbatch
   （模板在 `sbatch/dualview_gmask*.sbatch` 等）；跑前 `squeue`/`nvidia-smi` 看空闲；
   HF 离线（`HF_HUB_OFFLINE=1`，VAE 已缓存）；
3. 关键 ckpt：渲染器定版 `outputs/cross_embodiment_wm/dualview_gmask/g1low_100ep/dvdit_g.pt`
   （加载需 `GMASK=1 GMASK_LOW=1` 环境 + `DualViewDiTG` 类在 `__main__`，参照
   `exp_scel_dualview_gmaskcond.py` 的 RENDER_CKPT 模式）；②配对 ckpt
   `outputs/cross_embodiment_wm/dualview_comb/extra400_s2/wm_{align_wm,dummy5}_rh_N400.pt`。
   ⚠️ 本仓库大量代码未提交是常态，只 add 自己的文件；commit 不加 Co-Authored-By。

---

## 1. 时间线（用户每次说了什么 → 我们做了什么）

| 时刻 | 用户指令/讨论 | 行动 |
|---|---|---|
| 07-12 晚 | 读 spec（先误指 dualview formalize，更正为 combine-skel-world-action）+ 重读论文 + 读 memory | 精读 LaST-HD（arXiv 2606.23685，下载到仓库根）；发现 spec 里的 align 机制是论文的走样简化 |
| 07-12 晚 | 选定方案："加 align_wm 后一起 sweep" | 设计并 TDD 实现 align_wm（§3）；spec 更新 |
| 07-12 深夜（中途插话） | "A1 你不试试吗；目标=human-helps+精度双好；解决'只是把 action 条件换成预测运动条件'太简单的质疑；基础架构=双视角联合预测+DiT；我睡了你晚上做" | v3 全量 sweep + 移植到双视角（DualCombLWC）+ 端到端管线（§3-4） |
| 07-12 深夜（插话） | "GPU 有空就用 sbatch" | 杀本地任务，全部改 sbatch（seed 切分 array job） |
| 07-13 凌晨 | （自主）磁盘满事故、结果合并、显著性检验、补种子 | §7 事故；5 种子配对检验判"突破不显著" |
| 07-13 | "重新讲讲结论，什么没证成" ×2 轮追问 | 澄清"不付精度代价≠突破"（基线本来就不付代价；配对差 0.14px p=0.61） |
| 07-13 | "能不能证明 human 数据和 robot 数据接近等效（rh 数据量大于 ro 不公平）" | 数据汇率实验（§5） |
| 07-13 | "怎么看 OSCAR 等大规模 human 预训练" | 讨论定位：他们买覆盖、我们量密化汇率，同一条曲线两段（§5.3） |
| 07-13 | "我看着 gif 都很糊，渲染不行啊"；"can 之前就有挺好的渲染" | 诊断=agent 无条件化；发现老 detmem 好在有剪影条件（§6） |
| 07-13 | "确定（做剪影消融），同时查 human clip 腕点 bug" | 剪影消融大胜；腕点 bug 定位+新发现回填文件 26% 出画（§7.2） |
| 07-13 | "你自己跑我睡了，自我迭代吧" | 自主链：60ep 消融 → 100ep → 锐③端到端（§6.2-6.3） |
| 07-14 | "cam_high 还行但 low 不行；架构还有进化空间吗；DiT 容量够不够；其他选择" | 发现现成未接的低位剪影生成器；容量消融判死；warp-as-condition（§6.4-6.6） |
| 07-14 | "还是有些糊；gif 预测步长太短；物体区域捕捉得准吗" | VAE 天花板量化；metric 审计+修正+全量重评；44 步长程版（§6.7） |
| 07-14 | "维护详细 session 文档" | 本文件 |

---

## 2. 参考论文与每篇的具体用法

1. **LaST-HD**（arXiv 2606.23685，北大+CUHK+Simplexity，2026-06，PDF 在仓库根）——本 session 主论文。
   - 用法一（机制移植）：它的"独立冻结的、以动作为条件的世界模型产潜空间监督目标 + 余弦损失"
     → 我们的 align_wm 机制。三个从其消融（图3b：73>66 SigLIP>63 无动作条件>60 无监督）抠出的
     设计要点：目标须**独立冻结**（非同网络移动靶）、须**被真实未来验证过**（grounded）、
     **余弦 + 双域全样本监督**。
   - 用法二（数据经济学）：其"人手数据采集比机器人遥操作快 4-5 倍"用于把我们的汇率 α≈0.3 换算成
     单位采集时间价值（打平~1.5 倍）。
   - 用法三（对照表）：其 Mix-HD（50 机器人+50 手套 vs 100 机器人：68% vs 73%）本身就是一次
     汇率<1 的测量，与我们一致；其泛化实验（unseen 场景 human 数据带来最大增益）印证"human
     数据帮在机器人覆盖薄的地方"。
2. **OSCAR**（arXiv 2606.04463）——skel（骨架）动作特征的来源（腕速度+骨向量）；大规模混训+warm-start
   的代表作，在"大规模 human 预训练怎么看"讨论中作对照。
3. **DexWM**（见 `FLOW_REPR_SURVEY_2026-07-04.md`）——dummy5 虚拟夹爪五点星座的来源（双视角②的精度锚）。
4. **EgoScale / 其他大规模 human 数据工作**——讨论中作"覆盖派"代表（§5.3 定位论）。
5. 项目既有调研（未新读但引用）：`FLOW_REPR_SURVEY_2026-07-04.md`（11 篇 PDF 对照）、
   `CROSS_EMBODIMENT_WM_ATTEMPTS.md`（历史全记录）。

---

## 3. 实验线一：动作特征结合机制（align_wm 家族）

### 3.1 问题与动机
velocity_action 线已证帕累托两难：绝对编码（world/dummy5）精度最好但 human 帮不上；同域编码
（skel）human 帮助大但精度差。目标：一个机制同时拿两头（spec 判据1：最终误差 rh ≤ 绝对基线
且帮助 Δ 明显更大）。

### 3.2 机制族（全部在 `exp_scel_combine_action.py`(v3) / `exp_scel_dualview_comb.py`(双视角)）
| 机制 | 定义 | 来源/动机 |
|---|---|---|
| world / dummy5 | 绝对位置编码（锚，精度上界） | 既有 |
| skel | OSCAR 骨架编码（锚，迁移上界） | OSCAR |
| a1 | skel 主体 + 机器人专属绝对残差（human 样本残差置零 + L2 正则防独吞 + 0.1 小初始化） | spec 主攻 |
| align | 同网络双通路自蒸馏（L2 拉向 skel 通路的 detach，目标随训练移动）| spec 原案 = LaST-HD 的走样简化，**故意保留作对照** |
| **align_wm** | 冻结的混训 skel 世界模型作教师（复用 sweep 中本来要训的 skel 锚，同数据同种子），学生纯绝对编码主通路 + 余弦损失拉内部特征向教师 | **LaST-HD 忠实版（本 session 新增）** |
| warm | 两阶段：先 skel 混训整网 → 冻结主干只训绝对头（机器人数据） | spec |
| （未做候选）aux 多任务辅助头 / adversarial 梯度反转 / target-side Δ | — | spec 备选，未纳入；adversarial 高风险 |

关键实现决定：align_wm 的教师按臂分开（ro 臂教师只见机器人数据、rh 臂教师见混合数据）→ 保证
Δ 的度量不被教师泄漏污染。评价路径不传 is_h（缺省全机器人、残差全开）→ 评价代码零改动。

### 3.3 结果（v3：6 方法×4 数据量×最终 5 种子；双视角：4 方法×3 档×5 种子；详表在 COMBINE_ACTION_LOG.md §4.2）
- **机制排序（显著）**：align_wm 全面压制 a1/warm/align（v3 N=100 最终误差 8.79 vs 9.63/10.97/11.48，
  差距远超种子波动 ±0.5）；align 自蒸馏的 Δ 为**负**（−1.2）——"目标必须冻结且 grounded"的直接消融证据。
- **突破判定（不显著）**：align_wm vs 绝对锚，v3 N=100 配对差 −0.145±0.52（p=0.61），双视角
  N=400 配对差 −0.136（5 种子 4 负，p=0.42）。3 种子时的"突破"假象被 5 种子推翻。
- **附带发现**：can_dual 上绝对锚自带大 Δ（+4.7~4.9 vs v3 的 +0.67）——帕累托张力强弱是数据集性质。
- 教训（写给下个 session）：Δ 大可能来自更差的 ro 起点（align_wm ro 10.31 vs world 9.86），
  **只看 rh**；先量方差再下结论（spec 的 variance-first 规则救了我们一次）。

## 4. 实验线二：端到端管线（②预测运动 → ③渲染）

- 脚本 `exp_scel_dualview_e2e.py`（四列：真实 | 真实运动→③(天花板) | ②预测→③ | 朴素动作③）；
  后并入 `exp_scel_dualview_gmaskcond.py` 的 e2e_g 模式（锐③版）。
- 无泄漏配对 ckpt 结果（obj-LPIPS，高/低位）：天花板 0.299/0.301，②align_wm 0.309/0.314，
  ②dummy5 0.305/0.310，朴素 0.382/0.372。**②的 2.6px 预测误差几乎不损耗接口优势**。
- 锐③重跑（渲染修复后）：天花板 full-LPIPS 0.131，②列 0.137-0.138——贴住。
- 长程版（44 步，L48 完整片段，job 53317）：②只训过 16 步滚动，44 步是诚实压力测试——结果待收。
- ⚠️ 泄漏教训：最早的管线验证用了 L48 全量训练的 wm_dual.pt（对 L24 heldout 有时间窗重叠），
  数字偏乐观（ADE 2.1 vs 干净的 2.6/4.7）；正式数字一律用配对 L24-split ckpt。

## 5. 实验线三：human 数据 vs robot 数据的汇率

### 5.1 设计（用户质疑：rh 比 ro 多 1800 条数据，不公平）
补机器人独训曲线 ro(N) 至全量（v3 N∈{20..2550}，dual N∈{50..2460}，3-5 种子），对每个 rh(N)
找等效点 ro(N+M)=rh(N)，汇率 α=M/1800。脚本复用 sweep（ANCHORS/COMBINES 环境变量支持空集）。

### 5.2 结果（图 `outputs/cross_embodiment_wm/data_equiv/exchange_{v3,dual}.png`，两数据集一致）
- α ≈ 0.2~0.5，峰值在中等稀缺区（v3 N=400 处 0.35-0.47）：**约 3-5 条人手 ≈ 1 条机器人**。
- N≥1600 后 α→0；全量时 Δ≈0（点估计略负但 p=0.33/0.38 **不显著，禁说"有害"**——本 session
  曾口误说"转负"，已在 log/memory 更正）。
- 换算采集成本（LaST-HD 的 4-5 倍采集速度）：单位时间价值打平~1.5 倍。
- 免费单点证据：v3 上 rh(100)=9.07 好于 ro(400)=9.63（"100 机器人+1800 人手 胜 400 机器人"）。

### 5.3 与大规模 human 预训练论文的定位（用户问"怎么看 OSCAR 们"）
他们买"覆盖"（human 数据到达机器人数据到不了的场景/物体/任务；机器人在开放世界永远在曲线左端），
我们量"密化"（同分布内替代）的汇率与衰减——同一条曲线的两段，不矛盾；我们的曲线是他们没给出的
量化基础设施。未测部分（候选实验）："填洞 α ≫ 密化 α"验证：人为留出物体位置区间做覆盖空洞，
看 human 数据填洞的汇率是否远高于 0.3。

## 6. 实验线四：渲染治糊（用户投诉驱动，收益最大的一条线）

### 6.1 诊断
用户记得"can 之前渲染挺好"——查证：好的是老单视角 detmem 渲染器（full-LPIPS 0.147），其配方
含 **gmask 机械臂剪影条件**；新双视角 DiT 渲染器没继承 → 手臂靠猜 → 均值化黑影 = 糊的主因。

### 6.2 修复链（`exp_scel_dualview_gmaskcond.py`，每步单变量受控，60ep 对齐；全部 sbatch）
| 步骤 | 高位 obj/full-LPIPS | 低位 obj/full-LPIPS | 判决 |
|---|---|---|---|
| 原状 | 0.281/0.170 | 0.299/0.187 | 基线 |
| +高位剪影（maskgen_caneef） | 0.198/0.134 | 0.254/0.165 | **大胜 +2.8dB**；低位也涨=跨视角注意力自发传输 agent 信息（新发现，可写论文） |
| +低位剪影（maskgen_caneef_low，**发现它早已训好 IoU0.836 但从未被接**） | 0.192/0.132 | 0.233/0.154 | 低位 +1.2dB |
| +容量 2.5×（384×8→512×12） | 0.186/0.129 | 0.236/0.155 | **容量判死不是瓶颈**（答用户"DiT 容量够不够"） |
| +warp 搬运预览（刚体 Umeyama 把首帧罐子像素搬到预测位置作 RGB 草稿通道） | 0.184/0.130 | 0.222/0.151 | 温和有效；眼检罐子"流挂纹"变实体；印证"warp 战场在纹理物体"预测 |
| 100ep 定版（双剪影） | **0.170/0.122** | **0.212/0.146** | 超老 detmem（0.147）且双视角；低位 +1.65dB |
| 100ep 定版（双剪影+warp）job 53314 | 待收 | 待收 | |

### 6.3 质量的物理边界（答"还是有些糊"）
VAE 编码-解码往返 full-LPIPS 下限 = **0.072/0.078**（两视角实测）。我们 0.122/0.146 中约一半是
16× 压缩 VAE 在 128 分辨率的固有软化。**再上台阶需换 VAE（f8→f4）或 256 分辨率**——训练成本
数量级决策，待用户拍板。渲染器自身余量还剩 +0.05/+0.07（长程累积等）。

### 6.4 metric 审计（答"物体区域捕捉得准吗"）
发现真缺陷：footprint 凸包没滤不可见点（nan→图像中心会拽偏裁剪框，低位更重）。修正
（`fp_vis`：可见点过滤+检出率）后**全量重评 6 个配置：数字与原值几乎逐位一致、排序不变、
检出率 100%**——缺陷真实但数值影响可忽略，结论经受住审计。修正版已替换所有后续评测。

### 6.5 治糊候选清单（试过的与没试的）
| 候选 | 状态 | 结果/理由 |
|---|---|---|
| agent 剪影条件（双视角） | ✅ 采纳 | 主因，见上表 |
| 容量扩大 | ✅ 试过判死 | 2.5× 参数≈白给 |
| warp-as-condition | ✅ 采纳（温和） | 纹理物体有效 |
| 更长训练 100ep | ✅ 采纳 | 稳定小增益 |
| 换 f4 VAE / 256 分辨率 | ⏸ 待用户 | 天花板 0.072→更低，成本大 |
| 扩散采样替代确定性 | ✖ 不做 | detmem 线判过确定性赢；eval 方差大 |
| GAN 对抗损失 | ⏸ 最后手段 | 训练不稳 |
| 低位视角重新裁剪 | ✖ 证据不足 | 物体点已占画面 85%，"浪费像素"假设不成立 |
| 更多 prev 记忆帧 | ⏸ 候选 | 未试 |
| 低位 tracks 质量修复（投影播种噪声） | 移交 | 属另一 session 的数据管线 |

## 7. 事故与 bug（含教训）

1. **磁盘写满**（07-13 凌晨）：/scr2 100% → 一个 sweep 任务在 torch.save 时崩。处置：从 slurm log
   抢救数值（只真丢 1 格）、清 uv 包缓存 14G 应急（HF 缓存 84G 绝不能动——离线加载依赖它）、
   torch.save 全部包 try/except。教训：**长跑任务的保存必须容错；大实验前查 df**。
2. **人手腕点 bug**（用户+另一 session 发现，本 session 独立核验+补充）：clips 打包脚本
   `gen_flow_render_dataset_caneef.py::human_loader` 不读 parquet 真腕列，slot0 按构造=两指尖中点
   （距中点 0.13px；机器人是真腕 24.9px）。**本 session 新发现**：修复用的回填文件
   `human_wrist2d.npy` 有 26% 帧腕在画面外（手从画缘伸入）、7.8% 整段出画 → 混杂另一 session
   "真腕不帮"的负结论，复核需滤出画帧。对本 session 结论无影响（各编码共用同一数据，相对比较成立）；
   修好后 align_wm 教师质量或提升，值得重跑关键格。眼检图
   `outputs/cross_embodiment_wm/data_equiv/wrist_check.png`。
3. **组件版本泄漏教训**：早期管线验证误用 L48 全量 ② 对 L24 heldout（时间窗重叠）→ 数字偏乐观；
   正式结果全部换配对 ckpt 并声明版本。
4. **口误更正记录**："全量时 human 有害" → 实为 Δ≈0 不显著（p=0.33/0.38，种子方向不一致）。

## 7.5 追加（2026-07-14 深夜，EgoWAM 精读后的通宵迭代）

1. **EgoWAM（arXiv 2607.08436，Danfei Xu 组）精读**：世界表示受控对比（像素<DINO<3D flow），
   动作层共训在行为不对齐时塌穿基线而 flow 世界头稳。四层意义：主线三方确认 / 解释 formalize
   线的 eefsp 反超（对齐度高时强动作基线域内能打）/ 给出 stems 机制补做 / 差异化定位不冲突。
   用户钦定五篇核心论文：DexWM、OSCAR、IWS、LaST-HD、EgoWAM。笔记 memory `reference_egowam`。
2. **stems 机制判决**（每具身独立动作编码器+零显式对齐）：稀缺区（v3 N=100）不敌显式冻结教师
   （rh 9.53 vs align_wm 8.93，配对 +0.60，4/5 seed 同向 p=0.105），甚至略差于单共享编码器；
   充足区（dual N=400）三者打平。机制排序定稿见 COMBINE_ACTION_LOG §4.6。
   工程事故两起（空 regs stack / _save 无锚 KeyError）：训练值全部从 log 抢救、零损失，修复+补测试。
3. **44 步长程端到端**（回应"预测步长太短"）：天花板 full-LPIPS 0.126/0.134（与 20 步几乎同），
   ②预测列 0.132/0.145 贴住（②只训过 16 步滚动）；t=47 眼检夹爪罐子依然锐利。gif 已传
   `iws_evals/2026-07-14_gmask_agent_cond/长程44步端到端4列gif/`。
4. metric 审计（用户质疑"物体区域准吗"）：发现凸包未滤不可见点缺陷（nan→图像中心拽偏框），
   修正（fp_vis）后全量重评——数字几乎不变、排序不变、检出率 100%，结论经受审计。
5. VAE 天花板：编码-解码往返 full-LPIPS 0.072/0.078 = 渲染物理下限；当前 0.122/0.146 里约一半
   是 VAE 固有软化。再上台阶需换 f4 VAE 或 256 分辨率（待用户决策）。
6. 观察：另一 session 已把 warp 通道采纳进正式线（dvf_flowwarp job 在跑）。

## 8. 开放问题与待办（下个 session 从这里接）

1. （跑着）双剪影+warp 100ep 定版（job 53314）+ 44 步长程端到端（job 53317）——Monitor 自动收，
   若 session 结束未收：看 `outputs/cross_embodiment_wm/dualview_gmask/{final_warp_100ep,e2e_long44}/summary.txt`。
2. **⚠️ 需两 session 合看**：另一 session 的正式化表里，升级版朴素基线 eefsp 在高位相机上反超了
   flow 条件（0.2284 vs 0.2556，低位 flow 仍赢）——对"运动接口>朴素动作"主张有影响，需合并分析
   （他们的 `outputs/cross_embodiment_wm/dualview_dit_formal/ablation_table.md`）。
3. 腕点数据重生成（另一 session 管线）后：重跑 align_wm 关键格（教师质量或提升）+ 复核 dummy5w 负结果。
4. formalize 线是否采纳：gmask 双剪影通道 + warp 通道 + 修正版 fp_vis metric（建议全部采纳）。
5. 换 f4 VAE / 256 分辨率的质量跃迁（成本大，待用户）。
6. 数据汇率的"填洞实验"（覆盖空洞处 α 是否 ≫ 密化 α）——把大规模预训练的直觉变成可测定律的机会。
7. human 侧 agent 剪影（SAM2 手部 mask 作 human 的 gmask 通道）——现在 human 样本该通道置零。

## 9. 产物索引

- **Drive**（`iws_evals/`）：`2026-07-13_combine_alignwm_e2e/`（端到端五列/四列 gif、帕累托图、
  汇率曲线、眼检图）；`2026-07-13_gmask_agent_cond/`（治糊三列消融 gif、锐③端到端四列、100ep 版；
  长程 44 步版待 53317 完成后追加）。
- **本地结果树**：`outputs/cross_embodiment_wm/{combine_action,dualview_comb,dualview_e2e,dualview_gmask,data_equiv}/`
  每目录有 summary.txt / metrics.json / render_cache / gifs。
- **脚本**（全部有单元测试 tests/test_*）：`exp_scel_combine_action.py`（v3 机制）、
  `exp_scel_dualview_comb.py`（双视角机制）、`exp_scel_dualview_e2e.py`（端到端）、
  `exp_scel_dualview_gmaskcond.py`（渲染消融+锐③端到端+长程，env：GMASK/GMASK_LOW/WARP/DIM_G/
  DEPTH_G/HORIZON/RENDER_CKPT/E2E_CKPT/E2E_DS/WM2A/WM2B/COMPARE）。
- **文档**：`COMBINE_ACTION_LOG.md`（主研究日志）、`SESSION_SUMMARY_2026-07-14.md`（无黑话版）、
  本文件（维护版）。spec：`docs/superpowers/specs/2026-07-12-combine-skel-world-action-design.md`。
