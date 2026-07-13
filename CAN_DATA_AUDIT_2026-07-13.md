# can 数据管线体检报告(2026-07-13)

> 背景:发现 clips 层 human eef 丢真腕后,用户指示"修复,同时检查所有其他 bug"。
> 三个只读审计(标注层/clips 层/消费层)全部要求 python 实测数据验证断言。
> 详细证据:`.superpowers/sdd/audit-{annotation,clips,consumer}-layer.md`。
> spec:`docs/superpowers/specs/2026-07-13-can-eef-wrist-fix-and-audit.md`。

## 结论一览

| 层 | 判定 | 说明 |
|---|---|---|
| 标注层(build_eef_dataset_can) | **CLEAN** | 12 列语义/深度 lift z 范围/标定往返 6.7px/坐标系/GP 平滑+raw 列全过;检出率 95-98.5% |
| clips 层(gen_flow_render_dataset_caneef + gen_dualview_aligned) | **CLEAN** | fidx 帧对齐 bit 级精确;crop/归一化 overlay 全中;vis 真实(0.95-0.97);low_valid 一致过滤;grip 三峰合理 |
| 消费层(4 个实验脚本) | **1 个 CRITICAL** + 皆清 | 见下 |

## CRITICAL:e2e 数据泄漏(已修复,重测中)

- **问题**:②(exp_scel_dualview_wm)与 ③(formal)的 heldout 切分代码不等价(② 先 permute 再按 ok 过滤,③ 先过滤再 permute)→ ③ 的 24 条 e2e eval seq 有 **22 条落在 ② 的训练集里**。
- **影响范围**:只有 `DUALVIEW_DIT_REPORT.md` §3 的 e2e 列(② ADE 2.07/2.33px、e2e obj-LPIPS 0.270/0.291)偏乐观;消融表/门控/human-helps/crossview/IWS external 全部用 ③ 自己的干净 split,**不受影响**(审计逐一验证)。
- **修复**(commit 23d13de,review clean):`SPLIT=okfirst` 选项(默认 legacy 不变,另一 session 依赖旧 ckpt)+ `WM_PT` env;干净 split 的 ② 重训中(job 53259 → `outputs/cross_embodiment_wm/dualview_wm_cleansplit/`),训完 rerun e2e、report §3 以干净数字替换并标注勘误。
- **提醒另一 session**:其 e2e 管线若用同一 `wm_dual.pt` + 类似 eval 集,可能同样中招。

## 已修复:human 真腕缺失(sidecar,commit 01a3565,review clean)

- 根因:clips 打包层从 (中心,朝向,开度) 合成三点,slot0=指尖中点;上游 parquet 真腕+朝向齐全。
- 修复:**side-car**(旧 npz md5 不变)`outputs/flow_render_dataset_can_dual/wrist_sidecar_{human,robot}.npz`:
  `wrist2d_high`(HaMeR 真腕,不裁剪允许出画)、`wrist_valid`(74.03% 帧有效——26% 出画与另一 session 统计精确吻合)、`rot3d`(手/夹爪世界系朝向,正交性 6e-8;cam_low 的 approach 由 rot3d 投影合成)。
- 与另一 session 的 `human_wrist2d.npy` **bit 级一致**(独立推导互为交叉验证)。
- 消费须知:用 `wrist2d_high` 必须以 `wrist_valid` 过滤(出画帧是 out-of-support 输入;另一 session 的"真腕不帮"负结果即被此混杂,复核须过滤)。

## 脆弱点(无实测影响,列观察项)

1. **NaN 填充约定不一致**(0.0 via load_dual vs 0.5 via ② 系):当前 NaN 只出现在 `low_valid=False` 行且所有消费者都过滤 → 已证无影响;但同类不匹配已咬过一次(e2e 修复 523b7ae),新代码建议"保留 NaN + 显式 mask"而非各处自选填充值。
2. **flow_cond 的 vis 参数是死代码**(vis 只进被丢弃的通道,实测 real/zeros/ones bit 级同输出)→ 遮挡信息从未进 ③,"用最后观测 vis 保部署性"的注释叙述作废(不影响任何数字,三臂同待遇)。
3. 标注层 informational:raw 速度尖峰(~5-6 m/s 单帧)、~3s 未检出段线性插值后才 GP;wrist 出画 26%。
4. clips 层 gather 的 fidx 越界防御分支从未被真实数据触发(纯防御,未验证)。

## 已知且归属另一 session 的问题(本体检不重复处理)

- 双视角 48 track 点各自独立采样(非同一物理点)→ 三角化 3D 是假象(其 QC 已定位,修法=cam_high 点投影 cam_low 做种子)。
