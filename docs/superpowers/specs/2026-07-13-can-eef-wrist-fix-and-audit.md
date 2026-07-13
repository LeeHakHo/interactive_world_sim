# can 数据管线:human 真腕修复 + 全管线 bug 体检

> 背景:2026-07-13 定位 clips 打包层丢真腕(`gen_flow_render_dataset_caneef.py::human_loader` 从
> (p,R,w) 合成三点,slot0=指尖中点 0.125px)。上游 parquet 完好(HaMeR 真腕 `kpt2d_crop128_wrist` +
> `rotation_matrix_right` 朝向)。用户 2026-07-13 指示:修复,同时检查所有其他 bug。
> memory: [[project_human_eef_wrist_degenerate]](含根因)。

## 1. 修复设计(定案)

- **新版本数据集目录**(如 `outputs/flow_render_dataset_can_dual_v2/`),**旧 npz 一字节不动**
  (旧结果/旧 ckpt 全部引用旧版;正式化已交付结论不受影响)。
- 新版 human eef:**slot0 = 真腕**(HaMeR kpt0,双视角都要——cam_high 用 parquet 2D 列,cam_low 用
  3D 腕点(若有)经标定投影;若 parquet 无 3D 腕,用 depth-lift 或从 (p,R) 推,实现时核实列后定)。
  slot1/2 = 两指尖(保持)。另存 `rot`(手/夹爪朝向矩阵或 approach 单位向量)新字段,robot 侧同样补齐。
- robot 侧 slot 语义不变(已是真腕);两域 slot 语义在新版里第一次对齐。
- 验收:新版 slot0 距指尖中点 human 侧应显著 >0(接近 robot 侧的量级);overlay 眼检腕点落在解剖学
  手腕;新旧版指尖点应 bit 级一致。

## 2. 体检范围(audit,只读,先于修复跑)

三层各一个审计 agent,**要求用 python 对实际 npz/parquet 验证断言,不许只读代码下结论**:
1. **标注层**:`build_eef_dataset_can.py`(检测/depth-lift/平滑/坐标系/列语义,GP 平滑砍尾部的 raw 列),
   can 标定 json 的使用一致性。
2. **clips 层**:`gen_flow_render_dataset_caneef.py` + `gen_dualview_aligned.py`(crop/归一化/fidx 帧对齐/
   vis 语义/low_valid/grip=0.029 语义/L24 vs L48/track 种子/tracks3d valid/eef_low NaN 约定)。
3. **消费层不变量**:已知坑的系统化复查(NaN 填 0.0 vs 0.5 约定不一致——e2e 已咬过一次;human cam_high
   vis 是否全 1;双视角点不对应(已知,另 session 修);footprint 覆盖(已修 warp);load_dual pack 逻辑)。

产出:每层 findings 列表(bug/疑似/正常,附验证代码输出),汇总后按严重度决定哪些进修复 plan。

## 3. 非目标

- 不动旧数据集、不重训已交付的正式化实验;
- 下游 ② 动作表示实验(dummy5/skel 用真腕+approach)是修复后的**后续项目**,本期只交付干净数据+体检报告;
- v3 数据只做低成本 spot-check(同 builder 谱系),不全面重生成。
