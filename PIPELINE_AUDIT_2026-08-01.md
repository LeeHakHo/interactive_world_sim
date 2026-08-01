# 管线静默-bug 全面审计 + Go-Forward 正确性保证 (2026-08-01)

> 起因:发现 grip_ro/rh ③ 训在 `TLCAP=6` = 只前 ~21/48 帧(pick 阶段,place 被截)。用户令全面排查同类静默 bug。
> 三类边界并行审计(tL / L24-vs-L48 human / retrack / split)。**用户 scoping:只保证 go-forward 用的新 flow + L48 human 正确,不追 legacy。**

---

## 0. 一句话结论

- **tL6 是系统性的**:**整条 ③ 线(ro/rh/80k/scarce/warp/aux/flow-naive)全训在 tL6**,不是 grip 孤例 → 所有 ③ 指标只覆盖 pick 阶段(前 21 帧)。
- **另有三类静默不匹配**(L24 human 残留 / retrack 混用 / split 泄漏),但**大多在 legacy 脚本**;**go-forward 的干净路径存在且明确**。
- **修 = 用 L48+tL12+retrack+episode-split 重训 go-forward ③**(grip_ro_tL12/rh_tL12 已在跑),eval 显式传对口径。

---

## 1. 四类边界的审计结论

### ① tL 时序长度(最大发现,我亲查)
- **全部 ③ 模型训在 TLCAP=6**(ro/rh/ro_80k/rh_80k/scarce n100-300/warp/nowarp/aux 头/flow-vs-naive)→ tL6 = latent 0-5 ← RGB 0,4,8,12,16,20 = **前 ~21/48 帧**。
- eval(`eval_mh_render` 默认 6)一致 → deck 指标 pick-phase only(**覆盖缺,非推翻**;ro/rh 内部一致)。
- **渲染 tL12 在 tL6 模型上 = 外推**:`keyboard`(RENDER_H=48→tL12,用 ro_80k)、`render_compare_full`(12)、我的 `viz_skelflow_standard`(12)→ **后半是未训外推**;我 skelflow 的 cube 数把外推帧算进去了(confound)。

### ② L24-vs-L48 human(subagent)
- deck 的 ③ scarce/flow-naive(rh_n100 等,`sbatch_mh_train_L48`)用**正确 L48 human latent/cond**,只是 TLCAP=6 截了 → **不是 L24 confound**。
- **真·L24 残留在 legacy**:`sbatch_mh_train`/`he_train`/`rh_full_more*`、`human_e2e*`/`render_human_mh`、所有训练脚本**默认值**、aux/cond builder(`clips_human_L24`)、`exp_scel_dualview_wm` 的 **ACTION=skel/skelv3** 仍用 `skel_sidecar_human`(24,4)。**keyboard 不碰 human 数据,不受影响。**

### ③ retrack 混用(subagent)
- go-forward ③(grip)用 retrack cond(`cond_skel_grip_retrack`);② 用 `sbatch_dummy5_wm2_clean`=retrack。✓
- **陷阱**:`eval_mh_render` **默认 CONDP=非-retrack** `cond_skel_all`;deck 的 retrack eval sbatch(80k/scarce3/flowcond)都覆盖成 retrack(对),但基线 `sbatch_mh_eval_render`(评 `mh_full` ro_N2550/rh_N2550)**漏了 → 非-retrack 天花板评 retrack 模型**。`dummy5_e2e_render` 同图混 retrack/非-retrack(确定 bug)。

### ④ split 泄漏(subagent)
- go-forward(`grip3_mh_tL12`/`mh3_warp_retrack`/`dummy5_wm2_clean`/`submit_L48_chain`/`eval_mh_render`)= **干净 episode-split(100,102)**。✓
- **泄漏 live**:`sbatch_rh_full_more*`(07-28"full"跑)用 **clip-id split `HELDOUT=332,59,418,442`** → 训了 eval clip 的 ~147 个邻居 clip → **帧泄漏、held-out 偏乐观**(clip_heldout_leakage bug 仍在这些 config)。`wm2_generic/shard`(standalone 回落 okfirst)、`wm2_retarget_ab/retrack`(用 `HELDOUT_IDS` 非 `_VIDS` → okfirst)。

---

## 2. 影响映射(我们的结论哪些受影响)

| 结论 | split | 数据 | tL | 判断 |
|---|---|---|---|---|
| **② 全部**(mp/skelik/human-help/迁移;走 `dummy5_wm2_clean`) | 干净 episode | retrack+L48 | N/A | ✅ **不受影响** |
| **③ scarce human-help**(rh_n100/n300,headline) | 干净 episode | **L48**+retrack | **tL6** | 方向成立,**pick-phase only** |
| **③ flow-vs-naive / grip+warp** | 干净 episode | retrack | **tL6** | 同上,pick-phase |
| **③ full human-help**(ro_N2550/rh_N2550) | ⚠️**可能泄漏**(rh_full_more clip-split) | ⚠️非-retrack eval cond | tL6 | **最不可靠,三重 caveat,别信绝对值** |
| **我这季 skelflow cube**(mp vs skelik 穿③) | — | — | ⚠️tL12 外推 | cube 数含外推帧,confound,重跑 tL6/12 |

**净:没有结论被推翻;所有 ③ 结论 scoped 到 pick 阶段;full-③ 那一档最脏;② 全清白。**

---

## 3. ★ Go-Forward 正确性保证清单(用户只要这个)

**以后要用的每个组件,照这个跑就对:**

| 组件 | 正确做法 | 别做 |
|---|---|---|
| **② mp** | `sbatch_dummy5_wm2_clean`(retrack + L48 human + `HELDOUT_VIDS=100,102`) | 别 bare 跑 `exp_scel_dualview_wm`(默认非-retrack + okfirst) |
| **③ 渲染器** | `grip_ro_tL12`/`grip_rh_tL12`(重训中:L48+tL12+retrack cond+episode-split)✓已确认 55526 训在 tL=12 | 弃用旧 `grip_ro`/`ro_80k`(tL6) |
| **③ eval** | `eval_mh_render` **显式传 `TLCAP=12` + `CONDP=..._retrack`**(照 `sbatch_grip3_eval`/`scarce3_eval` 模式) | 别用裸默认(非-retrack+tL6);别用 `sbatch_mh_eval_render`(非-retrack) |
| **keyboard/replay** | 把 `CK_SKEL` 从 `ro_80k`(tL6)→ `grip_ro_tL12`;渲染 tL=RENDER_H//4=12 → 一致 | 别继续用 ro_80k(后半外推) |
| **IDM(新)** | `clips_robot_retrack` + L48 human trace + `HELDOUT_VIDS=100,102`(idm_data 已 hardcode) | — |
| **ACTION=skel/skelv3** | 若要用,先把 `skel_sidecar_human` 换成 L48 `_ik`(现只 skelik 用 _ik) | 别用现成 skel/skelv3(L24) |

**AVOID 名单(go-forward 别碰)**:`sbatch_rh_full_more*`(泄漏)、`sbatch_mh_train`/`he_train`(非-retrack+老默认)、`human_e2e*`/`render_human_mh`(L24)、裸 `train_*`/`exp_scel_dualview_wm`(默认全旧)。

---

## 4. 待办(修 go-forward)
- [进行中] `grip_ro_tL12`(55526)/`grip_rh_tL12`(55527)重训(L48+tL12+retrack+episode)。
- [ ] 训完 `TLCAP=12 CONDP=cond_skel_grip_retrack` 重 eval → **全 clip ③ 指标** + 复查 flow-vs-eef/human-help 在 place 阶段。
- [ ] keyboard/replay `CK_SKEL` 切 grip_ro_tL12,重跑演示(place 不再外推)。
- [ ] (可选)重跑我的 skelflow_standard 用 tL12 模型(去 confound)。
- [ ] IDM 继续(输入端已确认 retrack+L48+episode 干净)。
