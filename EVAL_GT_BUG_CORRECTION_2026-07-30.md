# ⚠️ 另 session 的 mp/ro_80k eval 用错 GT/flow — 修正 note (2026-07-30)

> 给另一个 session + 接手看。另 session 对"新模块(mp ② + ro_80k ③)退化"的结论**基于两个用错 GT/flow 的 eval**,独立复现后翻转。

## Bug: 用了老 GT/flow(应用 retrack)
另 session 的 replay/GT-flow eval 拿 **老 track(clips_robot 非 retrack,或更老)** 当 GT/flow,而 mp_rh_all/ro_80k 都训在 **retrack** 上 → GT 不匹配 → 误差/LPIPS 虚高。

## 独立复现对照(我用正确 retrack GT/flow)
| metric | 另 session(错:老GT/flow) | 我独立(对:retrack) | 脚本 |
|---|---|---|---|
| mp replay ②ADE cam_high / cam_low | 4.32 / 4.08 | **2.01 / 1.95** | CPU rollout_dual, GT=clips_robot_retrack |
| ro_80k GT-flow render-LPIPS | 0.0723 | **0.0620** | eval_mh_render, CONDP=cond_skel_all_retrack |
| ro_80k cube_px | 10.40 | **3.54** | 同上 |
| ro_80k 罐子消失 vanish | 11% | **0%** | 同上 |

结果 `outputs/video_arch_wm/mh_eval_ro80k_gtflow_indep/render_summary.txt`。

## 定论翻转
1. **ro_80k 是好渲染器,不是弱的**:0.0620/cube3.54/det1.00,甚至优于 ro_40k(0.0638/3.69)——warm-start resume 到 80k 确实提升(recon 0.056<0.076)。
2. **mp replay flow 是紧的**:2.0px,不是 4.3;甚至优于权威 heldout 均值 2.35(这 3 个 demo seq 本身不算难)。
3. **"新模块 mp+ro_80k 退化"结论站不住** —— 基于错误 eval。gif 看着"软/糊"很可能是 **gif 生成时也喂了老 flow cond**(需查 keyboard/replay 的 cond 源是不是 retrack)。

## 仍待独立验证(别再信未复核的数)
- **mp keyboard 合成控制 fb(报 mp 0.228 vs dummy5 0.091)**：这个不依赖 GT,但鉴于两个 eval 都错,独立复核后才算数。
- keyboard/replay gif 生成用的 **flow cond 是 retrack 还是老 track**?若老 → 渲染歪是 cond bug 非模型弱。

## 教训(同 [[feedback_cite_source_check_sibling_runs]])
报 metric 前确认 **GT/flow 源和模型训练数据一致(retrack)**;跨 session 的数独立复核再下定论。
