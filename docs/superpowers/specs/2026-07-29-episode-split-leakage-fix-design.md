# heldout split: clip-idx → IWS 式 episode(vid)分组(修帧泄漏)设计

日期: 2026-07-29
分支: phantom_dynamo
文件: `exp_scel_dualview_wm.py`(②)、`train_multihead_wm.py`(③)

## 0. 一句话
把 ②/③ 的 heldout 划分从**按 clip 下标**改成**按 `vid`(episode)整段留出**(照 IWS `play_eef_dataset.py:378` house standard),消除"重叠滑窗 + clip-idx split → 85% heldout 帧与训练重复"的泄漏,然后**决定性臂干净重训重评**拿诚实数。

## 1. 背景 / 泄漏根因(见 [[project_clip_heldout_leakage]])
- 切片 = 重叠滑窗:`gen_flow_render_dataset_caneef.py` CLIP_STRIDE=12 帧,窗口跨 L×S=192(robot)/96(human)帧 → 相邻 clip 共享逐帧相同源帧。
- split `split_okfirst`(`exp_scel_dualview_wm.py:528`)= `rng(0).permutation(np.where(ok)[0])[:150]`,**按 clip 下标、无 vid 分组** → 18/18 episode 横跨 train/heldout,**85% heldout clip 与训练 clip 共享 ≥50% 相同帧**。
- ③ multihead 用强制 heldout `{332,59,418,442}`(clip 下标)→ 同泄漏。
- IWS 原生 `play_eef_dataset.py:378` 已按 episode 分(干净)——只我们 bolt-on 的 ②/③ 偏离了。
- ★**切片(overlap)本身不改**:一旦 episode-split,训练内部 clip 重叠不算泄漏。slicing best-practice(随机起点/contact 加权/重 track)= Phase 2 独立立项。

## 2. 数据事实(已核实)
- robot: 18 episode `vid 100–117`,各 150 clip(valid 134–149)。
- human: 12 episode `vid 0–11`。
- 强制 demo seq 归属:**clip59→vid100;clip332/418/442→vid102** → 留出 vid100+vid102 覆盖全部 demo,keyboard 对齐自动保住。

## 3. 设计

### 3.1 新划分函数(替换 split_okfirst 的用途)
```python
def split_by_episode(ok, vid, heldout_vids):
    """IWS 式 episode 分组: heldout_vids 里的整段 episode 作 heldout,其余作训练池;
    都 ok-过滤。heldout episode 与训练 episode vid 完全不相交 → 无跨-episode 帧泄漏。"""
    hv = set(int(v) for v in heldout_vids)
    ho   = np.array([i for i in np.where(ok)[0] if int(vid[i]) in hv])
    pool = np.array([i for i in np.where(ok)[0] if int(vid[i]) not in hv])
    return ho, pool
```

### 3.2 env / main() 接线(②)
- 新 env `HELDOUT_VIDS`(默认 `"100,102"`);保留 `SPLIT=okfirst` 语义但走 vid 分组。
- `main()`:读 `vid = z["vid"]`;`ho, pool = split_by_episode(ok, vid, HELDOUT_VIDS)`。
- **废弃 `HELDOUT_IDS` 的 clip-idx 强制留出**(demo seq 已随 vid102/100 进 heldout,不需再按 clip 强制);若 `HELDOUT_IDS` 仍设则忽略并 warn(向后兼容不炸)。
- 逗号 env 先 shell `export` 再 `sbatch --export=ALL`([[feedback_sbatch_export_comma]])。
- human 侧(MIX=rh):human 是 cotrain,默认全用;新 env `HELDOUT_VIDS_H`(默认 `"0,1"`)从 human cotrain 排除 2 段作 human-in-domain eval(替代旧 `HELDOUT_H` 的 clip-idx)。

### 3.3 ③(train_multihead_wm)同口径
- 其 HELDOUT env(clip 下标)改成读 `vid` 按 `HELDOUT_VIDS` 分组(同 `split_by_episode`)。构造/latent 索引须映射到同一 vid 口径。

### 3.4 eval 兼容(eval_established_metrics)
- eval 复现 split 也走 `split_by_episode`(现共用 split 逻辑,改一处);老 ckpt 无关(eval 只重算 heldout 集合)。

## 4. 重训矩阵(今晚决定性 ~9 run,GPU1 直跑 + 队列)
留出:robot vid{100,102}(~279 clip)/ human vid{0,1};训练池 robot ~2157(16 ep)。
- **②**: `{ro, rh, rh_two}` 全量 + `{ro, rh}@NROB=100` + `{ro, rh}@NROB=300`(human-helps)= 7 run。
- **③**: `{ro, rh}` render = 2 run。
- 配置:`DS=clips_robot_retrack`、`DS_H=clips_human_L24_retrack_realwrist`、`SPLIT=okfirst`、`HELDOUT_VIDS=100,102`。
- ro 是 HEAD_MODE 无关(复用于 two 参照)。

## 5. 判决 / 读法
- **复核方向性判决**(泄漏对称,预期仍成立):two-head 无效(rh_two≈rh_single)、human 稀缺帮、human 全量害。
- **诚实 headline 绝对数**(预期比泄漏版**上升**):记录干净 ro/rh/rh_two + 稀缺 delta。
- 明确标注:**same-scene 天花板不变**(episode 仍同场景);**N-clip 因重叠而虚**(切片 Phase 2 再治)。

## 6. 测试
- 单元测试 `split_by_episode`:(a)heldout 与训练 vid 集合不相交;(b)demo seq {59,332,418,442} 全在 heldout;(c)heldout/pool 都 ok-过滤;(d)与旧 clip-idx split 对比,确认 heldout episode 的 clip 不再出现在 pool。
- ② smoke:`HELDOUT_VIDS=100,102` 跑通,日志打印 `heldout vids [100,102] n_ho=... pool=...`,metrics.json 记 heldout_vids。
- ③ smoke 同理。

## 7. 非目标(Phase 2,今晚不做)
- 切片 slicing best-practice(episode 内随机起点 / contact 加权 / 重 track)——要重建数据。
- 全量 sweep + mp/dummy5/cpt 干净重训(决定性臂出信号后再铺)。
- 量化 keyboard controllability 成功率(独立立项)。
- fps 降采样 / human-robot clip 长度统一 / horizon 曲线。
