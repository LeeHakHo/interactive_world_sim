# CKPT 清单(clean L48 episode-split)— 2026-07-30

> 每个 ckpt 的实验标注,防覆盖+防混淆。② heldout/config 来自 metrics.json;③ 来自训练 log 头。
> ★命名规范: <action>_<mix><_nN><_head><_sSEED>。mix: r=robot-only, rh=robot+human混训。nN=robot稀缺到N clip(无=全量2311)。head: two=域专属双头, film=软FiLM域, 无=单共享头。sSEED: 无=seed0, s1/s2=另两种子。

## ② object-flow WM (DualLWC) — `outputs/cross_embodiment_wm/epsplit_L48/`  ckpt=wm_dual.pt

| run | action | mix | robot N | human | head | seed | drift hi | drift lo |
|---|---|---|---|---|---|---|---|---|
| mp_r_all | mp | r | 2311 | 0 | single | 0 | 2.20 | 2.76 |
| mp_r_all_s1 | mp | r | 2311 | 0 | single | 1 | 2.29 | 2.68 |
| mp_r_all_s2 | mp | r | 2311 | 0 | single | 2 | 2.37 | 2.86 |
| mp_r_n100 | mp | r | 100 | 0 | single | 0 | 8.51 | 8.71 |
| mp_r_n300 | mp | r | 300 | 0 | single | 0 | 4.59 | 5.23 |
| mp_rh_all | mp | rh | 2311 | 1499 | single | 0 | 2.11 | 2.63 |
| mp_rh_all_film | mp | rh | 2311 | 1499 | film | 0 | 2.10 | 2.56 |
| mp_rh_all_film_s1 | mp | rh | 2311 | 1499 | film | 1 | 2.95 | 3.34 |
| mp_rh_all_film_s2 | mp | rh | 2311 | 1499 | film | 2 | 1.91 | 2.44 |
| mp_rh_all_s1 | mp | rh | 2311 | 1499 | single | 1 | 2.68 | 3.11 |
| mp_rh_all_s2 | mp | rh | 2311 | 1499 | single | 2 | 2.26 | 2.89 |
| mp_rh_all_two | mp | rh | 2311 | 1499 | two | 0 | 2.07 | 2.44 |
| mp_rh_all_two_s1 | mp | rh | 2311 | 1499 | two | 1 | 3.36 | 3.64 |
| mp_rh_all_two_s2 | mp | rh | 2311 | 1499 | two | 2 | 2.26 | 3.10 |
| mp_rh_n100 | mp | rh | 100 | 1499 | single | 0 | 3.60 | 4.32 |
| mp_rh_n100_film | mp | rh | 100 | 1499 | film | 0 | 4.06 | 4.56 |
| mp_rh_n300 | mp | rh | 300 | 1499 | single | 0 | 2.37 | 3.04 |
| mp_rh_n300_film | mp | rh | 300 | 1499 | film | 0 | 2.46 | 3.14 |

## ③ Wan 视频渲染器 (MultiHeadVideoWM) — `outputs/video_arch_wm/epsplit_L48_mh/`  ckpt=mh_ema.pt

| run | mix | robot N (OVERFIT) | human | NOFLOW | 训练步 | recon MSE | render-LPIPS | cube_px | ckpt |
|---|---|---|---|---|---|---|---|---|---|
| rh | human | 2311 | 1792 | 0 | 40000/40000 | 0.0848 | 0.0654 | 3.49 | ✅ |
| rh_n100 | human | 100 | 100 | 0 | 2200/40000 | — | — | — | 训练中 |
| rh_n300 | human | 300 | 300 | 0 | 2200/40000 | — | — | — | 训练中 |
| ro | robot-only | 2311 | 0 | 0 | 40000/40000 | 0.0759 | 0.0638 | 3.69 | ✅ |
| ro_n100 | robot-only | 100 | 0 | 0 | 2300/40000 | — | — | — | 训练中 |
| ro_n300 | robot-only | 300 | 0 | 0 | 2300/40000 | — | — | — | 训练中 |
| ro_noflow | robot-only | 2311 | 0 | 1 | 7200/40000 | 0.2064 | — | — | ✅ |

## 防覆盖检查
- ② 18 个 run + ③ 7 个 run 各自独立子目录,命名含 mix/N/head/seed,无同名覆盖。
- 旧泄漏 ckpt 另存 `mh_clean/`(③) 与老 `established_metrics` 对应 run,未被 L48 覆盖。
- heldout 一律 episode-split vid[100,102](见各 metrics.json / log 头)。
