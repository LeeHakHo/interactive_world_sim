"""M4 过夜编排: 等 latents+cond 就绪 -> 过拟合 sanity(gate) -> 通过则 launch 全量训练。
sanity gate: 过拟合 4 clip 1500 步, 末 loss 明显低于初始(<1.0, 初始~1.7 = mean((x1-x0)^2)) = 学得动。
通过 -> nohup 全量 STEPS=60000 后台; 不通过 -> 写 SANITY_FAILED 停下待人看。全程 log。
用 .venv_wan/bin/python 跑(训练也用它)。Bash run_in_background 启动 -> 退出时通知我。
"""
import os, sys, time, subprocess, re
sys.path.insert(0, ".")
PY = ".venv_wan/bin/python"
LAT = "outputs/video_arch_wm/wan_latents_can_dual/latents_all.npz"
COND = "outputs/video_arch_wm/cond_can_dual/cond_all.npz"
OUT = "outputs/video_arch_wm/m4_video_dit"; os.makedirs(OUT, exist_ok=True)
LOG = f"{OUT}/orchestrate.log"
GPU = os.environ.get("GPU", "0")


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    open(LOG, "a").write(line + "\n")


def wait_files(paths, timeout_s=4 * 3600):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if all(os.path.exists(p) for p in paths):
            time.sleep(20)  # 让 writer flush 完
            return True
        time.sleep(60)
    return False


def main():
    log(f"orchestrate 启动. 等 latents+cond ...")
    if not wait_files([LAT, COND]):
        log("TIMEOUT 等数据超时, 放弃"); return
    log(f"数据就绪: {LAT}, {COND}. 跑过拟合 sanity(4 clip, 1500 步)...")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, LAT=LAT, COND=COND,
               OUT=f"{OUT}/sanity", STEPS="1500", BS="4", OVERFIT="4", EVAL_EVERY="1500", D="512", DEPTH="12")
    r = subprocess.run([PY, "train_video_dit.py"], env=env, capture_output=True, text=True)
    slog = r.stdout + r.stderr
    open(f"{OUT}/sanity.log", "w").write(slog)
    losses = [float(x) for x in re.findall(r"loss (\d+\.\d+)", slog)]
    recon = re.findall(r"latent recon MSE (\d+\.\d+)", slog)
    if not losses:
        log(f"SANITY 无 loss 输出, 训练脚本可能崩. 见 sanity.log 末尾:\n{slog[-800:]}");
        open(f"{OUT}/SANITY_FAILED.txt", "w").write("no loss parsed\n" + slog[-2000:]); return
    l0, l1 = losses[0], losses[-1]
    rec = float(recon[-1]) if recon else -1
    log(f"sanity: loss {l0:.3f} -> {l1:.3f}  recon MSE {rec:.3f}  ({len(losses)} 记录)")
    if l1 < 1.0 and l1 < 0.8 * l0:
        log(f"SANITY PASS (loss 明显下降). launch 全量 STEPS=60000 后台 ...")
        fenv = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, LAT=LAT, COND=COND,
                    OUT=OUT, STEPS="60000", BS="16", OVERFIT="0", EVAL_EVERY="3000", D="512", DEPTH="12")
        f = open(f"{OUT}/m4_full.log", "w")
        subprocess.Popen([PY, "train_video_dit.py"], env=fenv, stdout=f, stderr=subprocess.STDOUT,
                         start_new_session=True)
        log(f"全量训练已 launch -> {OUT}/m4_full.log (ckpt {OUT}/video_dit_ema.pt)")
        open(f"{OUT}/SANITY_PASS_TRAINING_LAUNCHED.txt", "w").write(
            f"sanity loss {l0:.3f}->{l1:.3f} recon {rec:.3f}. full STEPS=60000 launched.\n")
    else:
        log(f"SANITY FAIL (loss 没明显降: {l0:.3f}->{l1:.3f}). 不 launch, 待人诊断。")
        open(f"{OUT}/SANITY_FAILED.txt", "w").write(
            f"loss {l0:.3f}->{l1:.3f} recon {rec:.3f} 未过 gate(需 l1<1.0 且 l1<0.8*l0)\n")
    log("orchestrate 结束")


if __name__ == "__main__":
    main()
