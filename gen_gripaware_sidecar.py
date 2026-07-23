"""生成 grip-aware skel sidecar (真实clips): FK(joint)+grip几何闭合(fk_skel2d_dual, 同keyboard推理) 全2700 clip。
让 skel③ 能在训练时看到夹爪随grip张合 -> 学到"指线收拢=渲闭合夹爪"(修inference-only无响应的负结果)。
输出 skel_sidecar_robot_gripaware.npz (skel2d_high/low 9pt随grip闭合, segments, grip)。conda phantom 跑。
"""
import numpy as np
from fk_skel2d import fk_skel2d_dual, ROBOT_SKEL

DS = "outputs/flow_render_dataset_can_dual/clips_robot.npz"
OUT = "outputs/flow_render_dataset_can_dual/skel_sidecar_robot_gripaware.npz"


def main():
    z = np.load(DS)
    J = np.asarray(z["joint"]); GRIP = np.asarray(z["grip"])          # (N,48,7),(N,48)
    N, L = J.shape[:2]
    skh = np.empty((N, L, 9, 2), np.float32); skl = np.empty((N, L, 9, 2), np.float32); seg = None
    for n in range(N):
        a, b, seg = fk_skel2d_dual(J[n], GRIP[n])                     # grip闭合两指点
        skh[n], skl[n] = a, b
        if n % 300 == 0:
            print(f"  gripaware FK {n}/{N}", flush=True)
    np.savez(OUT, skel2d_high=skh, skel2d_low=skl, segments=seg,
             joint_names=np.array(ROBOT_SKEL), grip=GRIP.astype(np.float32))
    # 自检: 指间距应随grip变
    car = skh[:, :, 6:8]; sp = np.linalg.norm(car[:, :, 0] - car[:, :, 1], axis=-1) * 128
    print(f"saved {OUT}: skel2d_high {skh.shape}; 指间距 {sp.min():.2f}..{sp.max():.2f}px (随grip闭合)", flush=True)
    print("=== gripaware sidecar DONE ===", flush=True)


if __name__ == "__main__":
    main()
