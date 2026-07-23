import os, sys, numpy as np

FOLLOW_TH = float(os.environ.get("FOLLOW_TH", "0.4"))

def blend_predtr(pred2_A, pred2_B, objA, objB, grasp, efA, efB):
    H = len(pred2_A); trajA = pred2_A.copy(); trajB = pred2_B.copy(); fb = np.zeros(H, bool)
    for h in range(H):
        if grasp[h]:
            ef_disp = np.linalg.norm(np.concatenate([efA[h].mean(0)-efA[0].mean(0), efB[h].mean(0)-efB[0].mean(0)]))
            pr_disp = np.linalg.norm(np.concatenate([pred2_A[h].mean(0)-pred2_A[0].mean(0), pred2_B[h].mean(0)-pred2_B[0].mean(0)]))
            if ef_disp > 1e-3 and pr_disp / ef_disp < FOLLOW_TH:
                trajA[h] = objA[h]; trajB[h] = objB[h]; fb[h] = True
    return trajA, trajB, fb
