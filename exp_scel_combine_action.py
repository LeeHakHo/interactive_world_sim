"""结合 skel(同域主体,human 帮) + world(绝对残差,robot 精度)的 ② 世界模型。
突破 velocity_action 线的 Pareto(单一 featurization 不可两全)。机制:A1 加性 residual /
align(LaST-HD 对齐) / warm(两阶段课程)。精度来源纯 robot,human 只从 shared 通道帮进来。
协议逐字复用 exp_scel_agentframe;SS trainer 复用 amplify_wm/velocity 超参。见 spec
docs/superpowers/specs/2026-07-12-combine-skel-world-action-design.md。
Output: outputs/cross_embodiment_wm/combine_action/"""
import os, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import amplify_wm as A
import eval_scheduled_sampling as SSm
import exp_scel_velocity_action as V
import exp_scel_agentframe as X
from amplify_wm import K, F, vel_to_class, device

Lw = K + F


def feat_world(eef3, objc):                                   # (B, Lw*6) 绝对星座(接触几何,disjoint)
    B = eef3.shape[0]
    return (eef3 - objc[:, :, None]).reshape(B, -1)


def feat_skel(eef3, objc):                                    # (B, Lw*10) OSCAR 骨架(同域)
    B = eef3.shape[0]
    wrist = eef3[:, :, 0]
    dwr = torch.cat([torch.zeros_like(wrist[:, :1]), wrist[:, 1:] - wrist[:, :-1]], 1)
    b1 = eef3[:, :, 1] - wrist; b2 = eef3[:, :, 2] - wrist
    db1 = torch.cat([torch.zeros_like(b1[:, :1]), b1[:, 1:] - b1[:, :-1]], 1)
    db2 = torch.cat([torch.zeros_like(b2[:, :1]), b2[:, 1:] - b2[:, :-1]], 1)
    return torch.cat([dwr, b1, b2, db1, db2], -1).reshape(B, -1)


class CombLWC(A.FlowWM_LWC):
    """双 action 头:act_shared(skel 主体) + act_world(world 残差)。forward 收可选 is_h;
    human 样本只走 shared(a1/warm world 残差置 0)。评价路径不传 is_h -> 全 robot,残差全开。"""
    def __init__(s, P, combine="a1", alpha=0.5, lam_res=1e-3, lam_align=0.3, **kw):
        super().__init__(P, **kw)
        s.combine, s.alpha, s.lam_res, s.lam_align = combine, alpha, lam_res, lam_align
        s.act_shared = nn.Linear(V.ACT_DIM["skel"], s.Dm)
        s.act_world = nn.Linear(V.ACT_DIM["world"], s.Dm)
        with torch.no_grad():                                 # 小初始化残差,防独吞
            s.act_world.weight.mul_(0.1); s.act_world.bias.zero_()
        s.warm_stage = 1
        s.aux = {}

    def _obj_tokens(s, hist):
        B, P = hist.shape[:2]
        anchor = hist[:, :, -1, :]
        obj = s.inp(torch.cat([(hist - anchor[:, :, None]).reshape(B, P, 2 * K), anchor], -1))
        objc = anchor.mean(1, keepdim=True)
        return obj, anchor, objc

    def trunk(s, hist, eef3, is_h=None):
        B, P = hist.shape[:2]
        obj, anchor, objc = s._obj_tokens(hist)
        if is_h is None:
            is_h = torch.zeros(B, dtype=torch.bool, device=hist.device)
        act_s = s.act_shared(feat_skel(eef3, objc))           # (B,Dm) 主体
        act_w = s.act_world(feat_world(eef3, objc))           # (B,Dm) 残差
        if s.combine in ("a1", "warm"):
            use_w = (~is_h).float()[:, None]                  # human -> 0
            if s.combine == "warm" and s.warm_stage == 1:
                use_w = use_w * 0.0                           # stage1 纯 shared
            residual = s.alpha * act_w * use_w
            s.aux["res_sq"] = residual.pow(2).mean()
            act = (act_s + residual)[:, None, :]
            x = s.tf(torch.cat([obj, act], 1))[:, :P]
            return x, anchor
        if s.combine == "align":
            x_s = s.tf(torch.cat([obj, act_s[:, None]], 1))[:, :P]
            x_w = s.tf(torch.cat([obj, act_w[:, None]], 1))[:, :P]
            robot = (~is_h).float()                            # (B,) align 只在 robot
            per = ((x_w - x_s.detach()) ** 2).mean((1, 2))     # (B,)
            s.aux["align"] = (per * robot).sum() / (robot.sum() + 1e-6)
            x = torch.where(is_h[:, None, None], x_s, x_w)      # robot 走 world(精度),human 走 shared
            return x, anchor
        raise ValueError(s.combine)

    def forward(s, hist, eef3, is_h=None):
        x, anchor = s.trunk(hist, eef3, is_h)
        B, P = x.shape[:2]
        logits = s.head(x).reshape(B, P, F, s.W * s.W)
        return logits, anchor
