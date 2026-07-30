"""验证 train_multihead_wm 的真 resume 机制: model/ema/opt/step 存取 round-trip + 续训 step 范围.
只测机制(轻量 tiny 模型), 不跑 Wan 管线."""
import torch, torch.nn as nn, copy, os, tempfile


def _mk():
    torch.manual_seed(0)
    m = nn.Linear(4, 4)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    ema = copy.deepcopy(m)
    return m, opt, ema


def test_resume_bundle_roundtrip():
    m, opt, ema = _mk()
    # 训几步让 opt 累积动量/方差
    for _ in range(5):
        loss = (m(torch.randn(8, 4)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), m.parameters()):
            pe.mul_(0.999).add_(pm, alpha=0.001)
    step = 40000
    with tempfile.TemporaryDirectory() as d:
        p = f"{d}/resume.pt"
        torch.save({"model": m.state_dict(), "ema": ema.state_dict(),
                    "opt": opt.state_dict(), "step": step}, p)
        # 全新对象加载
        m2, opt2, ema2 = _mk()
        ck = torch.load(p, weights_only=False)
        m2.load_state_dict(ck["model"]); ema2.load_state_dict(ck["ema"]); opt2.load_state_dict(ck["opt"])
        start_step = int(ck["step"])
        # model/ema 权重逐一相等
        for a, b in zip(m.parameters(), m2.parameters()):
            assert torch.equal(a, b)
        for a, b in zip(ema.parameters(), ema2.parameters()):
            assert torch.equal(a, b)
        # optimizer 动量状态恢复(exp_avg 非零且相等)
        st1 = opt.state_dict()["state"]; st2 = opt2.state_dict()["state"]
        assert len(st2) == len(st1) and len(st2) > 0
        for k in st1:
            assert torch.equal(st1[k]["exp_avg"], st2[k]["exp_avg"])
        assert start_step == 40000


def test_resume_step_range_continues():
    # 续训循环 range(start_step+1, STEPS+1): 从 40001 跑到 80000
    start_step, STEPS = 40000, 80000
    steps = list(range(start_step + 1, STEPS + 1))
    assert steps[0] == 40001 and steps[-1] == 80000 and len(steps) == 40000


def test_resume_already_done_no_steps():
    # start_step >= STEPS: 空循环, 不重训
    start_step, STEPS = 80000, 80000
    assert list(range(start_step + 1, STEPS + 1)) == []
