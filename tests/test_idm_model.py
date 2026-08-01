import sys
sys.path.insert(0, ".")
import torch, idm_model as M


def test_forward_shape():
    m = M.IDM(din=100)
    y = m(torch.randn(8, 100))
    assert y.shape == (8, 7)


def test_overfits_one_batch():
    torch.manual_seed(0)
    m = M.IDM(din=32)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    x = torch.randn(16, 32); y = torch.randn(16, 7)
    loss = None
    for _ in range(400):
        opt.zero_grad(); loss = ((m(x) - y) ** 2).mean(); loss.backward(); opt.step()
    assert loss.item() < 1e-3
