import torch

from lever.similarity import linear_cka


def test_cka_invariances():
    torch.manual_seed(0)
    x = torch.randn(200, 16)
    q, _ = torch.linalg.qr(torch.randn(16, 16))
    assert abs(linear_cka(x, x) - 1) < 1e-9
    assert abs(linear_cka(x, 3.0 * x @ q) - 1) < 1e-9  # rotation + isotropic scale: same representation
    assert linear_cka(x, torch.randn(200, 16)) < 0.2  # unrelated features: low
