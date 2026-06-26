import torch

from fps_uda.models import DY


def test_dy_forward_reset_and_correction():
    model = DY(class_num=3, feature_num=4)
    x = torch.randn(5, 4)
    prob, logits, _ = model(x)
    assert prob.shape == (5, 3)
    assert logits.shape == (5, 3)
    with torch.no_grad():
        model.db.copy_(torch.tensor([1.0, 0.0, -1.0]))
    prob_corr, _, _ = model(x, use_correction=True)
    assert not torch.allclose(prob, prob_corr)
    model.reset()
    assert torch.count_nonzero(model.M) == 0
    assert torch.count_nonzero(model.db) == 0
