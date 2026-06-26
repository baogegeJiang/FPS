import torch
from torch import nn

from fps_uda.models import DY, DomainAdaptationNet


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


def test_domain_adaptation_net_wraps_flat_backbone_features():
    class TinyBackbone(nn.Module):
        in_features = 4

        def forward(self, x):
            assert x.shape[1] == 3
            pooled = x.mean(dim=(2, 3))
            return torch.cat([pooled, pooled[:, :1]], dim=1)

    model = DomainAdaptationNet(TinyBackbone(), num_classes=3, embed_dim=5, dropout=0.0)
    x = torch.randn(4, 1, 6, 6)

    logits, features = model(x)

    assert logits.shape == (4, 3)
    assert features.shape == (4, 4)
