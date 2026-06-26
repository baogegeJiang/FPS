import sys
from types import SimpleNamespace

import pytest
import torch

from fps_uda.models.backbones import (
    BackboneConfig,
    PoolableBackbone,
    ResNetBackbone,
    TimmVisionBackbone,
    _select_encoder_state_dict,
    build_backbone,
)


class _TokenExtractor(torch.nn.Module):
    in_features = 3

    def forward(self, x):
        batch = x.shape[0]
        patches = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            dtype=x.dtype,
            device=x.device,
        ).unsqueeze(0)
        return patches.expand(batch, -1, -1)


class _SpatialExtractor(torch.nn.Module):
    in_features = 1

    def forward(self, x):
        out = torch.full((x.shape[0], 1, 4, 4), 2.0, dtype=x.dtype, device=x.device)
        out[:, :, 0, :] = 100.0
        return out


def _token_backbone():
    config = BackboneConfig.from_mapping(
        {
            "backend": "timm",
            "name": "tiny_vit",
            "pretrained": False,
            "in_features": 3,
            "pooling": {
                "feature_type": "token",
                "random_strategy": "token_channel_squared",
            },
        }
    )
    return PoolableBackbone(_TokenExtractor(), config)


def _spatial_backbone():
    config = BackboneConfig.from_mapping(
        {
            "backend": "torchvision",
            "name": "resnet50",
            "pretrained": False,
            "in_features": 1,
            "pooling": {
                "feature_type": "spatial",
                "random_strategy": "spatial_shared",
            },
        }
    )
    return PoolableBackbone(_SpatialExtractor(), config)


def test_token_random_pool_uses_channelwise_squared_strategy_and_varies_by_call():
    backbone = _token_backbone()
    x = torch.zeros(2, 3, 8, 8)

    deterministic = backbone(x, random_pool=False, mute_padding_in_pool=False)
    expected = torch.tensor([[5.5, 6.5, 7.5], [5.5, 6.5, 7.5]])
    torch.testing.assert_close(deterministic, expected)

    torch.manual_seed(1)
    pooled_1 = backbone(x, random_pool=True, water_level=0.0, mute_padding_in_pool=False)
    torch.manual_seed(2)
    pooled_2 = backbone(x, random_pool=True, water_level=0.0, mute_padding_in_pool=False)

    assert not torch.allclose(pooled_1, pooled_2)
    torch.manual_seed(1)
    manual_mask = torch.rand(2, 4, 3) ** 2
    manual_mask = manual_mask / manual_mask.sum(dim=1, keepdim=True).clamp_min(1e-12)
    manual = torch.einsum("blc,blc->bc", _TokenExtractor()(x), manual_mask)
    torch.testing.assert_close(pooled_1, manual)


def test_spatial_masked_pool_ignores_detected_padding_band():
    backbone = _spatial_backbone()
    x = torch.zeros(1, 3, 4, 4)
    x[:, :, 1:, :] = torch.arange(12, dtype=x.dtype).reshape(1, 1, 3, 4)

    unmasked = backbone(x, random_pool=False, mute_padding_in_pool=False)
    masked = backbone(x, random_pool=False, mute_padding_in_pool=True)

    torch.testing.assert_close(unmasked, torch.tensor([[26.5]]))
    torch.testing.assert_close(masked, torch.tensor([[2.0]]))


def test_extract_pooling_views_reuses_one_extractor_forward():
    class CountingExtractor(_TokenExtractor):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return super().forward(x)

    extractor = CountingExtractor()
    config = BackboneConfig.from_mapping(
        {
            "backend": "timm",
            "name": "tiny_vit",
            "pretrained": False,
            "in_features": 3,
            "pooling": {
                "feature_type": "token",
                "random_strategy": "token_channel_squared",
            },
        }
    )
    backbone = PoolableBackbone(extractor, config)
    x = torch.zeros(1, 3, 8, 8)

    torch.manual_seed(1)
    outputs = backbone.extract_pooling_views(
        x,
        poolings=("clean", "pool_a", "pool_b", "pool_c"),
        water_level=0.0,
        mute_padding_in_pool=False,
    )

    assert extractor.calls == 1
    assert set(outputs) == {"clean", "pool_a", "pool_b", "pool_c"}
    torch.testing.assert_close(outputs["clean"], torch.tensor([[5.5, 6.5, 7.5]]))
    assert outputs["pool_a"].shape == (1, 3)
    assert outputs["pool_b"].shape == (1, 3)
    assert outputs["pool_c"].shape == (1, 3)
    assert not torch.allclose(outputs["pool_a"], outputs["pool_b"])


def test_torchvision_resnet_loader_uses_explicit_weights(monkeypatch):
    calls = []

    class FakeEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Identity()
            self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.fc = torch.nn.Linear(2048, 1)

        def children(self):
            return iter([self.features, self.avgpool, self.fc])

    def fake_resnet50(**kwargs):
        calls.append(kwargs)
        return FakeEncoder()

    fake_torchvision = SimpleNamespace(models=SimpleNamespace(resnet50=fake_resnet50))
    monkeypatch.setitem(sys.modules, "torchvision", fake_torchvision)

    config = BackboneConfig.from_mapping(
        {
            "backend": "torchvision",
            "name": "resnet50",
            "pretrained": True,
            "weights": "IMAGENET1K_V1",
            "in_features": 2048,
            "pooling": {
                "feature_type": "spatial",
                "random_strategy": "spatial_shared",
            },
        }
    )
    backbone = build_backbone(config)

    assert isinstance(backbone, ResNetBackbone)
    assert calls == [{"weights": "IMAGENET1K_V1"}]
    assert backbone.config.pooling.random_strategy == "spatial_shared"


def test_timm_loader_uses_explicit_kwargs_and_checkpoint(monkeypatch):
    calls = []

    class FakeModel(torch.nn.Module):
        num_features = 768
        num_prefix_tokens = 1

        def forward_features(self, x):
            batch = x.shape[0]
            return torch.zeros(batch, 197, 768, dtype=x.dtype, device=x.device)

    def fake_create_model(name, **kwargs):
        calls.append((name, kwargs))
        return FakeModel()

    monkeypatch.setitem(sys.modules, "timm", SimpleNamespace(create_model=fake_create_model))

    backbone = TimmVisionBackbone(
        "vit_base_patch16_224.augreg2_in21k_ft_in1k",
        pretrained=True,
        checkpoint_path="checkpoint.pt",
        kwargs={"class_token": True, "global_pool": "token"},
    )

    assert backbone.in_features == 768
    assert backbone.config.pooling.random_strategy == "token_channel_squared"
    assert calls == [
        (
            "vit_base_patch16_224.augreg2_in21k_ft_in1k",
            {
                "pretrained": True,
                "class_token": True,
                "global_pool": "token",
                "checkpoint_path": "checkpoint.pt",
            },
        )
    ]


def test_build_resnet_backbone_forwards_checkpoint_path(monkeypatch, tmp_path):
    seen = {}

    def fake_init(
        self,
        name="resnet50",
        pretrained=True,
        checkpoint_path=None,
        *,
        weights=None,
        in_features=None,
        kwargs=None,
        pooling=None,
    ):
        torch.nn.Module.__init__(self)
        seen["name"] = name
        seen["pretrained"] = pretrained
        seen["checkpoint_path"] = checkpoint_path
        seen["weights"] = weights
        seen["pooling"] = pooling

    monkeypatch.setattr(ResNetBackbone, "__init__", fake_init)

    checkpoint = tmp_path / "resnet.pt"
    backbone = build_backbone(
        "resnet101", pretrained=False, checkpoint_path=str(checkpoint)
    )

    assert isinstance(backbone, ResNetBackbone)
    assert seen == {
        "name": "resnet101",
        "pretrained": False,
        "checkpoint_path": str(checkpoint),
        "weights": None,
        "pooling": {
            "feature_type": "spatial",
            "random_strategy": "spatial_shared",
        },
    }


def test_custom_backend_is_metadata_only_and_requires_poolable_wrapper():
    config = BackboneConfig.from_mapping(
        {
            "backend": "custom",
            "name": "my_private_model",
            "pretrained": False,
            "in_features": 8,
            "pooling": {
                "feature_type": "token",
                "random_strategy": "token_channel_squared",
            },
        }
    )

    assert config.metadata()["backbone_backend"] == "custom"
    with pytest.raises(ValueError, match="PoolableBackbone"):
        build_backbone(config)


def test_resnet_checkpoint_selector_accepts_full_model_prefix():
    encoder = torch.nn.Sequential(torch.nn.Linear(2, 3))
    full_model_state = {
        "backbone.encoder.0.weight": torch.ones(3, 2),
        "backbone.encoder.0.bias": torch.ones(3),
        "fc.weight": torch.zeros(1, 3),
    }

    selected = _select_encoder_state_dict(full_model_state, encoder)

    assert set(selected) == {"0.weight", "0.bias"}
