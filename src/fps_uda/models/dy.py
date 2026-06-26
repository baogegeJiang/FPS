from __future__ import annotations

import torch
from torch import nn


class DY(nn.Module):
    """Linear feature-space classifier used by FPS training."""

    def __init__(self, class_num: int, feature_num: int):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(class_num))
        self.M = nn.Parameter(torch.zeros((class_num, feature_num)))
        self.db = nn.Parameter(torch.zeros(class_num))
        self.dm = nn.Parameter(torch.zeros(class_num, feature_num))

    def forward(self, features: torch.Tensor, use_correction: bool = False):
        logits = self.b.unsqueeze(0) + torch.einsum("bf,cf->bc", features, self.M)
        if use_correction:
            logits = logits + self.db.unsqueeze(0) + torch.einsum("bf,cf->bc", features, self.dm)
        prob = torch.nn.functional.softmax(logits, dim=-1)
        return prob, logits, features.mean(dim=0) * 0

    def reset(self):
        self.b.data.zero_()
        self.M.data.zero_()
        self.db.data.zero_()
        self.dm.data.zero_()
        for param in (self.b, self.M, self.db, self.dm):
            if param.grad is not None:
                param.grad.zero_()


class BackboneDY(nn.Module):
    """Backbone plus DY classifier.

    This class supports both image inputs [B, C, H, W] and direct feature inputs
    [B, D], so the same object can be used for extraction and feature-space
    experiments.
    """

    def __init__(
        self,
        backbone: nn.Module,
        class_num: int,
        random_pool: bool = False,
        water_level: float = 0.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.random_pool = random_pool
        self.water_level = water_level
        if not hasattr(backbone, "in_features"):
            raise ValueError("backbone must expose an in_features attribute.")
        feature_num = int(backbone.in_features)
        self.Dy = DY(class_num=class_num, feature_num=feature_num)

    def forward(self, x, use_correction: bool = False, return_features: bool = False):
        if x.dim() == 4:
            features = self.backbone(
                x,
                random_pool=self.random_pool,
                water_level=self.water_level,
            )
        else:
            features = x
        if return_features:
            return features
        return self.Dy(features, use_correction=use_correction)

    def reset(self):
        self.Dy.reset()

    @property
    def b(self):
        return self.Dy.b

    @property
    def M(self):
        return self.Dy.M

    @property
    def db(self):
        return self.Dy.db

    @property
    def dm(self):
        return self.Dy.dm
