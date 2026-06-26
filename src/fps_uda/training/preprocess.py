from __future__ import annotations

import random
from dataclasses import replace
from typing import Optional

import numpy as np
import torch

from fps_uda.data import TensorFeatureSet
from fps_uda.training.config import FPSConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_means(features: torch.Tensor, labels: torch.Tensor):
    means = {}
    for cls in labels.unique():
        mask = labels == cls
        if mask.any():
            means[int(cls.item())] = features[mask].mean(dim=0, keepdim=True)
    return means


def normalize_features(data: TensorFeatureSet, cfg: FPSConfig) -> TensorFeatureSet:
    if cfg.normalize == "none":
        return data
    if cfg.normalize == "cross_norm":
        tgt_weight = float(cfg.cross_norm_target_weight)
        if not 0.0 <= tgt_weight <= 1.0:
            raise ValueError("cross_norm_target_weight must be in the interval [0, 1].")
        src_weight = 1.0 - tgt_weight
        src_mean = data.src_features.mean(dim=0, keepdim=True)
        tgt_mean = data.entropy_features.mean(dim=0, keepdim=True)
        mean = src_weight * src_mean + tgt_weight * tgt_mean
        src_var = data.src_features.var(dim=0, keepdim=True, unbiased=False)
        tgt_var = data.entropy_features.var(dim=0, keepdim=True, unbiased=False)
        var = (
            src_weight * (src_var + (src_mean - mean).pow(2))
            + tgt_weight * (tgt_var + (tgt_mean - mean).pow(2))
        )
        std = (var.clamp_min(0.0).sqrt() / float(cfg.cross_norm_scale)).clamp_min(1e-8)
        return replace(
            data,
            src_features=(data.src_features - mean) / std,
            entropy_features=(data.entropy_features - mean) / std,
            cr_features_1=(data.cr_features_1 - mean) / std,
            cr_features_2=(data.cr_features_2 - mean) / std,
            eval_features=None if data.eval_features is None else (data.eval_features - mean) / std,
        )
    if cfg.normalize == "self_norm":
        src_mean = data.src_features.mean(dim=0, keepdim=True)
        src_std = (
            data.src_features.std(dim=0, keepdim=True, unbiased=False)
            / float(cfg.self_norm_scale_src)
        ).clamp_min(1e-8)
        entropy_mean = data.entropy_features.mean(dim=0, keepdim=True)
        entropy_std = (
            data.entropy_features.std(dim=0, keepdim=True, unbiased=False)
            / float(cfg.self_norm_scale_tgt)
        ).clamp_min(1e-8)
        cr1_std = (
            data.cr_features_1.std(dim=0, keepdim=True, unbiased=False)
            / float(cfg.self_norm_scale_tgt)
        ).clamp_min(1e-8)
        cr2_std = (
            data.cr_features_2.std(dim=0, keepdim=True, unbiased=False)
            / float(cfg.self_norm_scale_tgt)
        ).clamp_min(1e-8)
        return replace(
            data,
            src_features=(data.src_features - src_mean) / src_std,
            entropy_features=(data.entropy_features - entropy_mean) / entropy_std,
            cr_features_1=(data.cr_features_1 - data.cr_features_1.mean(dim=0, keepdim=True))
            / cr1_std,
            cr_features_2=(data.cr_features_2 - data.cr_features_2.mean(dim=0, keepdim=True))
            / cr2_std,
            eval_features=None
            if data.eval_features is None
            else (data.eval_features - entropy_mean) / entropy_std,
        )
    raise ValueError("normalize must be one of: none, cross_norm, self_norm")


def apply_geom_alpha_beta(data: TensorFeatureSet, cfg: FPSConfig) -> TensorFeatureSet:
    if not cfg.use_geom_test or (float(cfg.geom_alpha) == 1.0 and float(cfg.geom_beta) == 1.0):
        return data
    if data.cr_labels is None or data.eval_labels is None or data.eval_features is None:
        raise ValueError("geom test requires cr_labels, eval_features, and eval_labels.")

    mu_s = compute_class_means(data.src_features, data.src_labels)
    mu_t = compute_class_means(data.eval_features, data.eval_labels)
    entropy_new = data.entropy_features.clone()
    cr1_new = data.cr_features_1.clone()
    cr2_new = data.cr_features_2.clone()
    eval_new = data.eval_features.clone()

    for cls, mu_s_cls in mu_s.items():
        if cls not in mu_t:
            continue
        mu_t_cls = mu_t[cls]
        mu_t_beta = mu_s_cls + float(cfg.geom_beta) * (mu_t_cls - mu_s_cls)
        mask_cr = data.cr_labels == cls
        if mask_cr.any():
            cr1_new[mask_cr] = mu_t_beta + float(cfg.geom_alpha) * (
                data.cr_features_1[mask_cr] - mu_t_cls
            )
            cr2_new[mask_cr] = mu_t_beta + float(cfg.geom_alpha) * (
                data.cr_features_2[mask_cr] - mu_t_cls
            )
        if data.entropy_labels is not None:
            mask_entropy = data.entropy_labels == cls
            if mask_entropy.any():
                entropy_new[mask_entropy] = mu_t_beta + float(cfg.geom_alpha) * (
                    data.entropy_features[mask_entropy] - mu_t_cls
                )
        mask_eval = data.eval_labels == cls
        if mask_eval.any():
            eval_new[mask_eval] = mu_t_beta + float(cfg.geom_alpha) * (
                data.eval_features[mask_eval] - mu_t_cls
            )
    return replace(
        data,
        entropy_features=entropy_new,
        cr_features_1=cr1_new,
        cr_features_2=cr2_new,
        eval_features=eval_new,
    )


def validate_sample_ratio(name: str, value: float) -> float:
    ratio = float(value)
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"{name} must be in the interval (0, 1].")
    return ratio


def sample_indices(n_samples: int, ratio: float, device: torch.device) -> Optional[torch.Tensor]:
    if ratio >= 1.0:
        return None
    n_take = max(1, int(round(float(n_samples) * ratio)))
    n_take = min(n_samples, n_take)
    return torch.randperm(n_samples, device=device)[:n_take]


def index_optional(value: Optional[torch.Tensor], indices: Optional[torch.Tensor]):
    if value is None or indices is None:
        return value
    return value.index_select(0, indices)
