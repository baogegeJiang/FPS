from __future__ import annotations

import math

import numpy as np
import torch

from fps_uda.training.config import FPSConfig


def effective_base_lr(cfg: FPSConfig) -> float:
    if cfg.base_lr is None:
        raise ValueError("base_lr is required.")
    return float(cfg.base_lr)


def build_optimizer(model: torch.nn.Module, cfg: FPSConfig) -> torch.optim.Optimizer:
    name = str(cfg.optimizer).strip().lower()
    base_lr = effective_base_lr(cfg)
    weight_decay = float(cfg.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=base_lr,
            momentum=float(cfg.momentum),
            nesterov=bool(cfg.nesterov),
            weight_decay=weight_decay,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=base_lr,
            betas=tuple(float(value) for value in cfg.adamw_betas),
            eps=float(cfg.adamw_eps),
            weight_decay=weight_decay,
        )
    raise ValueError("optimizer must be one of: sgd, adamw")


def lr_scheduler(optimizer: torch.optim.Optimizer, update_step: int, cfg: FPSConfig):
    mode = str(cfg.lr_schedule).strip().lower()
    base_lr = effective_base_lr(cfg)
    step = int(update_step)
    if step <= 0:
        raise ValueError("update_step must be positive.")
    if mode == "linear_step":
        lr = base_lr * step
    elif mode == "constant":
        lr = base_lr
    elif mode == "cosine":
        total_steps = max(1, int(cfg.iter_num))
        if total_steps <= 1:
            progress = 0.0
        else:
            progress = min(max((step - 1) / (total_steps - 1), 0.0), 1.0)
        min_lr = float(cfg.min_lr)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    else:
        raise ValueError("lr_schedule must be one of: linear_step, constant, cosine")
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def temperature_schedule(t0: float, t_final: float, t_max: int, step: int, k: float = 0.1):
    return {
        "linear": t0 + (t_final - t0) * (step / max(1, t_max)),
        "exp": t_final + (t0 - t_final) * np.exp(-k * step / 500),
    }


def hyperparameters(cfg: FPSConfig, step: int, device: torch.device):
    schedule_tau = max(float(cfg.schedule_tau), 1e-12)
    lcr_weight = (
        float(cfg.lambda_lcr)
        if cfg.lambda_lcr is not None
        else (
            float(cfg.randn_ratio)
            if cfg.randn_ratio is not None
            else 0.55  # FPS paper default LCR weight.
        )
    )
    if not cfg.dynamic_parameters:
        return {
            "beta": torch.tensor(cfg.beta, device=device),
            "rand": lcr_weight,
            "alpha": torch.tensor(cfg.alpha, device=device),
        }
    alpha_0 = float(cfg.alpha_0)
    beta_0 = float(cfg.beta_0)
    if cfg.baseline:
        return {
            "beta": 1.0,
            "rand": 0.0,
            "alpha": (1 - torch.exp(torch.tensor(-step / schedule_tau, device=device)))
            * (cfg.alpha - alpha_0)
            + alpha_0,
        }
    return {
        "beta": torch.exp(torch.tensor(-step / schedule_tau, device=device)) * (cfg.beta - beta_0)
        + beta_0,
        "rand": lcr_weight,
        "alpha": (1 - torch.exp(torch.tensor(-step / schedule_tau, device=device)))
        * (cfg.alpha - alpha_0)
        + alpha_0,
    }
