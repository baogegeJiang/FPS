from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

import fps_uda.losses as losses
from fps_uda.training.config import FPSConfig
from fps_uda.training.loss_api import LossContext, LossOutput, LossTerm


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(labels.to(torch.int64), num_classes=num_classes).float()


def sparse_weight(
    features: torch.Tensor,
    labels: Optional[torch.Tensor],
    sparse_weight_a,
    batch_size: int = 1024,
) -> torch.Tensor:
    if sparse_weight_a == "class_aware":
        if labels is None:
            raise ValueError("target labels are required when sparse_weight_a='class_aware'.")
        unique_labels, counts = torch.unique(labels, return_counts=True)
        inv_counts = 1.0 / counts.float()
        weights = torch.zeros_like(labels, dtype=torch.float32)
        for label, weight in zip(unique_labels, inv_counts):
            weights[labels == label] = weight
        return weights / weights.mean().clamp_min(1e-8)

    scale = float(sparse_weight_a)
    with torch.no_grad():
        normalized = features.detach()
        normalized = normalized / normalized.norm(dim=1, keepdim=True).clamp_min(1e-8)
        all_weight = []
        n_samples = normalized.size(0)
        for start in range(0, n_samples, batch_size):
            stop = min(start + batch_size, n_samples)
            batch = normalized[start:stop]
            sim = torch.einsum("bf,nf->bn", batch, normalized) * scale
            soft = torch.softmax(sim, dim=1)
            row = torch.arange(stop - start, device=normalized.device)
            col = torch.arange(start, stop, device=normalized.device)
            all_weight.append(soft[row, col])
        weight = torch.cat(all_weight, dim=0)
        return weight / weight.mean().clamp_min(1e-8)


def compute_margin_pair(
    cfg: FPSConfig,
    rand_y1: torch.Tensor,
    rand_y2: torch.Tensor,
    tgt_p: torch.Tensor,
    tgt_p2: torch.Tensor,
    tgt_feature: torch.Tensor,
    tgt_feature2: torch.Tensor,
    head_w: torch.Tensor,
):
    margin_weight_type = cfg.margin_weight_type
    margin_score_coef = float(cfg.margin_score_coef)
    if margin_weight_type == "top2_proto_gap_ratio":
        _, top2_idx_1 = torch.topk(rand_y1.detach().float(), k=2, dim=1)
        _, top2_idx_2 = torch.topk(rand_y2.detach().float(), k=2, dim=1)
        f1 = tgt_feature.detach().float()
        f2 = tgt_feature2.detach().float()
        p1 = tgt_p.detach().float()
        p2 = tgt_p2.detach().float()
        cls_mass_1 = p1.sum(dim=0)
        cls_mass_2 = p2.sum(dim=0)
        proto_1 = (p1.t() @ f1) / cls_mass_1.unsqueeze(1).clamp_min(1e-6)
        proto_2 = (p2.t() @ f2) / cls_mass_2.unsqueeze(1).clamp_min(1e-6)
        proto_1 = torch.where((cls_mass_1 > 1e-6).unsqueeze(1), proto_1, head_w)
        proto_2 = torch.where((cls_mass_2 > 1e-6).unsqueeze(1), proto_2, head_w)
        w1a, w1b = proto_1[top2_idx_1[:, 0]], proto_1[top2_idx_1[:, 1]]
        w2a, w2b = proto_2[top2_idx_2[:, 0]], proto_2[top2_idx_2[:, 1]]
        gap_1 = torch.norm(f1 - w1b, dim=1) - torch.norm(f1 - w1a, dim=1)
        gap_2 = torch.norm(f2 - w2b, dim=1) - torch.norm(f2 - w2a, dim=1)
        denom_1 = torch.norm(w1a - w1b, dim=1).clamp_min(1e-6)
        denom_2 = torch.norm(w2a - w2b, dim=1).clamp_min(1e-6)
        return margin_score_coef * gap_1 / denom_1, margin_score_coef * gap_2 / denom_2

    if margin_weight_type in {"distance_gap", "distance_gap_ratio"}:
        dist_1 = torch.cdist(tgt_feature.detach().float(), head_w, p=2)
        dist_2 = torch.cdist(tgt_feature2.detach().float(), head_w, p=2)
        d2_1, idx2_1 = torch.topk(dist_1, k=2, largest=False, dim=1)
        d2_2, idx2_2 = torch.topk(dist_2, k=2, largest=False, dim=1)
        gap_1 = d2_1[:, 1] - d2_1[:, 0]
        gap_2 = d2_2[:, 1] - d2_2[:, 0]
        if margin_weight_type == "distance_gap_ratio":
            head_dist = torch.cdist(head_w, head_w, p=2).clamp_min(1e-6)
            return (
                margin_score_coef * gap_1 / head_dist[idx2_1[:, 0], idx2_1[:, 1]],
                margin_score_coef * gap_2 / head_dist[idx2_2[:, 0], idx2_2[:, 1]],
            )
        return gap_1, gap_2

    head_dist = torch.cdist(head_w, head_w, p=2).clamp_min(1e-6)
    top2_val_1, top2_idx_1 = torch.topk(rand_y1.detach().float(), k=2, dim=1)
    top2_val_2, top2_idx_2 = torch.topk(rand_y2.detach().float(), k=2, dim=1)
    margin_1 = top2_val_1[:, 0] - top2_val_1[:, 1]
    margin_2 = top2_val_2[:, 0] - top2_val_2[:, 1]
    return (
        margin_1 / head_dist[top2_idx_1[:, 0], top2_idx_1[:, 1]],
        margin_2 / head_dist[top2_idx_2[:, 0], top2_idx_2[:, 1]],
    )


def convert_margin_to_weight_pair(cfg: FPSConfig, margin12_1, margin12_2, q_used: float = 0.15):
    tau = torch.tensor(float(cfg.margin_sigmoid_tau), device=margin12_1.device).clamp_min(1e-6)
    target = min(max(float(cfg.margin_sigmoid_boundary_weight), 1e-6), 0.499999)
    boundary_bias = math.log((1.0 - target) / target)
    mode = cfg.margin_convert_mode
    if mode in {"quantile_sigmoid", "by_type"} and cfg.margin_weight_type == "normalized_logit_gap":
        theta_1 = torch.quantile(margin12_1, q_used)
        theta_2 = torch.quantile(margin12_2, q_used)
        return (
            torch.sigmoid((margin12_1 - theta_1) / tau - boundary_bias),
            torch.sigmoid((margin12_2 - theta_2) / tau - boundary_bias),
        )
    if mode == "quantile_binary":
        theta_1 = torch.quantile(margin12_1, q_used)
        theta_2 = torch.quantile(margin12_2, q_used)
        return (margin12_1 > theta_1).float(), (margin12_2 > theta_2).float()
    if mode == "quantile_sharp_sigmoid":
        theta_1 = torch.quantile(margin12_1, q_used)
        theta_2 = torch.quantile(margin12_2, q_used)
        tau = torch.tensor(float(cfg.margin_sharp_tau), device=margin12_1.device).clamp_min(1e-6)
        return torch.sigmoid((margin12_1 - theta_1) / tau), torch.sigmoid(
            (margin12_2 - theta_2) / tau
        )
    if mode == "by_type" and cfg.margin_weight_type == "distance_gap":
        w1 = (margin12_1 - margin12_1.min()) / (margin12_1.max() - margin12_1.min() + 1e-6)
        w2 = (margin12_2 - margin12_2.min()) / (margin12_2.max() - margin12_2.min() + 1e-6)
        return w1.clamp(0.0, 1.0), w2.clamp(0.0, 1.0)
    return (
        torch.sigmoid(margin12_1 / tau - boundary_bias),
        torch.sigmoid(margin12_2 / tau - boundary_bias),
    )


def lcr_loss(
    cfg: FPSConfig,
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    prob1: torch.Tensor,
    prob2: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    mode = str(cfg.lcr_loss)
    if mode == "l2":
        per_sample = torch.linalg.vector_norm(logits1 - logits2, ord=2, dim=1) / logits1.shape[1]
    elif mode == "mse":
        per_sample = (logits1 - logits2).pow(2).mean(dim=1)
    else:
        raise ValueError("lcr_loss must be one of: mse, l2.")
    if sample_weight is not None:
        sample_weight = sample_weight.detach().to(device=per_sample.device, dtype=per_sample.dtype)
        per_sample = per_sample * sample_weight.reshape(-1)
    return per_sample.mean()


def lcr_sample_weight(
    cfg: FPSConfig,
    weight_tgt: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    mode = str(cfg.lcr_sample_weight)
    if mode in {"none", "false", "0"}:
        return None
    if mode == "density":
        return weight_tgt
    if mode == "margin":
        return mask
    if mode in {"density_margin", "margin_density", "paper"}:
        return weight_tgt if mask is None else weight_tgt * mask
    raise ValueError(
        "lcr_sample_weight must be one of: none, density, margin, density_margin, paper."
    )


def ldelta_loss(cfg: FPSConfig, model: torch.nn.Module, step: int, device: torch.device) -> torch.Tensor:
    decay_steps = max(float(cfg.ldelta_decay_steps), 1e-12)
    decay = torch.exp(torch.tensor(-0.05 * float(step) / decay_steps, device=device))
    reg_loss = torch.norm(model.db) + torch.norm(model.dm)
    return float(cfg.ldelta_weight) * decay * reg_loss


def _zero(ctx: LossContext) -> torch.Tensor:
    return torch.zeros((), device=ctx["device"])


@dataclass
class SupervisedLossTerm:
    name: str = "supervised"

    def __call__(self, ctx: LossContext) -> LossOutput:
        cfg: FPSConfig = ctx["cfg"]
        loss_sup = losses.supervise_loss(
            ctx["outputs"]["src.prob"],
            one_hot(ctx["batch"]["src.labels"], cfg.num_classes),
            weight_sup=None,
            weight=False,
        )
        beta = ctx["schedules"]["beta"]
        return LossOutput(
            name=self.name,
            value=loss_sup * beta,
            logs={"loss_sup": float(loss_sup.detach().cpu())},
        )


@dataclass
class EntropyConsistencyLossTerm:
    name: str = "entropy_consistency"

    def __call__(self, ctx: LossContext) -> LossOutput:
        cfg: FPSConfig = ctx["cfg"]
        loss_consis = _zero(ctx)
        if cfg.use_consistency_loss and not cfg.baseline:
            entropy_p = ctx["outputs"]["entropy.prob"]
            weight_entropy = ctx["weights"]["entropy.density"]
            temperature = ctx["schedules"].get("temperature")
            alpha = ctx["schedules"]["alpha"]
            if cfg.use_lse and cfg.use_lce:
                loss_consis = losses.consistency_loss(
                    entropy_p,
                    weight_entropy,
                    alpha,
                    temperature,
                    mask=None,
                    entropy_type=cfg.sample_entropy_type,
                    tsallis_q=cfg.tsallis_q,
                    legacy_mode=cfg.legacy_loss_mode,
                    adaptive_temp_margin_theta=cfg.adaptive_temp_margin_theta,
                    adaptive_temp_margin_tau=cfg.adaptive_temp_margin_tau,
                    adaptive_temp_t_min=cfg.adaptive_temp_t_min,
                    adaptive_temp_t_max=cfg.adaptive_temp_t_max,
                )
            elif cfg.use_lse:
                loss_consis = losses.consistency_loss(
                    entropy_p,
                    weight_entropy,
                    1.0,
                    temperature,
                    entropy_type=cfg.sample_entropy_type,
                    tsallis_q=cfg.tsallis_q,
                    legacy_mode=cfg.legacy_loss_mode,
                )
            elif cfg.use_lce:
                loss_consis = losses.consistency_loss(
                    entropy_p,
                    weight_entropy,
                    0.0,
                    temperature,
                    entropy_type=cfg.sample_entropy_type,
                    tsallis_q=cfg.tsallis_q,
                    legacy_mode=cfg.legacy_loss_mode,
                )
        ctx["state"]["loss_consistency_base"] = loss_consis
        return LossOutput(name=self.name, value=loss_consis, logs={})


@dataclass
class LDeltaLossTerm:
    name: str = "ldelta"

    def __call__(self, ctx: LossContext) -> LossOutput:
        cfg: FPSConfig = ctx["cfg"]
        loss_ldelta = _zero(ctx)
        if cfg.use_consistency_loss and not cfg.baseline and cfg.use_shift_constraint:
            loss_ldelta = ldelta_loss(cfg, ctx["model"], ctx["step"], ctx["device"])
        ctx["state"]["loss_ldelta"] = loss_ldelta
        return LossOutput(
            name=self.name,
            value=loss_ldelta,
            logs={"loss_ldelta": float(loss_ldelta.detach().cpu())},
        )


@dataclass
class PseudoMarginWeightTerm:
    name: str = "pseudo_margin_weight"

    def __call__(self, ctx: LossContext) -> LossOutput:
        cfg: FPSConfig = ctx["cfg"]
        if (
            not cfg.use_consistency_loss
            or cfg.baseline
            or not cfg.pseudo_margin
            or int(ctx["step"]) < int(cfg.margin_start_step)
        ):
            ctx["weights"]["cr.margin"] = None
            return LossOutput(name=self.name, value=_zero(ctx), logs={})

        cr_indices = ctx["indices"]["cr"]
        use_margin_cache = cr_indices is None
        update_interval = max(1, int(cfg.margin_weight_update_interval))
        current_margin_w1 = current_margin_w2 = None
        state = ctx["state"]
        if (
            not use_margin_cache
            or state.get("cached_margin_w1") is None
            or int(ctx["step"]) % update_interval == 0
        ):
            with torch.no_grad():
                model = ctx["model"]
                head_w = (
                    (model.M + model.dm).detach().float()
                    if cfg.use_correction
                    else model.M.detach().float()
                )
                margin12_1, margin12_2 = compute_margin_pair(
                    cfg,
                    ctx["outputs"]["cr1.logits"],
                    ctx["outputs"]["cr2.logits"],
                    ctx["outputs"]["cr1.prob"],
                    ctx["outputs"]["cr2.prob"],
                    ctx["batch"]["cr1.features"],
                    ctx["batch"]["cr2.features"],
                    head_w,
                )
                current_margin_w1, current_margin_w2 = convert_margin_to_weight_pair(
                    cfg, margin12_1, margin12_2, q_used=float(cfg.margin_quantile)
                )
                if use_margin_cache:
                    state["cached_margin_w1"] = current_margin_w1
                    state["cached_margin_w2"] = current_margin_w2
        if use_margin_cache:
            w_1 = state["cached_margin_w1"]
            w_2 = state["cached_margin_w2"]
        else:
            w_1 = current_margin_w1
            w_2 = current_margin_w2
        mask = (w_1 + w_2) / 2
        mask = mask / mask.mean().clamp_min(1e-8)
        ctx["weights"]["cr.margin"] = mask
        return LossOutput(name=self.name, value=_zero(ctx), logs={})


@dataclass
class LCRLossTerm:
    name: str = "lcr"

    def __call__(self, ctx: LossContext) -> LossOutput:
        cfg: FPSConfig = ctx["cfg"]
        loss_rand = _zero(ctx)
        if cfg.use_consistency_loss and not cfg.baseline and cfg.use_lcr:
            weight_cr = ctx["weights"]["cr.density"]
            mask = ctx["weights"].get("cr.margin")
            sample_weight = lcr_sample_weight(cfg, weight_cr, mask)
            ctx["weights"]["cr.paper"] = sample_weight
            loss_rand = lcr_loss(
                cfg,
                ctx["outputs"]["cr1.logits"],
                ctx["outputs"]["cr2.logits"],
                ctx["outputs"]["cr1.prob"],
                ctx["outputs"]["cr2.prob"],
                sample_weight=sample_weight,
            )
        return LossOutput(
            name=self.name,
            value=loss_rand,
            logs={"loss_lcr": float(loss_rand.detach().cpu())},
        )


def default_loss_terms() -> list[LossTerm]:
    return [
        SupervisedLossTerm(),
        EntropyConsistencyLossTerm(),
        LDeltaLossTerm(),
        PseudoMarginWeightTerm(),
        LCRLossTerm(),
    ]
