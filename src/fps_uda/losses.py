from __future__ import annotations

import torch
import torch.nn.functional as F


def supervise_loss(x, y, weight_sup=None, weight=False, legacy_mode=False):
    if not legacy_mode:
        x = x.clamp_min(1e-8)
    if weight:
        return -(y * torch.log(x) * weight_sup.unsqueeze(-1)).mean()
    return -(y * torch.log(x)).mean()


def _sample_entropy_term(p, weight_vec, entropy_type="shannon", tsallis_q=1.5, eps=1e-8):
    p = p.clamp_min(eps)
    w = weight_vec.unsqueeze(-1)
    if entropy_type == "tsallis":
        q = float(tsallis_q)
        if abs(q - 1.0) < 1e-6:
            ent_elem = -(p * torch.log(p))
        else:
            ent_elem = (p - torch.pow(p, q)) / (q - 1.0)
    else:
        ent_elem = -(p * torch.log(p))
    return (ent_elem * w).mean()


def _adaptive_temp_shannon_term(
    p,
    weight_vec,
    margin_theta=0.15,
    margin_tau=0.05,
    t_min=0.7,
    t_max=1.8,
    eps=1e-8,
):
    p = p.clamp_min(eps)
    top2_prob, _ = torch.topk(p, k=2, dim=1)
    p1 = top2_prob[:, 0]
    p2 = top2_prob[:, 1]
    margin = (p1 - p2) / (p1 + p2 + eps)
    tau = max(float(margin_tau), 1e-6)
    gate = torch.sigmoid((margin - float(margin_theta)) / tau)
    tmin = float(min(t_min, t_max))
    tmax = float(max(t_min, t_max))
    sample_temp = tmax - (tmax - tmin) * gate
    p_t = torch.pow(p, 1.0 / sample_temp.unsqueeze(-1).clamp_min(1e-6))
    p_t = p_t / p_t.sum(dim=1, keepdim=True).clamp_min(eps)
    ent = -(p_t * torch.log(p_t.clamp_min(eps)))
    return (ent * weight_vec.unsqueeze(-1)).mean()


def consistency_loss(
    p,
    weight_tgt,
    alpha,
    temperature=None,
    mask=None,
    entropy_type="shannon",
    tsallis_q=1.5,
    legacy_mode=False,
    adaptive_temp_margin_theta=0.15,
    adaptive_temp_margin_tau=0.05,
    adaptive_temp_t_min=0.7,
    adaptive_temp_t_max=1.8,
):
    if temperature is not None:
        p = p / temperature
    if legacy_mode and entropy_type == "shannon":
        tgt_p_mean = (p * weight_tgt.unsqueeze(-1)).mean(dim=0)
        if mask is None:
            return (
                -(p * torch.log(p) * weight_tgt.unsqueeze(-1)).mean()
                * alpha
            ) + (tgt_p_mean * tgt_p_mean.log()).mean() * (1 - alpha)
        return (
            -(p * torch.log(p) * weight_tgt.unsqueeze(-1) * mask.unsqueeze(-1)).mean()
            * alpha
        ) + (tgt_p_mean * tgt_p_mean.log()).mean() * (1 - alpha)

    p = p.clamp_min(1e-8)
    tgt_p_mean = (p * weight_tgt.unsqueeze(-1)).mean(dim=0)
    tgt_p_mean = tgt_p_mean.clamp_min(1e-8)
    inter_class = (tgt_p_mean * tgt_p_mean.log()).mean() * (1 - alpha)

    if entropy_type == "adaptive_temp_shannon":
        sample_entropy = _adaptive_temp_shannon_term(
            p,
            weight_tgt,
            margin_theta=adaptive_temp_margin_theta,
            margin_tau=adaptive_temp_margin_tau,
            t_min=adaptive_temp_t_min,
            t_max=adaptive_temp_t_max,
        )
        return sample_entropy * alpha + inter_class

    if entropy_type == "shannon":
        if mask is None:
            sample_entropy = -(p * torch.log(p) * weight_tgt.unsqueeze(-1)).mean()
        else:
            sample_entropy = -(
                p * torch.log(p) * weight_tgt.unsqueeze(-1) * mask.unsqueeze(-1)
            ).mean()
        return sample_entropy * alpha + inter_class

    sample_weight = weight_tgt if mask is None else weight_tgt * mask
    sample_entropy = _sample_entropy_term(
        p,
        sample_weight,
        entropy_type=entropy_type,
        tsallis_q=tsallis_q,
    )
    return sample_entropy * alpha + inter_class


def lcr_bi_kl(logits1, logits2, w=None, temperature=2.0, eps=1e-8):
    p1 = F.softmax(logits1 / temperature, dim=1)
    p2 = F.softmax(logits2 / temperature, dim=1)
    kl12 = (p1 * (torch.log(p1 + eps) - torch.log(p2 + eps))).sum(dim=1)
    kl21 = (p2 * (torch.log(p2 + eps) - torch.log(p1 + eps))).sum(dim=1)
    loss = 0.5 * (kl12 + kl21) * (temperature * temperature)
    if w is None:
        return loss.mean()
    w = w.detach()
    return (loss * w).sum() / w.sum().clamp_min(1.0)
