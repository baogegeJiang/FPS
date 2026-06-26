from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

import torch
from tqdm import trange

from fps_uda.data import FeatureSet
from fps_uda.models import DY
from fps_uda.training.config import FPSConfig
from fps_uda.training.evaluation import (
    class_r_mean as _class_r_mean,
    evaluate as _evaluate,
    view_accuracy as _view_accuracy,
    write_summary_csv as _write_summary_csv,
)
from fps_uda.training.loss_api import LossCallable, LossOutput, LossTerm
from fps_uda.training.loss_defs import (
    default_loss_terms,
    sparse_weight as _sparse_weight,
)
from fps_uda.training.optimization import (
    build_optimizer as _build_optimizer,
    effective_base_lr as _effective_base_lr,
    hyperparameters as _hyperparameters,
    lr_scheduler as _lr_scheduler,
    temperature_schedule as _temperature_schedule,
)
from fps_uda.training.preprocess import (
    apply_geom_alpha_beta as _apply_geom_alpha_beta,
    index_optional as _index_optional,
    normalize_features as _normalize,
    sample_indices as _sample_indices,
    set_seed as _set_seed,
    validate_sample_ratio as _validate_sample_ratio,
)
from fps_uda.training.result import TrainingResult


Callback = Callable[[int, dict], None]


def _as_loss_terms(
    loss_terms: Optional[Iterable[Union[LossTerm, LossCallable]]],
    extra_loss_terms: Optional[Iterable[Union[LossTerm, LossCallable]]],
) -> list:
    terms = list(default_loss_terms() if loss_terms is None else loss_terms)
    if extra_loss_terms is not None:
        terms.extend(extra_loss_terms)
    return terms


def _forward_outputs(model: torch.nn.Module, batch: dict, cfg: FPSConfig) -> dict:
    src_p, src_logits, src_extra = model(batch["src.features"])
    _, entropy_logits, entropy_extra = model(
        batch["entropy.features"],
        use_correction=cfg.use_correction,
    )
    entropy_p = torch.softmax(entropy_logits, dim=-1)
    _, cr1_logits, cr1_extra = model(
        batch["cr1.features"],
        use_correction=cfg.use_correction,
    )
    cr1_p = torch.softmax(cr1_logits, dim=-1)
    _, cr2_logits, cr2_extra = model(
        batch["cr2.features"],
        use_correction=cfg.use_correction,
    )
    cr2_p = torch.softmax(cr2_logits, dim=-1)
    return {
        "src.prob": src_p,
        "src.logits": src_logits,
        "src.extra": src_extra,
        "entropy.prob": entropy_p,
        "entropy.logits": entropy_logits,
        "entropy.extra": entropy_extra,
        "cr1.prob": cr1_p,
        "cr1.logits": cr1_logits,
        "cr1.extra": cr1_extra,
        "cr2.prob": cr2_p,
        "cr2.logits": cr2_logits,
        "cr2.extra": cr2_extra,
    }


def _empty_loss(device: torch.device) -> torch.Tensor:
    return torch.zeros((), device=device)


def _combine_loss_outputs(
    outputs: list[LossOutput],
    *,
    beta,
    lambda_lcr,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    total = _empty_loss(device)
    consistency = _empty_loss(device)
    logs = {
        "loss_sup": 0.0,
        "loss_consistency": 0.0,
        "loss_lcr": 0.0,
        "loss_ldelta": 0.0,
    }
    for output in outputs:
        logs.update(output.logs)
        if output.name == "supervised":
            total = total + output.value
        elif output.name in {"entropy_consistency", "ldelta"}:
            consistency = consistency + output.value
        elif output.name == "lcr":
            total = total + output.value * lambda_lcr * (1 - beta)
        elif output.name == "pseudo_margin_weight":
            total = total + output.value
        else:
            total = total + output.value
    total = total + consistency * (1 - beta)
    logs["loss_consistency"] = float(consistency.detach().cpu())
    return total, logs


def train_fps(
    features: FeatureSet,
    config: FPSConfig,
    *,
    output_dir: Optional[str] = None,
    callbacks: Optional[Iterable[Callback]] = None,
    model: Optional[torch.nn.Module] = None,
    loss_terms: Optional[Iterable[Union[LossTerm, LossCallable]]] = None,
    extra_loss_terms: Optional[Iterable[Union[LossTerm, LossCallable]]] = None,
) -> TrainingResult:
    """Train FPS directly from in-memory features.

    Custom Python losses can replace the default FPS losses with ``loss_terms``
    or extend them with ``extra_loss_terms``. Each loss receives a dict-like
    context containing step, config, model, data, batch, outputs, weights,
    schedules, and a mutable state dictionary.
    """
    require_consistency = bool(config.use_consistency_loss and not config.baseline)
    features.validate(
        feature_dim=config.feature_dim,
        require_consistency=require_consistency,
        require_target_labels=config.sparse_weight_a == "class_aware" or config.use_geom_test,
    )
    if config.feature_dim is None:
        config = replace(config, feature_dim=features.feature_dim)
    _effective_base_lr(config)
    if config.use_shift_constraint and not config.use_correction:
        config = replace(config, use_correction=True)
    src_sample_ratio = _validate_sample_ratio("src_sample_ratio", config.src_sample_ratio)
    target_sample_ratio = _validate_sample_ratio(
        "target_sample_ratio",
        config.target_sample_ratio,
    )
    device = config.validate_device()
    _set_seed(config.seed)
    callbacks = list(callbacks or [])
    active_loss_terms = _as_loss_terms(loss_terms, extra_loss_terms)

    data = features.as_tensors(device=device)
    data = _normalize(data, config)
    data = _apply_geom_alpha_beta(data, config)

    if model is None:
        model = DY(class_num=config.num_classes, feature_num=int(config.feature_dim))
    model.to(device)
    if hasattr(model, "reset"):
        model.reset()
    optimizer = _build_optimizer(model, config)

    if config.use_classes_weight:
        weight_entropy = _sparse_weight(
            data.entropy_features,
            data.entropy_labels,
            config.sparse_weight_a,
        ).to(device)
        weight_cr = _sparse_weight(
            data.cr_features_1,
            data.cr_labels,
            config.sparse_weight_a,
        ).to(device)
    else:
        weight_entropy = torch.ones(
            data.entropy_features.shape[0],
            device=device,
            dtype=data.entropy_features.dtype,
        )
        weight_cr = torch.ones(
            data.cr_features_1.shape[0],
            device=device,
            dtype=data.cr_features_1.dtype,
        )

    history = []
    best_score = None
    best_metric = "acc"
    best_cwc = None
    best_cwc_step = None
    predictions = None
    labels = None
    best_predictions = None
    best_labels = None
    best_cwc_predictions = None
    best_cwc_labels = None
    class_r_true = None
    if data.eval_features is not None and data.eval_labels is not None:
        class_r_true = _class_r_mean(
            data.eval_labels.detach().cpu().numpy(),
            data.eval_features.detach().cpu().numpy(),
        )

    loss_state: dict = {}
    iterator = trange(config.iter_num, disable=not config.progress)
    for step in iterator:
        current_lr = _lr_scheduler(optimizer, step + 1, config)
        model.train()
        optimizer.zero_grad()
        temperature = None
        if config.use_temperature:
            temperature = _temperature_schedule(
                config.begin_temperature, 1.0, config.iter_num, step
            )["exp"]
        hyper = _hyperparameters(config, step, device)

        src_indices = _sample_indices(data.src_features.shape[0], src_sample_ratio, device)
        entropy_indices = _sample_indices(
            data.entropy_features.shape[0], target_sample_ratio, device
        )
        cr_indices = _sample_indices(data.cr_features_1.shape[0], target_sample_ratio, device)
        batch = {
            "src.features": _index_optional(data.src_features, src_indices),
            "src.labels": _index_optional(data.src_labels, src_indices),
            "entropy.features": _index_optional(data.entropy_features, entropy_indices),
            "entropy.labels": _index_optional(data.entropy_labels, entropy_indices),
            "cr1.features": _index_optional(data.cr_features_1, cr_indices),
            "cr2.features": _index_optional(data.cr_features_2, cr_indices),
            "cr.labels": _index_optional(data.cr_labels, cr_indices),
        }
        weights = {
            "entropy.density": _index_optional(weight_entropy, entropy_indices),
            "cr.density": _index_optional(weight_cr, cr_indices),
            "cr.margin": None,
            "cr.paper": None,
        }
        schedules = {
            "alpha": hyper["alpha"],
            "beta": hyper["beta"],
            "lambda_lcr": hyper["rand"],
            "temperature": temperature,
            "lr": current_lr,
        }
        outputs = _forward_outputs(model, batch, config)
        ctx = {
            "step": step,
            "cfg": config,
            "model": model,
            "data": data,
            "batch": batch,
            "outputs": outputs,
            "weights": weights,
            "schedules": schedules,
            "state": loss_state,
            "indices": {"src": src_indices, "entropy": entropy_indices, "cr": cr_indices},
            "device": device,
        }

        loss_outputs = [term(ctx) for term in active_loss_terms]
        total_loss, loss_logs = _combine_loss_outputs(
            loss_outputs,
            beta=hyper["beta"],
            lambda_lcr=hyper["rand"],
            device=device,
        )
        total_loss.backward()
        optimizer.step()

        should_eval = step % max(1, config.eval_interval) == 0 or step == config.iter_num - 1
        row = {
            "step": float(step),
            "lr": float(current_lr),
            **loss_logs,
            "src_train_count": float(batch["src.features"].shape[0]),
            "entropy_train_count": float(batch["entropy.features"].shape[0]),
            "cr_train_count": float(batch["cr1.features"].shape[0]),
            "alpha": float(hyper["alpha"]),
            "beta": float(hyper["beta"]),
        }
        if should_eval:
            model.eval()
            with torch.no_grad():
                metrics, predictions, labels = _evaluate(model, data, config)
                view_accs = {
                    "src_acc": _view_accuracy(
                        model, data.src_features, data.src_labels, config
                    ),
                    "entropy_acc": _view_accuracy(
                        model, data.entropy_features, data.entropy_labels, config
                    ),
                    "cr1_acc": _view_accuracy(
                        model, data.cr_features_1, data.cr_labels, config
                    ),
                    "cr2_acc": _view_accuracy(
                        model, data.cr_features_2, data.cr_labels, config
                    ),
                }
            row.update({key: float(value) for key, value in metrics.items()})
            row.update(
                {key: float(value) for key, value in view_accs.items() if value is not None}
            )
            if (
                class_r_true is not None
                and predictions is not None
                and data.eval_features is not None
            ):
                class_r = _class_r_mean(
                    predictions.argmax(axis=-1), data.eval_features.detach().cpu().numpy()
                )
                row["invR"] = float(class_r / max(class_r_true, 1e-12))
            if "acc" in row and (best_score is None or row["acc"] > best_score):
                best_score = row["acc"]
                best_predictions = None if predictions is None else predictions.copy()
                best_labels = None if labels is None else labels.copy()
            if "class_wise_acc" in row and (
                best_cwc is None or row["class_wise_acc"] > best_cwc
            ):
                best_cwc = row["class_wise_acc"]
                best_cwc_step = row["step"]
                best_cwc_predictions = None if predictions is None else predictions.copy()
                best_cwc_labels = None if labels is None else labels.copy()
            if best_cwc is not None:
                row["best_cwc"] = float(best_cwc)
                row["best_cwc_step"] = float(best_cwc_step)
            if config.progress and "acc" in row:
                postfix = {"acc": f"{row['acc']:.4f}"}
                if "class_wise_acc" in row:
                    postfix["cwc"] = f"{row['class_wise_acc']:.4f}"
                if best_cwc is not None:
                    postfix["best_cwc"] = f"{best_cwc:.4f}"
                if "src_acc" in row:
                    postfix["src"] = f"{row['src_acc']:.4f}"
                if "entropy_acc" in row:
                    postfix["ent"] = f"{row['entropy_acc']:.4f}"
                if "cr1_acc" in row:
                    postfix["cr1"] = f"{row['cr1_acc']:.4f}"
                if "cr2_acc" in row:
                    postfix["cr2"] = f"{row['cr2_acc']:.4f}"
                if best_score is not None:
                    postfix["best"] = f"{best_score:.4f}"
                iterator.set_postfix(postfix)
        history.append(row)
        for callback in callbacks:
            callback(step, row)

    result = TrainingResult(
        best_metric=best_metric if best_score is not None else None,
        best_score=None if best_score is None else float(best_score),
        best_cwc=None if best_cwc is None else float(best_cwc),
        best_cwc_step=None if best_cwc_step is None else float(best_cwc_step),
        history=history,
        predictions=best_predictions if best_predictions is not None else predictions,
        labels=best_labels if best_labels is not None else labels,
        best_cwc_predictions=best_cwc_predictions,
        best_cwc_labels=best_cwc_labels,
        final_model_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
        config=config.to_dict(),
    )
    out = output_dir or config.output_dir
    if out is not None:
        result.save(out)
        _write_summary_csv(history, Path(out))
    return result
