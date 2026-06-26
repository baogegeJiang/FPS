import numpy as np
import torch

import fps_uda.training.trainer as trainer_module
from fps_uda import FPSConfig, FeatureSet, train_fps
from fps_uda.training.loss_api import LossOutput


def _features(seed: int = 7, *, target_labels: bool = True) -> FeatureSet:
    rng = np.random.default_rng(seed)
    labels = np.array([0, 1] * 5)
    target = np.array([0, 1] * 6)
    kwargs = {
        "src_features": rng.normal(size=(10, 4)).astype("float32"),
        "src_labels": labels,
        "entropy_features": rng.normal(size=(12, 4)).astype("float32"),
        "cr_features_1": rng.normal(size=(12, 4)).astype("float32"),
        "cr_features_2": (rng.normal(size=(12, 4)) + 0.5).astype("float32"),
        "eval_features": rng.normal(size=(6, 4)).astype("float32"),
    }
    if target_labels:
        kwargs.update(
            {
                "entropy_labels": target,
                "cr_labels": target,
                "eval_labels": np.array([0, 1, 0, 1, 0, 1]),
            }
        )
    return FeatureSet(**kwargs)


def test_train_fps_cpu_smoke():
    config = FPSConfig(
        num_classes=2,
        feature_dim=4,
        device="cpu",
        base_lr=0.01,
        iter_num=2,
        eval_interval=1,
        progress=False,
    )
    result = train_fps(_features(), config)

    assert result.best_metric == "acc"
    assert result.best_score is not None
    assert result.best_cwc is not None
    assert result.best_cwc_step is not None
    assert result.predictions.shape == (6, 2)
    assert result.best_cwc_predictions.shape == (6, 2)
    assert result.best_cwc_labels.shape == (6,)
    assert result.history[0]["lr"] == 0.01
    assert result.history[1]["lr"] == 0.02
    assert "best_cwc" in result.history[0]
    assert "best_cwc_step" in result.history[0]
    assert "src_acc" in result.history[0]
    assert "entropy_acc" in result.history[0]
    assert "cr1_acc" in result.history[0]
    assert "cr2_acc" in result.history[0]


def test_train_fps_records_invr_when_true_class_r_is_zero():
    features = _features(seed=17)
    features.eval_features = np.zeros_like(features.eval_features)
    config = FPSConfig(
        num_classes=2,
        feature_dim=4,
        device="cpu",
        base_lr=0.01,
        iter_num=1,
        eval_interval=1,
        progress=False,
        normalize="none",
        pseudo_margin=False,
        use_lcr=False,
    )

    result = train_fps(features, config)

    assert "invR" in result.history[0]
    assert result.history[0]["invR"] == 0.0


def test_train_fps_uses_configured_momentum(monkeypatch):
    captured = {}
    original_sgd = torch.optim.SGD

    class CapturingSGD(original_sgd):
        def __init__(self, params, *args, **kwargs):
            captured["momentum"] = kwargs.get("momentum")
            super().__init__(params, *args, **kwargs)

    monkeypatch.setattr(trainer_module.torch.optim, "SGD", CapturingSGD)
    config = FPSConfig(
        num_classes=2,
        feature_dim=4,
        device="cpu",
        base_lr=0.01,
        momentum=0.73,
        iter_num=1,
        eval_interval=1,
        progress=False,
    )

    train_fps(_features(), config)

    assert captured["momentum"] == 0.73


def test_lcr_contributes_to_optimizer_update():
    base_config = {
        "num_classes": 3,
        "feature_dim": 4,
        "device": "cpu",
        "base_lr": 0.1,
        "iter_num": 3,
        "eval_interval": 1,
        "progress": False,
        "seed": 7,
        "normalize": "none",
        "dynamic_parameters": False,
        "beta": 0.5,
        "alpha": 0.0,
        "use_lse": False,
        "use_lce": False,
        "use_lcr": True,
        "use_classes_weight": False,
        "pseudo_margin": False,
        "lcr_loss": "mse",
    }
    features = _features(seed=123)

    without_lcr_weight = train_fps(features, FPSConfig(**base_config, lambda_lcr=0.0))
    with_lcr_weight = train_fps(features, FPSConfig(**base_config, lambda_lcr=1.0))

    assert with_lcr_weight.history[1]["loss_lcr"] > 0
    state_abs_diff = sum(
        torch.sum(
            torch.abs(
                without_lcr_weight.final_model_state[key]
                - with_lcr_weight.final_model_state[key]
            )
        ).item()
        for key in without_lcr_weight.final_model_state
    )
    assert state_abs_diff > 1e-6


def test_train_sample_ratios_control_per_step_counts():
    config = FPSConfig(
        num_classes=2,
        feature_dim=4,
        device="cpu",
        base_lr=0.01,
        iter_num=2,
        eval_interval=1,
        progress=False,
        normalize="none",
        src_sample_ratio=0.5,
        target_sample_ratio=0.25,
        pseudo_margin=False,
        use_lcr=False,
    )

    result = train_fps(_features(seed=333), config)

    assert result.config["src_sample_ratio"] == 0.5
    assert result.config["target_sample_ratio"] == 0.25
    assert result.history[0]["src_train_count"] == 5
    assert result.history[0]["entropy_train_count"] == 3
    assert result.history[0]["cr_train_count"] == 3


def test_lcr_sample_weight_contributes_to_optimizer_update():
    base_config = {
        "num_classes": 3,
        "feature_dim": 4,
        "device": "cpu",
        "base_lr": 0.1,
        "iter_num": 3,
        "eval_interval": 1,
        "progress": False,
        "seed": 13,
        "normalize": "none",
        "dynamic_parameters": False,
        "beta": 0.5,
        "alpha": 0.0,
        "use_lse": False,
        "use_lce": False,
        "use_lcr": True,
        "lambda_lcr": 1.0,
        "use_classes_weight": True,
        "sparse_weight_a": 5.0,
        "pseudo_margin": False,
        "lcr_loss": "mse",
    }
    features = _features(seed=456)

    unweighted_lcr = train_fps(features, FPSConfig(**base_config, lcr_sample_weight="none"))
    weighted_lcr = train_fps(features, FPSConfig(**base_config, lcr_sample_weight="density"))

    assert unweighted_lcr.history[2]["loss_lcr"] != weighted_lcr.history[2]["loss_lcr"]
    state_abs_diff = sum(
        torch.sum(
            torch.abs(
                unweighted_lcr.final_model_state[key]
                - weighted_lcr.final_model_state[key]
            )
        ).item()
        for key in unweighted_lcr.final_model_state
    )
    assert state_abs_diff > 1e-7


def test_sparse_weight_contributes_to_optimizer_update():
    base_config = {
        "num_classes": 3,
        "feature_dim": 4,
        "device": "cpu",
        "base_lr": 0.1,
        "iter_num": 3,
        "eval_interval": 1,
        "progress": False,
        "seed": 11,
        "normalize": "none",
        "dynamic_parameters": False,
        "beta": 0.5,
        "alpha": 0.5,
        "use_lse": True,
        "use_lce": True,
        "use_lcr": False,
        "pseudo_margin": False,
    }
    features = _features(seed=321)

    unweighted = train_fps(features, FPSConfig(**base_config, use_classes_weight=False))
    weighted = train_fps(
        features,
        FPSConfig(**base_config, use_classes_weight=True, sparse_weight_a=5.0),
    )

    assert unweighted.history[2]["loss_consistency"] != weighted.history[2]["loss_consistency"]
    state_abs_diff = sum(
        torch.sum(
            torch.abs(unweighted.final_model_state[key] - weighted.final_model_state[key])
        ).item()
        for key in unweighted.final_model_state
    )
    assert state_abs_diff > 1e-7


def test_custom_extra_loss_reads_context_and_can_warm_up():
    calls = []

    def delayed_entropy_logit_penalty(ctx):
        calls.append(
            {
                "step": ctx["step"],
                "has_entropy_logits": "entropy.logits" in ctx["outputs"],
                "src_count": int(ctx["batch"]["src.features"].shape[0]),
            }
        )
        logits = ctx["outputs"]["entropy.logits"]
        if ctx["step"] < 1:
            value = logits.sum() * 0.0
        else:
            value = 0.01 * logits.pow(2).mean()
        return LossOutput(
            name="custom_entropy_logit_penalty",
            value=value,
            logs={"loss_custom": float(value.detach().cpu())},
        )

    base_config = {
        "num_classes": 2,
        "feature_dim": 4,
        "device": "cpu",
        "base_lr": 0.01,
        "iter_num": 2,
        "eval_interval": 1,
        "progress": False,
        "seed": 5,
        "normalize": "none",
        "pseudo_margin": False,
        "use_lcr": False,
    }
    features = _features(seed=654)
    baseline = train_fps(features, FPSConfig(**base_config))
    custom = train_fps(
        features,
        FPSConfig(**base_config),
        extra_loss_terms=[delayed_entropy_logit_penalty],
    )

    assert calls[0]["step"] == 0
    assert calls[0]["has_entropy_logits"] is True
    assert calls[0]["src_count"] == 10
    assert custom.history[0]["loss_custom"] == 0.0
    assert custom.history[1]["loss_custom"] > 0.0
    state_abs_diff = sum(
        torch.sum(
            torch.abs(baseline.final_model_state[key] - custom.final_model_state[key])
        ).item()
        for key in baseline.final_model_state
    )
    assert state_abs_diff > 1e-8
