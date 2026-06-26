from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from fps_uda import FPSConfig, FeatureSet
from fps_uda.cli import _resolve_feature_transform
from fps_uda.io.config import flatten_training_config
from fps_uda.losses import consistency_loss, lcr_bi_kl, supervise_loss
from fps_uda.metrics import calculate_auc, calculate_class_accuracy
from fps_uda.training.evaluation import evaluate
from fps_uda.training.loss_defs import (
    convert_margin_to_weight_pair as _convert_margin_to_weight_pair,
    lcr_loss as _lcr_loss,
    lcr_sample_weight as _lcr_sample_weight,
    ldelta_loss as _ldelta_loss,
)
from fps_uda.training.optimization import temperature_schedule
from fps_uda.training.preprocess import apply_geom_alpha_beta as _apply_geom_alpha_beta
from fps_uda.training.trainer import (
    _build_optimizer,
    _effective_base_lr,
    _hyperparameters,
    _lr_scheduler,
    _normalize,
)


def test_lr_scheduler_uses_tex_base_lr_steps():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    config = FPSConfig(num_classes=2, feature_dim=1, device="cpu", base_lr=0.001)

    assert _lr_scheduler(optimizer, 1, config) == pytest.approx(0.001)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.001)
    assert _lr_scheduler(optimizer, 2, config) == pytest.approx(0.002)
    assert _lr_scheduler(optimizer, 100, config) == pytest.approx(0.1)


def test_lr_scheduler_supports_constant_and_cosine():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    constant = FPSConfig(
        num_classes=2,
        feature_dim=1,
        device="cpu",
        base_lr=0.001,
        lr_schedule="constant",
    )
    assert _lr_scheduler(optimizer, 1, constant) == pytest.approx(0.001)
    assert _lr_scheduler(optimizer, 100, constant) == pytest.approx(0.001)

    cosine = FPSConfig(
        num_classes=2,
        feature_dim=1,
        device="cpu",
        base_lr=0.01,
        min_lr=0.001,
        iter_num=3,
        lr_schedule="cosine",
    )
    assert _lr_scheduler(optimizer, 1, cosine) == pytest.approx(0.01)
    assert _lr_scheduler(optimizer, 3, cosine) == pytest.approx(0.001)


def test_build_optimizer_supports_sgd_and_adamw():
    model = torch.nn.Linear(2, 2)
    sgd = _build_optimizer(
        model,
        FPSConfig(
            num_classes=2,
            feature_dim=2,
            device="cpu",
            base_lr=0.01,
            momentum=0.8,
            nesterov=True,
            weight_decay=0.01,
        ),
    )
    assert isinstance(sgd, torch.optim.SGD)
    assert sgd.param_groups[0]["momentum"] == pytest.approx(0.8)
    assert sgd.param_groups[0]["nesterov"] is True
    assert sgd.param_groups[0]["weight_decay"] == pytest.approx(0.01)

    model = torch.nn.Linear(2, 2)
    adamw = _build_optimizer(
        model,
        FPSConfig(
            num_classes=2,
            feature_dim=2,
            device="cpu",
            base_lr=0.01,
            optimizer="adamw",
            adamw_betas=(0.7, 0.8),
            adamw_eps=1e-7,
            weight_decay=0.02,
        ),
    )
    assert isinstance(adamw, torch.optim.AdamW)
    assert adamw.param_groups[0]["betas"] == pytest.approx((0.7, 0.8))
    assert adamw.param_groups[0]["eps"] == pytest.approx(1e-7)
    assert adamw.param_groups[0]["weight_decay"] == pytest.approx(0.02)


def test_base_lr_is_required():
    config = FPSConfig(num_classes=2, feature_dim=1, device="cpu")

    with pytest.raises(ValueError, match="base_lr is required"):
        _effective_base_lr(config)


def test_legacy_randn_ratio_maps_to_public_lambda_lcr():
    config = FPSConfig.from_mapping(
        {
            "num_classes": 2,
            "feature_dim": 1,
            "device": "cpu",
            "randn_ratio": 2.0,
            "base_lr": 0.001,
        }
    )

    assert config.lambda_lcr == pytest.approx(2.0)
    hyper = _hyperparameters(config, step=0, device=torch.device("cpu"))
    assert hyper["rand"] == pytest.approx(2.0)


def test_shift_constraint_enables_correction_and_matches_origin_weight():
    config = FPSConfig.from_mapping(
        {
            "num_classes": 2,
            "feature_dim": 2,
            "device": "cpu",
            "base_lr": 0.001,
            "use_shift_constraint": True,
            "ldelta_weight": 0.1,
            "ldelta_decay_steps": 500,
        }
    )
    model = SimpleNamespace(
        db=torch.tensor([3.0, 4.0]),
        dm=torch.tensor([[0.0, 12.0], [0.0, 0.0]]),
    )

    loss = _ldelta_loss(config, model, step=500, device=torch.device("cpu"))

    assert config.use_correction is True
    assert loss == pytest.approx(0.1 * np.exp(-0.05) * (5.0 + 12.0))


def test_baseline_hyperparameters_disable_target_regularization():
    config = FPSConfig(
        num_classes=2,
        feature_dim=1,
        device="cpu",
        base_lr=0.001,
        baseline=True,
        alpha=0.8,
        alpha_0=0.2,
    )

    hyper = _hyperparameters(config, step=100, device=torch.device("cpu"))

    assert hyper["beta"] == 1.0
    assert hyper["rand"] == 0.0
    assert float(hyper["alpha"]) > 0.2


def test_temperature_schedule_is_available_for_temperature_training():
    schedule = temperature_schedule(t0=30.0, t_final=1.0, t_max=100, step=50)

    assert schedule["linear"] == pytest.approx(15.5)
    assert 1.0 < schedule["exp"] < 30.0


def test_public_default_lcr_and_margin_parameters():
    config = FPSConfig(num_classes=2, feature_dim=1, device="cpu", base_lr=0.001)

    assert config.iter_num == 36000
    assert config.optimizer == "sgd"
    assert config.momentum == pytest.approx(0.9)
    assert config.nesterov is False
    assert config.weight_decay == pytest.approx(0.0)
    assert config.lr_schedule == "linear_step"
    assert config.min_lr == pytest.approx(0.0)
    assert config.cross_norm_scale == pytest.approx(2.5)
    assert config.cross_norm_target_weight == pytest.approx(0.5)
    hyper = _hyperparameters(config, step=0, device=torch.device("cpu"))
    assert hyper["rand"] == pytest.approx(0.55)
    assert config.schedule_tau == pytest.approx(1000.0)
    assert config.lcr_loss == "mse"
    assert config.lcr_sample_weight == "paper"
    assert config.margin_start_step == 100
    assert config.margin_quantile == pytest.approx(0.15)
    assert config.margin_convert_mode == "quantile_sigmoid"
    assert config.margin_sigmoid_tau == pytest.approx(0.2)
    assert config.margin_sigmoid_boundary_weight == pytest.approx(0.2)


def test_public_loss_defaults_match_config_legacy_mode():
    config = FPSConfig(num_classes=2, feature_dim=1, device="cpu", base_lr=0.001)
    p = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    weights = torch.ones(2)

    assert config.legacy_loss_mode is False
    assert torch.isfinite(supervise_loss(p, labels))
    assert torch.isfinite(consistency_loss(p, weights, alpha=0.5))


def test_consistency_loss_supports_public_entropy_variants():
    p = torch.tensor([[0.8, 0.2], [0.45, 0.55]], dtype=torch.float32)
    weights = torch.tensor([1.0, 0.5], dtype=torch.float32)

    tsallis = consistency_loss(
        p,
        weights,
        alpha=0.7,
        entropy_type="tsallis",
        tsallis_q=1.3,
    )
    adaptive = consistency_loss(
        p,
        weights,
        alpha=0.7,
        entropy_type="adaptive_temp_shannon",
    )

    assert torch.isfinite(tsallis)
    assert torch.isfinite(adaptive)


def test_lcr_bi_kl_public_helper_is_finite_and_weightable():
    logits_1 = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    logits_2 = torch.tensor([[1.5, 0.5], [0.5, 1.5]])
    weights = torch.tensor([0.25, 1.75])

    unweighted = lcr_bi_kl(logits_1, logits_2)
    weighted = lcr_bi_kl(logits_1, logits_2, w=weights)

    assert torch.isfinite(unweighted)
    assert torch.isfinite(weighted)
    assert weighted >= 0.0


def test_lcr_loss_mse_and_l2_are_logits_space():
    logits_1 = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    logits_2 = torch.tensor([[1.0, 1.0], [1.0, 5.0]])
    prob_1 = torch.softmax(logits_1, dim=1)
    prob_2 = torch.softmax(logits_2, dim=1)

    mse = _lcr_loss(
        FPSConfig(num_classes=2, feature_dim=1, device="cpu", base_lr=0.001, lcr_loss="mse"),
        logits_1,
        logits_2,
        prob_1,
        prob_2,
    )
    l2 = _lcr_loss(
        FPSConfig(num_classes=2, feature_dim=1, device="cpu", base_lr=0.001, lcr_loss="l2"),
        logits_1,
        logits_2,
        prob_1,
        prob_2,
    )

    assert mse == pytest.approx(float((logits_1 - logits_2).pow(2).mean(dim=1).mean()))
    expected_l2 = torch.linalg.vector_norm(logits_1 - logits_2, ord=2, dim=1).mean() / 2
    assert l2 == pytest.approx(float(expected_l2))


def test_weighted_lcr_loss_applies_sample_weights():
    config = FPSConfig(
        num_classes=2,
        feature_dim=1,
        device="cpu",
        base_lr=0.001,
        lcr_loss="mse",
    )
    logits_1 = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    logits_2 = torch.tensor([[1.0, 1.0], [1.0, 5.0]])
    prob_1 = torch.softmax(logits_1, dim=1)
    prob_2 = torch.softmax(logits_2, dim=1)
    sample_weight = torch.tensor([0.25, 1.75])

    loss = _lcr_loss(config, logits_1, logits_2, prob_1, prob_2, sample_weight=sample_weight)
    expected = ((logits_1 - logits_2).pow(2).mean(dim=1) * sample_weight).mean()

    assert loss == pytest.approx(float(expected))


def test_lcr_sample_weight_modes():
    config = FPSConfig(
        num_classes=2,
        feature_dim=1,
        device="cpu",
        base_lr=0.001,
        lcr_sample_weight="density",
    )
    weight_tgt = torch.tensor([0.5, 1.5])
    mask = torch.tensor([0.25, 1.75])

    torch.testing.assert_close(_lcr_sample_weight(config, weight_tgt, mask), weight_tgt)
    config = FPSConfig(
        num_classes=2,
        feature_dim=1,
        device="cpu",
        base_lr=0.001,
        lcr_sample_weight="margin",
    )
    torch.testing.assert_close(_lcr_sample_weight(config, weight_tgt, mask), mask)
    config = FPSConfig(
        num_classes=2,
        feature_dim=1,
        device="cpu",
        base_lr=0.001,
        lcr_sample_weight="density_margin",
    )
    torch.testing.assert_close(_lcr_sample_weight(config, weight_tgt, mask), weight_tgt * mask)


def test_margin_sigmoid_uses_quantile_and_boundary_weight():
    config = FPSConfig(num_classes=2, feature_dim=1, device="cpu", base_lr=0.001)
    margins = torch.tensor([0.0, 1.0, 2.0, 3.0])

    weights_1, weights_2 = _convert_margin_to_weight_pair(
        config, margins, margins, q_used=config.margin_quantile
    )

    theta = torch.quantile(margins, 0.15)
    bias = np.log((1.0 - 0.2) / 0.2)
    expected = torch.sigmoid((margins - theta) / 0.2 - bias)
    torch.testing.assert_close(weights_1, expected)
    torch.testing.assert_close(weights_2, expected)


def test_margin_conversion_public_modes():
    margins = torch.tensor([0.0, 1.0, 2.0, 3.0])
    binary_cfg = FPSConfig(
        num_classes=2,
        feature_dim=1,
        device="cpu",
        base_lr=0.001,
        margin_convert_mode="quantile_binary",
    )
    binary_1, binary_2 = _convert_margin_to_weight_pair(binary_cfg, margins, margins, q_used=0.5)

    torch.testing.assert_close(binary_1, torch.tensor([0.0, 0.0, 1.0, 1.0]))
    torch.testing.assert_close(binary_2, torch.tensor([0.0, 0.0, 1.0, 1.0]))

    distance_cfg = FPSConfig(
        num_classes=2,
        feature_dim=1,
        device="cpu",
        base_lr=0.001,
        margin_weight_type="distance_gap",
        margin_convert_mode="by_type",
    )
    distance_1, distance_2 = _convert_margin_to_weight_pair(
        distance_cfg, margins, margins, q_used=0.5
    )

    torch.testing.assert_close(distance_1, margins / 3.0)
    torch.testing.assert_close(distance_2, margins / 3.0)


def test_cross_norm_uses_weighted_source_target_domain_stats():
    features = FeatureSet(
        src_features=np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
        src_labels=np.array([0, 1]),
        entropy_features=np.array([[4.0, 5.0], [6.0, 7.0]], dtype=np.float32),
        cr_features_1=np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32),
        cr_features_2=np.array([[500.0, 600.0], [700.0, 800.0]], dtype=np.float32),
        eval_features=np.array([[900.0, 1000.0]], dtype=np.float32),
        eval_labels=np.array([1]),
    )
    tensors = features.as_tensors(device="cpu")
    config = FPSConfig(
        num_classes=2,
        feature_dim=2,
        device="cpu",
        base_lr=0.001,
        normalize="cross_norm",
        cross_norm_scale=1.0,
        cross_norm_target_weight=0.75,
    )

    normalized = _normalize(tensors, config)

    src_mean = features.src_features.mean(axis=0, keepdims=True)
    tgt_mean = features.entropy_features.mean(axis=0, keepdims=True)
    tgt_weight = 0.75
    src_weight = 1.0 - tgt_weight
    mean = src_weight * src_mean + tgt_weight * tgt_mean
    src_var = features.src_features.var(axis=0, keepdims=True, ddof=0)
    tgt_var = features.entropy_features.var(axis=0, keepdims=True, ddof=0)
    var = (
        src_weight * (src_var + (src_mean - mean) ** 2)
        + tgt_weight * (tgt_var + (tgt_mean - mean) ** 2)
    )
    std = np.sqrt(var)
    expected_src = (features.src_features - mean) / std
    expected_eval = (features.eval_features - mean) / std
    np.testing.assert_allclose(normalized.src_features.numpy(), expected_src, rtol=1e-6)
    np.testing.assert_allclose(normalized.eval_features.numpy(), expected_eval, rtol=1e-6)


def test_self_norm_uses_source_and_target_scale_fields():
    features = FeatureSet(
        src_features=np.array([[0.0, 2.0], [4.0, 6.0]], dtype=np.float32),
        src_labels=np.array([0, 1]),
        entropy_features=np.array([[10.0, 12.0], [14.0, 20.0]], dtype=np.float32),
        cr_features_1=np.array([[2.0, 4.0], [6.0, 12.0]], dtype=np.float32),
        cr_features_2=np.array([[4.0, 6.0], [8.0, 14.0]], dtype=np.float32),
        eval_features=np.array([[12.0, 16.0]], dtype=np.float32),
        eval_labels=np.array([1]),
    )
    tensors = features.as_tensors(device="cpu")
    config = FPSConfig(
        num_classes=2,
        feature_dim=2,
        device="cpu",
        base_lr=0.001,
        normalize="self_norm",
        self_norm_scale_src=2.0,
        self_norm_scale_tgt=4.0,
    )

    normalized = _normalize(tensors, config)

    src_mean = features.src_features.mean(axis=0, keepdims=True)
    src_std = features.src_features.std(axis=0, keepdims=True, ddof=0) / 2.0
    tgt_mean = features.entropy_features.mean(axis=0, keepdims=True)
    tgt_std = features.entropy_features.std(axis=0, keepdims=True, ddof=0) / 4.0
    np.testing.assert_allclose(
        normalized.src_features.numpy(),
        (features.src_features - src_mean) / src_std,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        normalized.eval_features.numpy(),
        (features.eval_features - tgt_mean) / tgt_std,
        rtol=1e-6,
    )


def test_geom_alpha_beta_adjusts_target_views_by_class_centers():
    features = FeatureSet(
        src_features=np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32),
        src_labels=np.array([0, 1]),
        entropy_features=np.array([[3.0, 0.0], [16.0, 0.0]], dtype=np.float32),
        entropy_labels=np.array([0, 1]),
        cr_features_1=np.array([[3.0, 0.0], [16.0, 0.0]], dtype=np.float32),
        cr_features_2=np.array([[4.0, 0.0], [18.0, 0.0]], dtype=np.float32),
        cr_labels=np.array([0, 1]),
        eval_features=np.array([[2.0, 0.0], [14.0, 0.0]], dtype=np.float32),
        eval_labels=np.array([0, 1]),
    )
    config = FPSConfig(
        num_classes=2,
        feature_dim=2,
        device="cpu",
        base_lr=0.001,
        use_geom_test=True,
        geom_alpha=0.5,
        geom_beta=0.25,
    )

    adjusted = _apply_geom_alpha_beta(features.as_tensors(device="cpu"), config)

    torch.testing.assert_close(
        adjusted.cr_features_1,
        torch.tensor([[1.0, 0.0], [12.0, 0.0]]),
    )
    torch.testing.assert_close(
        adjusted.eval_features,
        torch.tensor([[0.5, 0.0], [11.0, 0.0]]),
    )


def test_evaluate_binary_metrics_when_multi_class_is_false():
    class BinaryModel(torch.nn.Module):
        def forward(self, features, use_correction=False):
            return torch.softmax(features, dim=1), features, None

    features = FeatureSet(
        src_features=np.zeros((2, 2), dtype=np.float32),
        src_labels=np.array([0, 1]),
        entropy_features=np.zeros((2, 2), dtype=np.float32),
        cr_features_1=np.zeros((2, 2), dtype=np.float32),
        cr_features_2=np.zeros((2, 2), dtype=np.float32),
        eval_features=np.array([[3.0, 0.0], [0.0, 3.0], [2.0, 0.0], [0.0, 2.0]]),
        eval_labels=np.array([0, 1, 0, 1]),
    )
    config = FPSConfig(
        num_classes=2,
        feature_dim=2,
        device="cpu",
        base_lr=0.001,
        multi_class=False,
    )

    metrics, predictions, labels = evaluate(BinaryModel(), features.as_tensors(device="cpu"), config)

    assert labels.tolist() == [0, 1, 0, 1]
    assert predictions.shape == (4, 2)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert "class_wise_acc" not in metrics


def test_public_class_accuracy_and_auc_helpers():
    labels = np.array([0, 1, 0, 1])
    prob = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.7, 0.3],
            [0.1, 0.9],
        ]
    )

    class_acc, overall = calculate_class_accuracy(labels, prob, class_num=3)

    assert overall == pytest.approx(1.0)
    assert class_acc == {0: 1.0, 1: 1.0, 2: 0.0}
    assert calculate_auc(labels, prob) == pytest.approx(1.0)


def test_sectioned_training_config_flattens_for_internal_api():
    config = flatten_training_config(
        {
            "io": {
                "feature_bank": "bank.h5",
                "source_domain": "Art",
                "target_domain": "Product",
                "feature_transform": "sqrt",
                "num_classes": 2,
                "device": "cpu",
            },
            "views": {"src": {"key": "src_clean", "combine": "stack"}},
            "optimization": {
                "optimizer": "adamw",
                "base_lr": 0.001,
                "weight_decay": 0.01,
                "lr_schedule": "constant",
            },
            "schedule": {"iter_num": 2},
            "normalization": {"normalize": "none"},
            "losses": {"lcr_loss": "mse"},
            "eval": {"progress": False},
        }
    )

    assert config["feature_bank"] == "bank.h5"
    assert config["views"]["src"]["key"] == "src_clean"
    assert config["optimizer"] == "adamw"
    assert config["lr_schedule"] == "constant"


def test_flat_training_config_is_rejected():
    with pytest.raises(ValueError, match="sectioned schema"):
        flatten_training_config({"feature_bank": "bank.h5", "base_lr": 0.001})


def test_benchmark_training_yaml_uses_sectioned_schema():
    paths = sorted(Path("configs/training").rglob("*.yaml"))
    assert paths
    sections = {"io", "views", "optimization", "schedule", "normalization", "losses", "eval"}
    removed_top_level = {
        "feature_bank",
        "source_domain",
        "target_domain",
        "feature_transform",
        "num_classes",
        "device",
        "seed",
        "base_lr",
        "momentum",
        "alpha",
        "beta",
        "alpha_0",
        "beta_0",
        "schedule_tau",
        "lambda_lcr",
    }

    for path in paths:
        data = yaml.safe_load(path.read_text())
        assert sections.issubset(data), path
        assert set(data).issubset(sections), path
        assert removed_top_level.isdisjoint(data), path
        assert data["io"]["feature_bank"], path
        assert data["optimization"]["base_lr"] is not None, path
        assert data["optimization"]["optimizer"] in {"sgd", "adamw"}, path
        assert data["optimization"]["lr_schedule"] in {"linear_step", "constant", "cosine"}, path
        assert float(data["optimization"]["momentum"]) >= 0, path
        assert float(data["optimization"]["weight_decay"]) >= 0, path
        assert (
            isinstance(data["schedule"]["iter_num"], int)
            and data["schedule"]["iter_num"] > 0
        ), path
        assert float(data["normalization"]["cross_norm_scale"]) > 0, path
        assert float(data["losses"]["lambda_lcr"]) >= 0, path
        assert float(data["schedule"]["schedule_tau"]) > 0, path
        assert data["losses"]["lcr_loss"] in {"mse", "l2"}, path
        assert int(data["losses"]["margin_start_step"]) == 100, path
        assert data["losses"]["sample_entropy_type"] in {
            "shannon",
            "tsallis",
            "adaptive_temp_shannon",
        }, path
        views = data["views"]
        assert views["src"]["key"], path
        assert views["entropy"]["key"], path
        assert views["cr"]["view1"]["key"], path
        assert views["cr"]["view2"]["key"], path
        assert views["eval"]["key"], path
        assert views["src"]["combine"] in {"stack", "mean"}, path
        assert views["entropy"]["combine"] in {"stack", "mean"}, path
        assert views["cr"]["view1"]["combine"] in {"stack", "mean"}, path
        assert views["cr"]["view2"]["combine"] in {"stack", "mean"}, path
        assert views["eval"]["combine"] in {"stack", "mean"}, path


def test_configs_snapshot_before_restructure_exists():
    snapshot = Path("bak/configs_snapshot_before_restructure")

    assert snapshot.exists()
    assert (snapshot / "training").exists()
    assert (snapshot / "datasets").exists()


def test_cli_feature_transform_uses_yaml_unless_overridden():
    assert (
        _resolve_feature_transform(
            SimpleNamespace(feature_transform=None), {"feature_transform": "sqrt"}
        )
        == "sqrt"
    )
    assert (
        _resolve_feature_transform(
            SimpleNamespace(feature_transform="none"), {"feature_transform": "sqrt"}
        )
        == "none"
    )
