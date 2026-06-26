import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from fps_uda.cli import main
from fps_uda.io import save_feature_bank_h5


def _save_bank(path, *, labels: bool = True, multi: bool = False):
    rng = np.random.default_rng(10)
    source_views = {
        "src_clean": rng.normal(size=(8, 3)).astype("float32"),
    }
    target_views = {
        "clean": rng.normal(size=(8, 3)).astype("float32"),
        "pool_a": rng.normal(size=(8, 3)).astype("float32"),
        "pool_b": rng.normal(size=(8, 3)).astype("float32"),
    }
    if multi:
        source_views["src_clean_b"] = rng.normal(size=(8, 3)).astype("float32")
        target_views.update(
            {
                "clean_b": rng.normal(size=(8, 3)).astype("float32"),
                "pool_a_b": rng.normal(size=(8, 3)).astype("float32"),
                "pool_b_b": rng.normal(size=(8, 3)).astype("float32"),
            }
        )
    target = {"views": target_views}
    if labels:
        target["label"] = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    save_feature_bank_h5(
        {
            "Art": {
                "label": np.array([0, 1, 0, 1, 0, 1, 0, 1]),
                "views": source_views,
            },
            "Product": target,
        },
        str(path),
    )


def _base_training_config(bank_path):
    return {
        "io": {
            "feature_bank": str(bank_path),
            "source_domain": "Art",
            "target_domain": "Product",
            "feature_transform": "none",
            "num_classes": 2,
            "device": "cpu",
        },
        "views": {
            "src": {"key": "src_clean", "combine": "stack"},
            "entropy": {"key": "clean", "combine": "mean"},
            "cr": {
                "view1": {"key": "pool_a", "combine": "stack"},
                "view2": {"key": "pool_b", "combine": "stack"},
            },
            "eval": {"key": "clean", "combine": "mean"},
        },
        "optimization": {
            "optimizer": "sgd",
            "base_lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0,
            "lr_schedule": "linear_step",
        },
        "schedule": {"iter_num": 2},
        "normalization": {},
        "losses": {"lcr_loss": "mse"},
        "eval": {"eval_interval": 1, "progress": False},
    }


def test_cli_train_from_view_role_yaml(tmp_path):
    bank_path = tmp_path / "bank.h5"
    _save_bank(bank_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_base_training_config(bank_path)))
    out = tmp_path / "run"

    assert (
        main(
            [
                "train",
                "--config",
                str(config_path),
                "--out",
                str(out),
                "--src-sample-ratio",
                "0.5",
                "--target-sample-ratio",
                "0.75",
                "--cross-norm-target-weight",
                "0.65",
                "--momentum",
                "0.85",
                "--schedule-tau",
                "7",
                "--ldelta-weight",
                "0.2",
                "--ldelta-decay-steps",
                "7",
                "--use-shift-constraint",
                "--margin-quantile",
                "0.2",
            ]
        )
        == 0
    )
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["best_metric"] == "acc"
    assert metrics["best_cwc"] is not None
    assert metrics["best_cwc_step"] is not None
    assert metrics["config"]["lcr_loss"] == "mse"
    assert metrics["config"]["src_sample_ratio"] == 0.5
    assert metrics["config"]["target_sample_ratio"] == 0.75
    assert metrics["config"]["cross_norm_target_weight"] == 0.65
    assert metrics["config"]["momentum"] == 0.85
    assert metrics["config"]["schedule_tau"] == 7
    assert metrics["config"]["use_correction"] is True
    assert metrics["config"]["use_shift_constraint"] is True
    assert metrics["config"]["ldelta_weight"] == 0.2
    assert metrics["config"]["ldelta_decay_steps"] == 7
    assert metrics["config"]["margin_quantile"] == 0.2


def test_cli_train_requires_feature_bank(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "io": {"device": "cpu", "num_classes": 2},
                "optimization": {"base_lr": 0.01},
                "schedule": {"iter_num": 2},
            }
        )
    )
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "train",
                "--config",
                str(config_path),
                "--out",
                str(tmp_path / "run"),
            ]
        )
    assert "--feature-bank is required" in str(exc.value)


def test_cli_train_rejects_flat_yaml(tmp_path):
    config_path = tmp_path / "flat.yaml"
    config_path.write_text(yaml.safe_dump({"device": "cpu", "iter_num": 2, "base_lr": 0.01}))

    with pytest.raises(ValueError, match="sectioned schema"):
        main(["train", "--config", str(config_path), "--out", str(tmp_path / "run")])


def test_cli_train_from_feature_bank_cli_roles(tmp_path):
    bank_path = tmp_path / "bank.h5"
    _save_bank(bank_path)
    config_path = tmp_path / "config.yaml"
    config = _base_training_config(bank_path)
    config["io"].pop("feature_bank")
    config["io"].pop("source_domain")
    config["io"].pop("target_domain")
    config.pop("views")
    config_path.write_text(yaml.safe_dump(config))
    out = tmp_path / "run_bank"

    assert (
        main(
            [
                "train",
                "--feature-bank",
                str(bank_path),
                "--source-domain",
                "Art",
                "--target-domain",
                "Product",
                "--src-view",
                "src_clean",
                "--entropy-view",
                "clean",
                "--cr-view1",
                "pool_a",
                "--cr-view2",
                "pool_b",
                "--eval-view",
                "clean",
                "--config",
                str(config_path),
                "--out",
                str(out),
                "--lcr-loss",
                "l2",
                "--optimizer",
                "adamw",
                "--lr-schedule",
                "constant",
                "--weight-decay",
                "0.01",
            ]
        )
        == 0
    )
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["best_metric"] == "acc"
    assert metrics["config"]["lcr_loss"] == "l2"
    assert metrics["config"]["optimizer"] == "adamw"
    assert metrics["config"]["lr_schedule"] == "constant"
    assert metrics["config"]["weight_decay"] == 0.01


def test_cli_train_from_feature_bank_with_unlabeled_target(tmp_path):
    bank_path = tmp_path / "bank_unlabeled.h5"
    _save_bank(bank_path, labels=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_base_training_config(bank_path)))
    out = tmp_path / "run_unlabeled"

    assert main(["train", "--config", str(config_path), "--out", str(out)]) == 0
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["best_metric"] is None
    assert metrics["best_cwc"] is None
    assert metrics["has_labels"] is False


def test_cli_train_from_feature_bank_with_multi_view_combine(tmp_path):
    bank_path = tmp_path / "bank_multi.h5"
    _save_bank(bank_path, multi=True)
    config_path = tmp_path / "config.yaml"
    config = _base_training_config(bank_path)
    config["views"] = {
        "src": {"key": ["src_clean", "src_clean_b"], "combine": "stack"},
        "entropy": {"key": ["clean", "clean_b"], "combine": "mean"},
        "cr": {
            "view1": {"key": ["pool_a", "pool_a_b"], "combine": "stack"},
            "view2": {"key": ["pool_b", "pool_b_b"], "combine": "stack"},
        },
        "eval": {"key": ["clean", "clean_b"], "combine": "mean"},
    }
    config_path.write_text(yaml.safe_dump(config))
    out = tmp_path / "run_bank_multi"

    assert main(["train", "--config", str(config_path), "--out", str(out)]) == 0
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["config"]["feature_dim"] == 3


def test_cli_extract_feature_bank_uses_yaml_backbone_with_overrides(monkeypatch, tmp_path):
    dataset_config = {
        "name": "mini_bank",
        "root_dir": "data/mini",
        "backbone": {
            "backend": "timm",
            "name": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
            "pretrained": True,
            "weights": None,
            "checkpoint": None,
            "in_features": 768,
            "kwargs": {"class_token": True, "global_pool": "token"},
            "pooling": {
                "feature_type": "token",
                "random_strategy": "token_channel_squared",
            },
        },
        "domains": [
            {
                "name": "A",
                "kind": "manifest",
                "image_root": "A",
                "manifest": "A/annotations.txt",
                "path_column": 0,
                "label_column": 1,
            }
        ],
        "feature_bank": {
            "views": [
                {
                    "key": "v",
                    "pad_to_square": True,
                    "resize_size": 224,
                    "input_size": 224,
                    "crop": "direct",
                    "flip": "orig",
                }
            ]
        },
    }
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(yaml.safe_dump(dataset_config))
    seen = {}

    def fake_build_backbone(config):
        seen["config"] = config
        return SimpleNamespace(config=config)

    def fake_extract_feature_bank_to_h5(backbone, config, output_path, **kwargs):
        seen["summary_args"] = {
            "backbone": backbone,
            "config": config,
            "output_path": output_path,
            **kwargs,
        }
        return {"output": output_path, "views_per_domain": 0}

    monkeypatch.setattr("fps_uda.cli.build_backbone", fake_build_backbone)
    monkeypatch.setattr(
        "fps_uda.cli.extract_feature_bank_to_h5",
        fake_extract_feature_bank_to_h5,
    )

    assert (
        main(
            [
                "extract-feature-bank",
                "--dataset-config",
                str(config_path),
                "--out",
                str(tmp_path / "bank.h5"),
                "--backend",
                "torchvision",
                "--backbone",
                "resnet101",
                "--weights",
                "IMAGENET1K_V1",
                "--checkpoint",
                "resnet.pt",
                "--no-pretrained",
                "--backbone-kw",
                "progress=false",
            ]
        )
        == 0
    )

    backbone_config = seen["config"]
    assert backbone_config.backend == "torchvision"
    assert backbone_config.name == "resnet101"
    assert backbone_config.pretrained is False
    assert backbone_config.weights == "IMAGENET1K_V1"
    assert backbone_config.checkpoint == "resnet.pt"
    assert backbone_config.kwargs["progress"] is False
    assert backbone_config.pooling.feature_type == "spatial"
    assert backbone_config.pooling.random_strategy == "spatial_shared"
    assert seen["summary_args"]["backbone_name"] == "resnet101"
