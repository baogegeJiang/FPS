from pathlib import Path

import numpy as np
import yaml

import scripts.search_fps_hyperparams as search
from fps_uda.io import load_yaml_config, save_feature_bank_h5
from fps_uda.training import TrainingResult


def _separated(labels, scale=5.0):
    labels = np.asarray(labels, dtype="float32")
    return np.stack([labels * scale, labels * 0.1], axis=1).astype("float32")


def _noisy(labels):
    labels = np.asarray(labels, dtype="float32")
    return np.stack([labels * 0.1, labels * 0.1], axis=1).astype("float32")


def _save_search_bank(path: Path):
    src_labels = np.array([0, 2, 0, 2], dtype="int64")
    tgt_labels = np.array([0, 1, 2, 0, 1, 2], dtype="int64")
    src_good = _separated(src_labels)
    tgt_good = _separated(tgt_labels)
    save_feature_bank_h5(
        {
            "source": {
                "label": src_labels,
                "views": {
                    "src_good_orig_clean": {
                        "feature": src_good,
                        "attrs": {"pooling": "clean", "flip": "orig"},
                    },
                    "src_second_orig_clean": {
                        "feature": src_good + 0.01,
                        "attrs": {"pooling": "clean", "flip": "orig"},
                    },
                    "src_bad_hflip_clean": {
                        "feature": _noisy(src_labels),
                        "attrs": {"pooling": "clean", "flip": "hflip"},
                    },
                },
            },
            "target": {
                "label": tgt_labels,
                "views": {
                    "tgt_good_orig_clean": {
                        "feature": tgt_good,
                        "attrs": {"pooling": "clean", "flip": "orig"},
                    },
                    "tgt_good_orig_pool_a": {
                        "feature": tgt_good + 0.02,
                        "attrs": {"pooling": "pool_a", "flip": "orig"},
                    },
                    "tgt_good_orig_pool_b": {
                        "feature": tgt_good - 0.02,
                        "attrs": {"pooling": "pool_b", "flip": "orig"},
                    },
                    "tgt_bad_hflip_clean": {
                        "feature": _noisy(tgt_labels),
                        "attrs": {"pooling": "clean", "flip": "hflip"},
                    },
                },
            },
        },
        str(path),
        metadata={"backbone_backend": "timm", "backbone_name": "tiny_vit"},
    )


def _write_tiny_search_space(path: Path):
    data = {
        "basic": {
            "io": {"feature_transform": "none", "device": "cpu"},
            "views": {},
            "optimization": {
                "optimizer": "sgd",
                "base_lr": 0.01,
                "momentum": 0.9,
                "nesterov": False,
                "weight_decay": 0.0,
                "lr_schedule": "linear_step",
                "min_lr": 0.0,
            },
            "schedule": {
                "seed": 0,
                "iter_num": 2,
                "alpha": 0.1,
                "beta": 0.1,
                "alpha_0": 0.1,
                "beta_0": 0.1,
                "schedule_tau": 1000,
                "dynamic_parameters": True,
            },
            "normalization": {
                "normalize": "cross_norm",
                "cross_norm_scale": 2.5,
                "cross_norm_target_weight": 0.5,
                "self_norm_scale_src": 1,
                "self_norm_scale_tgt": 1,
            },
            "losses": {
                "use_consistency_loss": True,
                "use_lse": True,
                "use_lce": True,
                "use_lcr": True,
                "lambda_lcr": 0.1,
                "lcr_loss": "mse",
                "lcr_sample_weight": "paper",
                "pseudo_margin": True,
                "margin_start_step": 100,
                "sparse_weight_a": 5,
                "sample_entropy_type": "shannon",
                "margin_quantile": 0.01,
                "margin_weight_type": "normalized_logit_gap",
                "margin_convert_mode": "quantile_sigmoid",
                "margin_sigmoid_tau": 0.2,
                "margin_sigmoid_boundary_weight": 0.2,
            },
            "eval": {"multi_class": True, "eval_interval": 1, "progress": False},
        },
        "lr_candidates": [
            {
                "feature_transform": "none",
                "base_lr": 0.01,
                "optimizer": "sgd",
                "lr_schedule": "linear_step",
            }
        ],
        "lambda_lcr_grid": [0.1],
        "margin_candidates": [
            {
                "pseudo_margin": True,
                "margin_quantile": 0.01,
                "margin_weight_type": "normalized_logit_gap",
                "margin_convert_mode": "quantile_sigmoid",
                "margin_sigmoid_tau": 0.2,
                "margin_sigmoid_boundary_weight": 0.2,
                "margin_start_step": 100,
            }
        ],
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_search_view_selection_and_num_classes(tmp_path):
    bank = tmp_path / "bank.h5"
    _save_search_bank(bank)
    info = search._read_bank_for_selection(str(bank), "source", "target")

    class_info = search.infer_num_classes(info)
    selected = search.select_views(info, source_domain="source", target_domain="target")

    assert class_info["num_classes"] == 3
    assert class_info["unique_class_counts"] == {"source": 2, "target": 3}
    assert set(selected["src"]["key"].split(",")) == {
        "src_good_orig_clean",
        "src_second_orig_clean",
    }
    assert selected["entropy"]["key"] == "tgt_good_orig_clean"
    assert selected["cr"]["view1"]["key"] == "tgt_good_orig_pool_a"
    assert selected["cr"]["view2"]["key"] == "tgt_good_orig_pool_b"
    assert selected["lcr_enabled"] is True


def test_search_script_writes_best_yaml_and_summary(monkeypatch, tmp_path):
    bank = tmp_path / "bank.h5"
    space = tmp_path / "space.yaml"
    out = tmp_path / "search"
    _save_search_bank(bank)
    _write_tiny_search_space(space)
    calls = []

    def fake_train_fps(features, config):
        calls.append(config.to_dict())
        score = (
            float(config.base_lr) * 100.0
            + float(config.beta)
            + float(config.alpha) * 0.1
            + float(config.lambda_lcr or 0.0) * 0.01
        )
        return TrainingResult(
            best_metric="acc",
            best_score=score,
            best_cwc=score / 2.0,
            best_cwc_step=0.0,
            history=[{"acc": score, "class_wise_acc": score / 2.0}],
            config=config.to_dict(),
            early_stopped=True,
            early_stop_step=0.0,
        )

    monkeypatch.setattr(search, "train_fps", fake_train_fps)
    args = search.build_parser().parse_args(
        [
            "--feature-bank",
            str(bank),
            "--source-domain",
            "source",
            "--target-domain",
            "target",
            "--out",
            str(out),
            "--device",
            "cpu",
            "--rounds",
            "2",
            "--grid-min",
            "0.05",
            "--grid-max",
            "0.10",
            "--grid-step",
            "0.05",
            "--patience",
            "1",
            "--search-space",
            str(space),
            "--no-progress",
        ]
    )

    summary = search.run_search(args)

    assert summary["num_trials"] == 27
    assert [trial["stage"] for trial in summary["trials"][:5]] == [
        "source_only",
        "target_only",
        "lr",
        "beta_tied",
        "beta_tied",
    ]
    assert summary["trials"][0]["beta"] == 1.0
    assert summary["trials"][0]["beta_0"] == 1.0
    assert summary["trials"][0]["baseline"] is True
    assert summary["trials"][0]["use_consistency_loss"] is False
    assert summary["trials"][1]["source_domain"] == "target"
    assert summary["trials"][1]["src_view"] == "tgt_good_orig_clean"
    round2_stages = [trial["stage"] for trial in summary["trials"] if trial["round"] == 2]
    assert "beta_tied" not in round2_stages
    assert "alpha_tied" not in round2_stages
    assert calls[0]["num_classes"] == 3
    assert calls[0]["early_stop_patience"] == 1
    best_config = load_yaml_config(str(out / "best.yaml"))
    assert best_config["num_classes"] == 3
    assert best_config["feature_bank"] == str(bank)
    assert best_config["progress"] is True
    assert (out / "selected_views.yaml").exists()
    assert (out / "search_summary.csv").exists()
    assert (out / "search_summary.json").exists()
