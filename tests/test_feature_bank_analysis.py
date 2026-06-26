import csv
import json

import h5py
import numpy as np
import pytest
import yaml

from fps_uda.analysis.feature_bank import analyze_feature_bank, compute_domain_view_metrics
from fps_uda.cli import main
from fps_uda.io import load_feature_bank_h5, save_feature_bank_h5


def _good_source_features():
    return np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
        ],
        dtype="float32",
    )


def _compact_features():
    return np.array(
        [
            [1.0, 0.05],
            [1.0, -0.05],
            [-1.0, 0.05],
            [-1.0, -0.05],
        ],
        dtype="float32",
    )


def _noisy_features():
    return np.array(
        [
            [1.0, 0.0],
            [-0.2, 0.4],
            [0.2, -0.4],
            [-1.0, 0.0],
        ],
        dtype="float32",
    )


def _analysis_bank(path):
    labels = np.array([0, 0, 1, 1], dtype="int64")
    compact = _compact_features()
    save_feature_bank_h5(
        {
            "Source": {
                "label": labels,
                "views": {
                    "geom_orig_clean": _good_source_features(),
                    "noisy_orig_clean": _noisy_features(),
                },
            },
            "Target": {
                "label": labels,
                "views": {
                    "geom_orig_clean": compact,
                    "geom_orig_pool_a": compact + np.array([0.01, 0.0], dtype="float32"),
                    "geom_orig_pool_b": compact + np.array([-0.01, 0.0], dtype="float32"),
                    "noisy_orig_clean": _noisy_features(),
                    "noisy_orig_pool_a": _noisy_features(),
                    "noisy_orig_pool_b": _noisy_features(),
                },
            },
        },
        str(path),
    )


def test_domain_view_metrics_have_expected_center_geometry():
    row = compute_domain_view_metrics(
        _good_source_features(),
        np.array([0, 0, 1, 1]),
        domain="Source",
        view_key="geom_orig_clean",
    )

    assert row["intra_mean"] == pytest.approx(0.0)
    assert row["inter_mean"] == pytest.approx(2.0)
    assert row["inter_min"] == pytest.approx(2.0)
    assert row["nearest_center_margin_mean"] == pytest.approx(2.0)
    assert row["domain_quality_score"] > 1e6


def test_analyze_feature_bank_ranks_compact_views_and_recommends_siblings(tmp_path):
    bank = tmp_path / "bank.h5"
    out = tmp_path / "analysis"
    _analysis_bank(bank)

    summary = analyze_feature_bank(
        str(bank),
        str(out),
        source_domain="Source",
        target_domain="Target",
        top_k_domain=4,
        top_k_task=3,
        make_plots=False,
        progress=False,
    )

    assert (out / "domain_view_metrics.csv").exists()
    assert (out / "task_view_metrics.csv").exists()
    assert (out / "recommended_views.yaml").exists()
    assert (out / "summary.json").exists()
    views = summary["recommended"]["views"]
    assert views["src"]["key"] == "geom_orig_clean"
    assert views["entropy"]["key"] == "geom_orig_clean"
    assert views["cr"]["view1"]["key"] == "geom_orig_pool_a"
    assert views["cr"]["view2"]["key"] == "geom_orig_pool_b"
    assert views["eval"]["key"] == "geom_orig_clean"

    with (out / "domain_view_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    source_rows = [row for row in rows if row["domain"] == "Source"]
    best_source = max(source_rows, key=lambda row: float(row["domain_quality_score"]))
    assert best_source["view_key"] == "geom_orig_clean"


def test_cli_analyze_feature_bank_outputs_trainable_yaml(tmp_path):
    bank = tmp_path / "bank.h5"
    out = tmp_path / "analysis"
    _analysis_bank(bank)

    assert (
        main(
            [
                "analyze-feature-bank",
                "--feature-bank",
                str(bank),
                "--out",
                str(out),
                "--source-domain",
                "Source",
                "--target-domain",
                "Target",
                "--top-k-domain",
                "4",
                "--top-k-task",
                "2",
                "--no-plots",
                "--no-progress",
            ]
        )
        == 0
    )

    recommended = yaml.safe_load((out / "recommended_views.yaml").read_text())
    loaded = load_feature_bank_h5(
        str(bank),
        source_domain=recommended["source_domain"],
        target_domain=recommended["target_domain"],
        src_view=recommended["views"]["src"]["key"],
        entropy_view=recommended["views"]["entropy"]["key"],
        cr_view1=recommended["views"]["cr"]["view1"]["key"],
        cr_view2=recommended["views"]["cr"]["view2"]["key"],
        eval_view=recommended["views"]["eval"]["key"],
    )
    assert loaded.src_features.shape == (4, 2)
    assert loaded.cr_features_2.shape == (4, 2)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["task_view_count"] >= 1


def test_analyze_feature_bank_rejects_label_feature_mismatch(tmp_path):
    bank = tmp_path / "bad.h5"
    with h5py.File(bank, "w") as h5:
        domains = h5.create_group("domains")
        domain = domains.create_group("A")
        domain.create_dataset("label", data=np.array([0]))
        views = domain.create_group("views")
        view = views.create_group("clean")
        view.create_dataset("feature", data=np.ones((2, 3), dtype="float32"))

    with pytest.raises(ValueError, match="2 features but 1 labels"):
        analyze_feature_bank(str(bank), str(tmp_path / "out"), make_plots=False, progress=False)
