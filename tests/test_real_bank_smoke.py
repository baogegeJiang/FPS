import json
from pathlib import Path

import pytest
import yaml

from fps_uda.cli import main


FIXTURE = Path("tests/fixtures/office31_amazon_to_webcam_vit_smoke.h5")
BASE_CONFIG = Path("configs/examples/office31_amazon_to_webcam_vit_packaged_h5.yaml")


@pytest.mark.real_h5
def test_office31_amazon_to_webcam_vit_real_h5_smoke(tmp_path):
    assert FIXTURE.exists(), "packaged real-H5 smoke fixture is missing"

    config = yaml.safe_load(BASE_CONFIG.read_text())
    config["io"]["device"] = "cpu"
    config["schedule"]["iter_num"] = 2
    config["losses"]["pseudo_margin"] = False
    config["eval"]["eval_interval"] = 1
    config["eval"]["progress"] = False
    config_path = tmp_path / "office31_aw_vit_smoke.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    out = tmp_path / "run"

    assert main(["train", "--config", str(config_path), "--out", str(out)]) == 0

    metrics = json.loads((out / "metrics.json").read_text())
    history = (out / "history.csv").read_text()
    assert metrics["best_metric"] == "acc"
    assert metrics["best_score"] is not None
    assert metrics["best_cwc"] is not None
    assert metrics["has_best_cwc_predictions"] is True
    assert (out / "best_cwc_predictions.npy").exists()
    assert (out / "best_cwc_labels.npy").exists()
    assert metrics["config"]["feature_dim"] == 768
    assert metrics["config"]["num_classes"] == 31
    for field in ("loss_sup", "loss_consistency", "loss_lcr", "lr"):
        assert field in history
