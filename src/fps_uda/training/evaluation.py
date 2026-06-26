from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import f1_score

from fps_uda.data import TensorFeatureSet
from fps_uda.metrics import (
    binary_metrics,
    expected_calibration_error,
    top1_accuracy,
    top1_classwise_accuracy,
)
from fps_uda.training.config import FPSConfig


def class_r_mean(labels: np.ndarray, features: np.ndarray) -> float:
    values = []
    for label in np.unique(labels):
        class_features = features[labels == label]
        if len(class_features) == 0:
            continue
        values.append((class_features.std(axis=0) ** 2).sum() * len(class_features) / len(features))
    return float(np.sqrt(np.sum(values))) if values else 0.0


def evaluate(model: torch.nn.Module, features: TensorFeatureSet, cfg: FPSConfig):
    eval_features = features.eval_features
    eval_labels = features.eval_labels
    if eval_features is None:
        return {}, None, None
    prob, logits, _ = model(eval_features, use_correction=cfg.use_correction)
    prob_np = torch.softmax(logits, dim=1).detach().cpu().numpy()
    if eval_labels is None:
        return {}, prob_np, None
    labels_np = eval_labels.detach().cpu().numpy()
    metrics = {"acc": top1_accuracy(prob_np, labels_np)}
    if cfg.multi_class:
        pred = prob_np.argmax(axis=-1)
        metrics["macro_f1"] = float(f1_score(labels_np, pred, average="macro"))
        metrics["class_wise_acc"] = top1_classwise_accuracy(prob_np, labels_np)
        metrics["ece"] = expected_calibration_error(labels_np, prob_np)
    else:
        metrics.update(binary_metrics(labels_np, prob_np))
    return metrics, prob_np, labels_np


def view_accuracy(
    model: torch.nn.Module,
    view_features: Optional[torch.Tensor],
    view_labels: Optional[torch.Tensor],
    cfg: FPSConfig,
) -> Optional[float]:
    if view_features is None or view_labels is None:
        return None
    _, logits, _ = model(view_features, use_correction=cfg.use_correction)
    prob_np = torch.softmax(logits, dim=1).detach().cpu().numpy()
    labels_np = view_labels.detach().cpu().numpy()
    return top1_accuracy(prob_np, labels_np)


def write_summary_csv(history: list, output_dir: Path) -> None:
    if not history:
        return
    fieldnames = sorted({key for row in history for key in row.keys()})
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
