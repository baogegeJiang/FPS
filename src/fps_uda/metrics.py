from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import auc, roc_curve


def top1_accuracy(prob: np.ndarray, labels: np.ndarray) -> float:
    pred = prob.argmax(axis=-1)
    return float(np.mean(pred == labels))


def top1_classwise_accuracy(prob: np.ndarray, labels: np.ndarray) -> float:
    pred = prob.argmax(axis=-1)
    correct = pred == labels
    values = [correct[labels == i].mean() for i in np.unique(labels) if np.any(labels == i)]
    return float(np.mean(values)) if values else 0.0


def calculate_class_accuracy(
    y_true: np.ndarray, y_pred_prob: np.ndarray, class_num: int
) -> Tuple[Dict[int, float], float]:
    y_pred = np.argmax(y_pred_prob, axis=1)
    overall_accuracy = float(np.mean(y_pred == y_true))
    class_accuracies: Dict[int, float] = {}
    for class_idx in range(class_num):
        class_mask = y_true == class_idx
        if np.any(class_mask):
            class_accuracies[class_idx] = float(np.mean(y_pred[class_mask] == y_true[class_mask]))
        else:
            class_accuracies[class_idx] = 0.0
    return class_accuracies, overall_accuracy


def binary_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray) -> Dict[str, float]:
    y_pred = y_pred_prob.argmax(axis=-1)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 15) -> float:
    y_pred = prob.argmax(axis=1)
    confidences = prob.max(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        acc_bin = (y_pred[mask] == y_true[mask]).mean()
        conf_bin = confidences[mask].mean()
        ece += mask.mean() * abs(acc_bin - conf_bin)
    return float(ece)


def calculate_auc(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
    y_pred = y_pred_prob.argmax(axis=-1)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return float(auc(fpr, tpr))

