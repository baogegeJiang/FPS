"""Train with an extra Python loss term.

This example uses in-memory synthetic features so it can run without a feature
bank. Custom losses receive a dict-like context and can read features, logits,
probabilities, schedules, and step number by key.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

if __package__ is None or __package__ == "":
    repo_src = Path(__file__).resolve().parents[1] / "src"
    if repo_src.exists():
        sys.path.insert(0, str(repo_src))

from fps_uda import FPSConfig, FeatureSet, LossContext, LossOutput, train_fps


def make_features(seed: int = 0) -> FeatureSet:
    rng = np.random.default_rng(seed)
    src_labels = np.array([0, 1] * 8)
    target_labels = np.array([0, 1] * 10)
    return FeatureSet(
        src_features=rng.normal(size=(16, 6)).astype("float32"),
        src_labels=src_labels,
        entropy_features=rng.normal(size=(20, 6)).astype("float32"),
        entropy_labels=target_labels,
        cr_features_1=rng.normal(size=(20, 6)).astype("float32"),
        cr_features_2=(rng.normal(size=(20, 6)) + 0.2).astype("float32"),
        cr_labels=target_labels,
        eval_features=rng.normal(size=(10, 6)).astype("float32"),
        eval_labels=np.array([0, 1] * 5),
    )


class WarmupEntropyLogitPenalty:
    def __init__(self, *, start_step: int = 2, weight: float = 0.01):
        self.start_step = int(start_step)
        self.weight = float(weight)

    def __call__(self, ctx: LossContext) -> LossOutput:
        logits = ctx["outputs"]["entropy.logits"]
        if ctx["step"] < self.start_step:
            value = logits.sum() * 0.0
        else:
            value = self.weight * logits.pow(2).mean()
        return LossOutput(
            name="warmup_entropy_logit_penalty",
            value=value,
            logs={"loss_warmup_entropy_logit_penalty": float(value.detach().cpu())},
        )


def main() -> None:
    torch.set_num_threads(1)
    config = FPSConfig(
        num_classes=2,
        feature_dim=6,
        device="cpu",
        base_lr=0.01,
        iter_num=4,
        eval_interval=1,
        normalize="none",
        progress=False,
        pseudo_margin=False,
    )
    result = train_fps(
        make_features(),
        config,
        extra_loss_terms=[WarmupEntropyLogitPenalty(start_step=2, weight=0.01)],
    )
    for row in result.history:
        print(
            "step={step:.0f} acc={acc:.3f} custom={loss_warmup_entropy_logit_penalty:.8e}".format(
                **row
            )
        )


if __name__ == "__main__":
    main()
