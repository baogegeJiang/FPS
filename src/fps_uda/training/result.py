from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


@dataclass
class TrainingResult:
    best_metric: Optional[str]
    best_score: Optional[float]
    best_cwc: Optional[float] = None
    best_cwc_step: Optional[float] = None
    history: List[Dict[str, float]] = field(default_factory=list)
    predictions: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    final_model_state: Optional[Dict[str, torch.Tensor]] = None
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_metric": self.best_metric,
            "best_score": self.best_score,
            "best_cwc": self.best_cwc,
            "best_cwc_step": self.best_cwc_step,
            "history": self.history,
            "has_predictions": self.predictions is not None,
            "has_labels": self.labels is not None,
            "config": self.config,
        }

    def save(self, output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        with (out / "history.json").open("w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        if self.predictions is not None:
            np.save(out / "predictions.npy", self.predictions)
        if self.labels is not None:
            np.save(out / "labels.npy", self.labels)
        if self.final_model_state is not None:
            torch.save(self.final_model_state, out / "final_model.pt")
