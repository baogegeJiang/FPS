from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


def _shape(value: Optional[ArrayLike]) -> Optional[tuple]:
    if value is None:
        return None
    return tuple(value.shape)


def _to_tensor(value: Optional[ArrayLike], *, dtype: torch.dtype) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().clone().to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


@dataclass
class TensorFeatureSet:
    """FeatureSet converted to Torch tensors."""

    src_features: torch.Tensor
    src_labels: torch.Tensor
    entropy_features: torch.Tensor
    cr_features_1: torch.Tensor
    cr_features_2: torch.Tensor
    entropy_labels: Optional[torch.Tensor] = None
    cr_labels: Optional[torch.Tensor] = None
    eval_features: Optional[torch.Tensor] = None
    eval_labels: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def feature_dim(self) -> int:
        return int(self.src_features.shape[1])

    def to(self, device: Union[str, torch.device]) -> "TensorFeatureSet":
        return TensorFeatureSet(
            src_features=self.src_features.to(device),
            src_labels=self.src_labels.to(device),
            entropy_features=self.entropy_features.to(device),
            entropy_labels=None
            if self.entropy_labels is None
            else self.entropy_labels.to(device),
            cr_features_1=self.cr_features_1.to(device),
            cr_features_2=self.cr_features_2.to(device),
            cr_labels=None if self.cr_labels is None else self.cr_labels.to(device),
            eval_features=None if self.eval_features is None else self.eval_features.to(device),
            eval_labels=None if self.eval_labels is None else self.eval_labels.to(device),
            metadata=dict(self.metadata),
        )


@dataclass
class FeatureSet:
    """In-memory feature tensors for FPS training.

    The public training roles are explicit:
    - src: supervised source features and labels.
    - entropy: target features for LSE/LCE entropy terms.
    - cr1/cr2: paired target views for CR/LCR.
    - eval: optional target evaluation view.
    """

    src_features: ArrayLike
    src_labels: ArrayLike
    entropy_features: ArrayLike
    cr_features_1: ArrayLike
    cr_features_2: ArrayLike
    entropy_labels: Optional[ArrayLike] = None
    cr_labels: Optional[ArrayLike] = None
    eval_features: Optional[ArrayLike] = None
    eval_labels: Optional[ArrayLike] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def feature_dim(self) -> int:
        self.validate()
        return int(self.src_features.shape[1])

    def _check_feature_matrix(
        self,
        value: ArrayLike,
        *,
        name: str,
        dim: Optional[int] = None,
    ) -> int:
        if len(_shape(value) or ()) != 2:
            raise ValueError(f"{name} must be a 2D array [N, D].")
        current_dim = int(value.shape[1])
        if dim is not None and current_dim != dim:
            raise ValueError(f"{name} must share feature dimension {dim}.")
        return current_dim

    def _check_labels(
        self,
        labels: Optional[ArrayLike],
        features: ArrayLike,
        *,
        name: str,
    ) -> None:
        if labels is not None and labels.shape[0] != features.shape[0]:
            raise ValueError(f"{name} length must match its feature view.")

    def validate(
        self,
        *,
        feature_dim: Optional[int] = None,
        require_consistency: bool = False,
        require_target_labels: bool = False,
    ) -> "FeatureSet":
        dim = self._check_feature_matrix(self.src_features, name="src_features")
        if feature_dim is not None and int(feature_dim) != dim:
            raise ValueError(f"feature_dim={feature_dim} does not match data dimension {dim}.")
        if self.src_features.shape[0] != self.src_labels.shape[0]:
            raise ValueError("src_labels length must match src_features.")

        self._check_feature_matrix(self.entropy_features, name="entropy_features", dim=dim)
        self._check_feature_matrix(self.cr_features_1, name="cr_features_1", dim=dim)
        self._check_feature_matrix(self.cr_features_2, name="cr_features_2", dim=dim)
        if self.cr_features_1.shape != self.cr_features_2.shape:
            raise ValueError("cr_features_1 and cr_features_2 must have the same shape.")
        if require_consistency and (self.cr_features_1 is None or self.cr_features_2 is None):
            raise ValueError("cr_features_1 and cr_features_2 are required for consistency loss.")

        self._check_labels(self.entropy_labels, self.entropy_features, name="entropy_labels")
        self._check_labels(self.cr_labels, self.cr_features_1, name="cr_labels")
        if require_target_labels and self.entropy_labels is None and self.eval_labels is None:
            raise ValueError("Target labels are required for this configuration.")

        if self.eval_features is not None:
            self._check_feature_matrix(self.eval_features, name="eval_features", dim=dim)
            self._check_labels(self.eval_labels, self.eval_features, name="eval_labels")

        return self

    def as_tensors(
        self,
        *,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> TensorFeatureSet:
        tensors = TensorFeatureSet(
            src_features=_to_tensor(self.src_features, dtype=dtype),
            src_labels=_to_tensor(self.src_labels, dtype=torch.long),
            entropy_features=_to_tensor(self.entropy_features, dtype=dtype),
            entropy_labels=_to_tensor(self.entropy_labels, dtype=torch.long),
            cr_features_1=_to_tensor(self.cr_features_1, dtype=dtype),
            cr_features_2=_to_tensor(self.cr_features_2, dtype=dtype),
            cr_labels=_to_tensor(self.cr_labels, dtype=torch.long),
            eval_features=_to_tensor(self.eval_features, dtype=dtype),
            eval_labels=_to_tensor(self.eval_labels, dtype=torch.long),
            metadata=dict(self.metadata),
        )
        if device is not None:
            return tensors.to(device)
        return tensors
