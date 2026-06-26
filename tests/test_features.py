import numpy as np
import pytest
import torch

from fps_uda import FeatureSet


def test_feature_set_accepts_numpy_and_torch():
    features = FeatureSet(
        src_features=np.zeros((4, 3), dtype="float32"),
        src_labels=np.array([0, 1, 0, 1]),
        entropy_features=torch.zeros(5, 3),
        cr_features_1=torch.zeros(5, 3),
        cr_features_2=torch.ones(5, 3),
    )
    tensors = features.validate(require_consistency=True).as_tensors(device="cpu")
    assert tensors.src_features.dtype == torch.float32
    assert tensors.src_labels.dtype == torch.long
    assert tensors.feature_dim == 3


def test_feature_set_validates_shape_mismatch():
    features = FeatureSet(
        src_features=np.zeros((4, 3), dtype="float32"),
        src_labels=np.array([0, 1, 0, 1]),
        entropy_features=np.zeros((5, 4), dtype="float32"),
        cr_features_1=np.zeros((5, 3), dtype="float32"),
        cr_features_2=np.ones((5, 3), dtype="float32"),
    )
    with pytest.raises(ValueError, match="feature dimension"):
        features.validate()
