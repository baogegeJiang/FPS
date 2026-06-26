import numpy as np
import pytest

from fps_uda.io import (
    load_feature_bank_h5,
    save_feature_bank_h5,
)


def test_feature_bank_h5_roundtrip_and_sqrt_transform(tmp_path):
    path = tmp_path / "bank.h5"
    save_feature_bank_h5(
        {
            "Art": {
                "label": np.array([0, 1]),
                "views": {
                    "pad_resize224_input224_direct_orig_clean": {
                        "feature": np.array([[1.0, 4.0], [9.0, 16.0]], dtype="float32"),
                        "attrs": {
                            "pad_to_square": True,
                            "resize_size": 224,
                            "input_size": 224,
                            "crop": "direct",
                            "flip": "orig",
                            "pooling": "clean",
                            "water_level": 0.0,
                            "backbone": "tiny",
                            "preprocess": "resnet",
                        },
                    }
                },
            },
            "Product": {
                "label": np.array([1, 0]),
                "views": {
                    "clean": np.array([[1.0, 1.0], [4.0, 4.0]], dtype="float32"),
                    "pool_a": np.array([[9.0, 9.0], [16.0, 16.0]], dtype="float32"),
                    "pool_b": np.array([[25.0, 25.0], [36.0, 36.0]], dtype="float32"),
                },
            },
        },
        str(path),
    )

    loaded = load_feature_bank_h5(
        str(path),
        source_domain="Art",
        target_domain="Product",
        src_view="pad_resize224_input224_direct_orig_clean",
        entropy_view="clean",
        cr_view1="pool_a",
        cr_view2="pool_b",
        eval_view="clean",
        feature_transform="sqrt",
    )

    assert loaded.src_labels.tolist() == [0, 1]
    assert loaded.entropy_labels.tolist() == [1, 0]
    assert loaded.eval_labels.tolist() == [1, 0]
    assert loaded.src_features.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert loaded.cr_features_2.tolist() == [[5.0, 5.0], [6.0, 6.0]]
    assert loaded.metadata["schema"] == "feature_bank"


def test_feature_bank_missing_view_has_clear_error(tmp_path):
    path = tmp_path / "bank.h5"
    save_feature_bank_h5(
        {
            "A": {
                "label": np.array([0]),
                "views": {"clean": np.ones((1, 2), dtype="float32")},
            },
            "B": {
                "label": np.array([0]),
                "views": {"clean": np.ones((1, 2), dtype="float32")},
            },
        },
        str(path),
    )

    with pytest.raises(KeyError, match="Missing feature-bank view 'pool_a'"):
        load_feature_bank_h5(
            str(path),
            source_domain="A",
            target_domain="B",
            src_view="clean",
            entropy_view="clean",
            cr_view1="pool_a",
            cr_view2="clean",
            eval_view="clean",
        )


def test_feature_bank_can_stack_and_mean_multiple_views(tmp_path):
    path = tmp_path / "bank_multi.h5"
    save_feature_bank_h5(
        {
            "A": {
                "label": np.array([0, 1]),
                "views": {
                    "s1": np.array([[1.0, 1.0], [2.0, 2.0]], dtype="float32"),
                    "s2": np.array([[3.0, 3.0], [4.0, 4.0]], dtype="float32"),
                },
            },
            "B": {
                "label": np.array([1, 0]),
                "views": {
                    "t1": np.array([[10.0, 10.0], [20.0, 20.0]], dtype="float32"),
                    "t2": np.array([[30.0, 30.0], [40.0, 40.0]], dtype="float32"),
                    "u1": np.array([[11.0, 11.0], [21.0, 21.0]], dtype="float32"),
                    "u2": np.array([[31.0, 31.0], [41.0, 41.0]], dtype="float32"),
                    "e1": np.array([[100.0, 100.0], [200.0, 200.0]], dtype="float32"),
                    "e2": np.array([[300.0, 300.0], [400.0, 400.0]], dtype="float32"),
                },
            },
        },
        str(path),
    )

    loaded = load_feature_bank_h5(
        str(path),
        source_domain="A",
        target_domain="B",
        src_view=["s1", "s2"],
        entropy_view="t1,t2",
        cr_view1="t1,t2",
        cr_view2=["u1", "u2"],
        eval_view=["e1", "e2"],
        eval_combine="mean",
    )

    assert loaded.src_features.tolist() == [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
    assert loaded.src_labels.tolist() == [0, 1, 0, 1]
    assert loaded.cr_features_1.shape == loaded.cr_features_2.shape == (4, 2)
    assert loaded.cr_labels.tolist() == [1, 0, 1, 0]
    assert loaded.eval_features.tolist() == [[200.0, 200.0], [300.0, 300.0]]
    assert loaded.eval_labels.tolist() == [1, 0]
    assert loaded.metadata["src_combine"] == "stack"
    assert loaded.metadata["eval_combine"] == "mean"


def test_feature_bank_mean_combine_preserves_sample_count(tmp_path):
    path = tmp_path / "bank_mean.h5"
    save_feature_bank_h5(
        {
            "A": {
                "label": np.array([0, 1]),
                "views": {
                    "s1": np.array([[1.0, 1.0], [3.0, 3.0]], dtype="float32"),
                    "s2": np.array([[3.0, 3.0], [5.0, 5.0]], dtype="float32"),
                },
            },
            "B": {
                "label": np.array([0, 1]),
                "views": {
                    "t1": np.array([[2.0, 2.0], [4.0, 4.0]], dtype="float32"),
                    "t2": np.array([[4.0, 4.0], [6.0, 6.0]], dtype="float32"),
                    "u1": np.array([[3.0, 3.0], [5.0, 5.0]], dtype="float32"),
                    "u2": np.array([[5.0, 5.0], [7.0, 7.0]], dtype="float32"),
                },
            },
        },
        str(path),
    )

    loaded = load_feature_bank_h5(
        str(path),
        source_domain="A",
        target_domain="B",
        src_view=["s1", "s2"],
        entropy_view=["t1", "t2"],
        cr_view1=["t1", "t2"],
        cr_view2=["u1", "u2"],
        eval_view=["t1", "t2"],
        src_combine="mean",
        entropy_combine="mean",
        cr_view1_combine="mean",
        cr_view2_combine="mean",
        eval_combine="mean",
    )

    assert loaded.src_features.tolist() == [[2.0, 2.0], [4.0, 4.0]]
    assert loaded.entropy_features.tolist() == [[3.0, 3.0], [5.0, 5.0]]
    assert loaded.cr_features_2.tolist() == [[4.0, 4.0], [6.0, 6.0]]
    assert loaded.eval_features.tolist() == [[3.0, 3.0], [5.0, 5.0]]
    assert loaded.entropy_labels.tolist() == [0, 1]


def test_feature_bank_rejects_unaligned_target_view_combinations(tmp_path):
    path = tmp_path / "bank_bad_align.h5"
    save_feature_bank_h5(
        {
            "A": {
                "label": np.array([0, 1]),
                "views": {"s": np.ones((2, 2), dtype="float32")},
            },
            "B": {
                "label": np.array([0, 1]),
                "views": {
                    "t1": np.ones((2, 2), dtype="float32"),
                    "t2": np.ones((2, 2), dtype="float32"),
                    "u1": np.ones((2, 2), dtype="float32"),
                },
            },
        },
        str(path),
    )

    with pytest.raises(ValueError, match="cr_view1 and cr_view2 must produce aligned"):
        load_feature_bank_h5(
            str(path),
            source_domain="A",
            target_domain="B",
            src_view="s",
            entropy_view="t1",
            cr_view1=["t1", "t2"],
            cr_view2=["u1"],
            eval_view="t1",
        )


def test_feature_bank_rejects_label_feature_mismatch(tmp_path):
    with pytest.raises(ValueError, match="2 features but 1 labels"):
        save_feature_bank_h5(
            {
                "A": {
                    "label": np.array([0]),
                    "views": {"bad": np.ones((2, 3), dtype="float32")},
                }
            },
            str(tmp_path / "bad.h5"),
        )


def test_feature_bank_allows_unlabeled_target_domain(tmp_path):
    path = tmp_path / "bank_unlabeled_target.h5"
    save_feature_bank_h5(
        {
            "A": {
                "label": np.array([0, 1]),
                "views": {"clean": np.ones((2, 2), dtype="float32")},
            },
            "B": {
                "views": {
                    "clean": np.ones((2, 2), dtype="float32"),
                    "pool_a": np.ones((2, 2), dtype="float32"),
                    "pool_b": np.ones((2, 2), dtype="float32"),
                },
            },
        },
        str(path),
    )

    loaded = load_feature_bank_h5(
        str(path),
        source_domain="A",
        target_domain="B",
        src_view="clean",
        entropy_view="clean",
        cr_view1="pool_a",
        cr_view2="pool_b",
        eval_view="clean",
    )

    assert loaded.src_labels.tolist() == [0, 1]
    assert loaded.entropy_labels is None
    assert loaded.eval_labels is None


def test_feature_bank_requires_source_labels(tmp_path):
    path = tmp_path / "bank_unlabeled_source.h5"
    save_feature_bank_h5(
        {
            "A": {"views": {"clean": np.ones((2, 2), dtype="float32")}},
            "B": {
                "label": np.array([0, 1]),
                "views": {
                    "clean": np.ones((2, 2), dtype="float32"),
                    "pool_a": np.ones((2, 2), dtype="float32"),
                    "pool_b": np.ones((2, 2), dtype="float32"),
                },
            },
        },
        str(path),
    )

    with pytest.raises(KeyError, match="missing labels"):
        load_feature_bank_h5(
            str(path),
            source_domain="A",
            target_domain="B",
            src_view="clean",
            entropy_view="clean",
            cr_view1="pool_a",
            cr_view2="pool_b",
            eval_view="clean",
        )
