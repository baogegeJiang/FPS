import numpy as np

from fps_uda import FPSConfig, FeatureSet, train_fps


def main():
    rng = np.random.default_rng(0)
    src_x = rng.normal(size=(32, 8)).astype("float32")
    src_y = rng.integers(0, 3, size=(32,), dtype=np.int64)
    tgt_x = rng.normal(size=(32, 8)).astype("float32")
    tgt_x2 = tgt_x + rng.normal(scale=0.05, size=tgt_x.shape).astype("float32")
    tgt_eval = rng.normal(size=(16, 8)).astype("float32")
    tgt_eval_y = rng.integers(0, 3, size=(16,), dtype=np.int64)

    features = FeatureSet(
        src_features=src_x,
        src_labels=src_y,
        entropy_features=tgt_eval,
        entropy_labels=tgt_eval_y,
        cr_features_1=tgt_x,
        cr_features_2=tgt_x2,
        cr_labels=rng.integers(0, 3, size=(32,), dtype=np.int64),
        eval_features=tgt_eval,
        eval_labels=tgt_eval_y,
    )
    config = FPSConfig(
        num_classes=3,
        feature_dim=8,
        device="cpu",
        optimizer="sgd",
        base_lr=0.001,
        lr_schedule="linear_step",
        iter_num=5,
        eval_interval=1,
    )
    result = train_fps(features, config)
    print(result.best_metric, result.best_score)


if __name__ == "__main__":
    main()
