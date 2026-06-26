from pathlib import Path

import h5py
import pytest
import yaml

from fps_uda.adapters.datasets import (
    DatasetConfig,
    build_domain_dataset,
    discover_dataset_domains,
    load_dataset_config,
)


PRESETS = [
    "configs/datasets/office31_resnet.yaml",
    "configs/datasets/office31_vit.yaml",
    "configs/datasets/office_home_resnet.yaml",
    "configs/datasets/office_home_vit.yaml",
    "configs/datasets/visda17_resnet.yaml",
    "configs/datasets/visda17_vit.yaml",
]


def _vision_available():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    return pytest.importorskip("PIL.Image")


def _write_image(path: Path, color=(128, 64, 32)):
    Image = _vision_available()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=color).save(path)


def _manifest_root(tmp_path: Path, *, unlabeled_target: bool = False) -> Path:
    root = tmp_path / "office_like"
    for domain in ("Art", "Product"):
        lines = []
        for idx in range(2):
            image_name = f"img_{idx}.jpg"
            _write_image(root / domain / image_name, color=(idx * 40 + 10, 20, 30))
            if unlabeled_target and domain == "Product":
                lines.append(f"{image_name}\n")
            else:
                lines.append(f"{image_name} {idx % 2}\n")
        (root / domain / "annotations").mkdir(parents=True, exist_ok=True)
        (root / domain / "annotations" / "annotations.txt").write_text(
            "".join(lines),
            encoding="utf-8",
        )
    return root


def _class_folder_root(tmp_path: Path) -> Path:
    root = tmp_path / "class_folder"
    for domain in ("Art", "Product"):
        for label, class_name in enumerate(("Alarm_Clock", "Backpack")):
            _write_image(root / domain / class_name / "00001.jpg", color=(label * 50 + 20, 40, 60))
    return root


def _base_config(root: Path, *, unlabeled_target: bool = False):
    product_domain = {
        "name": "Product",
        "kind": "manifest",
        "image_root": "Product",
        "manifest": "Product/annotations/annotations.txt",
        "path_column": 0,
    }
    if not unlabeled_target:
        product_domain["label_column"] = 1
    return {
        "name": "mini_resnet",
        "root_dir": str(root),
        "loader": {
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "drop_last": False,
            "seed": 7,
        },
        "transform": {
            "interpolation": "bilinear",
            "antialias": True,
            "pad_fill": 127,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "domains": [
            {
                "name": "Art",
                "kind": "manifest",
                "image_root": "Art",
                "manifest": "Art/annotations/annotations.txt",
                "path_column": 0,
                "label_column": 1,
            },
            product_domain,
        ],
        "feature_bank": {
            "water_level": 0.0,
            "mute_padding_in_pool": False,
            "views": [
                {
                    "key": "pad_resize224_input224_direct_orig",
                    "pad_to_square": True,
                    "resize_size": 224,
                    "input_size": 224,
                    "crop": "direct",
                    "flip": "orig",
                }
            ],
        },
    }


def test_visda_resnet_pad_to_square_matches_origin_ceil(monkeypatch):
    Image = _vision_available()
    from fps_uda.adapters import datasets as dataset_module

    monkeypatch.setattr(dataset_module.random, "uniform", lambda _low, _high: 1.01)
    image = Image.new("RGB", (50, 100), color=(128, 128, 128))
    padded = dataset_module.pad_to_square(image, aspect_jitter=0.1)
    assert padded.size == (101, 100)


def test_dataset_presets_parse_without_identity_behavior_fields():
    for preset in PRESETS:
        raw = yaml.safe_load(Path(preset).read_text())
        assert "dataset" not in raw
        assert "model_family" not in raw
        assert "preprocess" not in raw
        config = load_dataset_config(preset)
        assert config.name
        assert config.source_domain is None
        assert config.target_domain is None
        assert config.feature_bank.water_level == 0.0
        assert config.feature_bank.views
        assert config.domains
        assert config.transform.mean
        assert config.backbone.backend in {"torchvision", "timm"}
        if preset.endswith("_resnet.yaml"):
            assert config.backbone.pooling.feature_type == "spatial"
            assert config.backbone.pooling.random_strategy == "spatial_shared"
        if preset.endswith("_vit.yaml"):
            assert config.backbone.pooling.feature_type == "token"
            assert config.backbone.pooling.random_strategy == "token_channel_squared"
        assert not Path(config.root_dir).is_absolute()


def test_dataset_config_overrides_loader_values():
    config = load_dataset_config(
        "configs/datasets/office_home_resnet.yaml",
        root_dir="data/office_home_override",
        batch_size=7,
        num_workers=0,
        seed=3,
    )
    assert config.root_dir == "data/office_home_override"
    assert config.batch_size == 7
    assert config.num_workers == 0
    assert config.seed == 3


def test_dataset_config_rejects_old_behavior_fields():
    with pytest.raises(ValueError, match="no longer supports behavior fields"):
        DatasetConfig.from_mapping(
            {
                "dataset": "office31",
                "name": "bad",
                "root_dir": "data/office31",
                "domains": [],
            }
        )


def test_manifest_domains_with_and_without_labels(tmp_path):
    _vision_available()
    config = DatasetConfig.from_mapping(
        _base_config(
            _manifest_root(tmp_path, unlabeled_target=True),
            unlabeled_target=True,
        )
    )
    assert discover_dataset_domains(config) == ("Art", "Product")

    transform = lambda image: image
    art = build_domain_dataset(config, "Art", transform)
    product = build_domain_dataset(config, "Product", transform)
    assert art.has_labels is True
    assert product.has_labels is False
    assert [label for _, label in art.samples] == [0, 1]
    assert [label for _, label in product.samples] == [None, None]


def test_class_folder_domain_loader(tmp_path):
    _vision_available()
    root = _class_folder_root(tmp_path)
    config = DatasetConfig.from_mapping(
        {
            "name": "class_folder",
            "root_dir": str(root),
            "source_domain": "Art",
            "target_domain": "Product",
            "loader": {"batch_size": 2, "num_workers": 0, "pin_memory": False},
            "domains": [
                {"name": "Art", "kind": "class_folder", "image_root": "Art"},
                {"name": "Product", "kind": "class_folder", "image_root": "Product"},
            ],
        }
    )
    dataset = build_domain_dataset(config, "Art", transform=lambda image: image)
    assert dataset.has_labels is True
    assert len(dataset.samples) == 2
    assert [label for _, label in dataset.samples] == [0, 1]


def test_feature_bank_preset_view_counts():
    from fps_uda.adapters.feature_extraction import build_feature_bank_view_specs

    def expected_view_count(config):
        return sum(
            1 + int(view.get("random_pooling_count", 2))
            for view in config.feature_bank.views
        )

    def expected_clean_keys(config):
        return {f"{view['key']}_clean" for view in config.feature_bank.views}

    def has_nopad_base_views(config):
        return any(not bool(view["pad_to_square"]) for view in config.feature_bank.views)

    office_home_resnet_config = load_dataset_config("configs/datasets/office_home_resnet.yaml")
    office31_resnet_config = load_dataset_config("configs/datasets/office31_resnet.yaml")
    visda_resnet_config = load_dataset_config("configs/datasets/visda17_resnet.yaml")
    visda_vit_config = load_dataset_config("configs/datasets/visda17_vit.yaml")
    office_home_vit_config = load_dataset_config("configs/datasets/office_home_vit.yaml")
    office_home_resnet_specs = build_feature_bank_view_specs(
        office_home_resnet_config, backbone="resnet50"
    )
    office31_resnet_specs = build_feature_bank_view_specs(
        office31_resnet_config, backbone="resnet50"
    )
    visda_resnet_specs = build_feature_bank_view_specs(
        visda_resnet_config, backbone="resnet50"
    )
    visda_vit_specs = build_feature_bank_view_specs(
        visda_vit_config, backbone="vit-base-patch16-224"
    )
    vit_specs = build_feature_bank_view_specs(
        office_home_vit_config, backbone="vit-base-patch16-224"
    )

    assert len(office_home_resnet_specs) == expected_view_count(office_home_resnet_config)
    assert len(office31_resnet_specs) == expected_view_count(office31_resnet_config)
    assert len(visda_resnet_specs) == expected_view_count(visda_resnet_config)
    assert len(visda_vit_specs) == expected_view_count(visda_vit_config)
    assert len(vit_specs) == expected_view_count(office_home_vit_config)
    office_home_resnet_keys = {spec.key for spec in office_home_resnet_specs}
    office31_resnet_keys = {spec.key for spec in office31_resnet_specs}
    visda_resnet_keys = {spec.key for spec in visda_resnet_specs}
    vit_keys = {spec.key for spec in vit_specs}
    assert expected_clean_keys(office_home_resnet_config).issubset(office_home_resnet_keys)
    assert expected_clean_keys(office31_resnet_config).issubset(office31_resnet_keys)
    assert expected_clean_keys(visda_resnet_config).issubset(visda_resnet_keys)
    assert expected_clean_keys(visda_vit_config).issubset(
        {spec.key for spec in visda_vit_specs}
    )
    assert expected_clean_keys(office_home_vit_config).issubset(vit_keys)
    assert any(key.endswith("_pool_b") for key in office31_resnet_keys)
    assert any(key.endswith("_pool_b") for key in vit_keys)
    assert any(key.startswith("nopad_") for key in office31_resnet_keys) == has_nopad_base_views(
        office31_resnet_config
    )
    assert any(key.startswith("nopad_") for key in visda_resnet_keys) == has_nopad_base_views(
        visda_resnet_config
    )


def test_feature_bank_view_random_pooling_count():
    from fps_uda.adapters.feature_extraction import build_feature_bank_view_specs

    data = _base_config("data")
    data["feature_bank"]["views"] = [
        {
            "key": "pad_resize256_input256_direct_orig",
            "pad_to_square": True,
            "resize_size": 256,
            "input_size": 256,
            "crop": "direct",
            "flip": "orig",
            "random_pooling_count": 4,
        }
    ]
    specs = build_feature_bank_view_specs(data, backbone="tiny")

    assert [spec.pooling for spec in specs] == [
        "clean",
        "pool_a",
        "pool_b",
        "pool_c",
        "pool_d",
    ]
    assert [spec.key for spec in specs][-1] == "pad_resize256_input256_direct_orig_pool_d"
    assert specs[-1].attrs()["random_pooling_count"] == 4


def test_feature_bank_view_config_rejects_missing_views(tmp_path):
    from fps_uda.adapters.feature_extraction import build_feature_bank_view_specs

    data = _base_config(_manifest_root(tmp_path))
    data["feature_bank"] = {"water_level": 0.0, "views": []}
    config = DatasetConfig.from_mapping(data)

    with pytest.raises(ValueError, match="feature_bank.views"):
        build_feature_bank_view_specs(config)


def test_feature_bank_view_config_rejects_invalid_crop(tmp_path):
    from fps_uda.adapters.feature_extraction import build_feature_bank_view_specs

    data = _base_config(_manifest_root(tmp_path))
    data["feature_bank"]["views"][0]["crop"] = "random"
    config = DatasetConfig.from_mapping(data)

    with pytest.raises(ValueError, match="crop must be one of"):
        build_feature_bank_view_specs(config)


def test_extract_feature_bank_to_h5_from_explicit_views_with_unlabeled_target(tmp_path):
    torch = pytest.importorskip("torch")
    _vision_available()
    from fps_uda.adapters.feature_extraction import extract_feature_bank_to_h5
    from fps_uda.io import load_feature_bank_h5
    from fps_uda.models import BackboneConfig

    class TinyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = BackboneConfig.from_mapping(
                {
                    "backend": "timm",
                    "name": "tiny",
                    "pretrained": False,
                    "in_features": 3,
                    "pooling": {
                        "feature_type": "token",
                        "random_strategy": "token_channel_squared",
                    },
                }
            )

        def extract_pooling_views(
            self,
            x,
            *,
            poolings,
            water_level=0.0,
            **kwargs,
        ):
            base = x.mean(dim=(2, 3))
            return {
                pooling: base + {"clean": 0.0, "pool_a": 1.0, "pool_b": 2.0}[pooling]
                for pooling in poolings
            }

    root = _manifest_root(tmp_path, unlabeled_target=True)
    data = _base_config(root, unlabeled_target=True)
    data["feature_bank"]["views"].append(
        {
            "key": "pad_resize256_input224_center_hflip",
            "pad_to_square": True,
            "resize_size": 256,
            "input_size": 224,
            "crop": "center",
            "flip": "hflip",
        }
    )
    config = DatasetConfig.from_mapping(data)
    out = tmp_path / "bank.h5"
    summary = extract_feature_bank_to_h5(
        TinyBackbone(),
        config,
        str(out),
        domains=("Art", "Product"),
        backbone_name="tiny",
        device="cpu",
        progress=False,
    )

    assert summary["views_per_domain"] == 6
    with h5py.File(out, "r") as h5:
        assert h5.attrs["schema"] == "fps_uda_feature_bank"
        assert h5.attrs["name"] == "mini_resnet"
        assert h5.attrs["backbone_backend"] == "timm"
        assert h5.attrs["backbone_name"] == "tiny"
        assert not h5.attrs["backbone_pretrained"]
        assert h5.attrs["backbone_weights"] == ""
        assert h5.attrs["pooling_feature_type"] == "token"
        assert h5.attrs["pooling_random_strategy"] == "token_channel_squared"
        assert h5["domains/Art"].attrs["has_labels"]
        assert not h5["domains/Product"].attrs["has_labels"]
        assert h5["domains/Art/label"][:].tolist() == [0, 1]
        assert "label" not in h5["domains/Product"]
        view = h5["domains/Art/views/pad_resize256_input224_center_hflip_pool_a"]
        assert view["feature"].shape == (2, 3)
        assert view.attrs["crop"] == "center"
        assert view.attrs["flip"] == "hflip"
        assert view.attrs["pooling"] == "pool_a"
        assert view.attrs["interpolation"] == "bilinear"
        assert view.attrs["mean"].tolist() == pytest.approx([0.485, 0.456, 0.406])

    loaded = load_feature_bank_h5(
        str(out),
        source_domain="Art",
        target_domain="Product",
        src_view="pad_resize224_input224_direct_orig_clean",
        entropy_view="pad_resize256_input224_center_hflip_clean",
        cr_view1="pad_resize256_input224_center_hflip_pool_a",
        cr_view2="pad_resize256_input224_center_hflip_pool_b",
        eval_view="pad_resize256_input224_center_hflip_clean",
    )
    assert loaded.src_features.shape == (2, 3)
    assert loaded.entropy_labels is None
    assert loaded.eval_labels is None
    assert loaded.cr_features_1.shape == loaded.cr_features_2.shape == (2, 3)
