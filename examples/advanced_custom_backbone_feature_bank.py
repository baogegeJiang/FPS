"""Pure Python feature-bank extraction with a fully custom backbone.

This advanced example intentionally avoids YAML. It writes a tiny image
dataset into /tmp, defines a custom PyTorch model, wraps it with FPS-UDA's
pooling adapter, and extracts a dataset-level H5 feature bank.

Replace ``TinyTokenBackbone`` and the dataset/domain definitions with your own
project code when integrating a private model.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import sys

import torch
from PIL import Image
from torch import nn

if __package__ is None or __package__ == "":
    repo_src = Path(__file__).resolve().parents[1] / "src"
    if repo_src.exists():
        sys.path.insert(0, str(repo_src))

from fps_uda.adapters.datasets import (
    DatasetConfig,
    DomainSpec,
    FeatureBankConfig,
    LoaderConfig,
    TransformConfig,
)
from fps_uda.adapters.feature_extraction import extract_feature_bank_to_h5
from fps_uda.models import BackboneConfig, PoolableBackbone


class TinyTokenBackbone(nn.Module):
    """Example user model returning token features [B, L, C]."""

    in_features = 8

    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, self.in_features, kernel_size=3, stride=8, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.proj(x)
        batch, channels, height, width = feature_map.shape
        return feature_map.reshape(batch, channels, height * width).transpose(1, 2)


def _write_tiny_domain(root: Path, domain: str, *, offset: int) -> None:
    domain_root = root / domain
    domain_root.mkdir(parents=True, exist_ok=True)
    lines = []
    for index in range(4):
        label = index % 2
        image_name = f"img_{index}.png"
        color = (
            40 + offset + index * 20,
            80 + label * 40,
            120 + index * 10,
        )
        Image.new("RGB", (48, 40), color=color).save(domain_root / image_name)
        lines.append(f"{image_name} {label}\n")
    (domain_root / "annotations.txt").write_text("".join(lines), encoding="utf-8")


def build_dataset_config(root: Path) -> DatasetConfig:
    return DatasetConfig(
        name="custom_tiny_token_bank",
        root_dir=str(root),
        backbone=BackboneConfig.from_mapping(
            {
                "backend": "custom",
                "name": "tiny_token_backbone",
                "pretrained": False,
                "in_features": 8,
                "pooling": {
                    "feature_type": "token",
                    "random_strategy": "token_channel_squared",
                },
            }
        ),
        loader=LoaderConfig(batch_size=2, num_workers=0, pin_memory=False),
        transform=TransformConfig(
            interpolation="bilinear",
            antialias=True,
            pad_fill=127,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
        domains=(
            DomainSpec(
                name="source",
                kind="manifest",
                image_root="source",
                manifest="source/annotations.txt",
                path_column=0,
                label_column=1,
            ),
            DomainSpec(
                name="target",
                kind="manifest",
                image_root="target",
                manifest="target/annotations.txt",
                path_column=0,
                label_column=1,
            ),
        ),
        feature_bank=FeatureBankConfig(
            water_level=0.0,
            mute_padding_in_pool=True,
            views=(
                {
                    "key": "pad_resize64_input64_direct_orig",
                    "pad_to_square": True,
                    "resize_size": 64,
                    "input_size": 64,
                    "crop": "direct",
                    "flip": "orig",
                    "random_pooling_count": 2,
                },
                {
                    "key": "pad_resize64_input64_direct_hflip",
                    "pad_to_square": True,
                    "resize_size": 64,
                    "input_size": 64,
                    "crop": "direct",
                    "flip": "hflip",
                    "random_pooling_count": 2,
                },
            ),
        ),
    )


def main() -> None:
    with TemporaryDirectory(prefix="fps_uda_custom_example_") as tmp:
        root = Path(tmp) / "data"
        _write_tiny_domain(root, "source", offset=0)
        _write_tiny_domain(root, "target", offset=30)
        config = build_dataset_config(root)

        custom_model = TinyTokenBackbone()
        backbone = PoolableBackbone(custom_model, config.backbone)

        output_path = Path(tmp) / "custom_feature_bank.h5"
        summary = extract_feature_bank_to_h5(
            backbone,
            config,
            str(output_path),
            domains=("source", "target"),
            backbone_name=config.backbone.name,
            device="cpu",
            progress=True,
        )
        print(summary)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
