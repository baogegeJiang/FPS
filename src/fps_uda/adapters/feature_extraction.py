from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Union

import h5py
import numpy as np
import torch
from tqdm import tqdm

from fps_uda.adapters.datasets import (
    DatasetConfig,
    TransformConfig,
    _make_loader,
    _vision_imports,
    build_domain_dataset,
    dataset_labels,
    discover_dataset_domains,
    pad_to_square,
)
from fps_uda.models import BackboneConfig


DEFAULT_RANDOM_POOLING_COUNT = 2
VALID_CROPS = {"direct", "tl", "tr", "bl", "br", "center"}
VALID_FLIPS = {"orig", "hflip"}


@dataclass(frozen=True)
class FeatureBankViewSpec:
    key: str
    pad_to_square: bool
    resize_size: int
    input_size: int
    crop: str
    flip: str
    pooling: str
    random_pooling_count: int
    water_level: float
    backbone: str
    transform: TransformConfig

    @property
    def transform_key(self) -> tuple:
        return (
            bool(self.pad_to_square),
            int(self.resize_size),
            int(self.input_size),
            self.crop,
            self.flip,
            self.transform.interpolation,
            bool(self.transform.antialias),
            int(self.transform.pad_fill),
            tuple(float(value) for value in self.transform.mean),
            tuple(float(value) for value in self.transform.std),
        )

    def attrs(self) -> Dict[str, object]:
        return {
            "pad_to_square": bool(self.pad_to_square),
            "resize_size": int(self.resize_size),
            "input_size": int(self.input_size),
            "crop": self.crop,
            "flip": self.flip,
            "pooling": self.pooling,
            "random_pooling_count": int(self.random_pooling_count),
            "water_level": float(self.water_level),
            "backbone": self.backbone,
            "interpolation": self.transform.interpolation,
            "antialias": bool(self.transform.antialias),
            "pad_fill": int(self.transform.pad_fill),
            "mean": np.asarray(self.transform.mean, dtype=np.float32),
            "std": np.asarray(self.transform.std, dtype=np.float32),
        }


@dataclass(frozen=True)
class _BaseFeatureBankView:
    key: str
    pad_to_square: bool
    resize_size: int
    input_size: int
    crop: str
    flip: str
    random_pooling_count: int


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _pooling_suffix(index: int) -> str:
    letters = []
    value = int(index) + 1
    while value > 0:
        value, rem = divmod(value - 1, 26)
        letters.append(chr(ord("a") + rem))
    return "".join(reversed(letters))


def _pooling_views(random_pooling_count: int) -> tuple[str, ...]:
    count = int(random_pooling_count)
    if count < 0:
        raise ValueError("random_pooling_count must be non-negative.")
    return ("clean",) + tuple(f"pool_{_pooling_suffix(index)}" for index in range(count))


def _has_pooling_suffix(key: str) -> bool:
    if key.endswith("_clean"):
        return True
    if "_pool_" in key:
        suffix = key.rsplit("_pool_", 1)[-1]
        return suffix.isalpha() and suffix.islower()
    return False


def _parse_base_view(view: Mapping[str, object], *, index: int) -> _BaseFeatureBankView:
    required = ("key", "pad_to_square", "resize_size", "input_size", "crop", "flip")
    missing = [key for key in required if key not in view]
    if missing:
        raise ValueError(f"feature_bank.views[{index}] is missing: {', '.join(missing)}.")
    key = str(view["key"]).strip()
    if not key:
        raise ValueError(f"feature_bank.views[{index}].key must be non-empty.")
    if _has_pooling_suffix(key):
        raise ValueError(
            f"feature_bank.views[{index}].key must be a base key without pooling suffix."
        )
    resize_size = int(view["resize_size"])
    input_size = int(view["input_size"])
    if resize_size <= 0 or input_size <= 0:
        raise ValueError("resize_size and input_size must be positive.")
    crop = str(view["crop"]).strip().lower()
    if crop not in VALID_CROPS:
        raise ValueError("crop must be one of: direct, tl, tr, bl, br, center.")
    if crop == "direct" and input_size != resize_size:
        raise ValueError("direct crop requires input_size == resize_size.")
    if crop != "direct" and input_size > resize_size:
        raise ValueError("cropped views require input_size <= resize_size.")
    flip = str(view["flip"]).strip().lower()
    if flip not in VALID_FLIPS:
        raise ValueError("flip must be one of: orig, hflip.")
    random_pooling_count = int(
        view.get("random_pooling_count", DEFAULT_RANDOM_POOLING_COUNT)
    )
    if random_pooling_count < 0:
        raise ValueError("random_pooling_count must be non-negative.")
    return _BaseFeatureBankView(
        key=key,
        pad_to_square=_as_bool(view["pad_to_square"]),
        resize_size=resize_size,
        input_size=input_size,
        crop=crop,
        flip=flip,
        random_pooling_count=random_pooling_count,
    )


def build_feature_bank_view_specs(
    config: Union[DatasetConfig, Mapping[str, object]],
    *,
    backbone: str = "unknown",
    water_level: Optional[float] = None,
) -> list[FeatureBankViewSpec]:
    """Build final feature-bank view specs from explicit YAML base views."""
    if not isinstance(config, DatasetConfig):
        config = DatasetConfig.from_mapping(config)
    views = config.feature_bank.views
    if not isinstance(views, Sequence) or isinstance(views, (str, bytes)) or not views:
        raise ValueError("dataset config must define a non-empty feature_bank.views list.")
    resolved_water_level = (
        float(water_level)
        if water_level is not None
        else float(config.feature_bank.water_level)
    )
    specs: list[FeatureBankViewSpec] = []
    seen: set[str] = set()
    for index, view in enumerate(views):
        if not isinstance(view, Mapping):
            raise ValueError(f"feature_bank.views[{index}] must be a mapping.")
        base = _parse_base_view(view, index=index)
        for pooling in _pooling_views(base.random_pooling_count):
            key = f"{base.key}_{pooling}"
            if key in seen:
                raise ValueError(f"Duplicate feature-bank view key after pooling expansion: {key}.")
            seen.add(key)
            specs.append(
                FeatureBankViewSpec(
                    key=key,
                    pad_to_square=base.pad_to_square,
                    resize_size=base.resize_size,
                    input_size=base.input_size,
                    crop=base.crop,
                    flip=base.flip,
                    pooling=pooling,
                    random_pooling_count=base.random_pooling_count,
                    water_level=resolved_water_level,
                    backbone=backbone,
                    transform=config.transform,
                )
            )
    return specs


def _deterministic_crop(image, *, size: int, crop: str):
    _, _, _, _, transforms, _ = _vision_imports()
    if crop == "direct":
        return image
    width, height = image.size
    if width < size or height < size:
        raise ValueError(f"Cannot crop {size} from image size {(width, height)}.")
    if crop == "center":
        top = int(round((height - size) / 2.0))
        left = int(round((width - size) / 2.0))
    elif crop == "tl":
        top, left = 0, 0
    elif crop == "tr":
        top, left = 0, width - size
    elif crop == "bl":
        top, left = height - size, 0
    elif crop == "br":
        top, left = height - size, width - size
    else:
        raise ValueError("crop must be one of: direct, tl, tr, bl, br, center.")
    return transforms.functional.crop(image, top, left, size, size)


def build_feature_bank_transform(spec: FeatureBankViewSpec):
    """Build one deterministic image transform for a feature-bank view."""
    _, _, _, _, transforms, InterpolationMode = _vision_imports()
    if spec.transform.interpolation == "bilinear":
        interpolation = InterpolationMode.BILINEAR
    elif spec.transform.interpolation == "bicubic":
        interpolation = InterpolationMode.BICUBIC
    elif spec.transform.interpolation == "nearest":
        interpolation = InterpolationMode.NEAREST
    else:
        raise ValueError("transform.interpolation must be bilinear, bicubic, or nearest.")
    normalize = transforms.Normalize(spec.transform.mean, spec.transform.std)
    ops = []
    if spec.pad_to_square:
        ops.append(
            transforms.Lambda(
                lambda img, fill=spec.transform.pad_fill: pad_to_square(img, fill=fill)
            )
        )
    ops.append(
        transforms.Resize(
            [spec.resize_size, spec.resize_size],
            interpolation=interpolation,
            antialias=spec.transform.antialias,
        )
    )
    if spec.crop != "direct":
        ops.append(
            transforms.Lambda(
                lambda img, size=spec.input_size, crop=spec.crop: _deterministic_crop(
                    img, size=size, crop=crop
                )
            )
        )
    if spec.flip == "hflip":
        ops.append(transforms.Lambda(lambda img: transforms.functional.hflip(img)))
    elif spec.flip != "orig":
        raise ValueError("flip must be 'orig' or 'hflip'.")
    ops.extend([transforms.ToTensor(), normalize])
    return transforms.Compose(ops)


def _group_view_specs(
    specs: Sequence[FeatureBankViewSpec],
) -> list[tuple[tuple, list[FeatureBankViewSpec]]]:
    grouped: Dict[tuple, list[FeatureBankViewSpec]] = {}
    for spec in specs:
        grouped.setdefault(spec.transform_key, []).append(spec)
    return [(key, grouped[key]) for key in grouped]


def _forward_pooling_views(
    backbone: torch.nn.Module,
    inputs: torch.Tensor,
    specs: Sequence[FeatureBankViewSpec],
    *,
    backbone_kwargs: Optional[Mapping[str, object]],
) -> Dict[str, torch.Tensor]:
    poolings = [spec.pooling for spec in specs]
    forward_kwargs = dict(backbone_kwargs or {})
    if hasattr(backbone, "extract_pooling_views"):
        return backbone.extract_pooling_views(
            inputs,
            poolings=poolings,
            water_level=float(specs[0].water_level),
            **forward_kwargs,
        )
    outputs = {}
    for spec in specs:
        outputs[spec.pooling] = backbone(
            inputs,
            random_pool=spec.pooling != "clean",
            water_level=spec.water_level,
            **forward_kwargs,
        )
    return outputs


def _progress_total(loader) -> Optional[int]:
    try:
        return len(loader)
    except TypeError:
        return None


@torch.no_grad()
def _collect_feature_bank_view_group(
    backbone: torch.nn.Module,
    loader,
    device: str,
    specs: Sequence[FeatureBankViewSpec],
    *,
    backbone_kwargs: Optional[Mapping[str, object]] = None,
    progress: bool = True,
    desc: str = "feature bank",
) -> tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
    backbone.eval().to(device)
    features = {spec.key: [] for spec in specs}
    labels = []
    has_labels = bool(getattr(loader.dataset, "has_labels", True))
    with tqdm(
        total=_progress_total(loader),
        desc=desc,
        unit="batch",
        dynamic_ncols=True,
        disable=not progress,
    ) as pbar:
        for inputs, batch_labels in loader:
            inputs = inputs.to(device)
            outputs = _forward_pooling_views(
                backbone,
                inputs,
                specs,
                backbone_kwargs=backbone_kwargs,
            )
            for spec in specs:
                features[spec.key].append(outputs[spec.pooling].detach().cpu().numpy())
            if has_labels:
                labels.append(batch_labels.detach().cpu().numpy())
            pbar.update(1)
    return (
        {key: np.concatenate(parts, axis=0) for key, parts in features.items()},
        np.concatenate(labels, axis=0) if labels else None,
    )


def _feature_bank_backbone_kwargs(config: DatasetConfig) -> Dict[str, object]:
    return {
        "mute_padding_in_pool": _as_bool(config.feature_bank.mute_padding_in_pool),
    }


def extract_feature_bank_to_h5(
    backbone: torch.nn.Module,
    config: Union[DatasetConfig, Mapping[str, object]],
    output_path: str,
    *,
    domains: Optional[Iterable[str]] = None,
    backbone_name: str = "unknown",
    device: str = "cuda",
    water_level: Optional[float] = None,
    view_specs: Optional[Sequence[FeatureBankViewSpec]] = None,
    backbone_kwargs: Optional[Mapping[str, object]] = None,
    progress: bool = True,
) -> Dict[str, object]:
    """Extract a dataset-level deterministic feature bank."""
    if not isinstance(config, DatasetConfig):
        config = DatasetConfig.from_mapping(config)
    selected_domains = tuple(domains) if domains is not None else discover_dataset_domains(config)
    specs = list(
        view_specs
        or build_feature_bank_view_specs(
            config,
            backbone=backbone_name,
            water_level=water_level,
        )
    )
    if not specs:
        raise ValueError("feature-bank extraction requires at least one view spec.")
    backbone_config = getattr(backbone, "config", None)
    if not isinstance(backbone_config, BackboneConfig):
        backbone_config = config.backbone.with_overrides(name=backbone_name)
    forward_kwargs = _feature_bank_backbone_kwargs(config)
    forward_kwargs.update(dict(backbone_kwargs or {}))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as h5:
        h5.attrs["schema"] = "fps_uda_feature_bank"
        h5.attrs["schema_version"] = 1
        h5.attrs["name"] = config.name
        h5.attrs["backbone"] = backbone_name
        for attr_key, attr_value in backbone_config.metadata().items():
            h5.attrs[attr_key] = attr_value
        domains_group = h5.create_group("domains")
        for domain in selected_domains:
            domain_group = domains_group.create_group(str(domain))
            views_group = domain_group.create_group("views")
            reference_labels: Optional[np.ndarray] = None
            reference_initialized = False
            for _, group_specs in _group_view_specs(specs):
                transform = build_feature_bank_transform(group_specs[0])
                dataset = build_domain_dataset(config, str(domain), transform)
                labels_list = dataset_labels(dataset)
                labels = (
                    None if labels_list is None else np.asarray(labels_list, dtype=np.int64)
                )
                if not reference_initialized:
                    reference_labels = labels
                    reference_initialized = True
                    domain_group.attrs["has_labels"] = labels is not None
                    if reference_labels is not None:
                        domain_group.create_dataset("label", data=reference_labels)
                elif labels is None or reference_labels is None:
                    if labels is not reference_labels:
                        raise ValueError(
                            "Label availability changed while extracting "
                            f"feature-bank domain '{domain}'."
                        )
                elif labels.shape != reference_labels.shape or not np.array_equal(
                    labels,
                    reference_labels,
                ):
                    raise ValueError(
                        f"Label order changed while extracting feature-bank domain '{domain}'."
                    )
                loader = _make_loader(dataset, config, shuffle=False, origin_compatible=False)
                view_features, batch_labels = _collect_feature_bank_view_group(
                    backbone,
                    loader,
                    device,
                    group_specs,
                    backbone_kwargs=forward_kwargs,
                    progress=progress,
                    desc=f"{domain} {group_specs[0].transform_key}",
                )
                if reference_labels is None:
                    if batch_labels is not None:
                        raise ValueError(
                            f"Loader label availability changed while extracting domain '{domain}'."
                        )
                elif (
                    batch_labels is None
                    or batch_labels.shape != reference_labels.shape
                    or not np.array_equal(batch_labels, reference_labels)
                ):
                    raise ValueError(
                        "Loader label order changed while extracting "
                        f"feature-bank domain '{domain}'."
                    )
                for spec in group_specs:
                    feature_array = view_features[spec.key]
                    expected_count = len(dataset)
                    if feature_array.shape[0] != expected_count:
                        raise ValueError(
                            f"Feature bank view '{domain}/{spec.key}' has "
                            f"{feature_array.shape[0]} features but "
                            f"{expected_count} samples."
                        )
                    view_group = views_group.create_group(spec.key)
                    view_group.create_dataset("feature", data=feature_array)
                    for attr_key, attr_value in spec.attrs().items():
                        view_group.attrs[attr_key] = attr_value
    return {
        "output": str(output),
        "domains": list(selected_domains),
        "views_per_domain": len(specs),
        "view_keys": [spec.key for spec in specs],
    }
