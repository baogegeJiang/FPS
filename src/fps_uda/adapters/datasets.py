from __future__ import annotations

import random
from dataclasses import dataclass, field, fields, replace
from math import ceil, sqrt
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import yaml

from fps_uda.models import BackboneConfig


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _vision_imports():
    try:
        import torch
        from PIL import Image
        from torch.utils.data import DataLoader, DistributedSampler
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
    except Exception as exc:
        raise RuntimeError(
            "Dataset adapters require the vision extra: pip install 'fps-uda[vision]'."
        ) from exc
    return (
        torch,
        Image,
        DataLoader,
        DistributedSampler,
        transforms,
        InterpolationMode,
    )


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_tuple3(value: Sequence[float], *, name: str) -> Tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values.")
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def _optional_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass(frozen=True)
class LoaderConfig:
    batch_size: int = 32
    num_workers: int = 4
    seed: int = 515
    strong_aug: bool = False
    distributed: bool = False
    pin_memory: bool = True
    drop_last: bool = False

    @classmethod
    def from_mapping(cls, mapping: Optional[Mapping[str, Any]]) -> "LoaderConfig":
        data = dict(mapping or {})
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown loader config fields: {', '.join(unknown)}.")
        cfg = cls(**data)
        if cfg.batch_size <= 0:
            raise ValueError("loader.batch_size must be positive.")
        if cfg.num_workers < 0:
            raise ValueError("loader.num_workers must be non-negative.")
        return replace(
            cfg,
            batch_size=int(cfg.batch_size),
            num_workers=int(cfg.num_workers),
            seed=int(cfg.seed),
            strong_aug=_as_bool(cfg.strong_aug),
            distributed=_as_bool(cfg.distributed),
            pin_memory=_as_bool(cfg.pin_memory),
            drop_last=_as_bool(cfg.drop_last),
        )


@dataclass(frozen=True)
class TransformConfig:
    interpolation: str = "bilinear"
    antialias: bool = True
    pad_fill: int = 127
    mean: Tuple[float, float, float] = IMAGENET_MEAN
    std: Tuple[float, float, float] = IMAGENET_STD

    @classmethod
    def from_mapping(cls, mapping: Optional[Mapping[str, Any]]) -> "TransformConfig":
        data = dict(mapping or {})
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown transform config fields: {', '.join(unknown)}.")
        if "mean" in data:
            data["mean"] = _as_tuple3(data["mean"], name="transform.mean")
        if "std" in data:
            data["std"] = _as_tuple3(data["std"], name="transform.std")
        cfg = cls(**data)
        interpolation = str(cfg.interpolation).strip().lower()
        if interpolation not in {"bilinear", "bicubic", "nearest"}:
            raise ValueError("transform.interpolation must be bilinear, bicubic, or nearest.")
        return replace(
            cfg,
            interpolation=interpolation,
            antialias=_as_bool(cfg.antialias),
            pad_fill=int(cfg.pad_fill),
            mean=_as_tuple3(cfg.mean, name="transform.mean"),
            std=_as_tuple3(cfg.std, name="transform.std"),
        )


@dataclass(frozen=True)
class DomainSpec:
    name: str
    kind: str = "manifest"
    image_root: Optional[str] = None
    manifest: Optional[str] = None
    path_column: int = 0
    label_column: Optional[int] = None
    class_to_idx: Union[str, Mapping[str, int]] = "auto"

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], *, index: int) -> "DomainSpec":
        data = dict(mapping)
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown domains[{index}] fields: {', '.join(unknown)}.")
        if "name" not in data or not str(data["name"]).strip():
            raise ValueError(f"domains[{index}].name must be non-empty.")
        cfg = cls(**data)
        kind = str(cfg.kind).strip().lower()
        if kind not in {"manifest", "class_folder"}:
            raise ValueError("domain kind must be 'manifest' or 'class_folder'.")
        if kind == "manifest" and not _optional_path(cfg.manifest):
            raise ValueError(f"domains[{index}].manifest is required for kind='manifest'.")
        label_column = None if cfg.label_column is None else int(cfg.label_column)
        if int(cfg.path_column) < 0 or (label_column is not None and label_column < 0):
            raise ValueError("path_column and label_column must be non-negative.")
        return replace(
            cfg,
            name=str(cfg.name).strip(),
            kind=kind,
            image_root=_optional_path(cfg.image_root) or str(cfg.name).strip(),
            manifest=_optional_path(cfg.manifest),
            path_column=int(cfg.path_column),
            label_column=label_column,
        )


@dataclass(frozen=True)
class FeatureBankConfig:
    water_level: float = 0.0
    mute_padding_in_pool: bool = False
    views: Tuple[Mapping[str, Any], ...] = ()

    @classmethod
    def from_mapping(cls, mapping: Optional[Mapping[str, Any]]) -> "FeatureBankConfig":
        data = dict(mapping or {})
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown feature_bank config fields: {', '.join(unknown)}.")
        views = data.get("views", ())
        if views is None:
            views = ()
        if isinstance(views, (str, bytes)) or not isinstance(views, Sequence):
            raise ValueError("feature_bank.views must be a list of view mappings.")
        data["views"] = tuple(dict(view) for view in views)
        cfg = cls(**data)
        return replace(
            cfg,
            water_level=float(cfg.water_level),
            mute_padding_in_pool=_as_bool(cfg.mute_padding_in_pool),
            views=tuple(cfg.views),
        )


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root_dir: str
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    domains: Tuple[DomainSpec, ...] = ()
    feature_bank: FeatureBankConfig = field(default_factory=FeatureBankConfig)
    source_domain: Optional[str] = None
    target_domain: Optional[str] = None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], **overrides: Any) -> "DatasetConfig":
        data = dict(mapping)
        forbidden = sorted(set(data).intersection({"dataset", "model_family", "preprocess"}))
        if forbidden:
            raise ValueError(
                "Dataset config no longer supports behavior fields: "
                + ", ".join(forbidden)
                + ". Use explicit domains and transform fields instead."
            )

        loader_data = dict(data.pop("loader", {}) or {})
        transform_data = dict(data.pop("transform", {}) or {})
        feature_bank_data = dict(data.pop("feature_bank", {}) or {})
        backbone_data = dict(data.pop("backbone", {}) or {})
        loader_fields = {field.name for field in fields(LoaderConfig)}

        for key in list(data):
            if key in loader_fields:
                loader_data[key] = data.pop(key)

        for key, value in overrides.items():
            if value is None:
                continue
            if key in loader_fields:
                loader_data[key] = value
            else:
                data[key] = value

        raw_domains = data.pop("domains", ())
        if not isinstance(raw_domains, Sequence) or isinstance(raw_domains, (str, bytes)):
            raise ValueError("domains must be a list of domain mappings.")
        domain_specs = tuple(
            DomainSpec.from_mapping(domain, index=index)
            for index, domain in enumerate(raw_domains)
        )

        known = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown dataset config fields: {', '.join(unknown)}.")
        if "name" not in data or not str(data["name"]).strip():
            raise ValueError("Dataset config requires a non-empty name.")
        if "root_dir" not in data or not str(data["root_dir"]).strip():
            raise ValueError("Dataset config requires root_dir.")

        cfg = cls(
            name=str(data["name"]).strip(),
            root_dir=str(data["root_dir"]),
            backbone=BackboneConfig.from_mapping(backbone_data),
            loader=LoaderConfig.from_mapping(loader_data),
            transform=TransformConfig.from_mapping(transform_data),
            domains=domain_specs,
            feature_bank=FeatureBankConfig.from_mapping(feature_bank_data),
            source_domain=_optional_path(data.get("source_domain")),
            target_domain=_optional_path(data.get("target_domain")),
        )
        return cfg.validate()

    def validate(self) -> "DatasetConfig":
        if not self.domains:
            raise ValueError("Dataset config must define at least one domain.")
        names = [domain.name for domain in self.domains]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate domain names: {', '.join(duplicates)}.")
        return self

    @property
    def batch_size(self) -> int:
        return self.loader.batch_size

    @property
    def num_workers(self) -> int:
        return self.loader.num_workers

    @property
    def seed(self) -> int:
        return self.loader.seed

    @property
    def strong_aug(self) -> bool:
        return self.loader.strong_aug

    @property
    def distributed(self) -> bool:
        return self.loader.distributed

    @property
    def pin_memory(self) -> bool:
        return self.loader.pin_memory

    @property
    def drop_last(self) -> bool:
        return self.loader.drop_last

    def domain_spec(self, name: str) -> DomainSpec:
        for domain in self.domains:
            if domain.name == name:
                return domain
        available = ", ".join(domain.name for domain in self.domains) or "<none>"
        raise KeyError(f"Unknown domain '{name}'. Available domains: {available}.")


def load_dataset_config(path: str, **overrides: Any) -> DatasetConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {path}")
    return DatasetConfig.from_mapping(data, **overrides)


def discover_dataset_domains(config: Union[DatasetConfig, Mapping[str, Any]]) -> Tuple[str, ...]:
    """Return explicitly configured domain names for feature-bank extraction."""
    if not isinstance(config, DatasetConfig):
        config = DatasetConfig.from_mapping(config)
    return tuple(domain.name for domain in config.domains)


def pad_to_square(image, fill: int = 255 // 2, aspect_jitter: float = 0.0):
    _, _, _, _, transforms, _ = _vision_imports()
    width, height = image.size
    base = max(width, height)
    if aspect_jitter > 0:
        ratio = random.uniform(max(1e-6, 1.0 - aspect_jitter), 1.0 + aspect_jitter)
    else:
        ratio = 1.0
    target_width = max(width, int(ceil(base * sqrt(ratio))))
    target_height = max(height, int(ceil(base / sqrt(ratio))))
    pad_width = target_width - width
    pad_height = target_height - height
    left = pad_width // 2
    top = pad_height // 2
    padding = (left, top, pad_width - left, pad_height - top)
    return transforms.functional.pad(image, padding, fill=fill)


def _path_under(root: Path, value: Optional[str]) -> Path:
    path = Path(value or "")
    return path if path.is_absolute() else root / path


def _resolve_manifest_image(root: Path, image_root: Path, image_token: str) -> Path:
    path = Path(image_token)
    if path.is_absolute():
        return path
    candidates = [
        image_root / path,
        image_root / "images" / path,
        root / path,
        root / "images" / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


class _ManifestDataset:
    def __init__(self, root_dir: str, spec: DomainSpec, transform):
        _, Image, *_ = _vision_imports()
        self._image_cls = Image
        self.root_dir = Path(root_dir)
        self.spec = spec
        self.transform = transform
        self.image_root = _path_under(self.root_dir, spec.image_root)
        self.manifest = _path_under(self.root_dir, spec.manifest)
        if not self.manifest.exists():
            raise FileNotFoundError(f"Missing domain manifest: {self.manifest}")
        self.samples: list[tuple[Path, Optional[int]]] = []
        with self.manifest.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                parts = line.strip().split()
                if not parts:
                    continue
                max_column = max(
                    spec.path_column,
                    spec.label_column if spec.label_column is not None else spec.path_column,
                )
                if len(parts) <= max_column:
                    raise ValueError(
                        f"Manifest {self.manifest}:{line_no} has {len(parts)} columns; "
                        f"needs column {max_column}."
                    )
                label = None
                if spec.label_column is not None:
                    label = int(parts[spec.label_column])
                image_path = _resolve_manifest_image(
                    self.root_dir,
                    self.image_root,
                    parts[spec.path_column],
                )
                self.samples.append((image_path, label))
        if not self.samples:
            raise FileNotFoundError(f"Domain manifest is empty: {self.manifest}")

    @property
    def has_labels(self) -> bool:
        return all(label is not None for _, label in self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = self._image_cls.open(path).convert("RGB")
        return self.transform(image), (-1 if label is None else int(label))


def _read_class_folder_samples(
    domain_dir: Path,
    *,
    class_to_idx: Union[str, Mapping[str, int]] = "auto",
):
    roots = []
    images_dir = domain_dir / "images"
    if images_dir.exists():
        roots.append(images_dir)
    roots.append(domain_dir)
    extensions = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
    for root in roots:
        class_dirs = [
            child
            for child in root.iterdir()
            if child.is_dir() and child.name not in {"annotations", "images"}
        ]
        if not class_dirs:
            continue
        if isinstance(class_to_idx, Mapping):
            class_items = [
                (class_dir, int(class_to_idx[class_dir.name]))
                for class_dir in class_dirs
                if class_dir.name in class_to_idx
            ]
            class_items = sorted(class_items, key=lambda item: item[1])
        else:
            class_items = [
                (class_dir, label)
                for label, class_dir in enumerate(
                    sorted(class_dirs, key=lambda path: path.name.lower())
                )
            ]
        samples = []
        for class_dir, label in class_items:
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in extensions:
                    samples.append((image_path, int(label)))
        if samples:
            return samples
    return []


class _ClassFolderDataset:
    def __init__(self, root_dir: str, spec: DomainSpec, transform):
        _, Image, *_ = _vision_imports()
        self._image_cls = Image
        self.root_dir = Path(root_dir)
        self.spec = spec
        self.transform = transform
        self.domain_dir = _path_under(self.root_dir, spec.image_root)
        self.samples = _read_class_folder_samples(
            self.domain_dir,
            class_to_idx=spec.class_to_idx,
        )
        if not self.samples:
            raise FileNotFoundError(f"Missing class-folder images for domain: {self.domain_dir}")

    @property
    def has_labels(self) -> bool:
        return True

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = self._image_cls.open(path).convert("RGB")
        return self.transform(image), int(label)


def build_domain_dataset(
    config: Union[DatasetConfig, Mapping[str, Any]],
    domain: str,
    transform,
):
    """Build one ordered domain dataset for feature-bank extraction."""
    if not isinstance(config, DatasetConfig):
        config = DatasetConfig.from_mapping(config)
    spec = config.domain_spec(domain)
    if spec.kind == "manifest":
        return _ManifestDataset(config.root_dir, spec, transform)
    if spec.kind == "class_folder":
        return _ClassFolderDataset(config.root_dir, spec, transform)
    raise ValueError(f"Unsupported domain kind: {spec.kind}")


def dataset_labels(dataset) -> Optional[list[int]]:
    if not hasattr(dataset, "samples"):
        raise ValueError("Dataset object does not expose ordered samples.")
    labels = [label for _, label in dataset.samples]
    if any(label is None for label in labels):
        return None
    return [int(label) for label in labels]


def _make_loader(
    dataset,
    config: DatasetConfig,
    *,
    shuffle: bool,
    origin_compatible: bool = False,
):
    torch, _, DataLoader, DistributedSampler, *_ = _vision_imports()
    sampler = DistributedSampler(dataset, shuffle=shuffle) if config.distributed else None
    generator = None
    if not origin_compatible:
        generator = torch.Generator()
        generator.manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=config.num_workers,
        drop_last=config.drop_last,
        pin_memory=config.pin_memory,
        generator=generator,
    )
