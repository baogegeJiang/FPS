from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, replace
from typing import Any, Mapping, Optional, Union

import torch
from torch import nn


DEFAULT_RESNET_BACKBONE = "resnet50"
DEFAULT_VIT_BACKBONE = "vit_base_patch16_224.augreg2_in21k_ft_in1k"

BACKBONE_BACKENDS = {"torchvision", "timm", "hf_vit", "clip", "custom"}
FEATURE_TYPES = {"spatial", "token", "flat"}
RANDOM_POOLING_STRATEGIES = {
    "spatial_shared",
    "token_channel_squared",
    "token_shared",
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _none_if_blank(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _is_random_pooling_name(pooling: str) -> bool:
    return str(pooling).startswith("pool_")


def _normalize_state_dict(state):
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must contain a state dict.")
    return {key.replace("module.", ""): value for key, value in state.items()}


def _select_encoder_state_dict(state, encoder: nn.Module):
    encoder_keys = set(encoder.state_dict().keys())
    candidates = [state]
    for prefix in ("encoder.", "backbone.encoder.", "backbone."):
        stripped = {
            key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)
        }
        if stripped:
            candidates.append(stripped)
    return max(candidates, key=lambda candidate: len(encoder_keys.intersection(candidate)))


@dataclass(frozen=True)
class BackbonePoolingConfig:
    feature_type: str = "spatial"
    random_strategy: str = "spatial_shared"

    @classmethod
    def from_mapping(
        cls, mapping: Optional[Mapping[str, Any]]
    ) -> "BackbonePoolingConfig":
        data = dict(mapping or {})
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown backbone.pooling fields: {', '.join(unknown)}.")
        cfg = cls(**data)
        feature_type = str(cfg.feature_type).strip().lower()
        random_strategy = str(cfg.random_strategy).strip().lower()
        if feature_type not in FEATURE_TYPES:
            raise ValueError(
                "backbone.pooling.feature_type must be spatial, token, or flat."
            )
        if random_strategy not in RANDOM_POOLING_STRATEGIES:
            raise ValueError(
                "backbone.pooling.random_strategy must be spatial_shared, "
                "token_channel_squared, or token_shared."
            )
        if feature_type == "spatial" and random_strategy != "spatial_shared":
            raise ValueError("spatial features require random_strategy=spatial_shared.")
        if feature_type == "token" and not random_strategy.startswith("token_"):
            raise ValueError("token features require a token random pooling strategy.")
        return replace(
            cfg,
            feature_type=feature_type,
            random_strategy=random_strategy,
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "feature_type": self.feature_type,
            "random_strategy": self.random_strategy,
        }


@dataclass(frozen=True)
class BackboneConfig:
    backend: str = "torchvision"
    name: str = DEFAULT_RESNET_BACKBONE
    pretrained: bool = True
    weights: Optional[str] = "IMAGENET1K_V1"
    checkpoint: Optional[str] = None
    in_features: Optional[int] = None
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    pooling: BackbonePoolingConfig = field(default_factory=BackbonePoolingConfig)

    @classmethod
    def from_mapping(cls, mapping: Optional[Mapping[str, Any]]) -> "BackboneConfig":
        data = dict(mapping or {})
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown backbone config fields: {', '.join(unknown)}.")
        pooling = BackbonePoolingConfig.from_mapping(data.pop("pooling", None))
        kwargs = dict(data.pop("kwargs", {}) or {})
        cfg = cls(**data, kwargs=kwargs, pooling=pooling)
        backend = str(cfg.backend).strip().lower()
        if backend not in BACKBONE_BACKENDS:
            raise ValueError(
                "backbone.backend must be torchvision, timm, hf_vit, clip, or custom."
            )
        name = str(cfg.name).strip()
        if not name:
            raise ValueError("backbone.name must be non-empty.")
        in_features = None if cfg.in_features is None else int(cfg.in_features)
        if in_features is not None and in_features <= 0:
            raise ValueError("backbone.in_features must be positive when set.")
        return replace(
            cfg,
            backend=backend,
            name=name,
            pretrained=_as_bool(cfg.pretrained),
            weights=_none_if_blank(cfg.weights),
            checkpoint=_none_if_blank(cfg.checkpoint),
            in_features=in_features,
            kwargs=kwargs,
            pooling=pooling,
        )

    @classmethod
    def legacy(
        cls,
        name: str,
        *,
        backend: Optional[str] = None,
        pretrained: bool = True,
        checkpoint_path: Optional[str] = None,
        in_features: Optional[int] = None,
        weights: Optional[str] = None,
        kwargs: Optional[Mapping[str, Any]] = None,
        pooling: Optional[Mapping[str, Any]] = None,
    ) -> "BackboneConfig":
        lower = str(name).lower()
        resolved_backend = (backend or "").strip().lower()
        resolved_kwargs = dict(kwargs or {})
        if not resolved_backend:
            if lower.startswith("resnet"):
                resolved_backend = "torchvision"
            elif "clip" in lower:
                resolved_backend = "clip"
            elif _is_hf_vit_path(name) or _looks_like_hf_vit_name(name):
                resolved_backend = "hf_vit"
            elif checkpoint_path and _is_hf_vit_path(checkpoint_path):
                resolved_backend = "hf_vit"
            else:
                resolved_backend = "timm"
        if pooling is None:
            if resolved_backend == "torchvision":
                pooling = {
                    "feature_type": "spatial",
                    "random_strategy": "spatial_shared",
                }
            else:
                pooling = {
                    "feature_type": "token",
                    "random_strategy": "token_channel_squared",
                }
        if resolved_backend == "timm":
            resolved_kwargs.setdefault("class_token", True)
            resolved_kwargs.setdefault("global_pool", "token")
        resolved_weights = weights
        if resolved_backend == "torchvision" and pretrained and resolved_weights is None:
            resolved_weights = "IMAGENET1K_V1"
        return cls.from_mapping(
            {
                "backend": resolved_backend,
                "name": name,
                "pretrained": pretrained,
                "weights": resolved_weights,
                "checkpoint": checkpoint_path,
                "in_features": in_features,
                "kwargs": resolved_kwargs,
                "pooling": pooling,
            }
        )

    def with_overrides(
        self,
        *,
        backend: Optional[str] = None,
        name: Optional[str] = None,
        pretrained: Optional[bool] = None,
        weights: Optional[str] = None,
        checkpoint: Optional[str] = None,
        in_features: Optional[int] = None,
        kwargs: Optional[Mapping[str, Any]] = None,
        pooling: Optional[Mapping[str, Any]] = None,
    ) -> "BackboneConfig":
        data = self.as_dict()
        if backend is not None:
            data["backend"] = backend
        if name is not None:
            data["name"] = name
        if pretrained is not None:
            data["pretrained"] = pretrained
        if weights is not None:
            data["weights"] = weights
        if checkpoint is not None:
            data["checkpoint"] = checkpoint
        if in_features is not None:
            data["in_features"] = in_features
        if kwargs:
            merged_kwargs = dict(data["kwargs"])
            merged_kwargs.update(dict(kwargs))
            data["kwargs"] = merged_kwargs
        if pooling:
            merged_pooling = dict(data["pooling"])
            merged_pooling.update(dict(pooling))
            data["pooling"] = merged_pooling
        return BackboneConfig.from_mapping(data)

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "name": self.name,
            "pretrained": self.pretrained,
            "weights": self.weights,
            "checkpoint": self.checkpoint,
            "in_features": self.in_features,
            "kwargs": dict(self.kwargs),
            "pooling": self.pooling.as_dict(),
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "backbone_backend": self.backend,
            "backbone_name": self.name,
            "backbone_pretrained": bool(self.pretrained),
            "backbone_weights": (
                "" if (not self.pretrained or self.weights is None) else self.weights
            ),
            "backbone_checkpoint": "" if self.checkpoint is None else self.checkpoint,
            "backbone_kwargs_json": json.dumps(dict(self.kwargs), sort_keys=True),
            "pooling_feature_type": self.pooling.feature_type,
            "pooling_random_strategy": self.pooling.random_strategy,
        }


class FeaturePooler(nn.Module):
    def __init__(self, feature_type: str, random_strategy: str):
        super().__init__()
        self.feature_type = feature_type
        self.random_strategy = random_strategy

    @staticmethod
    def _count_true_prefix(mask: torch.Tensor) -> torch.Tensor:
        _, length = mask.shape
        all_true = mask.all(dim=1)
        first_false = (~mask).to(torch.float32).argmax(dim=1)
        return torch.where(all_true, torch.full_like(first_false, length), first_false)

    def _detect_border_padding_mask(
        self,
        x: torch.Tensor,
        *,
        tol: float = 1e-6,
        downsample: int = 1,
    ) -> torch.Tensor:
        batch, _, height, width = x.shape
        if height <= 2 or width <= 2:
            return torch.ones((batch, 1, height, width), device=x.device, dtype=x.dtype)

        stride = max(1, int(downsample))
        if stride > 1:
            det_height = max(2, height // stride)
            det_width = max(2, width // stride)
            x_det = torch.nn.functional.interpolate(
                x, size=(det_height, det_width), mode="nearest"
            )
        else:
            x_det = x
            det_height, det_width = height, width

        row_const = (x_det - x_det[:, :, :, :1]).abs().amax(dim=(1, 3)) <= float(tol)
        col_const = (x_det - x_det[:, :, :1, :]).abs().amax(dim=(1, 2)) <= float(tol)
        top_det = self._count_true_prefix(row_const)
        bottom_det = self._count_true_prefix(torch.flip(row_const, dims=[1]))
        left_det = self._count_true_prefix(col_const)
        right_det = self._count_true_prefix(torch.flip(col_const, dims=[1]))

        top = torch.floor(top_det.to(torch.float32) * (height / float(det_height))).to(
            torch.long
        )
        bottom = torch.floor(
            bottom_det.to(torch.float32) * (height / float(det_height))
        ).to(torch.long)
        left = torch.floor(left_det.to(torch.float32) * (width / float(det_width))).to(
            torch.long
        )
        right = torch.floor(
            right_det.to(torch.float32) * (width / float(det_width))
        ).to(torch.long)

        bad_row = top + bottom >= height
        bad_col = left + right >= width
        top = torch.where(bad_row, torch.zeros_like(top), top)
        bottom = torch.where(bad_row, torch.zeros_like(bottom), bottom)
        left = torch.where(bad_col, torch.zeros_like(left), left)
        right = torch.where(bad_col, torch.zeros_like(right), right)

        rows = torch.arange(height, device=x.device).view(1, height, 1)
        cols = torch.arange(width, device=x.device).view(1, 1, width)
        valid_row = (rows >= top.view(batch, 1, 1)) & (
            rows < (height - bottom).view(batch, 1, 1)
        )
        valid_col = (cols >= left.view(batch, 1, 1)) & (
            cols < (width - right).view(batch, 1, 1)
        )
        return (valid_row & valid_col).unsqueeze(1).to(dtype=x.dtype)

    def _feature_mask(
        self,
        inputs: torch.Tensor,
        features: torch.Tensor,
        *,
        mute_padding_in_pool: bool,
        padding_detect_tol: float,
        padding_detect_downsample: int,
    ) -> Optional[torch.Tensor]:
        if not mute_padding_in_pool or self.feature_type == "flat":
            return None
        with torch.no_grad():
            input_mask = self._detect_border_padding_mask(
                inputs,
                tol=padding_detect_tol,
                downsample=padding_detect_downsample,
            )
        if self.feature_type == "spatial":
            if features.dim() != 4:
                return None
            return torch.nn.functional.interpolate(
                input_mask, size=features.shape[-2:], mode="nearest"
            ).to(dtype=features.dtype, device=features.device)
        if self.feature_type == "token":
            if features.dim() != 3:
                return None
            token_num = features.shape[1]
            grid_size = int(round(float(token_num) ** 0.5))
            if token_num <= 0 or grid_size * grid_size != token_num:
                return torch.ones(
                    (inputs.shape[0], token_num, 1),
                    dtype=features.dtype,
                    device=features.device,
                )
            token_mask = torch.nn.functional.interpolate(
                input_mask, size=(grid_size, grid_size), mode="nearest"
            )
            token_mask = token_mask.reshape(inputs.shape[0], 1, token_num).transpose(1, 2)
            return token_mask.to(dtype=features.dtype, device=features.device)
        return None

    def _clean_spatial(
        self, features: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if mask is None:
            return features.mean(dim=(2, 3))
        denom = mask.sum(dim=(2, 3)).clamp_min(1e-6)
        return (features * mask).sum(dim=(2, 3)) / denom

    def _clean_token(
        self, features: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if mask is None:
            return features.mean(dim=1)
        weight = mask.to(dtype=features.dtype, device=features.device)
        denom = weight.sum(dim=1).clamp_min(1e-6)
        return (features * weight).sum(dim=1) / denom

    def clean(self, features: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.feature_type == "spatial":
            if features.dim() == 4:
                return self._clean_spatial(features, mask)
            if features.dim() == 2:
                return features
        if self.feature_type == "token":
            if features.dim() == 3:
                return self._clean_token(features, mask)
            if features.dim() == 2:
                return features
        if self.feature_type == "flat":
            if features.dim() == 2:
                return features
            if features.dim() == 4:
                return features.mean(dim=(2, 3))
            if features.dim() == 3:
                return features.mean(dim=1)
        raise ValueError(
            f"Cannot clean-pool {self.feature_type} features with shape {tuple(features.shape)}."
        )

    def random_pool(
        self,
        features: torch.Tensor,
        water_level: float,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.random_strategy == "spatial_shared":
            if features.dim() != 4:
                return self.clean(features, mask)
            batch, _, height, width = features.shape
            pool_mask = torch.rand(batch, height, width, device=features.device)
            pool_mask = pool_mask + float(water_level)
            if mask is not None:
                pool_mask = pool_mask * mask.squeeze(1).to(
                    dtype=pool_mask.dtype, device=pool_mask.device
                )
            pool_mask = pool_mask / pool_mask.sum(dim=(1, 2), keepdim=True).clamp_min(
                1e-12
            )
            return torch.einsum("bchw,bhw->bc", features, pool_mask)

        if self.random_strategy == "token_channel_squared":
            if features.dim() != 3:
                return self.clean(features, mask)
            batch, length, channels = features.shape
            pool_mask = torch.rand(batch, length, channels, device=features.device) ** 2
            pool_mask = pool_mask + float(water_level)
            if mask is not None:
                pool_mask = pool_mask * mask.to(
                    dtype=pool_mask.dtype, device=pool_mask.device
                )
            pool_mask = pool_mask / pool_mask.sum(dim=1, keepdim=True).clamp_min(1e-12)
            return torch.einsum("blc,blc->bc", features, pool_mask)

        if self.random_strategy == "token_shared":
            if features.dim() != 3:
                return self.clean(features, mask)
            batch, length, _ = features.shape
            pool_mask = torch.rand(batch, length, device=features.device)
            pool_mask = pool_mask + float(water_level)
            if mask is not None:
                pool_mask = pool_mask * mask.squeeze(-1).to(
                    dtype=pool_mask.dtype, device=pool_mask.device
                )
            pool_mask = pool_mask / pool_mask.sum(dim=1, keepdim=True).clamp_min(1e-12)
            return torch.einsum("blc,bl->bc", features, pool_mask)

        raise ValueError(f"Unknown random pooling strategy: {self.random_strategy}")

    def pool(
        self,
        features: torch.Tensor,
        inputs: torch.Tensor,
        *,
        random_pool: bool,
        water_level: float,
        mute_padding_in_pool: bool,
        padding_detect_tol: float,
        padding_detect_downsample: int,
    ) -> torch.Tensor:
        mask = self._feature_mask(
            inputs,
            features,
            mute_padding_in_pool=mute_padding_in_pool,
            padding_detect_tol=padding_detect_tol,
            padding_detect_downsample=padding_detect_downsample,
        )
        if random_pool:
            return self.random_pool(features, water_level, mask)
        return self.clean(features, mask)

    def pool_many(
        self,
        features: torch.Tensor,
        inputs: torch.Tensor,
        *,
        poolings,
        water_level: float,
        mute_padding_in_pool: bool,
        padding_detect_tol: float,
        padding_detect_downsample: int,
    ) -> dict[str, torch.Tensor]:
        mask = self._feature_mask(
            inputs,
            features,
            mute_padding_in_pool=mute_padding_in_pool,
            padding_detect_tol=padding_detect_tol,
            padding_detect_downsample=padding_detect_downsample,
        )
        outputs = {}
        for pooling in poolings:
            if pooling == "clean":
                outputs[pooling] = self.clean(features, mask)
            elif _is_random_pooling_name(pooling):
                outputs[pooling] = self.random_pool(features, water_level, mask)
            else:
                raise ValueError("pooling must be 'clean' or start with 'pool_'.")
        return outputs


class _TorchvisionSpatialExtractor(nn.Module):
    def __init__(self, config: BackboneConfig):
        super().__init__()
        try:
            import torchvision
        except Exception as exc:
            raise RuntimeError("torchvision is required for torchvision backbones.") from exc
        if not hasattr(torchvision.models, config.name):
            raise ValueError(f"Unknown torchvision model: {config.name}")
        builder = getattr(torchvision.models, config.name)
        weights = config.weights if config.pretrained else None
        kwargs = dict(config.kwargs)
        encoder = builder(weights=weights, **kwargs)
        if config.checkpoint:
            try:
                state = torch.load(config.checkpoint, map_location="cpu", weights_only=True)
            except RuntimeError as exc:
                if "weights_only=True" in str(exc):
                    raise RuntimeError(
                        f"Cannot safely load legacy checkpoint '{config.checkpoint}'. "
                        "Convert it to a plain state_dict with a trusted script, then pass "
                        "the converted file as --checkpoint."
                    ) from exc
                raise
            state = _normalize_state_dict(state)
            state = _select_encoder_state_dict(state, encoder)
            message = encoder.load_state_dict(state, strict=False)
            print(
                f"Loaded torchvision checkpoint from {config.checkpoint}; "
                f"missing={len(message.missing_keys)}, "
                f"unexpected={len(message.unexpected_keys)}"
            )
        self.encoder = encoder
        self.features = nn.Sequential(*list(encoder.children())[:-2])
        self.in_features = int(config.in_features or encoder.fc.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class _TimmExtractor(nn.Module):
    def __init__(self, config: BackboneConfig):
        super().__init__()
        try:
            import timm
        except Exception as exc:
            raise RuntimeError("timm is required for timm backbones.") from exc
        kwargs = dict(config.kwargs)
        if config.checkpoint:
            kwargs["checkpoint_path"] = config.checkpoint
        self.model = timm.create_model(
            config.name,
            pretrained=config.pretrained,
            **kwargs,
        )
        self.feature_type = config.pooling.feature_type
        self.in_features = int(config.in_features or getattr(self.model, "num_features", 0))
        if self.in_features <= 0:
            raise ValueError("Could not infer timm backbone feature dimension.")

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "forward_features"):
            features = self.model.forward_features(x)
        else:
            features = self.model(x)
        if isinstance(features, dict):
            for key in ("last_hidden_state", "x", "features"):
                if key in features:
                    features = features[key]
                    break
            else:
                raise ValueError("Unsupported timm feature dict: no tensor found.")
        return features

    def _token_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 4:
            batch, channels, height, width = features.shape
            return features.reshape(batch, channels, height * width).transpose(1, 2)
        if features.dim() == 2:
            return features.unsqueeze(1)
        if features.dim() != 3:
            raise ValueError(
                "Expected timm features [B, L, C], [B, C, H, W], or [B, C], "
                f"got {tuple(features.shape)}"
            )
        prefix = int(getattr(self.model, "num_prefix_tokens", 0) or 0)
        if prefix > 0 and features.size(1) > prefix:
            candidate = features[:, prefix:, :]
            grid_size = int(round(float(candidate.size(1)) ** 0.5))
            if grid_size * grid_size == candidate.size(1):
                return candidate
        if features.size(1) > 1:
            candidate = features[:, 1:, :]
            grid_size = int(round(float(candidate.size(1)) ** 0.5))
            if grid_size * grid_size == candidate.size(1):
                return candidate
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._forward_features(x)
        if self.feature_type == "token":
            return self._token_features(features)
        if self.feature_type == "spatial":
            if features.dim() != 4:
                raise ValueError(
                    "timm spatial pooling expects [B, C, H, W] features, "
                    f"got {tuple(features.shape)}"
                )
            return features
        if self.feature_type == "flat":
            if features.dim() == 2:
                return features
            if features.dim() == 4:
                return features.mean(dim=(2, 3))
            if features.dim() == 3:
                return features.mean(dim=1)
        raise ValueError(f"Unsupported timm feature type: {self.feature_type}")


class _HuggingFaceViTExtractor(nn.Module):
    def __init__(self, config: BackboneConfig):
        super().__init__()
        try:
            from transformers import ViTModel
        except Exception as exc:
            raise RuntimeError("transformers is required for hf_vit backbones.") from exc
        model_name = config.checkpoint or config.name
        self.model = ViTModel.from_pretrained(model_name)
        self.in_features = int(
            config.in_features or getattr(self.model.config, "hidden_size", 0)
        )
        if self.in_features <= 0:
            raise ValueError("Could not infer HuggingFace ViT feature dimension.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x)
        tokens = out["last_hidden_state"] if isinstance(out, dict) else out.last_hidden_state
        if tokens.dim() != 3:
            raise ValueError(
                f"Expected HuggingFace ViT tokens [B, L, C], got {tuple(tokens.shape)}"
            )
        if tokens.size(1) > 1:
            candidate = tokens[:, 1:, :]
            grid_size = int(round(float(candidate.size(1)) ** 0.5))
            if grid_size * grid_size == candidate.size(1):
                return candidate
        return tokens


class _ClipVisionExtractor(nn.Module):
    def __init__(self, config: BackboneConfig):
        super().__init__()
        try:
            from transformers import CLIPModel
        except Exception as exc:
            raise RuntimeError("transformers is required for clip backbones.") from exc
        model_name = config.checkpoint or config.name
        self.model = CLIPModel.from_pretrained(model_name).vision_model
        self.in_features = int(config.in_features or 768)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x)
        tokens = out["last_hidden_state"] if isinstance(out, dict) else out.last_hidden_state
        if tokens.dim() != 3:
            raise ValueError(f"Expected CLIP tokens [B, L, C], got {tuple(tokens.shape)}")
        return tokens[:, 1:, :] if tokens.size(1) > 1 else tokens


class PoolableBackbone(nn.Module):
    def __init__(
        self,
        extractor: nn.Module,
        config: BackboneConfig,
    ):
        super().__init__()
        self.extractor = extractor
        self.config = config
        self.in_features = int(getattr(extractor, "in_features"))
        self.pooler = FeaturePooler(
            config.pooling.feature_type,
            config.pooling.random_strategy,
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.extractor(x)

    def forward(
        self,
        x: torch.Tensor,
        water_level: float = 0.0,
        random_pool: bool = False,
        mute_padding_in_pool: bool = False,
        padding_detect_tol: float = 1e-6,
        padding_detect_downsample: int = 1,
        **_: object,
    ) -> torch.Tensor:
        features = self.extract_features(x)
        return self.pooler.pool(
            features,
            x,
            random_pool=random_pool,
            water_level=water_level,
            mute_padding_in_pool=mute_padding_in_pool,
            padding_detect_tol=padding_detect_tol,
            padding_detect_downsample=padding_detect_downsample,
        )

    def extract_pooling_views(
        self,
        x: torch.Tensor,
        *,
        poolings,
        water_level: float = 0.0,
        mute_padding_in_pool: bool = False,
        padding_detect_tol: float = 1e-6,
        padding_detect_downsample: int = 1,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        features = self.extract_features(x)
        return self.pooler.pool_many(
            features,
            x,
            poolings=poolings,
            water_level=water_level,
            mute_padding_in_pool=mute_padding_in_pool,
            padding_detect_tol=padding_detect_tol,
            padding_detect_downsample=padding_detect_downsample,
        )


class ResNetBackbone(PoolableBackbone):
    def __init__(
        self,
        name: str = DEFAULT_RESNET_BACKBONE,
        pretrained: bool = True,
        checkpoint_path: Optional[str] = None,
        *,
        weights: Optional[str] = "IMAGENET1K_V1",
        in_features: Optional[int] = None,
        kwargs: Optional[Mapping[str, Any]] = None,
        pooling: Optional[Mapping[str, Any]] = None,
    ):
        config = BackboneConfig.legacy(
            name,
            backend="torchvision",
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
            in_features=in_features,
            weights=weights,
            kwargs=kwargs,
            pooling=pooling,
        )
        extractor = _TorchvisionSpatialExtractor(config)
        super().__init__(extractor, config)


class TimmVisionBackbone(PoolableBackbone):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        in_features: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        *,
        kwargs: Optional[Mapping[str, Any]] = None,
        pooling: Optional[Mapping[str, Any]] = None,
    ):
        config = BackboneConfig.legacy(
            model_name,
            backend="timm",
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
            in_features=in_features,
            kwargs=kwargs,
            pooling=pooling,
        )
        extractor = _TimmExtractor(config)
        super().__init__(extractor, config)


class HuggingFaceViTBackbone(PoolableBackbone):
    """ViT backbone loaded from a HuggingFace model id or local HF directory."""

    def __init__(
        self,
        model_name_or_path: str,
        in_features: Optional[int] = None,
        *,
        pooling: Optional[Mapping[str, Any]] = None,
    ):
        config = BackboneConfig.legacy(
            model_name_or_path,
            backend="hf_vit",
            in_features=in_features,
            checkpoint_path=None,
            pooling=pooling,
        )
        extractor = _HuggingFaceViTExtractor(config)
        super().__init__(extractor, config)


class ClipVisionBackbone(PoolableBackbone):
    def __init__(
        self,
        model_name_or_path: str,
        in_features: int = 768,
        *,
        pooling: Optional[Mapping[str, Any]] = None,
    ):
        config = BackboneConfig.legacy(
            model_name_or_path,
            backend="clip",
            in_features=in_features,
            checkpoint_path=None,
            pooling=pooling,
        )
        extractor = _ClipVisionExtractor(config)
        super().__init__(extractor, config)


class DomainAdaptationNet(nn.Module):
    """Backbone + bottleneck classifier used by feature extraction adapters."""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        embed_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not hasattr(backbone, "in_features"):
            raise ValueError("backbone must expose an in_features attribute.")
        self.backbone = backbone
        self.bottleneck = nn.Sequential(
            nn.Linear(int(backbone.in_features), embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        if x.shape[1] != 3:
            x = x.expand(-1, 3, -1, -1)
        features = self.backbone(x)
        logits = self.fc(self.bottleneck(features))
        return logits, features


def _is_hf_vit_path(path: Optional[str]) -> bool:
    if not path or not os.path.isdir(path):
        return False
    config_path = os.path.join(path, "config.json")
    if not os.path.isfile(config_path):
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    except Exception:
        return False
    return str(config.get("model_type", "")).lower() == "vit"


def _looks_like_hf_vit_name(name: str) -> bool:
    lower = name.lower()
    return lower.startswith("google/vit-") or lower in {
        "vit-base-patch16-224",
        "vit-base-patch16-224-in21k",
    }


def build_backbone(
    config_or_name: Union[BackboneConfig, Mapping[str, Any], str],
    *,
    backend: Optional[str] = None,
    pretrained: Optional[bool] = None,
    checkpoint_path: Optional[str] = None,
    in_features: Optional[int] = None,
    weights: Optional[str] = None,
    kwargs: Optional[Mapping[str, Any]] = None,
    pooling: Optional[Mapping[str, Any]] = None,
) -> nn.Module:
    if isinstance(config_or_name, BackboneConfig):
        config = config_or_name
    elif isinstance(config_or_name, Mapping):
        config = BackboneConfig.from_mapping(config_or_name)
    else:
        config = BackboneConfig.legacy(
            str(config_or_name),
            backend=backend,
            pretrained=True if pretrained is None else pretrained,
            checkpoint_path=checkpoint_path,
            in_features=in_features,
            weights=weights,
            kwargs=kwargs,
            pooling=pooling,
        )
    if any(
        value is not None
        for value in (backend, pretrained, checkpoint_path, in_features, weights, kwargs, pooling)
    ) and not isinstance(config_or_name, str):
        config = config.with_overrides(
            backend=backend,
            pretrained=pretrained,
            checkpoint=checkpoint_path,
            in_features=in_features,
            weights=weights,
            kwargs=kwargs,
            pooling=pooling,
        )

    if config.backend == "torchvision":
        return ResNetBackbone(
            name=config.name,
            pretrained=config.pretrained,
            checkpoint_path=config.checkpoint,
            weights=config.weights,
            in_features=config.in_features,
            kwargs=config.kwargs,
            pooling=config.pooling.as_dict(),
        )
    if config.backend == "timm":
        return TimmVisionBackbone(
            config.name,
            pretrained=config.pretrained,
            in_features=config.in_features,
            checkpoint_path=config.checkpoint,
            kwargs=config.kwargs,
            pooling=config.pooling.as_dict(),
        )
    if config.backend == "hf_vit":
        return HuggingFaceViTBackbone(
            config.checkpoint or config.name,
            in_features=config.in_features,
            pooling=config.pooling.as_dict(),
        )
    if config.backend == "clip":
        return ClipVisionBackbone(
            config.checkpoint or config.name,
            in_features=config.in_features or 768,
            pooling=config.pooling.as_dict(),
        )
    if config.backend == "custom":
        raise ValueError(
            "custom backend requires a user-created torch.nn.Module. "
            "Wrap it with PoolableBackbone(custom_model, backbone_config) instead "
            "of calling build_backbone()."
        )
    raise ValueError(f"Unsupported backbone backend: {config.backend}")
