from __future__ import annotations

from collections.abc import Iterable as IterableABC
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch

from fps_uda.data import FeatureSet


ViewSpec = Union[str, Sequence[str]]


def _transform_features(array: Optional[np.ndarray], transform: str) -> Optional[np.ndarray]:
    if array is None:
        return None
    if transform == "none":
        return array
    if transform == "sqrt":
        return np.sqrt(np.clip(array, a_min=0.0, a_max=None))
    raise ValueError("feature_transform must be 'none' or 'sqrt'.")


def _as_numpy(array):
    if torch.is_tensor(array):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _write_attrs(h5_obj, attrs: Optional[Mapping[str, object]]) -> None:
    for key, value in dict(attrs or {}).items():
        if value is None:
            value = ""
        if isinstance(value, (bool, int, float, str, np.number)):
            h5_obj.attrs[key] = value
        else:
            h5_obj.attrs[key] = str(value)


def _available_domains(h5: h5py.File) -> list[str]:
    if "domains" not in h5:
        return []
    return sorted(h5["domains"].keys())


def _available_views(h5: h5py.File, domain: str) -> list[str]:
    views_path = f"domains/{domain}/views"
    if views_path not in h5:
        return []
    return sorted(h5[views_path].keys())


def _read_bank_labels(h5: h5py.File, domain: str, *, required: bool) -> Optional[np.ndarray]:
    domain_path = f"domains/{domain}"
    if domain_path not in h5:
        available = ", ".join(_available_domains(h5)) or "<none>"
        raise KeyError(
            f"Missing domain '{domain}' in feature bank. Available domains: {available}."
        )
    label_path = f"domains/{domain}/label"
    if label_path not in h5:
        if not required:
            return None
        available = ", ".join(_available_domains(h5)) or "<none>"
        raise KeyError(
            f"Feature bank domain '{domain}' is missing labels. Available domains: {available}."
        )
    return h5[label_path][:]


def _read_bank_view(
    h5: h5py.File,
    domain: str,
    view_key: str,
    *,
    feature_transform: str,
) -> np.ndarray:
    feature_path = f"domains/{domain}/views/{view_key}/feature"
    if feature_path not in h5:
        available = ", ".join(_available_views(h5, domain)) or "<none>"
        raise KeyError(
            f"Missing feature-bank view '{view_key}' for domain '{domain}'. "
            f"Available views: {available}."
        )
    return _transform_features(h5[feature_path][:], feature_transform)


def _parse_view_keys(value: ViewSpec, *, name: str) -> Tuple[str, ...]:
    if isinstance(value, str):
        keys = tuple(token.strip() for token in value.split(",") if token.strip())
    elif isinstance(value, IterableABC):
        keys = tuple(str(token).strip() for token in value if str(token).strip())
    else:
        keys = ()
    if not keys:
        raise ValueError(f"{name} must contain at least one feature-bank view key.")
    return keys


def _resolve_view_combine(
    *,
    value: Optional[str],
    default: str,
    name: str,
) -> str:
    mode = default if value is None else str(value)
    mode = mode.strip().lower()
    if mode not in {"stack", "mean"}:
        raise ValueError(f"{name} must be one of: stack, mean.")
    return mode


def _combine_bank_views(
    h5: h5py.File,
    domain: str,
    view_spec: ViewSpec,
    labels: Optional[np.ndarray],
    *,
    feature_transform: str,
    combine: str,
    name: str,
) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[str, ...]]:
    view_keys = _parse_view_keys(view_spec, name=name)
    arrays = [
        _read_bank_view(h5, domain, view_key, feature_transform=feature_transform)
        for view_key in view_keys
    ]
    for view_key, array in zip(view_keys, arrays):
        if array.ndim != 2:
            raise ValueError(f"Feature bank view '{domain}/{view_key}' must be a 2D array.")
        if labels is not None and array.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Feature bank view '{domain}/{view_key}' has {array.shape[0]} "
                f"features but {labels.shape[0]} labels."
            )
    if combine == "mean":
        shape = arrays[0].shape
        for view_key, array in zip(view_keys, arrays):
            if array.shape != shape:
                raise ValueError(
                    f"{name} combine='mean' requires identical feature shapes; "
                    f"'{domain}/{view_key}' has {array.shape}, expected {shape}."
                )
        return (
            np.stack(arrays, axis=0).mean(axis=0),
            None if labels is None else labels.copy(),
            view_keys,
        )
    if combine == "stack":
        feature_dim = arrays[0].shape[1]
        for view_key, array in zip(view_keys, arrays):
            if array.shape[1] != feature_dim:
                raise ValueError(
                    f"{name} combine='stack' requires matching feature_dim; "
                    f"'{domain}/{view_key}' has dim {array.shape[1]}, expected {feature_dim}."
                )
        combined_labels = None if labels is None else np.concatenate([labels] * len(arrays))
        return np.concatenate(arrays, axis=0), combined_labels, view_keys
    raise ValueError(f"{name} combine mode must be one of: stack, mean.")


def save_feature_bank_h5(
    domains: Mapping[str, Mapping[str, object]],
    path: str,
    *,
    metadata: Optional[Mapping[str, object]] = None,
) -> None:
    """Save a dataset-level feature bank.

    Expected input shape:
    ``{"domain": {"label": labels, "views": {"view_key": {"feature": array, "attrs": {...}}}}}``.
    ``label`` is optional; unlabeled domains are written with ``has_labels=false``.
    A view value may also be ``(feature, attrs)`` or just a feature array.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as h5:
        h5.attrs["schema"] = "fps_uda_feature_bank"
        h5.attrs["schema_version"] = 1
        _write_attrs(h5, metadata)
        domains_group = h5.create_group("domains")
        for domain, domain_data in domains.items():
            labels = domain_data.get("label", domain_data.get("labels"))
            labels_np = None if labels is None else _as_numpy(labels)
            domain_group = domains_group.create_group(str(domain))
            domain_group.attrs["has_labels"] = labels_np is not None
            if labels_np is not None:
                domain_group.create_dataset("label", data=labels_np)
            views_group = domain_group.create_group("views")
            views = domain_data.get("views")
            if not isinstance(views, Mapping) or not views:
                raise ValueError(f"Feature bank domain '{domain}' must contain views.")
            expected_count = None if labels_np is None else labels_np.shape[0]
            for view_key, view_value in views.items():
                attrs = {}
                if isinstance(view_value, Mapping) and "feature" in view_value:
                    feature_array = view_value["feature"]
                    attrs = dict(view_value.get("attrs", {}))
                elif isinstance(view_value, tuple) and len(view_value) == 2:
                    feature_array, attrs = view_value
                    attrs = dict(attrs or {})
                else:
                    feature_array = view_value
                feature_np = _as_numpy(feature_array)
                if feature_np.ndim != 2:
                    raise ValueError(
                        f"Feature bank view '{domain}/{view_key}' must be a 2D array."
                    )
                if expected_count is None:
                    expected_count = feature_np.shape[0]
                if feature_np.shape[0] != expected_count:
                    suffix = "labels" if labels_np is not None else "samples"
                    raise ValueError(
                        f"Feature bank view '{domain}/{view_key}' has {feature_np.shape[0]} "
                        f"features but {expected_count} {suffix}."
                    )
                view_group = views_group.create_group(str(view_key))
                view_group.create_dataset("feature", data=feature_np)
                _write_attrs(view_group, attrs)


def load_feature_bank_h5(
    path: str,
    *,
    source_domain: str,
    target_domain: str,
    src_view: ViewSpec,
    entropy_view: ViewSpec,
    cr_view1: ViewSpec,
    cr_view2: ViewSpec,
    eval_view: ViewSpec,
    feature_transform: str = "none",
    src_combine: Optional[str] = None,
    entropy_combine: Optional[str] = None,
    cr_view1_combine: Optional[str] = None,
    cr_view2_combine: Optional[str] = None,
    eval_combine: Optional[str] = None,
) -> FeatureSet:
    """Load selected domain/view keys from a dataset-level feature bank."""
    src_combine = _resolve_view_combine(
        value=src_combine,
        default="stack",
        name="src_combine",
    )
    entropy_combine = _resolve_view_combine(
        value=entropy_combine,
        default="mean",
        name="entropy_combine",
    )
    cr1_combine = _resolve_view_combine(
        value=cr_view1_combine,
        default="stack",
        name="cr_view1_combine",
    )
    cr2_combine = _resolve_view_combine(
        value=cr_view2_combine,
        default="stack",
        name="cr_view2_combine",
    )
    eval_combine = _resolve_view_combine(
        value=eval_combine,
        default="mean",
        name="eval_combine",
    )
    with h5py.File(path, "r") as h5:
        if "domains" not in h5:
            raise KeyError("Feature bank H5 is missing the required '/domains' group.")
        src_domain_labels = _read_bank_labels(h5, source_domain, required=True)
        tgt_domain_labels = _read_bank_labels(h5, target_domain, required=False)
        src_features, src_labels, src_view_keys = _combine_bank_views(
            h5,
            source_domain,
            src_view,
            src_domain_labels,
            feature_transform=feature_transform,
            combine=src_combine,
            name="src_view",
        )
        entropy_features, entropy_labels, entropy_view_keys = _combine_bank_views(
            h5,
            target_domain,
            entropy_view,
            tgt_domain_labels,
            feature_transform=feature_transform,
            combine=entropy_combine,
            name="entropy_view",
        )
        cr_features_1, cr1_labels, cr1_view_keys = _combine_bank_views(
            h5,
            target_domain,
            cr_view1,
            tgt_domain_labels,
            feature_transform=feature_transform,
            combine=cr1_combine,
            name="cr_view1",
        )
        cr_features_2, cr2_labels, cr2_view_keys = _combine_bank_views(
            h5,
            target_domain,
            cr_view2,
            tgt_domain_labels,
            feature_transform=feature_transform,
            combine=cr2_combine,
            name="cr_view2",
        )
        eval_features, eval_labels, eval_view_keys = _combine_bank_views(
            h5,
            target_domain,
            eval_view,
            tgt_domain_labels,
            feature_transform=feature_transform,
            combine=eval_combine,
            name="eval_view",
        )
        if cr_features_2.shape != cr_features_1.shape:
            raise ValueError(
                "cr_view1 and cr_view2 must produce aligned feature arrays; "
                f"got {cr_features_1.shape} and {cr_features_2.shape}."
            )
        if (
            cr1_labels is not None
            and cr2_labels is not None
            and not np.array_equal(cr1_labels, cr2_labels)
        ):
            raise ValueError("cr_view1 and cr_view2 must produce aligned label order.")
        features = FeatureSet(
            src_features=src_features,
            src_labels=src_labels,
            entropy_features=entropy_features,
            entropy_labels=entropy_labels,
            cr_features_1=cr_features_1,
            cr_features_2=cr_features_2,
            cr_labels=cr1_labels,
            eval_features=eval_features,
            eval_labels=eval_labels,
            metadata={
                "source": str(path),
                "schema": "feature_bank",
                "source_domain": source_domain,
                "target_domain": target_domain,
                "src_view": list(src_view_keys),
                "entropy_view": list(entropy_view_keys),
                "cr_view1": list(cr1_view_keys),
                "cr_view2": list(cr2_view_keys),
                "eval_view": list(eval_view_keys),
                "src_combine": src_combine,
                "entropy_combine": entropy_combine,
                "cr_view1_combine": cr1_combine,
                "cr_view2_combine": cr2_combine,
                "eval_combine": eval_combine,
            },
        )
    return features.validate(require_consistency=True)
