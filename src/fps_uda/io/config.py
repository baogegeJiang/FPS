from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml


ALIASES = {
    "pseduo_margin": "pseudo_margin",
    "num_class": "num_classes",
    "feature_num": "feature_dim",
    "use_LSE": "use_lse",
    "use_LCE": "use_lce",
    "use_LCR": "use_lcr",
    "cr_loss": "lcr_loss",
    "lcr_type": "lcr_loss",
    "lcr_weight": "lcr_sample_weight",
    "cr_weight": "lcr_sample_weight",
    "rand_loss": "lcr_loss",
    "lambda": "lambda_lcr",
    "lcr_lambda": "lambda_lcr",
    "use_ldelta": "use_shift_constraint",
    "use_LDelta": "use_shift_constraint",
    "use_shift_constraint": "use_shift_constraint",
    "lambda_ldelta": "ldelta_weight",
    "ldelta_lambda": "ldelta_weight",
    "shift_constraint_weight": "ldelta_weight",
    "shift_constraint_decay_steps": "ldelta_decay_steps",
    "src_ratio": "src_sample_ratio",
    "source_ratio": "src_sample_ratio",
    "source_sample_ratio": "src_sample_ratio",
    "train_src_ratio": "src_sample_ratio",
    "target_ratio": "target_sample_ratio",
    "train_tgt_ratio": "target_sample_ratio",
    "margin_q": "margin_quantile",
    "q": "margin_quantile",
    "margin_rho": "margin_sigmoid_boundary_weight",
    "rho": "margin_sigmoid_boundary_weight",
}


TRAINING_CONFIG_SECTIONS = {
    "io",
    "views",
    "optimization",
    "schedule",
    "normalization",
    "losses",
    "eval",
}

SNAPSHOT_HINT = "See bak/configs_snapshot_before_restructure/ for the pre-restructure flat configs."


def normalize_config_keys(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy with public key names while accepting legacy spellings."""
    normalized: Dict[str, Any] = {}
    for key, value in config.items():
        new_key = ALIASES.get(key, key)
        normalized[new_key] = value
    return normalized


def flatten_training_config(config: Mapping[str, Any], *, path: str = "<config>") -> Dict[str, Any]:
    """Flatten a sectioned training config for the internal trainer API.

    Public training YAML is intentionally sectioned-only. This keeps benchmark
    configs readable and prevents old flat configs from silently mixing with the
    new schema.
    """
    if not isinstance(config, Mapping):
        raise ValueError(f"Training YAML must contain a mapping: {path}")
    keys = set(config)
    unknown_sections = sorted(keys - TRAINING_CONFIG_SECTIONS)
    if unknown_sections:
        raise ValueError(
            "Training YAML must use sectioned schema with only these top-level "
            f"sections: {', '.join(sorted(TRAINING_CONFIG_SECTIONS))}. "
            f"Unknown top-level key(s): {', '.join(unknown_sections)}. {SNAPSHOT_HINT}"
        )
    if not keys:
        return {}

    flattened: Dict[str, Any] = {}
    for section in ("io", "optimization", "schedule", "normalization", "losses", "eval"):
        section_data = config.get(section)
        if section_data is None:
            continue
        if not isinstance(section_data, Mapping):
            raise ValueError(f"Training config section '{section}' must be a mapping: {path}")
        for key, value in normalize_config_keys(dict(section_data)).items():
            if key in flattened:
                raise ValueError(
                    f"Training config key '{key}' is defined more than once "
                    f"after flattening: {path}"
                )
            flattened[key] = value

    views = config.get("views")
    if views is not None:
        if not isinstance(views, Mapping):
            raise ValueError(f"Training config section 'views' must be a mapping: {path}")
        flattened["views"] = dict(views)
    return flattened


def load_yaml_config(path: str, section: Optional[str] = None) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must contain a mapping: {path}")
    if section is not None:
        if section not in data:
            raise KeyError(f"Config section '{section}' not found in {path}.")
        data = data[section]
    return flatten_training_config(dict(data), path=path)
