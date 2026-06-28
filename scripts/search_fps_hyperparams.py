#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.is_dir():
    sys.path.insert(0, str(SRC_ROOT))

import h5py
import numpy as np
import yaml
from tqdm import tqdm

from fps_uda.analysis.feature_bank import compute_domain_view_metrics
from fps_uda.io import load_feature_bank_h5, load_yaml_config
from fps_uda.io.config import flatten_training_config
from fps_uda.training import FPSConfig, train_fps


DEFAULT_SEARCH_SPACE = REPO_ROOT / "configs" / "search" / "default_search_space.yaml"
LOSS_KEYS = {
    "use_consistency_loss",
    "use_correction",
    "use_shift_constraint",
    "use_classes_weight",
    "baseline",
    "use_lse",
    "use_lce",
    "use_lcr",
    "lambda_lcr",
    "lcr_loss",
    "lcr_sample_weight",
    "pseudo_margin",
    "margin_start_step",
    "use_temperature",
    "begin_temperature",
    "use_geom_test",
    "geom_alpha",
    "geom_beta",
    "sparse_weight_a",
    "sample_entropy_type",
    "tsallis_q",
    "margin_quantile",
    "margin_weight_type",
    "margin_convert_mode",
    "margin_sigmoid_tau",
    "margin_sigmoid_boundary_weight",
}


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return data


def _deepcopy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(value))


def _load_search_space(path: str) -> dict[str, Any]:
    data = _read_yaml(Path(path))
    if "basic" not in data:
        raise ValueError("Search space YAML requires a 'basic' mapping.")
    data.setdefault("lr_candidates", [])
    data.setdefault("lambda_lcr_grid", [])
    data.setdefault("margin_candidates", [])
    return data


def _as_float_list(value: Optional[str]) -> Optional[list[float]]:
    if value is None:
        return None
    return [float(token.strip()) for token in value.split(",") if token.strip()]


def _frange(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("grid-step must be positive.")
    values = []
    current = float(start)
    limit = float(stop) + step / 2.0
    while current <= limit:
        values.append(round(current, 10))
        current += step
    return values


def _parse_lr_candidates(value: Optional[str]) -> Optional[list[dict[str, Any]]]:
    if value is None:
        return None
    candidates = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(":")]
        if len(parts) < 2:
            raise ValueError(
                "--lr-candidates entries must be "
                "feature_transform:base_lr[:optimizer[:lr_schedule[:min_lr]]]."
            )
        candidates.append(
            {
                "feature_transform": parts[0],
                "base_lr": float(parts[1]),
                "optimizer": parts[2] if len(parts) > 2 and parts[2] else "sgd",
                "lr_schedule": parts[3] if len(parts) > 3 and parts[3] else "linear_step",
                "min_lr": float(parts[4]) if len(parts) > 4 and parts[4] else 0.0,
            }
        )
    if not candidates:
        raise ValueError("--lr-candidates did not contain any candidates.")
    return candidates


def _view_pooling(view_key: str, attrs: Mapping[str, Any]) -> str:
    pooling = attrs.get("pooling")
    if pooling is not None and str(pooling):
        return str(pooling)
    if view_key == "clean" or view_key.endswith("_clean"):
        return "clean"
    match = re.search(r"_(pool_[a-z]+)$", view_key)
    return match.group(1) if match else "unknown"


def _view_base_key(view_key: str) -> Optional[str]:
    if view_key == "clean" or re.fullmatch(r"pool_[a-z]+", view_key):
        return ""
    if view_key.endswith("_clean"):
        return view_key[: -len("_clean")]
    match = re.search(r"_(pool_[a-z]+)$", view_key)
    if match:
        return view_key[: -len(match.group(0))]
    return None


def _sibling_key(base_key: str, pooling: str) -> str:
    return pooling if base_key == "" else f"{base_key}_{pooling}"


def _is_orig_view(row: Mapping[str, Any]) -> bool:
    if str(row.get("flip", "")).lower() == "orig":
        return True
    return "_orig" in str(row["view_key"])


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _read_bank_for_selection(
    path: str,
    source_domain: str,
    target_domain: str,
) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        if "domains" not in h5:
            raise KeyError("Feature bank H5 is missing '/domains'.")
        for domain in (source_domain, target_domain):
            if domain not in h5["domains"]:
                available = ", ".join(sorted(h5["domains"].keys())) or "<none>"
                raise KeyError(f"Missing domain '{domain}'. Available domains: {available}.")
            if "label" not in h5[f"domains/{domain}"]:
                raise KeyError(f"Domain '{domain}' must have labels for hyperparameter search.")

        domains = {}
        rows = []
        for domain in (source_domain, target_domain):
            domain_group = h5[f"domains/{domain}"]
            labels = domain_group["label"][:]
            views_group = domain_group["views"]
            view_keys = sorted(views_group.keys())
            domains[domain] = {"labels": labels, "view_keys": view_keys}
            for view_key in view_keys:
                view_group = views_group[view_key]
                attrs = {key: value for key, value in view_group.attrs.items()}
                feature = view_group["feature"][:]
                row = compute_domain_view_metrics(
                    feature,
                    labels,
                    domain=domain,
                    view_key=view_key,
                    attrs=attrs,
                )
                row["fisher_ratio"] = float(row.get("fisher_ratio", 0.0))
                row["domain_quality_score"] = float(row.get("domain_quality_score", 0.0))
                rows.append(row)
        attrs = {key: _json_safe(value) for key, value in h5.attrs.items()}
    return {"attrs": attrs, "domains": domains, "rows": rows}


def infer_num_classes(bank_info: Mapping[str, Any]) -> dict[str, Any]:
    labels = []
    unique_counts = {}
    for domain, domain_data in bank_info["domains"].items():
        domain_labels = np.asarray(domain_data["labels"], dtype=np.int64)
        labels.append(domain_labels)
        unique_counts[domain] = int(np.unique(domain_labels).size)
    combined = np.concatenate(labels)
    return {
        "num_classes": int(np.max(combined)) + 1,
        "unique_class_counts": unique_counts,
    }


def _rank_domain_views(
    rows: Sequence[Mapping[str, Any]],
    *,
    domain: str,
    limit: int = 2,
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    warnings = []
    domain_rows = [row for row in rows if row["domain"] == domain]
    if not domain_rows:
        raise ValueError(f"No feature-bank views found for domain '{domain}'.")
    clean_rows = [row for row in domain_rows if row.get("pooling") == "clean"]
    candidates = clean_rows or domain_rows
    if not clean_rows:
        warnings.append(f"Domain '{domain}' has no clean views; ranking all views.")
    orig_rows = [row for row in candidates if _is_orig_view(row)]
    if orig_rows:
        candidates = orig_rows
    else:
        warnings.append(f"Domain '{domain}' has no orig views; ranking available candidates.")
    ranked = sorted(
        candidates,
        key=lambda row: (
            -float(row.get("fisher_ratio", 0.0)),
            -float(row.get("domain_quality_score", 0.0)),
            str(row["view_key"]),
        ),
    )
    top = ranked[: max(1, int(limit))]
    return [str(row["view_key"]) for row in top], [dict(row) for row in ranked], warnings


def select_views(
    bank_info: Mapping[str, Any],
    *,
    source_domain: str,
    target_domain: str,
) -> dict[str, Any]:
    rows = list(bank_info["rows"])
    src_keys, src_ranked, src_warnings = _rank_domain_views(
        rows,
        domain=source_domain,
        limit=2,
    )
    tgt_keys, tgt_ranked, tgt_warnings = _rank_domain_views(
        rows,
        domain=target_domain,
        limit=2,
    )
    target_available = set(bank_info["domains"][target_domain]["view_keys"])
    cr1_keys = []
    cr2_keys = []
    warnings = src_warnings + tgt_warnings
    for key in tgt_keys:
        base = _view_base_key(key)
        if base is None:
            warnings.append(f"Cannot infer pool siblings for target view '{key}'.")
            continue
        pool_a = _sibling_key(base, "pool_a")
        pool_b = _sibling_key(base, "pool_b")
        if pool_a in target_available and pool_b in target_available:
            cr1_keys.append(pool_a)
            cr2_keys.append(pool_b)
        else:
            warnings.append(f"Missing pool_a/pool_b siblings for target view '{key}'.")
    lcr_enabled = bool(cr1_keys and cr2_keys)
    if not lcr_enabled:
        cr1_keys = list(tgt_keys)
        cr2_keys = list(tgt_keys)
        warnings.append("No CR pool siblings found; disabling LCR search and using clean CR views.")
    return {
        "src": {"key": ",".join(src_keys), "combine": "stack"},
        "entropy": {"key": ",".join(tgt_keys), "combine": "stack"},
        "cr": {
            "view1": {"key": ",".join(cr1_keys), "combine": "stack"},
            "view2": {"key": ",".join(cr2_keys), "combine": "stack"},
        },
        "eval": {"key": ",".join(tgt_keys), "combine": "mean"},
        "lcr_enabled": lcr_enabled,
        "warnings": warnings,
        "ranked": {
            source_domain: src_ranked,
            target_domain: tgt_ranked,
        },
    }


def _section_for_key(key: str) -> str:
    if key in {
        "feature_bank",
        "source_domain",
        "target_domain",
        "feature_transform",
        "num_classes",
        "device",
    }:
        return "io"
    if key in {
        "optimizer",
        "base_lr",
        "momentum",
        "nesterov",
        "weight_decay",
        "lr_schedule",
        "min_lr",
        "adamw_betas",
        "adamw_eps",
    }:
        return "optimization"
    if key in {
        "seed",
        "iter_num",
        "alpha",
        "beta",
        "alpha_0",
        "beta_0",
        "schedule_tau",
        "dynamic_parameters",
        "src_sample_ratio",
        "target_sample_ratio",
    }:
        return "schedule"
    if key in {
        "normalize",
        "cross_norm_scale",
        "cross_norm_target_weight",
        "self_norm_scale_src",
        "self_norm_scale_tgt",
    }:
        return "normalization"
    if key in LOSS_KEYS:
        return "losses"
    if key in {"eval_interval", "multi_class", "progress"}:
        return "eval"
    raise KeyError(f"Unsupported search config key: {key}")


def _apply_updates(config: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    updated = _deepcopy_mapping(config)
    for key, value in updates.items():
        section = _section_for_key(key)
        updated.setdefault(section, {})
        updated[section][key] = value
    return updated


def _build_initial_config(
    search_space: Mapping[str, Any],
    *,
    feature_bank: str,
    source_domain: str,
    target_domain: str,
    num_classes: int,
    views: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = _deepcopy_mapping(search_space["basic"])
    config.setdefault("io", {})
    config["io"].update(
        {
            "feature_bank": feature_bank,
            "source_domain": source_domain,
            "target_domain": target_domain,
            "num_classes": int(num_classes),
        }
    )
    config["views"] = _deepcopy_mapping(views)
    config = _apply_updates(
        config,
        {
            "device": args.device,
            "seed": args.seed,
            "cross_norm_scale": args.cross_norm_scale,
        },
    )
    if args.iter_num is not None:
        config = _apply_updates(config, {"iter_num": args.iter_num})
    if args.eval_interval is not None:
        config = _apply_updates(config, {"eval_interval": args.eval_interval})
    return config


def _flat_for_trial(
    config: Mapping[str, Any],
    *,
    feature_dim: int,
    metric: str,
    patience: int,
) -> dict[str, Any]:
    flat = flatten_training_config(config, path="<search-trial>")
    flat["feature_dim"] = int(feature_dim)
    flat["early_stop_metric"] = "class_wise_acc" if metric == "cwc" else "acc"
    flat["early_stop_patience"] = int(patience)
    flat["early_stop_min_delta"] = 0.0
    return flat


def _load_features_cached(
    cache: dict[str, Any],
    config: Mapping[str, Any],
    *,
    feature_transform: str,
):
    io = config["io"]
    views = config["views"]
    cache_key = (
        str(feature_transform),
        str(io["feature_bank"]),
        str(io["source_domain"]),
        str(io["target_domain"]),
        str(views["src"]["key"]),
        str(views["src"].get("combine")),
        str(views["entropy"]["key"]),
        str(views["entropy"].get("combine")),
        str(views["cr"]["view1"]["key"]),
        str(views["cr"]["view1"].get("combine")),
        str(views["cr"]["view2"]["key"]),
        str(views["cr"]["view2"].get("combine")),
        str(views["eval"]["key"]),
        str(views["eval"].get("combine")),
    )
    if cache_key not in cache:
        cache[cache_key] = load_feature_bank_h5(
            str(io["feature_bank"]),
            source_domain=str(io["source_domain"]),
            target_domain=str(io["target_domain"]),
            src_view=views["src"]["key"],
            entropy_view=views["entropy"]["key"],
            cr_view1=views["cr"]["view1"]["key"],
            cr_view2=views["cr"]["view2"]["key"],
            eval_view=views["eval"]["key"],
            feature_transform=feature_transform,
            src_combine=views["src"].get("combine"),
            entropy_combine=views["entropy"].get("combine"),
            cr_view1_combine=views["cr"]["view1"].get("combine"),
            cr_view2_combine=views["cr"]["view2"].get("combine"),
            eval_combine=views["eval"].get("combine"),
        )
    return cache[cache_key]


def _score_result(result, metric: str) -> Optional[float]:
    if metric == "cwc":
        return result.best_cwc
    return result.best_score


def _trial_summary(config: Mapping[str, Any]) -> dict[str, Any]:
    flat = flatten_training_config(config, path="<search-summary>")
    keys = [
        "source_domain",
        "target_domain",
        "feature_transform",
        "optimizer",
        "base_lr",
        "lr_schedule",
        "min_lr",
        "alpha",
        "alpha_0",
        "beta",
        "beta_0",
        "lambda_lcr",
        "use_lcr",
        "baseline",
        "use_consistency_loss",
        "pseudo_margin",
        "margin_quantile",
        "margin_convert_mode",
        "margin_sigmoid_boundary_weight",
    ]
    summary = {key: flat.get(key) for key in keys if key in flat}
    views = config.get("views", {})
    if isinstance(views, Mapping):
        src_view = views.get("src", {})
        entropy_view = views.get("entropy", {})
        if isinstance(src_view, Mapping):
            summary["src_view"] = src_view.get("key")
            summary["src_combine"] = src_view.get("combine")
        if isinstance(entropy_view, Mapping):
            summary["entropy_view"] = entropy_view.get("key")
            summary["entropy_combine"] = entropy_view.get("combine")
    return summary


def _supervised_only_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return _apply_updates(
        config,
        {
            "beta": 1.0,
            "beta_0": 1.0,
            "alpha": 0.0,
            "alpha_0": 0.0,
            "lambda_lcr": 0.0,
            "baseline": True,
            "use_consistency_loss": False,
            "use_lse": False,
            "use_lce": False,
            "use_lcr": False,
            "use_shift_constraint": False,
            "pseudo_margin": False,
        },
    )


def _target_only_config(
    config: Mapping[str, Any],
    *,
    target_domain: str,
    target_src_view: Mapping[str, Any],
) -> dict[str, Any]:
    updated = _supervised_only_config(config)
    updated["io"]["source_domain"] = target_domain
    updated["views"]["src"] = {
        "key": target_src_view["key"],
        "combine": target_src_view.get("combine", "stack"),
    }
    return updated


def _run_trial(
    *,
    trial_id: int,
    stage: str,
    round_id: int,
    config: Mapping[str, Any],
    feature_cache: dict[str, Any],
    metric: str,
    patience: int,
) -> tuple[dict[str, Any], float]:
    feature_transform = str(config["io"].get("feature_transform", "none"))
    features = _load_features_cached(feature_cache, config, feature_transform=feature_transform)
    flat = _flat_for_trial(
        config,
        feature_dim=features.feature_dim,
        metric=metric,
        patience=patience,
    )
    start = time.time()
    result = train_fps(features, FPSConfig.from_mapping(flat))
    elapsed = time.time() - start
    score = _score_result(result, metric)
    score_value = float("-inf") if score is None else float(score)
    row = {
        "trial": trial_id,
        "round": round_id,
        "stage": stage,
        "metric": metric,
        "score": None if score is None else float(score),
        "best_acc": result.best_score,
        "best_cwc": result.best_cwc,
        "best_cwc_step": result.best_cwc_step,
        "early_stopped": result.early_stopped,
        "early_stop_step": result.early_stop_step,
        "history_steps": len(result.history),
        "elapsed_sec": elapsed,
    }
    row.update(_trial_summary(config))
    return row, score_value


def _ordered_lr_candidates(
    candidates: Sequence[Mapping[str, Any]],
    bank_attrs: Mapping[str, Any],
) -> list[dict[str, Any]]:
    backend = str(bank_attrs.get("backbone_backend", "")).lower()
    name = str(bank_attrs.get("backbone_name", bank_attrs.get("backbone", ""))).lower()
    prefer_sqrt = backend == "torchvision" or "resnet" in name

    def sort_key(candidate: Mapping[str, Any]):
        transform = str(candidate.get("feature_transform", "none"))
        preferred = transform == ("sqrt" if prefer_sqrt else "none")
        return (0 if preferred else 1,)

    return [dict(candidate) for candidate in sorted(candidates, key=sort_key)]


def _candidate_grid(key: str, values: Sequence[float]) -> list[dict[str, float]]:
    return [{key: value} for value in values]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: _json_safe(row.get(key)) for key in fieldnames} for row in rows])


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.tmp.",
        delete=False,
    ) as handle:
        handle.write(text)
        tmp_name = handle.name
    Path(tmp_name).replace(path)


def _write_yaml_with_header(path: Path, config: Mapping[str, Any], header: Sequence[str]) -> None:
    text = "\n".join(f"# {line}" for line in header)
    text += "\n"
    text += yaml.safe_dump(_json_safe(dict(config)), sort_keys=False, allow_unicode=True)
    _write_text_atomic(path, text)


def _final_config_for_output(config: Mapping[str, Any]) -> dict[str, Any]:
    final_config = _deepcopy_mapping(config)
    final_config.setdefault("eval", {})
    final_config["eval"]["progress"] = True
    return final_config


def run_search(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    search_space = _load_search_space(args.search_space)
    bank_info = _read_bank_for_selection(args.feature_bank, args.source_domain, args.target_domain)
    class_info = infer_num_classes(bank_info)
    selected = select_views(
        bank_info,
        source_domain=args.source_domain,
        target_domain=args.target_domain,
    )
    views = {
        key: value
        for key, value in selected.items()
        if key in {"src", "entropy", "cr", "eval"}
    }
    config = _build_initial_config(
        search_space,
        feature_bank=args.feature_bank,
        source_domain=args.source_domain,
        target_domain=args.target_domain,
        num_classes=class_info["num_classes"],
        views=views,
        args=args,
    )
    if not selected["lcr_enabled"]:
        config = _apply_updates(config, {"use_lcr": False, "pseudo_margin": False})

    lr_candidates = _parse_lr_candidates(args.lr_candidates)
    if lr_candidates is None:
        lr_candidates = list(search_space.get("lr_candidates", []))
    lr_candidates = _ordered_lr_candidates(lr_candidates, bank_info["attrs"])
    if not lr_candidates:
        lr_candidates = [
            {
                "feature_transform": config["io"].get("feature_transform", "none"),
                "base_lr": config["optimization"]["base_lr"],
                "optimizer": config["optimization"].get("optimizer", "sgd"),
                "lr_schedule": config["optimization"].get("lr_schedule", "linear_step"),
                "min_lr": config["optimization"].get("min_lr", 0.0),
            }
        ]

    lambda_grid = _as_float_list(args.lambda_lcr_grid)
    if lambda_grid is None:
        lambda_grid = [float(value) for value in search_space.get("lambda_lcr_grid", [])]
    margin_candidates = [dict(item) for item in search_space.get("margin_candidates", [])]
    values = _frange(args.grid_min, args.grid_max, args.grid_step)
    feature_cache: dict[str, Any] = {}
    rows = []
    trial_id = 0
    current_score = float("-inf")
    best_trial = None

    def evaluate(
        candidate_config: Mapping[str, Any],
        stage: str,
        round_id: int,
        *,
        update_best: bool = True,
    ):
        nonlocal trial_id, best_trial
        trial_id += 1
        row, score = _run_trial(
            trial_id=trial_id,
            stage=stage,
            round_id=round_id,
            config=candidate_config,
            feature_cache=feature_cache,
            metric=args.metric,
            patience=args.patience,
        )
        rows.append(row)
        if update_best and (best_trial is None or score > float(best_trial["score_value"])):
            best_trial = {
                "row": row,
                "score_value": score,
                "config": _deepcopy_mapping(candidate_config),
            }
        return row, score

    first_lr = lr_candidates[0]
    baseline_lr_updates = {
        "feature_transform": first_lr.get("feature_transform", "none"),
        "base_lr": float(first_lr["base_lr"]),
        "optimizer": first_lr.get("optimizer", "sgd"),
        "lr_schedule": first_lr.get("lr_schedule", "linear_step"),
        "min_lr": float(first_lr.get("min_lr", 0.0)),
    }
    baseline_config = _apply_updates(config, baseline_lr_updates)
    evaluate(_supervised_only_config(baseline_config), "source_only", 0, update_best=False)
    evaluate(
        _target_only_config(
            baseline_config,
            target_domain=args.target_domain,
            target_src_view=views["entropy"],
        ),
        "target_only",
        0,
        update_best=False,
    )

    lr_iter = tqdm(lr_candidates, desc="search lr", disable=args.no_progress)
    for candidate in lr_iter:
        updates = {
            "feature_transform": candidate.get("feature_transform", "none"),
            "base_lr": float(candidate["base_lr"]),
            "optimizer": candidate.get("optimizer", "sgd"),
            "lr_schedule": candidate.get("lr_schedule", "linear_step"),
            "min_lr": float(candidate.get("min_lr", 0.0)),
        }
        row, score = evaluate(_apply_updates(config, updates), "lr", 0)
        if score > current_score:
            current_score = score
            config = _apply_updates(config, updates)
        lr_iter.set_postfix({"best": f"{current_score:.4f}"})

    def search_stage(stage: str, candidates: Sequence[Mapping[str, Any]], round_id: int) -> None:
        nonlocal config, current_score
        if not candidates:
            return
        stage_best_config = _deepcopy_mapping(config)
        stage_best_score = current_score
        iterator = tqdm(candidates, desc=f"round {round_id} {stage}", disable=args.no_progress)
        for updates in iterator:
            row, score = evaluate(_apply_updates(config, updates), stage, round_id)
            if score > stage_best_score:
                stage_best_score = score
                stage_best_config = _apply_updates(config, updates)
            iterator.set_postfix({"best": f"{stage_best_score:.4f}", "last": row["score"]})
        config = stage_best_config
        current_score = stage_best_score

    for round_id in range(1, int(args.rounds) + 1):
        if round_id == 1:
            search_stage("beta_tied", [{"beta": v, "beta_0": v} for v in values], round_id)
        search_stage("beta", _candidate_grid("beta", values), round_id)
        search_stage("beta_0", _candidate_grid("beta_0", values), round_id)
        if round_id == 1:
            search_stage("alpha_tied", [{"alpha": v, "alpha_0": v} for v in values], round_id)
        search_stage("alpha", _candidate_grid("alpha", values), round_id)
        search_stage("alpha_0", _candidate_grid("alpha_0", values), round_id)
        if selected["lcr_enabled"]:
            search_stage("lambda_lcr", _candidate_grid("lambda_lcr", lambda_grid), round_id)
            search_stage("margin", margin_candidates, round_id)

    final_config = _final_config_for_output(config)
    header = [
        "Generated by scripts/search_fps_hyperparams.py",
        f"metric: {args.metric}",
        f"best_score: {None if best_trial is None else best_trial['row']['score']}",
        f"source_domain: {args.source_domain}",
        f"target_domain: {args.target_domain}",
        f"num_classes: {class_info['num_classes']}",
        f"best_trial: {None if best_trial is None else best_trial['row']['trial']}",
        f"best_stage: {None if best_trial is None else best_trial['row']['stage']}",
    ]
    _write_yaml_with_header(output / "best.yaml", final_config, header)

    selected_payload = {
        "feature_bank": args.feature_bank,
        "source_domain": args.source_domain,
        "target_domain": args.target_domain,
        "num_classes": class_info["num_classes"],
        "unique_class_counts": class_info["unique_class_counts"],
        "bank_attrs": bank_info["attrs"],
        "views": views,
        "lcr_enabled": selected["lcr_enabled"],
        "warnings": selected["warnings"],
        "ranked_views": selected["ranked"],
    }
    _write_text_atomic(
        output / "selected_views.yaml",
        yaml.safe_dump(_json_safe(selected_payload), sort_keys=False, allow_unicode=True),
    )
    _write_csv(output / "search_summary.csv", rows)
    summary = {
        "best": None if best_trial is None else best_trial["row"],
        "metric": args.metric,
        "num_trials": len(rows),
        "selected_views": selected_payload,
        "search_space": {
            "lr_candidates": lr_candidates,
            "lambda_lcr_grid": lambda_grid,
            "margin_candidates": margin_candidates,
        },
        "trials": rows,
    }
    _write_text_atomic(output / "search_summary.json", json.dumps(_json_safe(summary), indent=2))
    # Validate the emitted config before returning.
    load_yaml_config(str(output / "best.yaml"))
    return _json_safe(summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search FPS-UDA hyperparameters from one feature-bank task."
    )
    parser.add_argument("--feature-bank", required=True)
    parser.add_argument("--source-domain", required=True)
    parser.add_argument("--target-domain", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metric", choices=["acc", "cwc"], default="acc")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--grid-min", type=float, default=0.05)
    parser.add_argument("--grid-max", type=float, default=1.0)
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--iter-num", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--cross-norm-scale", type=float, default=2.5)
    parser.add_argument("--lr-candidates", default=None)
    parser.add_argument("--lambda-lcr-grid", default=None)
    parser.add_argument("--search-space", default=str(DEFAULT_SEARCH_SPACE))
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_search(args)
    print(json.dumps({"best": summary["best"], "num_trials": summary["num_trials"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
