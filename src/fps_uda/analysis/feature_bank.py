from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
from tqdm import tqdm
import yaml


EPS = 1e-12


def _transform_features(array: np.ndarray, transform: str) -> np.ndarray:
    if transform == "none":
        return np.asarray(array, dtype=np.float64)
    if transform == "sqrt":
        return np.sqrt(np.clip(np.asarray(array, dtype=np.float64), a_min=0.0, a_max=None))
    raise ValueError("feature_transform must be 'none' or 'sqrt'.")


def _safe_float(value) -> Optional[float]:
    value = float(value)
    if math.isfinite(value):
        return value
    return None


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return _safe_float(value)
    return value


def _view_pooling(view_key: str, attrs: Mapping[str, object]) -> str:
    pooling = attrs.get("pooling")
    if pooling is not None and str(pooling):
        return str(pooling)
    if view_key == "clean" or view_key.endswith("_clean"):
        return "clean"
    if re.fullmatch(r"pool_[a-z]+", view_key):
        return view_key
    match = re.search(r"_(pool_[a-z]+)$", view_key)
    if match:
        return match.group(1)
    return "unknown"


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


def _sorted_unique(labels: np.ndarray) -> np.ndarray:
    return np.asarray(sorted(int(label) for label in np.unique(labels)), dtype=np.int64)


def _class_centers(features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    classes = _sorted_unique(labels)
    centers = []
    for class_id in classes:
        mask = labels == class_id
        if not np.any(mask):
            continue
        centers.append(features[mask].mean(axis=0))
    if not centers:
        raise ValueError("Cannot compute class centers without labels.")
    return classes, np.stack(centers, axis=0)


def _pairwise_distances(values: np.ndarray) -> np.ndarray:
    if values.shape[0] < 2:
        return np.empty((0,), dtype=np.float64)
    pairs = []
    for i in range(values.shape[0]):
        for j in range(i + 1, values.shape[0]):
            pairs.append(float(np.linalg.norm(values[i] - values[j])))
    return np.asarray(pairs, dtype=np.float64)


def _same_class_distances(
    features: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    class_to_index = {int(class_id): idx for idx, class_id in enumerate(classes)}
    distances = np.zeros((features.shape[0],), dtype=np.float64)
    for idx, label in enumerate(labels):
        center = centers[class_to_index[int(label)]]
        distances[idx] = float(np.linalg.norm(features[idx] - center))
    return distances


def _nearest_other_center_distances(
    features: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    if len(classes) < 2:
        return np.full((features.shape[0],), np.nan, dtype=np.float64)
    class_to_index = {int(class_id): idx for idx, class_id in enumerate(classes)}
    values = []
    for feature, label in zip(features, labels):
        own_index = class_to_index[int(label)]
        other = np.delete(centers, own_index, axis=0)
        distances = np.linalg.norm(other - feature[None, :], axis=1)
        values.append(float(np.min(distances)))
    return np.asarray(values, dtype=np.float64)


def compute_domain_view_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    domain: str,
    view_key: str,
    attrs: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Compute supervised analysis-only quality metrics for one bank view."""
    labels = np.asarray(labels)
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"Feature bank view '{domain}/{view_key}' must be a 2D array.")
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Feature bank view '{domain}/{view_key}' has {features.shape[0]} features "
            f"but {labels.shape[0]} labels."
        )

    attrs = dict(attrs or {})
    classes, centers = _class_centers(features, labels)
    own_dist = _same_class_distances(features, labels, classes, centers)
    nearest_other = _nearest_other_center_distances(features, labels, classes, centers)
    margin = nearest_other - own_dist
    inter = _pairwise_distances(centers)

    intra_mean = float(np.mean(own_dist))
    inter_mean = float(np.mean(inter)) if inter.size else 0.0
    nearest_margin_mean = float(np.nanmean(margin)) if np.any(np.isfinite(margin)) else 0.0
    fisher_ratio = inter_mean / max(intra_mean, EPS)
    center_margin_score = nearest_margin_mean / max(intra_mean, EPS)
    domain_quality_score = fisher_ratio + center_margin_score

    row: Dict[str, object] = {
        "domain": domain,
        "view_key": view_key,
        "pooling": _view_pooling(view_key, attrs),
        "num_samples": int(features.shape[0]),
        "num_classes": int(len(classes)),
        "feature_dim": int(features.shape[1]),
        "intra_mean": intra_mean,
        "intra_p90": float(np.percentile(own_dist, 90)),
        "nearest_center_margin_mean": nearest_margin_mean,
        "inter_mean": inter_mean,
        "inter_min": float(np.min(inter)) if inter.size else 0.0,
        "fisher_ratio": fisher_ratio,
        "center_margin_score": center_margin_score,
        "domain_quality_score": domain_quality_score,
        "raw_euclidean_intra_mean": intra_mean,
        "raw_euclidean_intra_p90": float(np.percentile(own_dist, 90)),
        "raw_euclidean_inter_mean": inter_mean,
        "raw_euclidean_inter_min": float(np.min(inter)) if inter.size else 0.0,
    }
    for key in (
        "pad_to_square",
        "resize_size",
        "input_size",
        "crop",
        "flip",
        "water_level",
        "backbone",
        "interpolation",
        "antialias",
        "pad_fill",
        "mean",
        "std",
    ):
        if key in attrs:
            value = attrs[key]
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            if isinstance(value, np.ndarray):
                value = value.tolist()
            row[key] = value.item() if isinstance(value, np.generic) else value
    return row


def _centers_for_alignment(
    features: np.ndarray, labels: np.ndarray
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    classes, centers = _class_centers(features, labels)
    return {int(class_id): centers[idx] for idx, class_id in enumerate(classes)}, features


def compute_task_view_metrics(
    *,
    feature_bank: Mapping[str, Mapping[str, object]],
    domain_rows_by_key: Mapping[Tuple[str, str], Mapping[str, object]],
    source_domain: str,
    target_domain: str,
    source_view: str,
    target_eval_view: str,
    target_view: str,
    target2_view: str,
    center_cache: Optional[Dict[Tuple[str, str], Tuple[Dict[int, np.ndarray], np.ndarray]]] = None,
) -> Dict[str, object]:
    """Compute source-target view-combination metrics from selected clean views."""
    src_data = feature_bank[source_domain]
    tgt_data = feature_bank[target_domain]
    src_labels = np.asarray(src_data["labels"])
    tgt_labels = np.asarray(tgt_data["labels"])
    center_cache = center_cache if center_cache is not None else {}
    src_key = (source_domain, source_view)
    tgt_key = (target_domain, target_eval_view)
    if src_key not in center_cache:
        src_features = src_data["views"][source_view]["feature"]
        center_cache[src_key] = _centers_for_alignment(src_features, src_labels)
    if tgt_key not in center_cache:
        tgt_features = tgt_data["views"][target_eval_view]["feature"]
        center_cache[tgt_key] = _centers_for_alignment(tgt_features, tgt_labels)
    src_centers, src_features = center_cache[src_key]
    tgt_centers, tgt_features = center_cache[tgt_key]
    overlap = sorted(set(src_centers).intersection(tgt_centers))
    if not overlap:
        raise ValueError(
            f"No overlapping labels between '{source_domain}' and '{target_domain}'."
        )
    distances = np.asarray(
        [
            float(np.linalg.norm(src_centers[class_id] - tgt_centers[class_id]))
            for class_id in overlap
        ],
        dtype=np.float64,
    )
    src_global = src_features.mean(axis=0)
    tgt_global = tgt_features.mean(axis=0)
    global_center_gap = float(np.linalg.norm(src_global - tgt_global))
    src_quality = float(domain_rows_by_key[(source_domain, source_view)]["domain_quality_score"])
    tgt_quality = float(domain_rows_by_key[(target_domain, target_eval_view)]["domain_quality_score"])
    align_mean = float(np.mean(distances))
    align_max = float(np.max(distances))
    task_score = src_quality + tgt_quality - align_mean - global_center_gap
    return {
        "source_domain": source_domain,
        "target_domain": target_domain,
        "src_view": source_view,
        "entropy_view": target_eval_view,
        "cr_view1": target_view,
        "cr_view2": target2_view,
        "eval_view": target_eval_view,
        "source_quality_score": src_quality,
        "target_quality_score": tgt_quality,
        "class_center_align_mean": align_mean,
        "class_center_align_max": align_max,
        "global_center_gap": global_center_gap,
        "overlap_num_classes": int(len(overlap)),
        "task_score": task_score,
    }


def _read_feature_bank(
    path: str,
    *,
    domains: Optional[Iterable[str]],
    feature_transform: str,
) -> Dict[str, Dict[str, object]]:
    data: Dict[str, Dict[str, object]] = {}
    with h5py.File(path, "r") as h5:
        if "domains" not in h5:
            raise KeyError("Feature bank H5 is missing the required '/domains' group.")
        selected = list(domains) if domains is not None else sorted(h5["domains"].keys())
        for domain in selected:
            if domain not in h5["domains"]:
                available = ", ".join(sorted(h5["domains"].keys())) or "<none>"
                raise KeyError(
                    f"Missing domain '{domain}' in feature bank. Available domains: {available}."
                )
            group = h5["domains"][domain]
            if "views" not in group:
                raise KeyError(f"Feature bank domain '{domain}' is missing '/views'.")
            labels = group["label"][:] if "label" in group else None
            views = {}
            for view_key in sorted(group["views"].keys()):
                view_group = group["views"][view_key]
                if "feature" not in view_group:
                    raise KeyError(f"Feature bank view '{domain}/{view_key}' is missing feature.")
                feature = _transform_features(view_group["feature"][:], feature_transform)
                if labels is not None and feature.shape[0] != labels.shape[0]:
                    raise ValueError(
                        f"Feature bank view '{domain}/{view_key}' has {feature.shape[0]} "
                        f"features but {labels.shape[0]} labels."
                    )
                views[view_key] = {
                    "feature": feature,
                    "attrs": {key: value for key, value in view_group.attrs.items()},
                }
            data[domain] = {"labels": labels, "views": views}
    return data


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean_row = {}
            for key in fieldnames:
                value = row.get(key, "")
                clean_row[key] = "" if value is None else value
            writer.writerow(clean_row)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "domain"


def _top_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    key: str,
    limit: int,
) -> List[Mapping[str, object]]:
    return sorted(rows, key=lambda row: float(row.get(key, 0.0)), reverse=True)[:limit]


def _plot_outputs(
    output_dir: Path,
    domain_rows: Sequence[Mapping[str, object]],
    task_rows: Sequence[Mapping[str, object]],
    *,
    source_domain: Optional[str],
    target_domain: Optional[str],
) -> Tuple[bool, Optional[str], List[str]]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return False, f"matplotlib unavailable: {exc}", []

    paths = []
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    by_domain: Dict[str, List[Mapping[str, object]]] = {}
    for row in domain_rows:
        by_domain.setdefault(str(row["domain"]), []).append(row)
    for domain, rows in by_domain.items():
        top = _top_rows(rows, key="domain_quality_score", limit=min(20, len(rows)))
        fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.28)))
        labels = [str(row["view_key"]) for row in reversed(top)]
        values = [float(row["domain_quality_score"]) for row in reversed(top)]
        ax.barh(range(len(labels)), values)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("domain_quality_score")
        ax.set_title(f"{domain} top feature-bank views")
        fig.tight_layout()
        path = plots_dir / f"domain_{_safe_name(domain)}_top_views.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))

    if task_rows and source_domain and target_domain:
        top = _top_rows(task_rows, key="task_score", limit=min(40, len(task_rows)))
        src_keys = sorted({str(row["src_view"]) for row in top})
        tgt_keys = sorted({str(row["eval_view"]) for row in top})
        matrix = np.full((len(src_keys), len(tgt_keys)), np.nan, dtype=np.float64)
        src_index = {key: idx for idx, key in enumerate(src_keys)}
        tgt_index = {key: idx for idx, key in enumerate(tgt_keys)}
        for row in top:
            matrix[src_index[str(row["src_view"])], tgt_index[str(row["eval_view"])]] = float(
                row["task_score"]
            )
        fig, ax = plt.subplots(figsize=(max(5, len(tgt_keys) * 0.45), max(4, len(src_keys) * 0.35)))
        image = ax.imshow(matrix, aspect="auto")
        ax.set_xticks(range(len(tgt_keys)))
        ax.set_xticklabels(tgt_keys, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(src_keys)))
        ax.set_yticklabels(src_keys, fontsize=7)
        ax.set_title(f"{source_domain} to {target_domain} task view score")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        path = plots_dir / f"task_{_safe_name(source_domain)}_to_{_safe_name(target_domain)}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))

    return True, None, paths


DOMAIN_FIELDS = [
    "domain",
    "view_key",
    "pooling",
    "pad_to_square",
    "resize_size",
    "input_size",
    "crop",
    "flip",
    "water_level",
    "backbone",
    "interpolation",
    "antialias",
    "pad_fill",
    "mean",
    "std",
    "num_samples",
    "num_classes",
    "feature_dim",
    "intra_mean",
    "intra_p90",
    "nearest_center_margin_mean",
    "inter_mean",
    "inter_min",
    "fisher_ratio",
    "center_margin_score",
    "domain_quality_score",
    "raw_euclidean_intra_mean",
    "raw_euclidean_intra_p90",
    "raw_euclidean_inter_mean",
    "raw_euclidean_inter_min",
]

TASK_FIELDS = [
    "source_domain",
    "target_domain",
    "src_view",
    "entropy_view",
    "cr_view1",
    "cr_view2",
    "eval_view",
    "source_quality_score",
    "target_quality_score",
    "class_center_align_mean",
    "class_center_align_max",
    "global_center_gap",
    "overlap_num_classes",
    "task_score",
]


def analyze_feature_bank(
    feature_bank_path: str,
    output_dir: str,
    *,
    domains: Optional[Iterable[str]] = None,
    source_domain: Optional[str] = None,
    target_domain: Optional[str] = None,
    feature_transform: str = "none",
    top_k_domain: int = 10,
    top_k_task: int = 10,
    make_plots: bool = True,
    progress: bool = True,
) -> Dict[str, object]:
    """Analyze feature-bank views and write CSV/JSON/YAML reports."""
    if top_k_domain <= 0:
        raise ValueError("top_k_domain must be positive.")
    if top_k_task <= 0:
        raise ValueError("top_k_task must be positive.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    selected_domains = list(domains) if domains is not None else None
    if source_domain and target_domain:
        selected = set(selected_domains or [])
        selected.add(source_domain)
        selected.add(target_domain)
        selected_domains = sorted(selected)
    elif source_domain or target_domain:
        raise ValueError("source_domain and target_domain must be provided together.")

    bank = _read_feature_bank(
        feature_bank_path,
        domains=selected_domains,
        feature_transform=feature_transform,
    )

    domain_rows: List[Dict[str, object]] = []
    warnings: List[str] = []
    total_views = sum(len(domain_data["views"]) for domain_data in bank.values())
    domain_iter = tqdm(
        bank.items(),
        total=len(bank),
        desc="domains",
        dynamic_ncols=True,
        disable=not progress,
    )
    view_progress = tqdm(
        total=total_views,
        desc="domain view metrics",
        unit="view",
        dynamic_ncols=True,
        disable=not progress,
    )
    for domain, domain_data in domain_iter:
        if domain_data["labels"] is None:
            warnings.append(
                f"Domain '{domain}' has no labels; skipping supervised feature-bank analysis."
            )
            view_progress.update(len(domain_data["views"]))
            continue
        labels = np.asarray(domain_data["labels"])
        unique_labels = np.unique(labels)
        if unique_labels.size < 2:
            warnings.append(
                f"Domain '{domain}' has only {unique_labels.size} class in feature-bank labels; "
                "view metrics and recommendations are not meaningful."
            )
        for view_key, view_data in domain_data["views"].items():
            domain_rows.append(
                compute_domain_view_metrics(
                    view_data["feature"],
                    labels,
                    domain=domain,
                    view_key=view_key,
                    attrs=view_data.get("attrs"),
                )
            )
            view_progress.update(1)
    view_progress.close()
    domain_rows = sorted(
        domain_rows,
        key=lambda row: (str(row["domain"]), -float(row["domain_quality_score"]), str(row["view_key"])),
    )
    domain_rows_by_key = {(str(row["domain"]), str(row["view_key"])): row for row in domain_rows}

    task_rows: List[Dict[str, object]] = []
    if source_domain and target_domain:
        if source_domain not in bank or target_domain not in bank:
            raise KeyError(f"Missing source/target domains: {source_domain}, {target_domain}.")
        if bank[source_domain]["labels"] is None or bank[target_domain]["labels"] is None:
            warnings.append(
                f"Cannot compute task metrics for '{source_domain}' to '{target_domain}' "
                "because one or both domains have no labels."
            )
        else:
            source_clean = [
                row
                for row in domain_rows
                if row["domain"] == source_domain and row["pooling"] == "clean"
            ]
            target_clean = [
                row
                for row in domain_rows
                if row["domain"] == target_domain and row["pooling"] == "clean"
            ]
            source_clean = _top_rows(source_clean, key="domain_quality_score", limit=top_k_domain)
            target_clean = _top_rows(target_clean, key="domain_quality_score", limit=top_k_domain)
            target_views = bank[target_domain]["views"]
            center_cache: Dict[Tuple[str, str], Tuple[Dict[int, np.ndarray], np.ndarray]] = {}
            task_candidates = [(src_row, tgt_row) for src_row in source_clean for tgt_row in target_clean]
            for src_row, tgt_clean_row in tqdm(
                task_candidates,
                desc="task view metrics",
                unit="pair",
                dynamic_ncols=True,
                disable=not progress,
            ):
                base = _view_base_key(str(tgt_clean_row["view_key"]))
                if base is None:
                    warnings.append(
                        f"Cannot infer pool_a/pool_b siblings for target view "
                        f"{target_domain}/{tgt_clean_row['view_key']}."
                    )
                    continue
                cr_view1 = _sibling_key(base, "pool_a")
                cr_view2 = _sibling_key(base, "pool_b")
                if cr_view1 not in target_views or cr_view2 not in target_views:
                    warnings.append(
                        f"Missing pool siblings for target clean view "
                        f"{target_domain}/{tgt_clean_row['view_key']}."
                    )
                    continue
                task_rows.append(
                    compute_task_view_metrics(
                        feature_bank=bank,
                        domain_rows_by_key=domain_rows_by_key,
                        source_domain=source_domain,
                        target_domain=target_domain,
                        source_view=str(src_row["view_key"]),
                        target_eval_view=str(tgt_clean_row["view_key"]),
                        target_view=cr_view1,
                        target2_view=cr_view2,
                        center_cache=center_cache,
                    )
                )
            task_rows = sorted(task_rows, key=lambda row: -float(row["task_score"]))

    _write_csv(output / "domain_view_metrics.csv", domain_rows, DOMAIN_FIELDS)
    _write_csv(output / "task_view_metrics.csv", task_rows, TASK_FIELDS)

    top_domains = {}
    for domain in bank:
        domain_specific = [row for row in domain_rows if row["domain"] == domain]
        top_domains[domain] = [
            {
                "view_key": row["view_key"],
                "pooling": row["pooling"],
                "domain_quality_score": _safe_float(row["domain_quality_score"]),
            }
            for row in _top_rows(domain_specific, key="domain_quality_score", limit=top_k_domain)
        ]

    recommended = {
        "feature_bank": feature_bank_path,
        "feature_transform": feature_transform,
        "analysis_only_uses_labels": True,
    }
    if source_domain and target_domain and task_rows:
        best = task_rows[0]
        recommended.update(
            {
                "source_domain": source_domain,
                "target_domain": target_domain,
                "views": {
                    "src": {"key": best["src_view"], "combine": "stack"},
                    "entropy": {"key": best["entropy_view"], "combine": "mean"},
                    "cr": {
                        "view1": {"key": best["cr_view1"], "combine": "stack"},
                        "view2": {"key": best["cr_view2"], "combine": "stack"},
                    },
                    "eval": {"key": best["eval_view"], "combine": "mean"},
                },
                "task_score": _safe_float(best["task_score"]),
            }
        )
    recommended["top_domain_views"] = top_domains
    if task_rows:
        recommended["top_task_views"] = [
            {key: _json_safe(row[key]) for key in TASK_FIELDS}
            for row in task_rows[:top_k_task]
        ]

    with (output / "recommended_views.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_json_safe(recommended), handle, sort_keys=False, allow_unicode=True)

    plots_enabled = False
    plots_error = None
    plot_paths: List[str] = []
    if make_plots:
        if progress:
            print("writing plots...", flush=True)
        plots_enabled, plots_error, plot_paths = _plot_outputs(
            output,
            domain_rows,
            task_rows,
            source_domain=source_domain,
            target_domain=target_domain,
        )

    summary = {
        "feature_bank_path": feature_bank_path,
        "output_dir": str(output),
        "domains": list(bank.keys()),
        "feature_transform": feature_transform,
        "top_k_domain": int(top_k_domain),
        "top_k_task": int(top_k_task),
        "domain_view_count": len(domain_rows),
        "task_view_count": len(task_rows),
        "warnings": warnings,
        "plots_enabled": plots_enabled,
        "plots_error": plots_error,
        "plots": plot_paths,
        "recommended": recommended,
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, indent=2)
    return _json_safe(summary)
