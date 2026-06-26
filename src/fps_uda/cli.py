from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Iterable, List, Mapping, Optional

import numpy as np
import yaml

from fps_uda.adapters import extract_feature_bank_to_h5, load_dataset_config
from fps_uda.analysis import analyze_feature_bank
from fps_uda.io import load_feature_bank_h5, load_yaml_config
from fps_uda.io.config import normalize_config_keys
from fps_uda.models import build_backbone
from fps_uda.training import FPSConfig, train_fps


def _compact_result_dict(result, output_dir: str) -> dict:
    last_eval = next((row for row in reversed(result.history) if "acc" in row), None)
    data = {
        "best_metric": result.best_metric,
        "best_score": result.best_score,
        "best_cwc": result.best_cwc,
        "best_cwc_step": result.best_cwc_step,
        "history_steps": len(result.history),
        "has_predictions": result.predictions is not None,
        "has_labels": result.labels is not None,
        "has_best_cwc_predictions": result.best_cwc_predictions is not None,
        "has_best_cwc_labels": result.best_cwc_labels is not None,
        "output_dir": output_dir,
    }
    if last_eval is not None:
        for key in (
            "step",
            "acc",
            "macro_f1",
            "class_wise_acc",
            "best_cwc",
            "best_cwc_step",
            "ece",
            "invR",
        ):
            if key in last_eval:
                data[f"last_{key}"] = last_eval[key]
    return data


def _parse_grid(value: str) -> List[float]:
    return [float(token.strip()) for token in value.split(",") if token.strip()]


def _infer_config(config_dict: Mapping[str, object], features, args) -> FPSConfig:
    config_data = normalize_config_keys(dict(config_dict))
    if "num_classes" not in config_data:
        config_data["num_classes"] = int(np.max(features.src_labels)) + 1
    if "feature_dim" not in config_data:
        config_data["feature_dim"] = features.feature_dim
    overrides = {
        "device": getattr(args, "device", None),
        "seed": getattr(args, "seed", None),
        "iter_num": getattr(args, "iter_num", None),
        "output_dir": getattr(args, "out", None),
        "optimizer": getattr(args, "optimizer", None),
        "base_lr": getattr(args, "base_lr", None),
        "momentum": getattr(args, "momentum", None),
        "nesterov": getattr(args, "nesterov", None),
        "weight_decay": getattr(args, "weight_decay", None),
        "adamw_betas": getattr(args, "adamw_betas", None),
        "adamw_eps": getattr(args, "adamw_eps", None),
        "lr_schedule": getattr(args, "lr_schedule", None),
        "min_lr": getattr(args, "min_lr", None),
        "alpha": getattr(args, "alpha", None),
        "beta": getattr(args, "beta", None),
        "alpha_0": getattr(args, "alpha_0", None),
        "beta_0": getattr(args, "beta_0", None),
        "schedule_tau": getattr(args, "schedule_tau", None),
        "src_sample_ratio": getattr(args, "src_sample_ratio", None),
        "target_sample_ratio": getattr(args, "target_sample_ratio", None),
        "normalize": getattr(args, "normalize", None),
        "cross_norm_scale": getattr(args, "cross_norm_scale", None),
        "cross_norm_target_weight": getattr(args, "cross_norm_target_weight", None),
        "lcr_loss": getattr(args, "lcr_loss", None),
        "lambda_lcr": getattr(args, "lambda_lcr", None),
        "lcr_sample_weight": getattr(args, "lcr_sample_weight", None),
        "sparse_weight_a": getattr(args, "sparse_weight_a", None),
        "sample_entropy_type": getattr(args, "sample_entropy_type", None),
        "tsallis_q": getattr(args, "tsallis_q", None),
        "use_correction": getattr(args, "use_correction", None),
        "use_shift_constraint": getattr(args, "use_shift_constraint", None),
        "ldelta_weight": getattr(args, "ldelta_weight", None),
        "ldelta_decay_steps": getattr(args, "ldelta_decay_steps", None),
        "pseudo_margin": getattr(args, "pseudo_margin", None),
        "margin_start_step": getattr(args, "margin_start_step", None),
        "margin_quantile": getattr(args, "margin_quantile", None),
        "margin_sigmoid_tau": getattr(args, "margin_sigmoid_tau", None),
        "margin_sigmoid_boundary_weight": getattr(
            args,
            "margin_sigmoid_boundary_weight",
            None,
        ),
        "use_temperature": getattr(args, "use_temperature", None),
        "begin_temperature": getattr(args, "begin_temperature", None),
        "eval_interval": getattr(args, "eval_interval", None),
        "progress": getattr(args, "progress", None),
    }
    return FPSConfig.from_mapping(config_data, **overrides)


def _resolve_feature_transform(args, config_dict: Mapping[str, object]) -> str:
    value = args.feature_transform
    if value is None:
        value = normalize_config_keys(dict(config_dict)).get("feature_transform", "none")
    if value not in {"none", "sqrt"}:
        raise SystemExit("feature_transform must be one of: none, sqrt.")
    return str(value)


def _config_value(args, config_dict: Mapping[str, object], arg_name: str, *config_names: str):
    value = getattr(args, arg_name, None)
    if value is not None:
        return value
    for name in config_names or (arg_name,):
        if name in config_dict:
            return config_dict[name]
    return None


def _role_view_from_config(config_dict: Mapping[str, object], role: str) -> dict:
    views = config_dict.get("views", {})
    if not isinstance(views, Mapping):
        return {}
    if role == "cr_view1":
        cr = views.get("cr", {})
        value = cr.get("view1", {}) if isinstance(cr, Mapping) else {}
    elif role == "cr_view2":
        cr = views.get("cr", {})
        value = cr.get("view2", {}) if isinstance(cr, Mapping) else {}
    else:
        value = views.get(role, {})
    if isinstance(value, str):
        return {"key": value}
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _role_view(args, config_dict: Mapping[str, object], role: str, cli_key: str, cli_combine: str):
    config_role = _role_view_from_config(config_dict, role)
    key = getattr(args, cli_key, None)
    if key is None:
        key = config_role.get("key")
    combine = getattr(args, cli_combine, None)
    if combine is None:
        combine = config_role.get("combine")
    return key, combine


def _load_training_features(args, config_dict: Mapping[str, object]):
    feature_transform = _resolve_feature_transform(args, config_dict)
    feature_bank_path = _config_value(
        args,
        config_dict,
        "feature_bank",
        "feature_bank",
        "feature_bank_path",
    )
    if not feature_bank_path:
        raise SystemExit("--feature-bank is required for train/sweep.")

    source_domain = _config_value(args, config_dict, "source_domain")
    target_domain = _config_value(args, config_dict, "target_domain")
    src_view, src_combine = _role_view(args, config_dict, "src", "src_view", "src_combine")
    entropy_view, entropy_combine = _role_view(
        args,
        config_dict,
        "entropy",
        "entropy_view",
        "entropy_combine",
    )
    cr_view1, cr_view1_combine = _role_view(
        args,
        config_dict,
        "cr_view1",
        "cr_view1",
        "cr_view1_combine",
    )
    cr_view2, cr_view2_combine = _role_view(
        args,
        config_dict,
        "cr_view2",
        "cr_view2",
        "cr_view2_combine",
    )
    eval_view, eval_combine = _role_view(
        args,
        config_dict,
        "eval",
        "eval_view",
        "eval_combine",
    )
    required = {
        "source_domain": source_domain,
        "target_domain": target_domain,
        "src_view": src_view,
        "entropy_view": entropy_view,
        "cr_view1": cr_view1,
        "cr_view2": cr_view2,
        "eval_view": eval_view,
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise SystemExit(
            "--feature-bank training requires "
            + ", ".join(f"--{key.replace('_', '-')}" for key in missing)
        )

    return load_feature_bank_h5(
        str(feature_bank_path),
        source_domain=str(source_domain),
        target_domain=str(target_domain),
        src_view=src_view,
        entropy_view=entropy_view,
        cr_view1=cr_view1,
        cr_view2=cr_view2,
        eval_view=eval_view,
        feature_transform=feature_transform,
        src_combine=src_combine,
        entropy_combine=entropy_combine,
        cr_view1_combine=cr_view1_combine,
        cr_view2_combine=cr_view2_combine,
        eval_combine=eval_combine,
    )


def _cmd_train(args) -> int:
    config_dict = load_yaml_config(args.config, section=args.section) if args.config else {}
    features = _load_training_features(args, config_dict)
    config = _infer_config(config_dict, features, args)
    result = train_fps(features, config, output_dir=args.out)
    print(json.dumps(_compact_result_dict(result, args.out), indent=2))
    return 0


def _cmd_sweep(args) -> int:
    config_dict = load_yaml_config(args.config, section=args.section) if args.config else {}
    features = _load_training_features(args, config_dict)
    base_config = _infer_config(config_dict, features, args)
    output_root = Path(args.out)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for alpha in _parse_grid(args.alpha_grid):
        for beta in _parse_grid(args.beta_grid):
            for seed in args.seeds:
                run_dir = output_root / f"alpha_{alpha:g}_beta_{beta:g}_seed_{seed}"
                config = replace(base_config, alpha=alpha, beta=beta, seed=seed)
                result = train_fps(features, config, output_dir=str(run_dir))
                rows.append(
                    {
                        "alpha": alpha,
                        "beta": beta,
                        "seed": seed,
                        "best_metric": result.best_metric,
                        "best_score": result.best_score,
                        "best_cwc": result.best_cwc,
                        "best_cwc_step": result.best_cwc_step,
                        "run_dir": str(run_dir),
                    }
                )
    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "alpha",
                "beta",
                "seed",
                "best_metric",
                "best_score",
                "best_cwc",
                "best_cwc_step",
                "run_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    return 0


def _cmd_extract_feature_bank(args) -> int:
    config = load_dataset_config(
        args.dataset_config,
        root_dir=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    backbone_kwargs = _parse_backbone_kwargs(args.backbone_kw)
    pooling_override = _default_pooling_for_backend(args.backend)
    backbone_config = config.backbone.with_overrides(
        backend=args.backend,
        name=args.backbone,
        pretrained=False if args.no_pretrained else None,
        weights=args.weights,
        checkpoint=args.checkpoint,
        kwargs=backbone_kwargs,
        pooling=pooling_override,
    )
    backbone = build_backbone(backbone_config)
    summary = extract_feature_bank_to_h5(
        backbone,
        config,
        args.out,
        domains=args.domains,
        backbone_name=backbone_config.name,
        device=args.device,
        water_level=args.water_level,
        progress=not args.no_progress,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _parse_backbone_kwargs(values: Optional[Iterable[str]]) -> dict:
    kwargs = {}
    for value in values or []:
        if "=" not in value:
            raise SystemExit("--backbone-kw values must use KEY=VALUE syntax.")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit("--backbone-kw requires a non-empty key.")
        kwargs[key] = yaml.safe_load(raw)
    return kwargs


def _default_pooling_for_backend(backend: Optional[str]) -> Optional[dict]:
    if backend == "torchvision":
        return {"feature_type": "spatial", "random_strategy": "spatial_shared"}
    if backend in {"timm", "hf_vit", "clip"}:
        return {
            "feature_type": "token",
            "random_strategy": "token_channel_squared",
        }
    return None


def _cmd_analyze_feature_bank(args) -> int:
    if bool(args.source_domain) != bool(args.target_domain):
        raise SystemExit("--source-domain and --target-domain must be provided together.")
    summary = analyze_feature_bank(
        args.feature_bank,
        args.out,
        domains=args.domains,
        source_domain=args.source_domain,
        target_domain=args.target_domain,
        feature_transform=args.feature_transform,
        top_k_domain=args.top_k_domain,
        top_k_task=args.top_k_task,
        make_plots=not args.no_plots,
        progress=not args.no_progress,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _add_common_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--feature-bank", type=str, default=None)
    parser.add_argument("--source-domain", type=str, default=None)
    parser.add_argument("--target-domain", type=str, default=None)
    parser.add_argument("--src-view", type=str, default=None)
    parser.add_argument("--src-combine", choices=["stack", "mean"], default=None)
    parser.add_argument("--entropy-view", type=str, default=None)
    parser.add_argument("--entropy-combine", choices=["stack", "mean"], default=None)
    parser.add_argument("--cr-view1", type=str, default=None)
    parser.add_argument("--cr-view1-combine", choices=["stack", "mean"], default=None)
    parser.add_argument("--cr-view2", type=str, default=None)
    parser.add_argument("--cr-view2-combine", choices=["stack", "mean"], default=None)
    parser.add_argument("--eval-view", type=str, default=None)
    parser.add_argument("--eval-combine", choices=["stack", "mean"], default=None)
    parser.add_argument("--feature-transform", choices=["none", "sqrt"], default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--section", type=str, default=None)
    parser.add_argument("--out", type=str, required=True)

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--iter-num", type=int, default=None)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default=None)
    parser.add_argument("--base-lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--nesterov", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--adamw-betas", type=float, nargs=2, default=None)
    parser.add_argument("--adamw-eps", type=float, default=None)
    parser.add_argument(
        "--lr-schedule",
        choices=["linear_step", "constant", "cosine"],
        default=None,
    )
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--alpha-0", type=float, default=None)
    parser.add_argument("--beta-0", type=float, default=None)
    parser.add_argument("--schedule-tau", type=float, default=None)
    parser.add_argument("--src-sample-ratio", type=float, default=None)
    parser.add_argument("--target-sample-ratio", type=float, default=None)
    parser.add_argument("--normalize", choices=["none", "cross_norm", "self_norm"], default=None)
    parser.add_argument("--cross-norm-scale", type=float, default=None)
    parser.add_argument("--cross-norm-target-weight", type=float, default=None)

    parser.add_argument("--lcr-loss", choices=["mse", "l2"], default=None)
    parser.add_argument("--lambda-lcr", type=float, default=None)
    parser.add_argument(
        "--lcr-sample-weight",
        choices=["none", "density", "margin", "density_margin", "paper"],
        default=None,
    )
    parser.add_argument("--sparse-weight-a", type=str, default=None)
    parser.add_argument(
        "--sample-entropy-type",
        choices=["shannon", "tsallis", "adaptive_temp_shannon"],
        default=None,
    )
    parser.add_argument("--tsallis-q", type=float, default=None)
    parser.add_argument("--use-correction", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--use-shift-constraint",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--ldelta-weight", type=float, default=None)
    parser.add_argument("--ldelta-decay-steps", type=float, default=None)
    parser.add_argument("--pseudo-margin", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--margin-start-step", type=int, default=None)
    parser.add_argument("--margin-quantile", type=float, default=None)
    parser.add_argument("--margin-sigmoid-tau", type=float, default=None)
    parser.add_argument("--margin-sigmoid-boundary-weight", type=float, default=None)
    parser.add_argument("--use-temperature", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--begin-temperature", type=float, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fps-uda")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train FPS from a dataset-level feature bank.")
    _add_common_train_args(train)
    train.set_defaults(func=_cmd_train)

    sweep = sub.add_parser("sweep", help="Run alpha/beta sweeps from a feature bank.")
    _add_common_train_args(sweep)
    sweep.add_argument("--alpha-grid", type=str, required=True)
    sweep.add_argument("--beta-grid", type=str, required=True)
    sweep.add_argument("--seeds", type=int, nargs="+", default=[0])
    sweep.set_defaults(func=_cmd_sweep)

    bank = sub.add_parser(
        "extract-feature-bank",
        help="Extract a dataset-level H5 feature bank with deterministic views.",
    )
    bank.add_argument("--dataset-config", type=str, required=True)
    bank.add_argument("--dataset-root", type=str, default=None)
    bank.add_argument("--domains", type=str, nargs="+", default=None)
    bank.add_argument("--backend", choices=["torchvision", "timm", "hf_vit", "clip"], default=None)
    bank.add_argument("--backbone", type=str, default=None)
    bank.add_argument("--weights", type=str, default=None)
    bank.add_argument("--checkpoint", type=str, default=None)
    bank.add_argument("--backbone-kw", action="append", default=None)
    bank.add_argument("--out", type=str, required=True)
    bank.add_argument("--device", type=str, default="cuda")
    bank.add_argument("--batch-size", type=int, default=None)
    bank.add_argument("--num-workers", type=int, default=None)
    bank.add_argument("--seed", type=int, default=None)
    bank.add_argument("--water-level", type=float, default=None)
    bank.add_argument("--no-pretrained", action="store_true")
    bank.add_argument("--no-progress", action="store_true")
    bank.set_defaults(func=_cmd_extract_feature_bank)

    analyze = sub.add_parser(
        "analyze-feature-bank",
        help="Analyze feature-bank views and recommend source/target view keys.",
    )
    analyze.add_argument("--feature-bank", type=str, required=True)
    analyze.add_argument("--out", type=str, required=True)
    analyze.add_argument("--domains", type=str, nargs="+", default=None)
    analyze.add_argument("--source-domain", type=str, default=None)
    analyze.add_argument("--target-domain", type=str, default=None)
    analyze.add_argument("--feature-transform", choices=["none", "sqrt"], default="none")
    analyze.add_argument("--top-k-domain", type=int, default=10)
    analyze.add_argument("--top-k-task", type=int, default=10)
    analyze.add_argument("--no-plots", action="store_true")
    analyze.add_argument("--no-progress", action="store_true")
    analyze.set_defaults(func=_cmd_analyze_feature_bank)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
