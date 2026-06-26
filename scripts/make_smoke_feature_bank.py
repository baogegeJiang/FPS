#!/usr/bin/env python3
"""Create a compressed real-H5 smoke fixture from a full feature bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import yaml


DEFAULT_CONFIG = Path("configs/training/office31/amazon_to_webcam/vit.yaml")
DEFAULT_OUT = Path("tests/fixtures/office31_amazon_to_webcam_vit_smoke.h5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Office31 amazon->webcam ViT smoke feature bank."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-bank", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--compression", choices=["gzip", "lzf", "none"], default="gzip")
    parser.add_argument("--compression-level", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _split_views(value: str) -> list[str]:
    return [token.strip() for token in str(value).split(",") if token.strip()]


def _load_spec(config_path: Path) -> tuple[dict, list[str]]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    views = config["views"]
    keys = []
    keys.extend(_split_views(views["src"]["key"]))
    keys.extend(_split_views(views["entropy"]["key"]))
    keys.extend(_split_views(views["cr"]["view1"]["key"]))
    keys.extend(_split_views(views["cr"]["view2"]["key"]))
    keys.extend(_split_views(views["eval"]["key"]))
    unique_keys = sorted(set(keys))
    return config, unique_keys


def _copy_attrs(src, dst) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def _compression_kwargs(args: argparse.Namespace) -> dict:
    if args.compression == "none":
        return {}
    kwargs = {"compression": args.compression, "shuffle": True}
    if args.compression == "gzip":
        kwargs["compression_opts"] = int(args.compression_level)
    return kwargs


def build_smoke_bank(args: argparse.Namespace) -> dict:
    config, view_keys = _load_spec(args.config)
    source_bank = args.source_bank or Path(config["io"]["feature_bank"])
    source_domain = str(config["io"]["source_domain"])
    target_domain = str(config["io"]["target_domain"])
    domains = [source_domain, target_domain]

    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} already exists. Use --force to overwrite it.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    if tmp.exists():
        tmp.unlink()

    compression_kwargs = _compression_kwargs(args)
    dtype = np.float16 if args.dtype == "float16" else np.float32
    total_samples = {}

    with h5py.File(source_bank, "r") as src, h5py.File(tmp, "w") as dst:
        _copy_attrs(src, dst)
        dst.attrs["smoke_fixture"] = True
        dst.attrs["smoke_source_config"] = str(args.config)
        dst.attrs["smoke_source_bank"] = str(source_bank)
        dst.attrs["smoke_domains_json"] = json.dumps(domains)
        dst.attrs["smoke_view_keys_json"] = json.dumps(view_keys)
        dst.attrs["smoke_storage_dtype"] = args.dtype
        domains_group = dst.create_group("domains")
        for domain in domains:
            src_domain = src["domains"][domain]
            dst_domain = domains_group.create_group(domain)
            _copy_attrs(src_domain, dst_domain)
            labels = src_domain["label"][:]
            total_samples[domain] = int(labels.shape[0])
            dst_domain.create_dataset("label", data=labels, **compression_kwargs)
            views_group = dst_domain.create_group("views")
            for view_key in view_keys:
                src_view = src_domain["views"][view_key]
                dst_view = views_group.create_group(view_key)
                _copy_attrs(src_view, dst_view)
                feature = src_view["feature"][:].astype(dtype, copy=False)
                chunks = (min(256, feature.shape[0]), feature.shape[1])
                dst_view.create_dataset(
                    "feature",
                    data=feature,
                    chunks=chunks,
                    **compression_kwargs,
                )
    tmp.replace(args.out)
    return {
        "source_bank": str(source_bank),
        "out": str(args.out),
        "domains": domains,
        "samples": total_samples,
        "views": len(view_keys),
        "dtype": args.dtype,
        "compression": args.compression,
        "size_mib": round(args.out.stat().st_size / 1024 / 1024, 3),
    }


def main() -> int:
    args = parse_args()
    summary = build_smoke_bank(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
