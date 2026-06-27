#!/usr/bin/env python3
"""Download released FPS-UDA feature banks from Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
import shutil
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


DEFAULT_REPO_ID = "baogege1995/FPS_H5"
DEFAULT_REPO_TYPE = "dataset"
DEFAULT_OUT_DIR = Path("fps_h5cache/banks")
DEFAULT_REMOTE_PREFIX = "banks"
HF_ENDPOINT = "https://huggingface.co"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"

BANKS = {
    "office31_resnet": "office31_resnet50.h5",
    "office31_vit": "office31_vit.h5",
    "office_home_resnet": "office_home_resnet50.h5",
    "office_home_vit": "office_home_vit.h5",
    "visda17_resnet": "visda17_resnet101.h5",
    "visda17_vit": "visda17_vit.h5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download FPS-UDA benchmark dataset-level H5 feature banks."
    )
    parser.add_argument(
        "banks",
        nargs="+",
        help="Bank keys to download, or 'all'. Valid keys: " + ", ".join(BANKS),
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face repo id.")
    parser.add_argument(
        "--repo-type",
        default=DEFAULT_REPO_TYPE,
        choices=("dataset", "model", "space"),
        help="Hugging Face repo type.",
    )
    parser.add_argument("--revision", default=None, help="Optional Hugging Face revision.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory where H5 bank files are materialized.",
    )
    parser.add_argument(
        "--remote-prefix",
        default=DEFAULT_REMOTE_PREFIX,
        help=(
            "Path prefix inside the HF repo. The benchmark repo stores banks under 'banks/'. "
            "Pass an empty string only if files are stored at repo root."
        ),
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only files already present in the local Hugging Face cache.",
    )
    parser.add_argument(
        "--endpoint",
        default="auto",
        help=(
            "Download endpoint: 'auto' tries HF_ENDPOINT if set, then Hugging Face, "
            "then hf-mirror; 'hf' uses huggingface.co; 'hf-mirror' uses hf-mirror.com; "
            "or pass a full endpoint URL."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing local files.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without downloading.")
    return parser.parse_args()


def select_banks(names: list[str]) -> list[tuple[str, str]]:
    if "all" in names:
        if len(names) > 1:
            raise SystemExit("Use either 'all' or explicit bank keys, not both.")
        return list(BANKS.items())

    invalid = sorted(set(names) - set(BANKS))
    if invalid:
        raise SystemExit(
            "Unknown bank key(s): "
            + ", ".join(invalid)
            + "\nValid keys: all, "
            + ", ".join(BANKS)
        )
    return [(name, BANKS[name]) for name in names]


def remote_candidates(filename: str, remote_prefix: str) -> list[str]:
    prefix = remote_prefix.strip("/")
    if prefix:
        return [f"{prefix}/{filename}"]
    return [filename]


def import_hf_download():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'huggingface_hub'. Install it with one of:\n"
            '  pip install -e ".[hf]"\n'
            '  pip install "huggingface_hub"\n'
            'Use pip install -e ".[hf-fast]" only when you want the optional hf_xet downloader.'
        ) from exc
    return hf_hub_download


def normalize_endpoint(endpoint: str | None) -> str | None:
    if endpoint is None:
        return None
    value = endpoint.strip()
    if not value:
        return None
    aliases = {
        "hf": HF_ENDPOINT,
        "huggingface": HF_ENDPOINT,
        "huggingface.co": HF_ENDPOINT,
        "hf-mirror": HF_MIRROR_ENDPOINT,
        "mirror": HF_MIRROR_ENDPOINT,
        "cn": HF_MIRROR_ENDPOINT,
    }
    return aliases.get(value, value.rstrip("/"))


def endpoint_candidates(endpoint: str) -> list[tuple[str, str]]:
    endpoint = endpoint.strip()
    if endpoint == "auto":
        candidates: list[tuple[str, str]] = []
        env_endpoint = normalize_endpoint(os.environ.get("HF_ENDPOINT"))
        if env_endpoint:
            candidates.append(("HF_ENDPOINT", env_endpoint))
        candidates.extend(
            [
                ("huggingface", HF_ENDPOINT),
                ("hf-mirror", HF_MIRROR_ENDPOINT),
            ]
        )
    else:
        resolved = normalize_endpoint(endpoint)
        if resolved is None:
            raise SystemExit("--endpoint must be 'auto', 'hf', 'hf-mirror', or a URL.")
        label = endpoint if endpoint in {"hf", "hf-mirror"} else resolved
        candidates = [(label, resolved)]

    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for label, resolved in candidates:
        if resolved not in seen:
            deduped.append((label, resolved))
            seen.add(resolved)
    return deduped


def direct_resolve_url(
    endpoint_url: str,
    repo_id: str,
    repo_type: str,
    revision: str | None,
    remote_path: str,
) -> str:
    repo_prefix = {"dataset": "datasets/", "space": "spaces/", "model": ""}[repo_type]
    repo_path = urllib.parse.quote(repo_id.strip("/"), safe="/")
    revision_path = urllib.parse.quote(revision or "main", safe="")
    file_path = urllib.parse.quote(remote_path.strip("/"), safe="/")
    return f"{endpoint_url.rstrip('/')}/{repo_prefix}{repo_path}/resolve/{revision_path}/{file_path}"


def download_direct_url(url: str, output_path: Path, force: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        print(f"[skip] {output_path} already exists")
        return

    tmp_path = output_path.with_name(output_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "fps-uda-feature-bank-downloader"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            with tmp_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100.0 / total
                        print(
                            f"\r  {downloaded / 1024 / 1024:.1f} MiB / "
                            f"{total / 1024 / 1024:.1f} MiB ({pct:.1f}%)",
                            end="",
                        )
            if total:
                print()
    except urllib.error.HTTPError as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"HTTP {exc.code} for {url}") from exc
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    tmp_path.replace(output_path)
    print(f"[ok] downloaded {url} -> {output_path}")


def materialize_file(cache_path: str | os.PathLike[str], output_path: Path, force: bool) -> None:
    cache_path = Path(cache_path)
    try:
        source_path = cache_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Downloaded cache path does not resolve to a real file: {cache_path}"
        ) from exc
    if not source_path.is_file():
        raise RuntimeError(f"Downloaded cache path is not a regular file: {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if (output_path.exists() or output_path.is_symlink()) and not force:
        print(f"[skip] {output_path} already exists")
        return

    tmp_path = output_path.with_name(output_path.name + ".tmp")
    if tmp_path.exists() or tmp_path.is_symlink():
        tmp_path.unlink()

    try:
        os.link(source_path, tmp_path)
        action = "linked"
    except OSError:
        shutil.copy2(source_path, tmp_path)
        action = "copied"

    tmp_path.replace(output_path)
    print(f"[ok] {action} {source_path} -> {output_path}")


def download_one(args: argparse.Namespace, key: str, filename: str) -> None:
    output_path = args.out_dir / filename
    if output_path.exists() and not args.force:
        print(f"[skip] {key}: {output_path} already exists")
        return

    hf_hub_download = import_hf_download()
    candidates = remote_candidates(filename, args.remote_prefix)
    endpoints = endpoint_candidates(args.endpoint)
    errors: list[str] = []

    for endpoint_label, endpoint_url in endpoints:
        for remote_path in candidates:
            try:
                print(f"[try] {key}: {remote_path} via {endpoint_label} ({endpoint_url})")
                cache_path = hf_hub_download(
                    repo_id=args.repo_id,
                    filename=remote_path,
                    repo_type=args.repo_type,
                    revision=args.revision,
                    local_files_only=args.local_files_only,
                    force_download=args.force,
                    endpoint=endpoint_url,
                )
            except Exception as exc:  # pragma: no cover - depends on network/HF state.
                errors.append(f"  - {endpoint_label} {remote_path} via hub API: {exc}")
                if args.local_files_only:
                    continue

                direct_url = direct_resolve_url(
                    endpoint_url,
                    args.repo_id,
                    args.repo_type,
                    args.revision,
                    remote_path,
                )
                try:
                    print(f"[try-direct] {key}: {direct_url}")
                    download_direct_url(direct_url, output_path, args.force)
                    return
                except Exception as direct_exc:  # pragma: no cover - depends on network/HF state.
                    errors.append(
                        f"  - {endpoint_label} {remote_path} via direct URL: {direct_exc}"
                    )
                    continue

            materialize_file(cache_path, output_path, args.force)
            return

    details = "\n".join(errors)
    raise RuntimeError(f"Failed to download {key} ({filename}). Tried:\n{details}")


def main() -> None:
    args = parse_args()
    selected = select_banks(args.banks)

    print(f"HF repo: {args.repo_id} ({args.repo_type})")
    print(f"Output:  {args.out_dir}")
    print("Endpoints:")
    for endpoint_label, endpoint_url in endpoint_candidates(args.endpoint):
        print(f"  {endpoint_label}: {endpoint_url}")
    for key, filename in selected:
        candidates = ", ".join(remote_candidates(filename, args.remote_prefix))
        print(f"  {key}: {filename}  [remote: {candidates}]")

    if args.dry_run:
        return

    for key, filename in selected:
        download_one(args, key, filename)


if __name__ == "__main__":
    main()
