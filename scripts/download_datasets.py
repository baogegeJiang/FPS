#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}

DEFAULT_URLS = {
    "office31": "https://people.eecs.berkeley.edu/~jhoffman/domainadapt/office_31.tar.gz",
    "office_home": "https://www.hemanthdv.org/OfficeHome-Dataset/OfficeHomeDataset_10072016.zip",
    "visda17_train": "https://csr.bu.edu/ftp/visda17/clf/train.tar",
    "visda17_validation": "https://csr.bu.edu/ftp/visda17/clf/validation.tar",
}

OFFICE31_DOMAINS = ("amazon", "dslr", "webcam")
OFFICE_HOME_DOMAINS = ("Art", "Clipart", "Product", "Real World")
VISDA17_SPLITS = ("train", "validation")
VISDA17_CLASSES = (
    "aeroplane",
    "bicycle",
    "bus",
    "car",
    "horse",
    "knife",
    "motorcycle",
    "person",
    "plant",
    "skateboard",
    "train",
    "truck",
)


def _safe_destination(root: Path, member_name: str) -> Path:
    destination = root / member_name
    root_resolved = root.resolve()
    destination_resolved = destination.resolve()
    if destination_resolved != root_resolved and root_resolved not in destination_resolved.parents:
        raise ValueError(f"Archive member escapes extraction root: {member_name}")
    return destination


def _download(url: str, destination: Path, *, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        print(f"Using existing archive: {destination}")
        return destination

    tmp = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as f:
        total = int(response.headers.get("Content-Length", "0") or 0)
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100.0 / total
                print(f"\r  {downloaded / 1024 / 1024:.1f} MiB / {total / 1024 / 1024:.1f} MiB ({pct:.1f}%)", end="")
        if total:
            print()
    tmp.replace(destination)
    return destination


def _extract_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive} -> {destination}")
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                _safe_destination(destination, member.filename)
            zf.extractall(destination)
        return
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                _safe_destination(destination, member.name)
            tf.extractall(destination)
        return
    raise ValueError(f"Unsupported archive format: {archive}")


def _class_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        (
            child
            for child in root.iterdir()
            if child.is_dir() and child.name not in {"annotations", "images"}
        ),
        key=lambda path: path.name.lower(),
    )


def _looks_like_domain_dir(path: Path) -> bool:
    image_root = path / "images"
    roots = [image_root] if image_root.exists() else []
    roots.append(path)
    return any(_class_dirs(root) for root in roots)


def _find_domain_candidate(search_root: Path, name: str) -> Path | None:
    candidates = [
        path
        for path in search_root.rglob(name)
        if path.is_dir() and _looks_like_domain_dir(path)
    ]
    candidates = sorted(candidates, key=lambda path: (len(path.parts), str(path)))
    return candidates[0] if candidates else None


def _promote_named_dirs(search_root: Path, dataset_root: Path, names: Sequence[str]) -> None:
    dataset_root.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = dataset_root / name
        if _looks_like_domain_dir(target):
            continue
        candidate = _find_domain_candidate(search_root, name)
        if candidate is None:
            raise FileNotFoundError(
                f"Could not find directory '{name}' under {search_root}. "
                "Check the archive layout or place the data manually and rerun with --skip-download."
            )
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(candidate), str(target))


def _image_root(domain_dir: Path) -> Path:
    images = domain_dir / "images"
    return images if images.exists() and _class_dirs(images) else domain_dir


def _class_names_for_domains(dataset_root: Path, domains: Sequence[str]) -> list[str]:
    class_names = set()
    for domain in domains:
        root = _image_root(dataset_root / domain)
        class_names.update(path.name for path in _class_dirs(root))
    if not class_names:
        raise FileNotFoundError(f"No class folders found under {dataset_root}")
    return sorted(class_names, key=str.lower)


def _class_to_idx(class_names: Iterable[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(class_names)}


def _iter_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def generate_class_folder_manifest(
    domain_dir: Path,
    manifest_path: Path,
    class_to_idx: Mapping[str, int],
) -> int:
    root = _image_root(domain_dir)
    rows: list[str] = []
    for class_name, label in sorted(class_to_idx.items(), key=lambda item: item[1]):
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        for image_path in _iter_images(class_dir):
            rel_path = image_path.relative_to(domain_dir).as_posix()
            rows.append(f"{rel_path} {label}\n")
    if not rows:
        raise FileNotFoundError(f"No images found for manifest: {manifest_path}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("".join(rows), encoding="utf-8")
    return len(rows)


def _write_office_manifests(dataset_root: Path, domains: Sequence[str]) -> dict[str, int]:
    class_names = _class_names_for_domains(dataset_root, domains)
    mapping = _class_to_idx(class_names)
    counts = {}
    for domain in domains:
        manifest = dataset_root / domain / "annotations" / "annotations.txt"
        counts[str(manifest)] = generate_class_folder_manifest(
            dataset_root / domain,
            manifest,
            mapping,
        )
    (dataset_root / "class_to_idx.json").write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return counts


def _visda_class_mapping(split_root: Path) -> dict[str, int]:
    present = {path.name for path in _class_dirs(split_root)}
    if present.issubset(set(VISDA17_CLASSES)) and present:
        return {name: idx for idx, name in enumerate(VISDA17_CLASSES)}
    return _class_to_idx(sorted(present, key=str.lower))


def _write_visda_manifest(split_root: Path) -> int:
    mapping = _visda_class_mapping(split_root)
    manifest = split_root / "image_list.txt"
    rows = []
    for class_name, label in sorted(mapping.items(), key=lambda item: item[1]):
        class_dir = split_root / class_name
        if not class_dir.exists():
            continue
        for image_path in _iter_images(class_dir):
            rows.append(f"{image_path.relative_to(split_root).as_posix()} {label}\n")
    if not rows:
        raise FileNotFoundError(f"No VisDA17 images found under {split_root}")
    manifest.write_text("".join(rows), encoding="utf-8")
    return len(rows)


def _prepare_archive(
    *,
    name: str,
    url: str,
    archive_name: str,
    download_dir: Path,
    staging_root: Path,
    skip_download: bool,
    force: bool,
) -> Path:
    staging = staging_root / name
    if skip_download:
        return staging
    archive = _download(url, download_dir / archive_name, force=force)
    if force and staging.exists():
        shutil.rmtree(staging)
    _extract_archive(archive, staging)
    return staging


def prepare_office31(args) -> dict[str, object]:
    dataset_root = args.root / "office31"
    download_dir = getattr(args, "download_dir", args.root / "downloads")
    staging = _prepare_archive(
        name="office31",
        url=getattr(args, "office31_url", DEFAULT_URLS["office31"]),
        archive_name="office_31.tar.gz",
        download_dir=download_dir,
        staging_root=download_dir / "extracted",
        skip_download=args.skip_download,
        force=getattr(args, "force", False),
    )
    search_root = dataset_root if args.skip_download else staging
    _promote_named_dirs(search_root, dataset_root, OFFICE31_DOMAINS)
    manifests = _write_office_manifests(dataset_root, OFFICE31_DOMAINS)
    return {"dataset": "office31", "root": str(dataset_root), "manifests": manifests}


def prepare_office_home(args) -> dict[str, object]:
    dataset_root = args.root / "office_home"
    download_dir = getattr(args, "download_dir", args.root / "downloads")
    staging = _prepare_archive(
        name="office_home",
        url=getattr(args, "office_home_url", DEFAULT_URLS["office_home"]),
        archive_name="OfficeHomeDataset_10072016.zip",
        download_dir=download_dir,
        staging_root=download_dir / "extracted",
        skip_download=args.skip_download,
        force=getattr(args, "force", False),
    )
    search_root = dataset_root if args.skip_download else staging
    _promote_named_dirs(search_root, dataset_root, OFFICE_HOME_DOMAINS)
    manifests = _write_office_manifests(dataset_root, OFFICE_HOME_DOMAINS)
    return {"dataset": "office_home", "root": str(dataset_root), "manifests": manifests}


def prepare_visda17(args) -> dict[str, object]:
    dataset_root = args.root / "visda17"
    download_dir = getattr(args, "download_dir", args.root / "downloads")
    dataset_root.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        train_archive = _download(
            getattr(args, "visda17_train_url", DEFAULT_URLS["visda17_train"]),
            download_dir / "visda17_train.tar",
            force=getattr(args, "force", False),
        )
        validation_archive = _download(
            getattr(args, "visda17_validation_url", DEFAULT_URLS["visda17_validation"]),
            download_dir / "visda17_validation.tar",
            force=getattr(args, "force", False),
        )
        _extract_archive(train_archive, dataset_root)
        _extract_archive(validation_archive, dataset_root)
    _promote_named_dirs(dataset_root, dataset_root, VISDA17_SPLITS)
    manifests = {}
    for split in VISDA17_SPLITS:
        manifest = dataset_root / split / "image_list.txt"
        manifests[str(manifest)] = _write_visda_manifest(dataset_root / split)
    return {"dataset": "visda17", "root": str(dataset_root), "manifests": manifests}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download FPS-UDA image datasets and generate manifest files."
    )
    parser.add_argument(
        "dataset",
        choices=["all", "office31", "office_home", "visda17"],
        help="Dataset to prepare.",
    )
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--download-dir", type=Path, default=Path("data/downloads"))
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not download/extract; only generate manifests from existing directories.",
    )
    parser.add_argument("--force", action="store_true", help="Redownload and re-extract archives.")
    parser.add_argument("--office31-url", default=DEFAULT_URLS["office31"])
    parser.add_argument("--office-home-url", default=DEFAULT_URLS["office_home"])
    parser.add_argument("--visda17-train-url", default=DEFAULT_URLS["visda17_train"])
    parser.add_argument(
        "--visda17-validation-url",
        default=DEFAULT_URLS["visda17_validation"],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.root = args.root.expanduser()
    args.download_dir = args.download_dir.expanduser()

    selected = (
        ("office31", "office_home", "visda17")
        if args.dataset == "all"
        else (args.dataset,)
    )
    summaries = []
    for dataset in selected:
        if dataset == "office31":
            summaries.append(prepare_office31(args))
        elif dataset == "office_home":
            summaries.append(prepare_office_home(args))
        elif dataset == "visda17":
            summaries.append(prepare_visda17(args))
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
