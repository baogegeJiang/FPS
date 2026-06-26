from pathlib import Path
from types import SimpleNamespace
import importlib.util


def _load_script_module():
    script_path = Path("scripts/download_datasets.py").resolve()
    spec = importlib.util.spec_from_file_location("download_datasets", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake image")


def test_visda17_default_download_urls_use_https():
    module = _load_script_module()

    assert module.DEFAULT_URLS["visda17_train"].startswith("https://")
    assert module.DEFAULT_URLS["visda17_validation"].startswith("https://")


def test_prepare_office_home_skip_download_generates_annotations(tmp_path):
    module = _load_script_module()
    root = tmp_path / "data"
    for domain in ("Art", "Clipart", "Product", "Real World"):
        _touch(root / "office_home" / domain / "back_pack" / "0001.jpg")
        _touch(root / "office_home" / domain / "bike" / "0002.png")
    args = SimpleNamespace(root=root, skip_download=True)

    summary = module.prepare_office_home(args)

    manifest = root / "office_home" / "Art" / "annotations" / "annotations.txt"
    assert summary["manifests"][str(manifest)] == 2
    assert manifest.read_text().splitlines() == [
        "back_pack/0001.jpg 0",
        "bike/0002.png 1",
    ]


def test_prepare_office31_skip_download_generates_shared_class_mapping(tmp_path):
    module = _load_script_module()
    root = tmp_path / "data"
    for domain in ("amazon", "dslr", "webcam"):
        _touch(root / "office31" / domain / "images" / "keyboard" / "a.jpg")
        _touch(root / "office31" / domain / "images" / "monitor" / "b.jpg")
    args = SimpleNamespace(root=root, skip_download=True)

    module.prepare_office31(args)

    manifest = root / "office31" / "dslr" / "annotations" / "annotations.txt"
    assert manifest.read_text().splitlines() == [
        "images/keyboard/a.jpg 0",
        "images/monitor/b.jpg 1",
    ]


def test_prepare_visda17_skip_download_generates_image_lists(tmp_path):
    module = _load_script_module()
    root = tmp_path / "data"
    for split in ("train", "validation"):
        _touch(root / "visda17" / split / "aeroplane" / "a.jpg")
        _touch(root / "visda17" / split / "car" / "c.jpg")
    args = SimpleNamespace(root=root, skip_download=True)

    module.prepare_visda17(args)

    manifest = root / "visda17" / "validation" / "image_list.txt"
    assert manifest.read_text().splitlines() == [
        "aeroplane/a.jpg 0",
        "car/c.jpg 3",
    ]
