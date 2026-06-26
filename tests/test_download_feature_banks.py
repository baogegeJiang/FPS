from pathlib import Path
import importlib.util


def _load_script_module():
    script_path = Path("scripts/download_feature_banks.py").resolve()
    spec = importlib.util.spec_from_file_location("download_feature_banks", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_endpoint_candidates_auto_tries_hf_then_mirror(monkeypatch):
    module = _load_script_module()
    monkeypatch.delenv("HF_ENDPOINT", raising=False)

    assert module.endpoint_candidates("auto") == [
        ("huggingface", "https://huggingface.co"),
        ("hf-mirror", "https://hf-mirror.com"),
    ]


def test_endpoint_candidates_auto_respects_env_and_deduplicates(monkeypatch):
    module = _load_script_module()
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com/")

    assert module.endpoint_candidates("auto") == [
        ("HF_ENDPOINT", "https://hf-mirror.com"),
        ("huggingface", "https://huggingface.co"),
    ]


def test_endpoint_candidates_accepts_alias_and_custom_url():
    module = _load_script_module()

    assert module.endpoint_candidates("hf-mirror") == [
        ("hf-mirror", "https://hf-mirror.com")
    ]
    assert module.endpoint_candidates("https://example.invalid/hf/") == [
        ("https://example.invalid/hf", "https://example.invalid/hf")
    ]


def test_direct_resolve_url_uses_dataset_prefix_and_main_revision():
    module = _load_script_module()

    assert module.direct_resolve_url(
        "https://hf-mirror.com/",
        "baogege1995/FPS_H5",
        "dataset",
        None,
        "banks/office31_resnet50.h5",
    ) == (
        "https://hf-mirror.com/datasets/baogege1995/FPS_H5/resolve/main/"
        "banks/office31_resnet50.h5"
    )


def test_direct_resolve_url_encodes_revision_and_file_path():
    module = _load_script_module()

    assert module.direct_resolve_url(
        "https://huggingface.co",
        "org/model",
        "model",
        "refs/pr/1",
        "dir/file with space.h5",
    ) == "https://huggingface.co/org/model/resolve/refs%2Fpr%2F1/dir/file%20with%20space.h5"
