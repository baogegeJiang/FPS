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
