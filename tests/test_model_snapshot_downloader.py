import importlib.util
import json
from pathlib import Path


def load_downloader_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "download_model_snapshot.py"
    spec = importlib.util.spec_from_file_location("download_model_snapshot", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_local_model_dir_maps_model_id_under_models_root(tmp_path):
    downloader = load_downloader_module()

    path = downloader.local_model_dir(tmp_path, "Qwen/Qwen3.5-0.8B")

    assert path == tmp_path / "Qwen3.5-0.8B"


def test_refuses_destination_outside_allowed_root(tmp_path):
    downloader = load_downloader_module()

    try:
        downloader.ensure_under_root(tmp_path / "models", tmp_path / "other" / "Qwen3.5-0.8B")
    except ValueError as exc:
        assert "outside allowed model root" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_existing_snapshot_is_reported_without_download(tmp_path, monkeypatch):
    downloader = load_downloader_module()
    model_root = tmp_path / "models"
    model_dir = model_root / "Qwen3.5-0.8B"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}")
    (model_dir / "tokenizer_config.json").write_text("{}")
    called = False

    def fake_snapshot_download(**kwargs):
        nonlocal called
        called = True

    report = downloader.download_or_reuse_snapshot(
        model_id="Qwen/Qwen3.5-0.8B",
        model_root=model_root,
        provider="modelscope",
        revision=None,
        allow_patterns=["*.json"],
        ignore_patterns=[],
        snapshot_download=fake_snapshot_download,
    )

    assert report["status"] == "FOUND"
    assert report["provider"] == "modelscope"
    assert report["download_attempted"] is False
    assert report["model_path"] == str(model_dir)
    assert called is False


def test_download_passes_provider_agnostic_snapshot_request(tmp_path):
    downloader = load_downloader_module()
    model_root = tmp_path / "models"
    requests = []

    def fake_snapshot_download(**kwargs):
        requests.append(kwargs)
        path = Path(kwargs["local_dir"])
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}")
        (path / "tokenizer_config.json").write_text("{}")
        return str(path)

    report = downloader.download_or_reuse_snapshot(
        model_id="Qwen/Qwen3-0.6B",
        model_root=model_root,
        provider="modelscope",
        revision="master",
        allow_patterns=["*.json"],
        ignore_patterns=["*.msgpack"],
        snapshot_download=fake_snapshot_download,
    )

    assert report["status"] == "DOWNLOADED"
    assert report["provider"] == "modelscope"
    assert requests == [
        {
            "model_id": "Qwen/Qwen3-0.6B",
            "revision": "master",
            "local_dir": str(model_root / "Qwen3-0.6B"),
            "allow_patterns": ["*.json"],
            "ignore_patterns": ["*.msgpack"],
        }
    ]


def test_huggingface_snapshot_adapter_preserves_filter_arguments(tmp_path):
    downloader = load_downloader_module()
    calls = []

    def fake_hf_snapshot_download(**kwargs):
        calls.append(kwargs)
        return kwargs["local_dir"]

    wrapped = downloader.huggingface_snapshot_adapter(fake_hf_snapshot_download)
    result = wrapped(
        model_id="Qwen/Qwen3-0.6B",
        revision="main",
        local_dir=str(tmp_path / "Qwen3-0.6B"),
        allow_patterns=["*.json"],
        ignore_patterns=["*.msgpack"],
    )

    assert result == str(tmp_path / "Qwen3-0.6B")
    assert calls == [
        {
            "repo_id": "Qwen/Qwen3-0.6B",
            "revision": "main",
            "local_dir": str(tmp_path / "Qwen3-0.6B"),
            "local_dir_use_symlinks": False,
            "resume_download": True,
            "allow_patterns": ["*.json"],
            "ignore_patterns": ["*.msgpack"],
        }
    ]


def test_modelscope_snapshot_adapter_uses_model_id_and_local_dir(tmp_path):
    downloader = load_downloader_module()
    calls = []

    def fake_modelscope_snapshot_download(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return kwargs["local_dir"]

    wrapped = downloader.modelscope_snapshot_adapter(fake_modelscope_snapshot_download)
    result = wrapped(
        model_id="Qwen/Qwen3-0.6B",
        revision="master",
        local_dir=str(tmp_path / "Qwen3-0.6B"),
        allow_patterns=["*.json"],
        ignore_patterns=["*.msgpack"],
    )

    assert result == str(tmp_path / "Qwen3-0.6B")
    assert calls == [
        (
            "Qwen/Qwen3-0.6B",
            {
                "revision": "master",
                "local_dir": str(tmp_path / "Qwen3-0.6B"),
            },
        )
    ]


def test_download_report_writes_artifact_and_markdown(tmp_path, monkeypatch):
    downloader = load_downloader_module()
    demo_root = tmp_path / "demo"
    model_root = tmp_path / "models"

    def fake_snapshot_download(**kwargs):
        path = Path(kwargs["local_dir"])
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}")
        (path / "tokenizer_config.json").write_text("{}")
        return str(path)

    report = downloader.download_or_reuse_snapshot(
        model_id="Qwen/Qwen3.5-0.8B",
        model_root=model_root,
        provider="huggingface",
        revision="main",
        allow_patterns=["*.json"],
        ignore_patterns=["*.msgpack"],
        snapshot_download=fake_snapshot_download,
    )
    downloader.write_outputs(report, demo_root, run_label="qwen35")

    payload = json.loads((demo_root / "artifacts/model_snapshot_qwen35.json").read_text())
    markdown = (demo_root / "reports/model-snapshot-qwen35.md").read_text()
    assert payload["status"] == "DOWNLOADED"
    assert payload["download_attempted"] is True
    assert "Qwen/Qwen3.5-0.8B" in markdown
    assert "model-snapshot" in markdown
