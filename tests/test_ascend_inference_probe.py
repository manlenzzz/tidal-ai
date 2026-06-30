import importlib.util
import json
from pathlib import Path


def load_probe():
    script = Path(__file__).resolve().parents[1] / "scripts" / "ascend_inference_probe.py"
    spec = importlib.util.spec_from_file_location("ascend_inference_probe", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_find_local_model_candidates_detects_snapshot_layout(tmp_path):
    probe = load_probe()
    snapshot = tmp_path / "models--Qwen--Qwen3-0.6B" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "tokenizer.json").write_text("{}")

    candidates = probe.find_local_model_candidates(["Qwen/Qwen3-0.6B"], [tmp_path])

    assert candidates == [
        {
            "model_id": "Qwen/Qwen3-0.6B",
            "status": "FOUND",
            "path": str(snapshot),
            "source": "huggingface_cache",
        }
    ]


def test_find_local_model_candidates_marks_missing_model(tmp_path):
    probe = load_probe()

    candidates = probe.find_local_model_candidates(["Qwen/Qwen3-0.6B"], [tmp_path])

    assert candidates == [
        {
            "model_id": "Qwen/Qwen3-0.6B",
            "status": "MISSING",
            "path": None,
            "source": None,
        }
    ]


def test_write_probe_report_outputs_json_and_markdown(tmp_path):
    probe = load_probe()
    report = {
        "status": "MODEL_MISSING",
        "packages": {"torch": {"status": "FOUND", "version": "2.9.0+cpu"}},
        "models": [{"model_id": "Qwen/Qwen3-0.6B", "status": "MISSING", "path": None}],
        "vllm_ascend": {"status": "FOUND", "modules": ["__editable___vllm_ascend_dev_finder"]},
    }

    probe.write_probe_report(report, tmp_path)

    assert json.loads((tmp_path / "artifacts/ascend_inference_probe.json").read_text())["status"] == "MODEL_MISSING"
    markdown = (tmp_path / "reports/ascend-inference-probe.md").read_text()
    assert "Qwen/Qwen3-0.6B" in markdown
    assert "MODEL_MISSING" in markdown


def test_vllm_ascend_info_ignores_unrelated_ascend_scripts(monkeypatch):
    probe = load_probe()

    class Module:
        def __init__(self, name):
            self.name = name

    monkeypatch.setattr(
        probe.pkgutil,
        "iter_modules",
        lambda: [
            Module("ascend_inference_probe"),
            Module("run_parallel_ascend_benchmarks"),
            Module("__editable___vllm_ascend_0_1_dev_finder"),
        ],
    )

    info = probe.vllm_ascend_info()

    assert info == {
        "status": "FOUND",
        "modules": ["__editable___vllm_ascend_0_1_dev_finder"],
    }
