import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def load_fixture_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "create_tiny_qwen_fixture.py"
    spec = importlib.util.spec_from_file_location("create_tiny_qwen_fixture", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_export_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "export_tiny_qwen_compressed_serving.py"
    spec = importlib.util.spec_from_file_location("export_tiny_qwen_compressed_serving", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_markdown_report_describes_dense_serving_tradeoff():
    exporter = load_export_module()

    text = exporter.markdown_report(
        {
            "status": "PASS",
            "model_id": "TinyQwen3-Offline",
            "device": "cpu",
            "dtype": "torch.float32",
            "exports": {
                "baseline": {
                    "status": "PASS",
                    "path": "/demo/serving_exports/baseline",
                    "targeted_layers": 7,
                    "targeted_params_original": 4096,
                    "dense_linears": 7,
                    "load_check": "PASS",
                },
                "cap": {
                    "status": "PASS",
                    "path": "/demo/serving_exports/cap",
                    "targeted_layers": 7,
                    "targeted_compression_ratio": 19.0,
                    "exported_dense_linears": 7,
                    "load_check": "PASS",
                },
                "qpruner": {
                    "status": "PASS",
                    "path": "/demo/serving_exports/qpruner",
                    "targeted_layers": 7,
                    "average_bits": 3.789,
                    "exported_dense_linears": 7,
                    "load_check": "PASS",
                },
            },
        }
    )

    assert "Serving-compatible dense export" in text
    assert "does not preserve compressed storage" in text
    assert "| Baseline | PASS | 7 | 1.0x | 7 | PASS |" in text
    assert "| CAP | PASS | 7 | 19.0x | 7 | PASS |" in text
    assert "| QPruner | PASS | 7 | 3.789 avg bits | 7 | PASS |" in text


def test_export_tiny_qwen_compressed_serving_runs_offline_cpu(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)
    script = Path(__file__).resolve().parents[1] / "scripts" / "export_tiny_qwen_compressed_serving.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--demo-root",
            str(demo_root),
            "--model-path",
            str(model_dir),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--cap-budget",
            "4096",
            "--cap-max-iter",
            "2",
            "--cap-policy-steps",
            "1",
            "--cap-samples-per-step",
            "1",
            "--qpruner-average-bits",
            "4.0",
            "--run-label",
            "cpu_test",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads((demo_root / "artifacts" / "tiny_qwen_serving_export_cpu_test.json").read_text())
    assert payload["status"] == "PASS"
    assert payload["exports"]["baseline"]["status"] == "PASS"
    assert payload["exports"]["baseline"]["load_check"] == "PASS"
    assert payload["exports"]["baseline"]["targeted_params_original"] > 0
    assert payload["exports"]["cap"]["status"] == "PASS"
    assert payload["exports"]["cap"]["exported_dense_linears"] > 0
    assert payload["exports"]["cap"]["load_check"] == "PASS"
    assert payload["exports"]["qpruner"]["status"] == "PASS"
    assert payload["exports"]["qpruner"]["average_bits"] <= 4.0
    assert payload["exports"]["qpruner"]["load_check"] == "PASS"
    for method in ("baseline", "cap", "qpruner"):
        export_dir = demo_root / "serving_exports" / "cpu_test" / method
        assert (export_dir / "config.json").exists()
        assert any(path.name.startswith("model") for path in export_dir.iterdir())
        assert (export_dir / "tokenizer_config.json").exists()
    assert (demo_root / "reports" / "tiny-qwen-serving-export-cpu_test.md").exists()
