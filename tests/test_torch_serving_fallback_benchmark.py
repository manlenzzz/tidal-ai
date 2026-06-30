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


def load_benchmark_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "torch_serving_fallback_benchmark.py"
    spec = importlib.util.spec_from_file_location("torch_serving_fallback_benchmark", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_markdown_report_summarizes_baseline_cap_and_qpruner_serving_metrics():
    bench = load_benchmark_module()

    text = bench.markdown_report(
        {
            "status": "PASS",
            "run_label": "serving_fallback",
            "export_run_label": "tiny_qwen3_serving_export_npu",
            "backend": "torch_generate",
            "device": "npu",
            "summary": {
                "best_method": "qpruner",
                "best_tokens_per_s": 123.0,
                "cap_vs_baseline_speedup": 1.111,
                "qpruner_vs_baseline_speedup": 1.367,
                "cap_vs_qpruner_speedup": 0.9,
            },
            "exports": {
                "baseline": {
                    "status": "PASS",
                    "exists": True,
                    "transformers_load": "PASS",
                    "latency_ms": 24.0,
                    "tokens_per_s": 90.0,
                    "peak_mem_mb": 39.0,
                },
                "cap": {
                    "status": "PASS",
                    "exists": True,
                    "transformers_load": "PASS",
                    "latency_ms": 20.0,
                    "tokens_per_s": 100.0,
                    "peak_mem_mb": 40.0,
                },
                "qpruner": {
                    "status": "PASS",
                    "exists": True,
                    "transformers_load": "PASS",
                    "latency_ms": 18.0,
                    "tokens_per_s": 123.0,
                    "peak_mem_mb": 42.0,
                },
            },
        }
    )

    assert "Torch Serving Fallback Benchmark" in text
    assert "| Baseline | PASS | True | PASS | 24.0 | 90.0 | 39.0 |" in text
    assert "| CAP | PASS | True | PASS | 20.0 | 100.0 | 40.0 |" in text
    assert "| QPruner | PASS | True | PASS | 18.0 | 123.0 | 42.0 |" in text
    assert "cap_vs_baseline_speedup" in text
    assert "qpruner_vs_baseline_speedup" in text
    assert "best_method" in text
    assert "qpruner" in text


def test_missing_export_report_marks_model_missing(tmp_path):
    bench = load_benchmark_module()

    report = bench.run_benchmark(
        demo_root=tmp_path / "demo",
        export_run_label="missing",
        run_label="fallback",
        device_name="cpu",
        dtype_name="float32",
        max_new_tokens=2,
        iters=1,
        warmup=0,
    )

    assert report["status"] == "MODEL_MISSING"
    assert report["exports"]["baseline"]["status"] == "MISSING"
    assert report["exports"]["cap"]["status"] == "MISSING"
    assert report["exports"]["qpruner"]["status"] == "MISSING"


def test_torch_serving_fallback_benchmark_runs_on_cpu_exports(tmp_path):
    fixture = load_fixture_module()
    exporter = load_export_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)

    export_args = type(
        "Args",
        (),
        {
            "demo_root": str(demo_root),
            "model_id": "TinyQwen3-Offline",
            "model_path": str(model_dir),
            "device": "cpu",
            "dtype": "float32",
            "cap_budget": 4096,
            "cap_max_iter": 2,
            "cap_policy_steps": 1,
            "cap_samples_per_step": 1,
            "cap_rpca_backend": "numpy",
            "qpruner_average_bits": 4.0,
            "seed": 0,
            "run_label": "cpu_export",
        },
    )()
    export_report = exporter.run_export(export_args)
    exporter.write_outputs(export_report, demo_root, "cpu_export")

    script = Path(__file__).resolve().parents[1] / "scripts" / "torch_serving_fallback_benchmark.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--demo-root",
            str(demo_root),
            "--export-run-label",
            "cpu_export",
            "--run-label",
            "cpu_fallback",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--max-new-tokens",
            "2",
            "--iters",
            "1",
            "--warmup",
            "0",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads((demo_root / "artifacts" / "torch_serving_fallback_cpu_fallback.json").read_text())
    assert payload["status"] == "PASS"
    assert payload["backend"] == "torch_generate"
    assert payload["exports"]["baseline"]["status"] == "PASS"
    assert payload["exports"]["cap"]["status"] == "PASS"
    assert payload["exports"]["qpruner"]["status"] == "PASS"
    assert payload["summary"]["best_method"] in {"baseline", "cap", "qpruner"}
    assert payload["summary"]["best_tokens_per_s"] > 0
    assert payload["summary"]["cap_vs_baseline_speedup"] is not None
    assert payload["summary"]["qpruner_vs_baseline_speedup"] is not None
    assert (demo_root / "reports" / "torch-serving-fallback-cpu_fallback.md").exists()
