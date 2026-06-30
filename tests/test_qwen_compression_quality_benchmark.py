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


def load_quality_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "qwen_compression_quality_benchmark.py"
    spec = importlib.util.spec_from_file_location("qwen_compression_quality_benchmark", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_qwen_quality_markdown_reports_loss_and_target_layer_limit():
    bench = load_quality_module()

    text = bench.markdown_report(
        {
            "status": "PASS",
            "model_id": "Qwen/Qwen3-0.6B",
            "model_path": "/mnt/nvme/622/models/Qwen3-0.6B",
            "device": "npu",
            "dtype": "torch.float16",
            "target_layer_limit": 2,
            "targeted_layers_total": 196,
            "targeted_layer_names_sample": ["model.layers.0.self_attn.q_proj"],
            "baseline": {"status": "PASS", "loss": 12.0, "latency_ms": 20.0, "tokens_per_s": 100.0},
            "cap": {
                "status": "PASS",
                "loss": 12.1,
                "loss_delta": 0.1,
                "latency_ms": 10.0,
                "tokens_per_s": 200.0,
                "latency_speedup": 2.0,
                "targeted_compression_ratio": 24.0,
                "targeted_layers": 2,
                "compression_time_s": 1.5,
            },
            "wanda": {
                "status": "PASS",
                "loss": 12.08,
                "loss_delta": 0.08,
                "latency_ms": 13.0,
                "tokens_per_s": 153.8,
                "latency_speedup": 1.538,
                "targeted_sparsity": 0.5,
                "targeted_param_reduction_pct": 50.0,
                "targeted_layers": 2,
                "compression_time_s": 0.2,
            },
            "sparsegpt": {
                "status": "PASS",
                "loss": 12.07,
                "loss_delta": 0.07,
                "latency_ms": 12.5,
                "tokens_per_s": 160.0,
                "latency_speedup": 1.6,
                "targeted_sparsity": 0.5,
                "targeted_param_reduction_pct": 50.0,
                "targeted_layers": 2,
                "compression_time_s": 0.3,
            },
            "qpruner": {
                "status": "PASS",
                "loss": 12.05,
                "loss_delta": 0.05,
                "latency_ms": 11.0,
                "tokens_per_s": 181.8,
                "latency_speedup": 1.818,
                "average_bits": 4.0,
                "targeted_layers": 2,
                "compression_time_s": 0.4,
            },
            "peak_mem_mb": 1200.0,
        }
    )

    assert "Qwen CAP/QPruner Compression Quality Benchmark" in text
    assert "| Metric | Baseline | CAP | WANDA | SparseGPT | QPruner |" in text
    assert "Loss delta" in text
    assert "50.000% sparse" in text
    assert "SparseGPT" in text
    assert "Target layer limit" in text
    assert "2 / 196" in text
    assert "model.layers.0.self_attn.q_proj" in text


def test_qwen_compression_quality_benchmark_runs_offline_cpu_with_layer_limit(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)
    script = Path(__file__).resolve().parents[1] / "scripts" / "qwen_compression_quality_benchmark.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--demo-root",
            str(demo_root),
            "--model-id",
            "Qwen/Qwen3-0.6B",
            "--model-path",
            str(model_dir),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--max-length",
            "32",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--cap-budget",
            "4096",
            "--cap-max-iter",
            "1",
            "--cap-policy-steps",
            "1",
            "--cap-samples-per-step",
            "1",
            "--wanda-sparsity",
            "0.5",
            "--sparsegpt-sparsity",
            "0.5",
            "--target-layer-limit",
            "2",
            "--target-layer-pattern",
            "self_attn\\.(q_proj|k_proj)$",
            "--enable-inference-cache",
            "--inplace-compression",
            "--run-label",
            "cpu_test",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads((demo_root / "artifacts" / "qwen_compression_quality_cpu_test.json").read_text())
    assert payload["status"] == "PASS"
    assert payload["backend"] == "torch_forward"
    assert payload["model_id"] == "Qwen/Qwen3-0.6B"
    assert payload["target_layer_limit"] == 2
    assert payload["target_layer_pattern"] == "self_attn\\.(q_proj|k_proj)$"
    assert payload["targeted_layers_total"] >= 2
    assert payload["baseline"]["status"] == "PASS"
    assert payload["baseline"]["loss"] > 0
    assert payload["baseline"]["targeted_layers"] == 2
    assert payload["cap"]["status"] == "PASS"
    assert payload["cap"]["targeted_layers"] == 2
    assert payload["cap"]["compression_time_s"] >= 0
    assert "loss_delta" in payload["cap"]
    assert payload["wanda"]["status"] == "PASS"
    assert payload["wanda"]["targeted_layers"] == 2
    assert payload["wanda"]["targeted_sparsity"] == 0.5
    assert payload["wanda"]["targeted_param_reduction_pct"] == 50.0
    assert payload["wanda"]["compression_time_s"] >= 0
    assert "loss_delta" in payload["wanda"]
    assert payload["sparsegpt"]["status"] == "PASS"
    assert payload["sparsegpt"]["targeted_layers"] == 2
    assert payload["sparsegpt"]["targeted_sparsity"] == 0.5
    assert payload["sparsegpt"]["targeted_param_reduction_pct"] == 50.0
    assert payload["sparsegpt"]["compression_time_s"] >= 0
    assert "loss_delta" in payload["sparsegpt"]
    assert payload["qpruner"]["status"] == "PASS"
    assert payload["qpruner"]["targeted_layers"] == 2
    assert payload["qpruner"]["average_bits"] <= 4.0
    assert "loss_delta" in payload["qpruner"]
    report = (demo_root / "reports" / "qwen-compression-quality-cpu_test.md").read_text()
    assert "Qwen CAP/QPruner Compression Quality Benchmark" in report
    assert "WANDA" in report
    assert "SparseGPT" in report
    assert "Loss delta" in report
    assert "self_attn\\.(q_proj|k_proj)$" in report
