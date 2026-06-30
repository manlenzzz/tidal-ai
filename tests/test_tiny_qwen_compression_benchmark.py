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


def load_compression_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "tiny_qwen_compression_benchmark.py"
    spec = importlib.util.spec_from_file_location("tiny_qwen_compression_benchmark", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_qwen_projection_filter_selects_modern_projection_names():
    bench = load_compression_module()

    assert bench.qwen_projection_filter("model.layers.0.self_attn.q_proj")
    assert bench.qwen_projection_filter("model.layers.0.mlp.down_proj")
    assert not bench.qwen_projection_filter("model.embed_tokens")


def test_tiny_qwen_compression_benchmark_runs_offline_cpu(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)
    script = Path(__file__).resolve().parents[1] / "scripts" / "tiny_qwen_compression_benchmark.py"

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
            "--seq-len",
            "8",
            "--batch-size",
            "1",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--cap-budget",
            "4096",
            "--cap-max-iter",
            "2",
            "--cap-policy-steps",
            "1",
            "--cap-samples-per-step",
            "1",
            "--run-label",
            "cpu_test",
            "--enable-inference-cache",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads((demo_root / "artifacts" / "tiny_qwen_compression_cpu_test.json").read_text())
    assert payload["status"] == "PASS"
    assert payload["download_attempted"] is False
    assert payload["baseline"]["latency_ms"] > 0
    assert payload["cap"]["status"] == "PASS"
    assert payload["cap"]["targeted_compression_ratio"] > 1
    assert payload["qpruner"]["status"] == "PASS"
    assert payload["qpruner"]["average_bits"] <= 4.0
    assert payload["inference_cache_enabled"] is True
    assert payload["cap_cached"]["status"] == "PASS"
    assert payload["cap_cached"]["latency_speedup"] > 0
    assert payload["qpruner_cached"]["status"] == "PASS"
    assert payload["qpruner_cached"]["latency_speedup"] > 0
