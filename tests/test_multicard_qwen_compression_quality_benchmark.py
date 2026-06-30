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


def load_multicard_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "multicard_qwen_compression_quality_benchmark.py"
    spec = importlib.util.spec_from_file_location("multicard_qwen_compression_quality_benchmark", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def rank_payload(rank: int, *, world_size: int = 2) -> dict:
    return {
        "status": "PASS",
        "rank": rank,
        "world_size": world_size,
        "backend": "gloo",
        "device": "cpu",
        "card": rank,
        "model_id": "Qwen/Qwen3-0.6B",
        "model_path": "/models/Qwen3-0.6B",
        "benchmark": {
            "status": "PASS",
            "backend": "torch_forward",
            "model_id": "Qwen/Qwen3-0.6B",
            "model_path": "/models/Qwen3-0.6B",
            "target_layer_pattern": r"self_attn\.(q_proj|k_proj)$",
            "target_layer_limit": 2,
            "targeted_layers_total": 196,
            "peak_mem_mb": 3489.3 + rank,
            "baseline": {
                "status": "PASS",
                "loss": 12.0 + rank,
                "latency_ms": 20.0 + rank,
                "tokens_per_s": 100.0 + rank,
                "targeted_layers": 2,
            },
            "cap": {
                "status": "PASS",
                "loss": 12.1 + rank,
                "loss_delta": 0.1,
                "latency_ms": 10.0 + rank,
                "tokens_per_s": 200.0 + rank,
                "latency_speedup": 2.0,
                "targeted_compression_ratio": 3.0,
                "targeted_layers": 2,
            },
            "wanda": {
                "status": "PASS",
                "loss": 12.08 + rank,
                "loss_delta": 0.08,
                "latency_ms": 13.0 + rank,
                "tokens_per_s": 153.8 + rank,
                "latency_speedup": 1.538,
                "targeted_sparsity": 0.5,
                "targeted_param_reduction_pct": 50.0,
                "targeted_layers": 2,
            },
            "sparsegpt": {
                "status": "PASS",
                "loss": 12.07 + rank,
                "loss_delta": 0.07,
                "latency_ms": 12.5 + rank,
                "tokens_per_s": 160.0 + rank,
                "latency_speedup": 1.6,
                "targeted_sparsity": 0.5,
                "targeted_param_reduction_pct": 50.0,
                "targeted_layers": 2,
            },
            "qpruner": {
                "status": "PASS",
                "loss": 12.05 + rank,
                "loss_delta": 0.05,
                "latency_ms": 11.0 + rank,
                "tokens_per_s": 181.8 + rank,
                "latency_speedup": 1.818,
                "average_bits": 4.0,
                "targeted_layers": 2,
            },
        },
        "distributed_totals": {
            "baseline_tokens_per_s": 201.0,
            "cap_tokens_per_s": 401.0,
            "wanda_tokens_per_s": 308.6,
            "sparsegpt_tokens_per_s": 321.0,
            "qpruner_tokens_per_s": 364.6,
            "peak_mem_mb": 6979.6,
            "pass_count": 2,
            "baseline_loss_sum": 25.0,
            "cap_loss_sum": 25.2,
            "wanda_loss_sum": 25.16,
            "sparsegpt_loss_sum": 25.14,
            "qpruner_loss_sum": 25.1,
            "cap_loss_delta_sum": 0.2,
            "wanda_loss_delta_sum": 0.16,
            "sparsegpt_loss_delta_sum": 0.14,
            "qpruner_loss_delta_sum": 0.1,
        },
    }


def test_write_summary_aggregates_qwen_compression_quality_metrics(tmp_path):
    bench = load_multicard_module()
    artifacts = tmp_path / "artifacts"
    reports = tmp_path / "reports"
    artifacts.mkdir()
    reports.mkdir()
    for rank in range(2):
        (artifacts / f"multicard_qwen_compression_quality_run_rank{rank}.json").write_text(
            json.dumps(rank_payload(rank))
        )

    summary = bench.write_summary(
        artifacts_dir=artifacts,
        reports_dir=reports,
        run_label="run",
        spawn_error=None,
    )

    assert summary["status"] == "PASS"
    assert summary["model_id"] == "Qwen/Qwen3-0.6B"
    assert summary["world_size"] == 2
    assert summary["aggregate"]["baseline_tokens_per_s_total"] == 201.0
    assert summary["aggregate"]["cap_tokens_per_s_total"] == 401.0
    assert summary["aggregate"]["wanda_tokens_per_s_total"] == 308.6
    assert summary["aggregate"]["sparsegpt_tokens_per_s_total"] == 321.0
    assert summary["aggregate"]["qpruner_tokens_per_s_total"] == 364.6
    assert summary["aggregate"]["baseline_loss_avg"] == 12.5
    assert summary["aggregate"]["cap_loss_delta_avg"] == 0.1
    assert summary["aggregate"]["wanda_loss_delta_avg"] == 0.08
    assert summary["aggregate"]["sparsegpt_loss_delta_avg"] == 0.07
    assert summary["aggregate"]["qpruner_loss_delta_avg"] == 0.05
    assert summary["aggregate"]["distributed_reduce_consistent"] is True
    assert summary["target_layer_limit"] == 2
    assert summary["target_layer_pattern"] == r"self_attn\.(q_proj|k_proj)$"
    assert summary["targeted_layers_total"] == 196
    assert (artifacts / "multicard_qwen_compression_quality_run.json").exists()
    text = (reports / "multicard-qwen-compression-quality-run.md").read_text()
    assert "Multi-Card Qwen CAP/QPruner Compression Quality Benchmark" in text
    assert "2 / 196" in text
    assert r"self_attn\.(q_proj|k_proj)$" in text
    assert "| 0 | PASS | cpu |" in text
    assert "WANDA loss delta avg" in text
    assert "SparseGPT loss delta avg" in text
    assert "SparseGPT tokens/s total" in text
    assert "loss averages" in text
    assert "not extrapolated from one card" in text


def test_cpu_gloo_multicard_qwen_compression_quality_runs_two_processes(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)
    script = Path(__file__).resolve().parents[1] / "scripts" / "multicard_qwen_compression_quality_benchmark.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--demo-root",
            str(demo_root),
            "--model-id",
            "TinyQwen3-Offline",
            "--model-path",
            str(model_dir),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--backend",
            "gloo",
            "--cards",
            "0,1",
            "--max-length",
            "32",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--target-layer-limit",
            "2",
            "--target-layer-pattern",
            r"self_attn\.(q_proj|k_proj)$",
            "--cap-budget",
            "4096",
            "--cap-max-iter",
            "1",
            "--cap-policy-steps",
            "1",
            "--cap-samples-per-step",
            "1",
            "--enable-inference-cache",
            "--inplace-compression",
            "--timeout-seconds",
            "120",
            "--run-label",
            "cpu_test",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=240,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads((demo_root / "artifacts" / "multicard_qwen_compression_quality_cpu_test.json").read_text())
    assert summary["status"] == "PASS"
    assert summary["world_size"] == 2
    assert summary["target_layer_pattern"] == r"self_attn\.(q_proj|k_proj)$"
    assert summary["aggregate"]["pass_count"] == 2
    assert summary["aggregate"]["baseline_tokens_per_s_total"] > 0
    assert summary["aggregate"]["wanda_tokens_per_s_total"] > 0
    assert summary["aggregate"]["sparsegpt_tokens_per_s_total"] > 0
    assert summary["aggregate"]["baseline_loss_avg"] > 0
    assert summary["aggregate"]["distributed_reduce_consistent"] is True
    assert all(row["benchmark"]["wanda"]["status"] == "PASS" for row in summary["ranks"])
    assert all(row["benchmark"]["sparsegpt"]["status"] == "PASS" for row in summary["ranks"])
    assert all(row["benchmark"]["cap"]["status"] == "PASS" for row in summary["ranks"])
