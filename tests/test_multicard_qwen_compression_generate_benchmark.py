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
    script = Path(__file__).resolve().parents[1] / "scripts" / "multicard_qwen_compression_generate_benchmark.py"
    spec = importlib.util.spec_from_file_location("multicard_qwen_compression_generate_benchmark", script)
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
            "model_id": "Qwen/Qwen3-0.6B",
            "model_path": "/models/Qwen3-0.6B",
            "target_layer_pattern": r"self_attn\.(q_proj|k_proj)$",
            "target_layer_limit": 2,
            "targeted_layers_total": 196,
            "prompt_repeat": 4,
            "per_rank_prompt_count": 8,
            "per_rank_generated_tokens_target": 64,
            "peak_mem_mb": 3489.3 + rank,
            "baseline": {
                "status": "PASS",
                "latency_ms": 990.0 + rank,
                "tokens_per_s": 8.0 + rank,
                "targeted_layers": 2,
            },
            "cap": {
                "status": "PASS",
                "latency_ms": 310.0 + rank,
                "tokens_per_s": 25.0 + rank,
                "latency_speedup": 3.19,
                "targeted_compression_ratio": 3.0,
                "targeted_layers": 2,
            },
            "qpruner": {
                "status": "PASS",
                "latency_ms": 283.0 + rank,
                "tokens_per_s": 28.0 + rank,
                "latency_speedup": 3.50,
                "average_bits": 4.0,
                "targeted_layers": 2,
            },
        },
        "distributed_totals": {
            "baseline_tokens_per_s": 17.0,
            "cap_tokens_per_s": 51.0,
            "qpruner_tokens_per_s": 57.0,
            "peak_mem_mb": 6979.6,
            "pass_count": 2,
        },
    }


def test_write_summary_aggregates_qwen_compression_generate_metrics(tmp_path):
    bench = load_multicard_module()
    artifacts = tmp_path / "artifacts"
    reports = tmp_path / "reports"
    artifacts.mkdir()
    reports.mkdir()
    for rank in range(2):
        (artifacts / f"multicard_qwen_compression_generate_run_rank{rank}.json").write_text(
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
    assert summary["aggregate"]["baseline_tokens_per_s_total"] == 17.0
    assert summary["aggregate"]["cap_tokens_per_s_total"] == 51.0
    assert summary["aggregate"]["qpruner_tokens_per_s_total"] == 57.0
    assert summary["aggregate"]["peak_mem_mb_total"] == 6979.6
    assert summary["aggregate"]["pass_count"] == 2
    assert summary["aggregate"]["distributed_reduce_consistent"] is True
    assert summary["target_layer_limit"] == 2
    assert summary["target_layer_pattern"] == r"self_attn\.(q_proj|k_proj)$"
    assert summary["targeted_layers_total"] == 196
    assert summary["workload"]["prompt_repeat"] == 4
    assert summary["workload"]["per_rank_prompt_count"] == 8
    assert summary["workload"]["per_rank_generated_tokens_target"] == 64
    assert summary["workload"]["total_generated_tokens_target"] == 128
    assert (artifacts / "multicard_qwen_compression_generate_run.json").exists()
    text = (reports / "multicard-qwen-compression-generate-run.md").read_text()
    assert "Multi-Card Qwen CAP/QPruner Generate Benchmark" in text
    assert "2 / 196" in text
    assert "Per-rank prompt count" in text
    assert "Total generated-token target" in text
    assert r"self_attn\.(q_proj|k_proj)$" in text
    assert "| 0 | PASS | cpu |" in text
    assert "all-reduce" in text
    assert "not extrapolated from one card" in text


def test_aggregate_allows_small_float32_peak_memory_roundoff(tmp_path):
    bench = load_multicard_module()
    artifacts = tmp_path / "artifacts"
    reports = tmp_path / "reports"
    artifacts.mkdir()
    reports.mkdir()
    for rank in range(2):
        payload = rank_payload(rank)
        payload["benchmark"]["peak_mem_mb"] = 3489.7
        payload["distributed_totals"]["peak_mem_mb"] = 6979.40234375
        (artifacts / f"multicard_qwen_compression_generate_run_rank{rank}.json").write_text(json.dumps(payload))

    summary = bench.write_summary(
        artifacts_dir=artifacts,
        reports_dir=reports,
        run_label="run",
        spawn_error=None,
    )

    assert summary["aggregate"]["distributed_reduce_consistent"] is True


def test_cpu_gloo_multicard_qwen_compression_generate_runs_two_processes(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)
    script = Path(__file__).resolve().parents[1] / "scripts" / "multicard_qwen_compression_generate_benchmark.py"

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
            "--max-new-tokens",
            "1",
            "--prompt-repeat",
            "2",
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
            "--export-dense-for-serving",
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
    summary = json.loads((demo_root / "artifacts" / "multicard_qwen_compression_generate_cpu_test.json").read_text())
    assert summary["status"] == "PASS"
    assert summary["world_size"] == 2
    assert summary["workload"]["prompt_repeat"] == 2
    assert summary["workload"]["per_rank_prompt_count"] == 4
    assert summary["workload"]["total_generated_tokens_target"] == 8
    assert summary["target_layer_pattern"] == r"self_attn\.(q_proj|k_proj)$"
    assert summary["aggregate"]["pass_count"] == 2
    assert summary["aggregate"]["baseline_tokens_per_s_total"] > 0
    assert summary["aggregate"]["cap_tokens_per_s_total"] > 0
    assert all(row["benchmark"]["cap"]["status"] == "PASS" for row in summary["ranks"])
