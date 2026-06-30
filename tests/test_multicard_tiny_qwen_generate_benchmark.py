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
    script = Path(__file__).resolve().parents[1] / "scripts" / "multicard_tiny_qwen_generate_benchmark.py"
    spec = importlib.util.spec_from_file_location("multicard_tiny_qwen_generate_benchmark", script)
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
        "benchmark": {
            "status": "PASS",
            "peak_mem_mb": 0.0,
            "baseline": {"status": "PASS", "latency_ms": 10.0 + rank, "tokens_per_s": 100.0 + rank},
            "cap": {
                "status": "PASS",
                "latency_ms": 9.0 + rank,
                "tokens_per_s": 110.0 + rank,
                "latency_speedup": 1.1,
                "targeted_compression_ratio": 19.0,
            },
            "qpruner": {
                "status": "PASS",
                "latency_ms": 8.0 + rank,
                "tokens_per_s": 120.0 + rank,
                "latency_speedup": 1.2,
                "average_bits": 3.789,
            },
        },
        "distributed_totals": {
            "baseline_tokens_per_s": 201.0,
            "cap_tokens_per_s": 221.0,
            "qpruner_tokens_per_s": 241.0,
            "pass_count": 2,
        },
    }


def test_write_summary_aggregates_rank_generate_metrics(tmp_path):
    bench = load_multicard_module()
    artifacts = tmp_path / "artifacts"
    reports = tmp_path / "reports"
    artifacts.mkdir()
    reports.mkdir()
    for rank in range(2):
        (artifacts / f"multicard_tiny_qwen_generate_run_rank{rank}.json").write_text(
            json.dumps(rank_payload(rank))
        )

    summary = bench.write_summary(
        artifacts_dir=artifacts,
        reports_dir=reports,
        run_label="run",
        spawn_error=None,
    )

    assert summary["status"] == "PASS"
    assert summary["world_size"] == 2
    assert summary["aggregate"]["baseline_tokens_per_s_total"] == 201.0
    assert summary["aggregate"]["cap_tokens_per_s_total"] == 221.0
    assert summary["aggregate"]["qpruner_tokens_per_s_total"] == 241.0
    assert summary["aggregate"]["distributed_reduce_consistent"] is True
    assert (artifacts / "multicard_tiny_qwen_generate_run.json").exists()
    text = (reports / "multicard-tiny-qwen-generate-run.md").read_text()
    assert "Multi-Card TinyQwen Compressed Generate Benchmark" in text
    assert "| 0 | PASS | cpu |" in text
    assert "all-reduce" in text


def test_cpu_gloo_multicard_tiny_qwen_generate_runs_two_processes(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)
    script = Path(__file__).resolve().parents[1] / "scripts" / "multicard_tiny_qwen_generate_benchmark.py"

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
            "--backend",
            "gloo",
            "--cards",
            "0,1",
            "--max-new-tokens",
            "1",
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
            "--enable-inference-cache",
            "--export-dense-for-serving",
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
    summary = json.loads((demo_root / "artifacts" / "multicard_tiny_qwen_generate_cpu_test.json").read_text())
    assert summary["status"] == "PASS"
    assert summary["world_size"] == 2
    assert summary["aggregate"]["pass_count"] == 2
    assert summary["aggregate"]["baseline_tokens_per_s_total"] > 0
    assert all(row["benchmark"]["cap"]["status"] == "PASS" for row in summary["ranks"])
