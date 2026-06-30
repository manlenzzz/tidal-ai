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
    script = Path(__file__).resolve().parents[1] / "scripts" / "multicard_qwen_inference_benchmark.py"
    spec = importlib.util.spec_from_file_location("multicard_qwen_inference_benchmark", script)
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
            "backend": "torch",
            "model_id": "Qwen/Qwen3-0.6B",
            "batch_size": 2,
            "prompt_tokens": 8,
            "generated_tokens": 4 + rank,
            "latency_ms": 20.0 + rank,
            "tokens_per_s": 200.0 + rank,
            "peak_mem_mb": 10.0 + rank,
        },
        "distributed_totals": {
            "tokens_per_s": 401.0,
            "generated_tokens": 9,
            "peak_mem_mb": 21.0,
            "pass_count": 2,
        },
    }


def test_write_summary_aggregates_rank_inference_metrics(tmp_path):
    bench = load_multicard_module()
    artifacts = tmp_path / "artifacts"
    reports = tmp_path / "reports"
    artifacts.mkdir()
    reports.mkdir()
    for rank in range(2):
        (artifacts / f"multicard_qwen_inference_run_rank{rank}.json").write_text(json.dumps(rank_payload(rank)))

    summary = bench.write_summary(
        artifacts_dir=artifacts,
        reports_dir=reports,
        run_label="run",
        spawn_error=None,
    )

    assert summary["status"] == "PASS"
    assert summary["world_size"] == 2
    assert summary["aggregate"]["tokens_per_s_total"] == 401.0
    assert summary["aggregate"]["generated_tokens_total"] == 9
    assert summary["aggregate"]["peak_mem_mb_total"] == 21.0
    assert summary["aggregate"]["pass_count"] == 2
    assert summary["aggregate"]["distributed_reduce_consistent"] is True
    assert (artifacts / "multicard_qwen_inference_run.json").exists()
    text = (reports / "multicard-qwen-inference-run.md").read_text()
    assert "Multi-Card Qwen Inference Benchmark" in text
    assert "| 0 | PASS | cpu |" in text
    assert "not extrapolated from one card" in text


def test_cpu_gloo_multicard_qwen_inference_runs_two_processes(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)
    script = Path(__file__).resolve().parents[1] / "scripts" / "multicard_qwen_inference_benchmark.py"

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
            "--iters",
            "1",
            "--warmup",
            "0",
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
    summary = json.loads((demo_root / "artifacts" / "multicard_qwen_inference_cpu_test.json").read_text())
    assert summary["status"] == "PASS"
    assert summary["world_size"] == 2
    assert summary["aggregate"]["pass_count"] == 2
    assert summary["aggregate"]["tokens_per_s_total"] > 0
    assert all(row["benchmark"]["status"] == "PASS" for row in summary["ranks"])
