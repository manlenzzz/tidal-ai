import importlib.util
import json
import sys
from pathlib import Path


def load_runner():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_parallel_ascend_benchmarks.py"
    spec = importlib.util.spec_from_file_location("run_parallel_ascend_benchmarks", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_benchmark(path: Path, *, device: str, speedup: float, ratio: float) -> None:
    path.write_text(
        json.dumps(
            {
                "device": device,
                "baseline_latency_ms": 10.0,
                "compressed_latency_ms": round(10.0 / speedup, 4),
                "baseline_tokens_per_s": 6400.0,
                "compressed_tokens_per_s": round(6400.0 * speedup, 3),
                "latency_speedup": speedup,
                "targeted_compression_ratio": ratio,
                "targeted_param_reduction_pct": 100.0 * (1.0 - 1.0 / ratio),
                "compression_time_s": 1.25,
                "peak_mem_mb": 42.0,
            }
        )
    )


def test_collect_benchmark_reports_adds_card_and_status(tmp_path):
    runner = load_runner()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    write_benchmark(artifacts / "cap_benchmark_npu3.json", device="npu", speedup=1.2, ratio=4.0)

    reports = runner.collect_benchmark_reports(artifacts)

    assert reports == [
        {
            "benchmark": "cap_benchmark",
            "run_label": "",
            "card": 3,
            "status": "PASS",
            "device": "npu",
            "baseline_latency_ms": 10.0,
            "compressed_latency_ms": 8.3333,
            "baseline_tokens_per_s": 6400.0,
            "compressed_tokens_per_s": 7680.0,
            "latency_speedup": 1.2,
            "targeted_compression_ratio": 4.0,
            "targeted_param_reduction_pct": 75.0,
            "compression_time_s": 1.25,
            "peak_mem_mb": 42.0,
        }
    ]


def test_collect_benchmark_reports_ignores_workflow_smoke_json(tmp_path):
    runner = load_runner()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    write_benchmark(artifacts / "cap_benchmark_npu4.json", device="npu", speedup=1.1, ratio=4.0)
    (artifacts / "cap_npu0.json").write_text(json.dumps({"status": "PASS", "device": "npu"}))
    (artifacts / "qpruner_npu1.json").write_text(json.dumps({"status": "PASS", "device": "npu"}))

    reports = runner.collect_benchmark_reports(artifacts)

    assert [row["benchmark"] for row in reports] == ["cap_benchmark"]
    assert [row["card"] for row in reports] == [4]


def test_write_aggregate_outputs_json_and_markdown(tmp_path):
    runner = load_runner()
    artifacts = tmp_path / "artifacts"
    reports = tmp_path / "reports"
    artifacts.mkdir()
    reports.mkdir()
    rows = [
        {
            "benchmark": "cap_benchmark",
            "card": 3,
            "status": "PASS",
            "device": "npu",
            "baseline_latency_ms": 10.0,
            "compressed_latency_ms": 8.0,
            "baseline_tokens_per_s": 6400.0,
            "compressed_tokens_per_s": 8000.0,
            "latency_speedup": 1.25,
            "targeted_compression_ratio": 4.0,
            "targeted_param_reduction_pct": 75.0,
            "compression_time_s": 1.25,
            "peak_mem_mb": 42.0,
        }
    ]

    aggregate = runner.write_aggregate(rows, artifacts_dir=artifacts, reports_dir=reports)

    assert aggregate["status"] == "PASS"
    assert (artifacts / "parallel_ascend_benchmarks.json").exists()
    markdown = (reports / "parallel-ascend-benchmarks.md").read_text()
    assert "| cap_benchmark |  | 3 | PASS | npu | 1.250 | 4.000 | 8000.000 | 42.000 |" in markdown


def test_run_one_uses_run_label_in_artifact_names(tmp_path, monkeypatch):
    runner = load_runner()
    popen_calls = []

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            popen_calls.append((cmd, kwargs))

    monkeypatch.setattr(runner.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(runner.sys, "executable", sys.executable)

    process = runner.run_one(
        project_root=tmp_path,
        demo_root=tmp_path / "demo",
        card=4,
        benchmark="cap_benchmark",
        run_label="fastpath",
        device="npu",
        dtype="float32",
        budget=8000,
        seq_len=32,
        batch_size=2,
        iters=5,
        warmup=2,
        max_iter=5,
        policy_steps=1,
        samples_per_step=1,
        seed=0,
    )

    assert isinstance(process, FakePopen)
    cmd, kwargs = popen_calls[0]
    assert str(tmp_path / "demo" / "artifacts" / "cap_benchmark_fastpath_npu4.json") in cmd
    assert kwargs["stdout"].name.endswith("cap_benchmark_fastpath_npu4.log")
