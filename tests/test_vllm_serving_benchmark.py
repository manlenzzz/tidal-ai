import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path


def load_benchmark_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "vllm_serving_benchmark.py"
    assert script.exists(), f"missing benchmark script: {script}"
    spec = importlib.util.spec_from_file_location("vllm_serving_benchmark", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_export_dirs(demo_root: Path, run_label: str = "serving_export") -> None:
    for method in ("baseline", "cap", "qpruner"):
        export_dir = demo_root / "serving_exports" / run_label / method
        export_dir.mkdir(parents=True)
        (export_dir / "config.json").write_text("{}")


def install_fake_vllm(monkeypatch):
    calls = []

    class FakeCompletion:
        def __init__(self, token_count: int):
            self.token_ids = list(range(token_count))
            self.text = " ".join(f"tok_{idx}" for idx in range(token_count))

    class FakeRequestOutput:
        def __init__(self, token_count: int):
            self.outputs = [FakeCompletion(token_count)]

    class FakeLLM:
        def __init__(self, *args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})

        def generate(self, prompts, sampling_params):
            return [FakeRequestOutput(sampling_params.max_tokens) for _ in prompts]

    class FakeSamplingParams:
        def __init__(self, *, max_tokens: int, temperature: float):
            self.max_tokens = max_tokens
            self.temperature = temperature

    fake_vllm = types.SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm_ascend", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vllm_ascend.patch", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vllm_ascend.patch.platform", types.SimpleNamespace())
    return calls


def test_parse_npu_smi_process_memory_extracts_peak_process_hbm():
    bench = load_benchmark_module()

    sample = """
+------------------------------------------------------------------------------+
| NPU     Chip     | Process id      | Process name             | Process memory |
+------------------------------------------------------------------------------+
| 0       0        | 43210           | python                   | 8192 MB        |
| 1       0        | 43211           | python                   | 4096 MB        |
| 2       0        | 0               | -                        | 0 MB           |
+------------------------------------------------------------------------------+
"""

    assert bench.parse_npu_smi_process_memory_mb(sample) == 8192.0


def test_parse_npu_smi_process_memory_can_filter_visible_card():
    bench = load_benchmark_module()

    sample = """
| NPU     Chip     | Process id      | Process name             | Process memory |
| 0       0        | 43210           | python                   | 8192 MB        |
| 1       0        | 43211           | python                   | 4096 MB        |
"""

    assert bench.parse_npu_smi_process_memory_mb(sample, cards={"1"}) == 4096.0


def test_select_memory_measurement_prefers_npu_smi_over_zero_torch_allocator():
    bench = load_benchmark_module()

    measurement = bench.select_memory_measurement(torch_peak_mb=0.0, npu_smi_process_mb=6144.0)

    assert measurement == {
        "peak_mem_mb": 6144.0,
        "memory_measurement_source": "npu-smi process table",
        "torch_peak_mem_mb": 0.0,
        "npu_smi_process_mem_mb": 6144.0,
    }


def test_select_memory_measurement_marks_unavailable_when_all_sources_are_zero():
    bench = load_benchmark_module()

    measurement = bench.select_memory_measurement(torch_peak_mb=0.0, npu_smi_process_mb=None)

    assert measurement["peak_mem_mb"] is None
    assert measurement["memory_measurement_source"] == "unavailable"
    assert measurement["torch_peak_mem_mb"] == 0.0
    assert measurement["npu_smi_process_mem_mb"] is None


def test_markdown_report_summarizes_vllm_selector_shim_benchmark():
    bench = load_benchmark_module()

    text = bench.markdown_report(
        {
            "status": "PASS",
            "run_label": "tiny_qwen3_serving_vllm_selector_shim_npu",
            "export_run_label": "tiny_qwen3_serving_export_npu",
            "backend": "vllm_ascend_generate",
            "dtype": "float16",
            "preload_vllm_ascend_patch": True,
            "preload_vllm_ascend_selector_shim": True,
            "preload_vllm_ascend_metadata_shim": True,
            "summary": {
                "best_method": "qpruner",
                "best_tokens_per_s": 125.0,
                "cap_vs_baseline_speedup": 1.1,
                "qpruner_vs_baseline_speedup": 1.25,
                "cap_vs_qpruner_speedup": 0.88,
            },
            "exports": {
                "baseline": {
                    "status": "PASS",
                    "exists": True,
                    "vllm_load": "PASS",
                    "load_time_s": 1.2,
                    "latency_ms": 24.0,
                    "tokens_per_s": 100.0,
                    "peak_mem_mb": 512.0,
                    "memory_measurement_source": "npu-smi process table",
                    "npu_smi_process_mem_mb": 512.0,
                    "torch_peak_mem_mb": 0.0,
                },
                "cap": {
                    "status": "PASS",
                    "exists": True,
                    "vllm_load": "PASS",
                    "load_time_s": 1.1,
                    "latency_ms": 22.0,
                    "tokens_per_s": 110.0,
                    "peak_mem_mb": 512.0,
                    "memory_measurement_source": "npu-smi process table",
                    "npu_smi_process_mem_mb": 512.0,
                    "torch_peak_mem_mb": 0.0,
                },
                "qpruner": {
                    "status": "PASS",
                    "exists": True,
                    "vllm_load": "PASS",
                    "load_time_s": 1.0,
                    "latency_ms": 20.0,
                    "tokens_per_s": 125.0,
                    "peak_mem_mb": 512.0,
                    "memory_measurement_source": "npu-smi process table",
                    "npu_smi_process_mem_mb": 512.0,
                    "torch_peak_mem_mb": 0.0,
                },
            },
        }
    )

    assert "vLLM-Ascend Serving Benchmark" in text
    assert "| Baseline | PASS | True | PASS | 1.200 | 24.000 | 100.000 | 512.000 | npu-smi process table | 512.000 | 0.000 |" in text
    assert "| CAP | PASS | True | PASS | 1.100 | 22.000 | 110.000 | 512.000 | npu-smi process table | 512.000 | 0.000 |" in text
    assert "| QPruner | PASS | True | PASS | 1.000 | 20.000 | 125.000 | 512.000 | npu-smi process table | 512.000 | 0.000 |" in text
    assert "preload_vllm_ascend_patch" in text
    assert "preload_vllm_ascend_selector_shim" in text
    assert "preload_vllm_ascend_metadata_shim" in text
    assert "qpruner_vs_baseline_speedup" in text


def test_markdown_report_summarizes_failed_engine_root_cause():
    bench = load_benchmark_module()

    text = bench.markdown_report(
        {
            "status": "FAIL",
            "run_label": "tiny_qwen3_serving_vllm_selector_shim_npu",
            "export_run_label": "tiny_qwen3_serving_export_npu",
            "backend": "vllm_ascend_generate",
            "dtype": "float16",
            "preload_vllm_ascend_patch": True,
            "preload_vllm_ascend_selector_shim": True,
            "summary": {},
            "exports": {
                "baseline": {
                    "status": "FAIL",
                    "vllm_load": "FAIL",
                    "error_type": "EngineDeadError",
                    "error": "EngineCore encountered an issue.",
                    "stdout_tail": (
                        "ERROR attention_v1.py line 897\n"
                        "AttributeError: 'list' object has no attribute 'slot_mapping'\n"
                    ),
                },
                "cap": {
                    "status": "FAIL",
                    "vllm_load": "FAIL",
                    "error_type": "EngineDeadError",
                    "error": "EngineCore encountered an issue.",
                    "stdout_tail": (
                        "ERROR attention_v1.py line 897\n"
                        "AttributeError: 'list' object has no attribute 'slot_mapping'\n"
                    ),
                },
                "qpruner": {
                    "status": "FAIL",
                    "vllm_load": "FAIL",
                    "error_type": "EngineDeadError",
                    "error": "EngineCore encountered an issue.",
                    "stdout_tail": (
                        "ERROR attention_v1.py line 897\n"
                        "AttributeError: 'list' object has no attribute 'slot_mapping'\n"
                    ),
                },
            },
        }
    )

    assert "Root cause | baseline, CAP, QPruner: EngineDeadError" in text
    assert "AttributeError: 'list' object has no attribute 'slot_mapping'" in text


def test_missing_export_report_marks_model_missing(tmp_path):
    bench = load_benchmark_module()

    report = bench.run_benchmark(
        demo_root=tmp_path / "demo",
        export_run_label="missing",
        run_label="vllm_missing",
        dtype_name="float16",
        max_new_tokens=2,
        iters=1,
        warmup=0,
        gpu_memory_utilization=0.05,
        preload_vllm_ascend_patch=True,
        preload_vllm_ascend_selector_shim=True,
        preload_vllm_ascend_metadata_shim=True,
        subprocess_per_method=False,
    )

    assert report["status"] == "MODEL_MISSING"
    assert report["exports"]["baseline"]["status"] == "MISSING"
    assert report["exports"]["cap"]["status"] == "MISSING"
    assert report["exports"]["qpruner"]["status"] == "MISSING"


def test_vllm_benchmark_runs_with_fake_vllm_and_selector_shim(monkeypatch, tmp_path):
    bench = load_benchmark_module()
    calls = install_fake_vllm(monkeypatch)
    demo_root = tmp_path / "demo"
    write_export_dirs(demo_root)

    monkeypatch.setattr(bench, "install_selector_shim", lambda enabled: {"enabled": enabled, "status": "PATCHED"})
    monkeypatch.setattr(bench, "install_metadata_shim", lambda enabled: {"enabled": enabled, "status": "PATCHED"})
    monkeypatch.setattr(bench, "reset_peak_memory", lambda: None)
    monkeypatch.setattr(bench, "peak_memory_mb", lambda: 256.0)
    monkeypatch.setattr(bench, "npu_smi_process_memory_mb", lambda: 1024.0)
    monkeypatch.setattr(bench, "synchronize_backend", lambda: None)

    report = bench.run_benchmark(
        demo_root=demo_root,
        export_run_label="serving_export",
        run_label="fake_vllm",
        dtype_name="float16",
        max_new_tokens=3,
        iters=2,
        warmup=1,
        gpu_memory_utilization=0.05,
        preload_vllm_ascend_patch=True,
        preload_vllm_ascend_selector_shim=True,
        preload_vllm_ascend_metadata_shim=True,
        subprocess_per_method=False,
    )

    assert report["status"] == "PASS"
    assert report["backend"] == "vllm_ascend_generate"
    assert report["preload_vllm_ascend_patch"] is True
    assert report["preload_vllm_ascend_selector_shim"] is True
    assert report["preload_vllm_ascend_metadata_shim"] is True
    assert report["selector_shim"]["status"] == "PATCHED"
    assert report["metadata_shim"]["status"] == "PATCHED"
    assert len(calls) == 3
    assert calls[0]["kwargs"]["gpu_memory_utilization"] == 0.05
    assert calls[0]["kwargs"]["dtype"] == "float16"
    assert report["exports"]["baseline"]["status"] == "PASS"
    assert report["exports"]["cap"]["status"] == "PASS"
    assert report["exports"]["qpruner"]["status"] == "PASS"
    assert report["exports"]["baseline"]["memory_measurement_source"] == "npu-smi process table"
    assert report["exports"]["baseline"]["npu_smi_process_mem_mb"] == 1024.0
    assert report["exports"]["baseline"]["torch_peak_mem_mb"] == 256.0
    assert report["exports"]["baseline"]["peak_mem_mb"] == 1024.0
    assert report["exports"]["cap"]["generated_tokens"] == 6
    assert report["exports"]["qpruner"]["tokens_per_s"] > 0
    assert report["summary"]["best_method"] in {"baseline", "cap", "qpruner"}
    assert report["summary"]["cap_vs_baseline_speedup"] is not None
    assert report["summary"]["qpruner_vs_baseline_speedup"] is not None


def test_benchmark_export_does_not_touch_memory_api_before_vllm_load(monkeypatch, tmp_path):
    bench = load_benchmark_module()
    events = []
    export_dir = tmp_path / "baseline"
    export_dir.mkdir()

    class FakeCompletion:
        token_ids = [1, 2]

    class FakeRequestOutput:
        outputs = [FakeCompletion()]

    class FakeLLM:
        def __init__(self, *args, **kwargs):
            events.append("llm_init")

        def generate(self, prompts, sampling_params):
            events.append("generate")
            return [FakeRequestOutput() for _ in prompts]

    class FakeSamplingParams:
        def __init__(self, *, max_tokens: int, temperature: float):
            self.max_tokens = max_tokens
            self.temperature = temperature

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams))
    monkeypatch.setattr(bench, "reset_peak_memory", lambda: events.append("reset_peak_memory"))
    monkeypatch.setattr(bench, "peak_memory_mb", lambda: 128.0)
    monkeypatch.setattr(bench, "npu_smi_process_memory_mb", lambda: None)
    monkeypatch.setattr(bench, "synchronize_backend", lambda: events.append("synchronize"))

    result = bench.benchmark_export(
        path=export_dir,
        method="baseline",
        dtype_name="float16",
        prompts=["hello"],
        max_new_tokens=2,
        iters=1,
        warmup=0,
        gpu_memory_utilization=0.05,
        max_model_len=128,
    )

    assert result["status"] == "PASS"
    assert result["memory_measurement_source"] == "torch allocator"
    assert result["torch_peak_mem_mb"] == 128.0
    assert result["npu_smi_process_mem_mb"] is None
    assert events[0] == "llm_init"
    assert "reset_peak_memory" in events


def test_run_benchmark_can_isolate_methods_in_subprocess(monkeypatch, tmp_path):
    bench = load_benchmark_module()
    demo_root = tmp_path / "demo"
    write_export_dirs(demo_root)
    commands = []
    environments = []

    original_run = subprocess.run

    def fake_run(command, check=False, text=False, capture_output=False, env=None, **kwargs):
        if not isinstance(command, list) or "--single-method" not in command:
            return original_run(command, check=check, text=text, capture_output=capture_output, env=env, **kwargs)
        commands.append(command)
        environments.append(env)
        method = command[command.index("--single-method") + 1]
        tokens = {"baseline": 100.0, "cap": 110.0, "qpruner": 125.0}[method]
        payload = {
            "status": "PASS",
            "method": method,
            "exists": True,
            "path": command[command.index("--method-path") + 1],
            "vllm_load": "PASS",
            "load_time_s": 1.0,
            "latency_ms": 20.0,
            "generated_tokens": 6,
            "tokens_per_s": tokens,
            "peak_mem_mb": 256.0,
        }
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="noise\nVLLM_METHOD_BENCHMARK " + json.dumps(payload) + "\n",
            stderr="",
        )

    monkeypatch.setattr(bench.subprocess, "run", fake_run)

    report = bench.run_benchmark(
        demo_root=demo_root,
        export_run_label="serving_export",
        run_label="subprocess_vllm",
        dtype_name="float16",
        max_new_tokens=3,
        iters=2,
        warmup=1,
        gpu_memory_utilization=0.05,
        preload_vllm_ascend_patch=True,
        preload_vllm_ascend_selector_shim=True,
        preload_vllm_ascend_metadata_shim=True,
        subprocess_per_method=True,
    )

    assert report["status"] == "PASS"
    assert len(commands) == 3
    assert all("--single-method" in command for command in commands)
    assert all("--method-path" in command for command in commands)
    assert all("--preload-vllm-ascend-metadata-shim" in command for command in commands)
    assert all(env is not None for env in environments)
    assert all(env["TIDAL_VLLM_ASCEND_METADATA_SHIM"] == "1" for env in environments)
    assert all(str(bench.REPO_ROOT) in env["PYTHONPATH"].split(":") for env in environments)
    assert report["exports"]["baseline"]["tokens_per_s"] == 100.0
    assert report["exports"]["cap"]["tokens_per_s"] == 110.0
    assert report["exports"]["qpruner"]["tokens_per_s"] == 125.0
    assert report["summary"]["qpruner_vs_baseline_speedup"] == 1.25


def test_metadata_shim_environment_preserves_acl_runtime_pythonpath(monkeypatch):
    bench = load_benchmark_module()
    monkeypatch.setenv("PYTHONPATH", "/existing")

    def fake_apply_acl_pythonpath_to_env(env):
        env["PYTHONPATH"] = env["PYTHONPATH"] + os.pathsep + "/cann/python"
        return ["/cann/python"]

    monkeypatch.setattr(bench, "apply_acl_pythonpath_to_env", fake_apply_acl_pythonpath_to_env)

    env = bench.metadata_shim_environment(True)

    entries = env["PYTHONPATH"].split(os.pathsep)
    assert entries[0] == str(bench.REPO_ROOT)
    assert "/existing" in entries
    assert "/cann/python" in entries


def test_run_benchmark_can_run_methods_in_parallel_on_distinct_cards(monkeypatch, tmp_path):
    bench = load_benchmark_module()
    demo_root = tmp_path / "demo"
    write_export_dirs(demo_root)
    commands = []
    environments = []

    class FakePopen:
        def __init__(self, command, text=False, stdout=None, stderr=None, env=None, **kwargs):
            self.command = command
            self.env = env
            self.returncode = 0
            commands.append(command)
            environments.append(env)

        def communicate(self):
            method = self.command[self.command.index("--single-method") + 1]
            tokens = {"baseline": 100.0, "cap": 110.0, "qpruner": 125.0}[method]
            payload = {
                "status": "PASS",
                "method": method,
                "exists": True,
                "path": self.command[self.command.index("--method-path") + 1],
                "vllm_load": "PASS",
                "load_time_s": 1.0,
                "latency_ms": 20.0,
                "generated_tokens": 6,
                "tokens_per_s": tokens,
                "peak_mem_mb": 1024.0,
                "memory_measurement_source": "npu-smi process table",
                "npu_smi_process_mem_mb": 1024.0,
                "torch_peak_mem_mb": 0.0,
            }
            return "VLLM_METHOD_BENCHMARK " + json.dumps(payload) + "\n", ""

    monkeypatch.setattr(bench.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(bench.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(bench, "import_vllm_ascend_patch", lambda enabled: {"enabled": enabled, "status": "IMPORTED"})
    monkeypatch.setattr(bench, "install_selector_shim", lambda enabled: {"enabled": enabled, "status": "PATCHED"})
    monkeypatch.setattr(bench, "install_metadata_shim", lambda enabled: {"enabled": enabled, "status": "PATCHED"})

    report = bench.run_benchmark(
        demo_root=demo_root,
        export_run_label="serving_export",
        run_label="parallel_vllm",
        dtype_name="float16",
        max_new_tokens=3,
        iters=2,
        warmup=1,
        gpu_memory_utilization=0.05,
        preload_vllm_ascend_patch=True,
        preload_vllm_ascend_selector_shim=True,
        preload_vllm_ascend_metadata_shim=True,
        subprocess_per_method=True,
        parallel_methods=True,
        method_cards=[0, 1, 2],
    )

    assert report["status"] == "PASS"
    assert report["parallel_methods"] is True
    assert report["method_cards"] == [0, 1, 2]
    assert len(commands) == 3
    assert [env["ASCEND_RT_VISIBLE_DEVICES"] for env in environments] == ["0", "1", "2"]
    assert all(env["TIDAL_VLLM_ASCEND_METADATA_SHIM"] == "1" for env in environments)
    assert report["exports"]["baseline"]["visible_card"] == 0
    assert report["exports"]["cap"]["visible_card"] == 1
    assert report["exports"]["qpruner"]["visible_card"] == 2
    assert report["summary"]["qpruner_vs_baseline_speedup"] == 1.25


def test_subprocess_failure_payload_keeps_log_tails(monkeypatch, tmp_path):
    bench = load_benchmark_module()
    export_dir = tmp_path / "baseline"
    export_dir.mkdir()

    def fake_run(command, check=False, text=False, capture_output=False, env=None, **kwargs):
        payload = {
            "status": "FAIL",
            "method": "baseline",
            "exists": True,
            "path": str(export_dir),
            "vllm_load": "FAIL",
            "error_type": "EngineDeadError",
            "error": "EngineCore encountered an issue.",
        }
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="root stack line\nVLLM_METHOD_BENCHMARK " + json.dumps(payload) + "\n",
            stderr="stderr stack line",
        )

    monkeypatch.setattr(bench.subprocess, "run", fake_run)

    payload = bench.benchmark_export_subprocess(
        path=export_dir,
        method="baseline",
        dtype_name="float16",
        max_new_tokens=4,
        iters=1,
        warmup=0,
        gpu_memory_utilization=0.05,
        max_model_len=128,
        preload_vllm_ascend_patch=True,
        preload_vllm_ascend_selector_shim=True,
        preload_vllm_ascend_metadata_shim=True,
    )

    assert payload["status"] == "FAIL"
    assert payload["subprocess_returncode"] == 1
    assert "root stack line" in payload["stdout_tail"]
    assert "stderr stack line" in payload["stderr_tail"]


def test_write_outputs_uses_vllm_artifact_and_report_names(tmp_path):
    bench = load_benchmark_module()
    report = {
        "status": "MODEL_MISSING",
        "run_label": "sample",
        "export_run_label": "missing",
        "backend": "vllm_ascend_generate",
        "exports": {},
        "summary": {},
    }

    bench.write_outputs(report, tmp_path / "demo", "sample")

    artifact = tmp_path / "demo" / "artifacts" / "vllm_serving_benchmark_sample.json"
    report_path = tmp_path / "demo" / "reports" / "vllm-serving-benchmark-sample.md"
    assert json.loads(artifact.read_text())["status"] == "MODEL_MISSING"
    assert "vLLM-Ascend Serving Benchmark" in report_path.read_text()


def test_single_method_main_writes_output_json(monkeypatch, tmp_path):
    bench = load_benchmark_module()
    output_json = tmp_path / "single.json"
    method_dir = tmp_path / "baseline"
    method_dir.mkdir()

    monkeypatch.setattr(bench, "import_vllm_ascend_patch", lambda enabled: {"enabled": enabled, "status": "IMPORTED"})
    monkeypatch.setattr(bench, "install_selector_shim", lambda enabled: {"enabled": enabled, "status": "INSTALLED"})
    monkeypatch.setattr(bench, "install_metadata_shim", lambda enabled: {"enabled": enabled, "status": "INSTALLED"})
    monkeypatch.setattr(
        bench,
        "benchmark_export",
        lambda **kwargs: {
            "status": "PASS",
            "method": kwargs["method"],
            "tokens_per_s": 12.3,
            "vllm_load": "PASS",
        },
    )

    rc = bench.main(
        [
            "--single-method",
            "baseline",
            "--method-path",
            str(method_dir),
            "--output-json",
            str(output_json),
            "--preload-vllm-ascend-metadata-shim",
        ]
    )

    assert rc == 0
    payload = json.loads(output_json.read_text())
    assert payload["status"] == "PASS"
    assert payload["method"] == "baseline"
    assert payload["metadata_shim"]["status"] == "INSTALLED"
