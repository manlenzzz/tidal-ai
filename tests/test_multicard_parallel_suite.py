import importlib.util
import json
import sys
from pathlib import Path


def load_suite():
    script = Path(__file__).resolve().parents[1] / "scripts" / "multicard_parallel_suite.py"
    spec = importlib.util.spec_from_file_location("multicard_parallel_suite", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_assign_workers_repeats_tasks_across_cards(tmp_path):
    suite = load_suite()

    workers = suite.build_worker_specs(
        project_root=tmp_path / "project",
        demo_root=tmp_path / "demo",
        cards=[0, 2, 4, 6],
        tasks=["workflow_cap", "workflow_qpruner"],
        run_label="sync_suite",
        device="npu",
        dtype="float16",
        export_run_label="tiny_qwen3_serving_export_npu",
        iters=3,
        warmup=1,
        max_new_tokens=4,
        seed=11,
    )

    assert [(worker.rank, worker.card, worker.task) for worker in workers] == [
        (0, 0, "workflow_cap"),
        (1, 2, "workflow_qpruner"),
        (2, 4, "workflow_cap"),
        (3, 6, "workflow_qpruner"),
    ]
    assert workers[0].visible_devices == "0"
    assert workers[0].payload_json.name == "multicard_parallel_suite_sync_suite_rank0_workflow_cap_npu0_payload.json"
    assert workers[1].log_path.name == "multicard_parallel_suite_sync_suite_rank1_workflow_qpruner_npu2.log"
    assert "ascend_npu_workflow_smoke.py" in workers[0].command[1]
    assert "--workflow" in workers[0].command
    assert "cap" in workers[0].command


def test_serving_worker_uses_torch_fallback_artifact_name(tmp_path):
    suite = load_suite()

    worker = suite.build_worker_specs(
        project_root=tmp_path / "project",
        demo_root=tmp_path / "demo",
        cards=[3],
        tasks=["torch_serving_fallback"],
        run_label="sync_suite",
        device="npu",
        dtype="float16",
        export_run_label="tiny_qwen3_serving_export_npu",
        iters=2,
        warmup=0,
        max_new_tokens=4,
        seed=0,
    )[0]

    assert "torch_serving_fallback_benchmark.py" in worker.command[1]
    assert "--export-run-label" in worker.command
    assert "tiny_qwen3_serving_export_npu" in worker.command
    assert worker.payload_json.name == "torch_serving_fallback_sync_suite_rank0_npu3.json"
    assert "sync_suite_rank0_npu3" in worker.command


def test_vllm_serving_worker_uses_single_method_export_and_metadata_shim(tmp_path):
    suite = load_suite()

    worker = suite.build_worker_specs(
        project_root=tmp_path / "project",
        demo_root=tmp_path / "demo",
        cards=[5],
        tasks=["vllm_serving_benchmark"],
        run_label="sync_vllm",
        device="npu",
        dtype="float16",
        export_run_label="tiny_qwen3_serving_export_npu",
        iters=1,
        warmup=0,
        max_new_tokens=1,
        seed=0,
    )[0]

    assert "vllm_serving_benchmark.py" in worker.command[1]
    assert "--single-method" in worker.command
    assert worker.command[worker.command.index("--single-method") + 1] == "qpruner"
    assert "--method-path" in worker.command
    assert str(tmp_path / "demo" / "serving_exports" / "tiny_qwen3_serving_export_npu" / "qpruner") in worker.command
    assert "--output-json" in worker.command
    assert worker.command[worker.command.index("--output-json") + 1] == str(worker.payload_json)
    assert "--preload-vllm-ascend-metadata-shim" in worker.command
    assert worker.payload_json.name == "vllm_serving_benchmark_sync_vllm_rank0_qpruner_npu5.json"


def test_vllm_serving_workers_rotate_methods_by_rank(tmp_path):
    suite = load_suite()

    workers = suite.build_worker_specs(
        project_root=tmp_path / "project",
        demo_root=tmp_path / "demo",
        cards=[0, 1, 2, 3],
        tasks=["vllm_serving_benchmark"],
        run_label="sync_vllm",
        device="npu",
        dtype="float16",
        export_run_label="tiny_qwen3_serving_export_npu",
        iters=1,
        warmup=0,
        max_new_tokens=1,
        seed=0,
    )

    methods = [worker.command[worker.command.index("--single-method") + 1] for worker in workers]
    assert methods == ["qpruner", "cap", "baseline", "qpruner"]


def test_vllm_worker_env_enables_startup_metadata_shim(tmp_path, monkeypatch):
    suite = load_suite()
    project_root = tmp_path / "project"
    demo_root = tmp_path / "demo"
    project_root.mkdir()
    observed = {}

    worker = suite.build_worker_specs(
        project_root=project_root,
        demo_root=demo_root,
        cards=[2],
        tasks=["vllm_serving_benchmark"],
        run_label="sync_vllm",
        device="npu",
        dtype="float16",
        export_run_label="tiny_qwen3_serving_export_npu",
        iters=1,
        warmup=0,
        max_new_tokens=1,
        seed=0,
    )[0]

    class FakeProc:
        def __init__(self, command, cwd=None, env=None, stdout=None, stderr=None, text=None):
            observed["command"] = command
            observed["env"] = env
            self.returncode = 0

        def wait(self):
            return self.returncode

    monkeypatch.setattr(suite.subprocess, "Popen", FakeProc)
    suite.run_worker(worker, project_root=project_root, demo_root=demo_root, dry_run=False, sync_start_target_ts=1234.5)

    assert observed["env"]["TIDAL_VLLM_ASCEND_METADATA_SHIM"] == "1"
    assert str(project_root) in observed["env"]["PYTHONPATH"].split(":")
    assert "--preload-vllm-ascend-metadata-shim" in observed["command"]


def test_extract_metrics_includes_vllm_serving_benchmark_summary():
    suite = load_suite()

    metrics = suite.extract_metrics(
        "vllm_serving_benchmark",
        {
            "status": "PASS",
            "method": "qpruner",
            "tokens_per_s": 12.399,
            "latency_ms": 161.3,
            "peak_mem_mb": 512.0,
            "metadata_shim": {"status": "INSTALLED", "backend_forward_status": "INSTALLED"},
        },
    )

    assert metrics == {
        "method": "qpruner",
        "tokens_per_s": 12.399,
        "latency_ms": 161.3,
        "peak_mem_mb": 512.0,
        "metadata_shim_status": "INSTALLED",
        "metadata_backend_forward_status": "INSTALLED",
    }


def test_write_summary_marks_synchronized_parallel_suite(tmp_path):
    suite = load_suite()
    demo_root = tmp_path / "demo"
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    artifacts.mkdir(parents=True)
    reports.mkdir(parents=True)
    workers = []
    for rank, card, task, started, tps in [
        (0, 0, "workflow_cap", 1000.00, None),
        (1, 1, "torch_serving_fallback", 1000.08, 456.0),
    ]:
        worker = {
            "status": "PASS",
            "rank": rank,
            "card": card,
            "task": task,
            "payload_status": "PASS",
            "started_at": started,
            "sync_start_target_ts": 999.95,
            "released_at": started + 0.02,
            "release_lag_s": 0.07 if rank == 0 else 0.15,
            "finished_at": started + 1.0,
            "runtime_s": 1.0,
            "log_path": f"logs/worker{rank}.log",
            "payload_json": f"artifacts/payload{rank}.json",
        }
        if tps is not None:
            worker["metrics"] = {"best_tokens_per_s": tps}
        workers.append(worker)

    summary = suite.write_summary(
        workers,
        demo_root=demo_root,
        run_label="sync_suite",
        cards=[0, 1],
        max_start_skew_seconds=0.25,
    )

    assert summary["status"] == "PASS"
    assert summary["world_size"] == 2
    assert summary["launched_synchronously"] is True
    assert summary["sync_start_target_ts"] == 999.95
    assert summary["start_window_s"] == 0.08
    assert summary["release_lag_window_s"] == 0.08
    assert summary["task_counts"] == {"torch_serving_fallback": 1, "workflow_cap": 1}
    assert summary["aggregate"]["pass_count"] == 2
    assert summary["aggregate"]["best_tokens_per_s"] == 456.0
    assert (artifacts / "multicard_parallel_suite_sync_suite.json").exists()
    markdown = (reports / "multicard-parallel-suite-sync_suite.md").read_text()
    assert "Synchronized Multi-Card Parallel Suite" in markdown
    assert "| Sync start target | 999.950 |" in markdown
    assert "| Release lag window seconds | 0.080 |" in markdown
    assert "| 1 | 1 | torch_serving_fallback | PASS | PASS | 1.000 |" in markdown


def test_run_worker_uses_common_start_gate_before_command(tmp_path, monkeypatch):
    suite = load_suite()
    project_root = tmp_path / "project"
    demo_root = tmp_path / "demo"
    project_root.mkdir()
    observed = {}

    worker = suite.build_worker_specs(
        project_root=project_root,
        demo_root=demo_root,
        cards=[3],
        tasks=["workflow_cap"],
        run_label="sync_suite",
        device="cpu",
        dtype="float16",
        export_run_label="tiny_qwen3_serving_export_npu",
        iters=1,
        warmup=0,
        max_new_tokens=1,
        seed=7,
    )[0]

    class FakeProc:
        def __init__(self, command, cwd=None, env=None, stdout=None, stderr=None, text=None):
            observed["command"] = command
            observed["cwd"] = cwd
            observed["env"] = env
            observed["text"] = text
            self.returncode = 0

        def wait(self):
            return self.returncode

    monkeypatch.setattr(suite.subprocess, "Popen", FakeProc)
    entry = suite.run_worker(
        worker,
        project_root=project_root,
        demo_root=demo_root,
        dry_run=False,
        sync_start_target_ts=1234.5,
    )

    assert entry["sync_start_target_ts"] == 1234.5
    assert observed["command"][:2] == [sys.executable, "-c"]
    wrapper = observed["command"][2]
    assert "TIDAL_SYNC_START_TARGET_TS" in wrapper
    assert "subprocess.call(command)" in wrapper
    assert "--workflow" in observed["command"][3:]
    assert observed["env"]["TIDAL_SYNC_START_TARGET_TS"] == "1234.500000"
    assert observed["env"]["ASCEND_RT_VISIBLE_DEVICES"] == "3"


def test_main_dry_run_writes_worker_specs_and_summary(tmp_path, monkeypatch):
    suite = load_suite()
    monkeypatch.setattr(suite.sys, "executable", sys.executable)

    rc = suite.main(
        [
            "--project-root",
            str(tmp_path / "project"),
            "--demo-root",
            str(tmp_path / "demo"),
            "--cards",
            "0,1",
            "--tasks",
            "workflow_cap,workflow_qpruner",
            "--run-label",
            "sync_suite",
            "--device",
            "cpu",
            "--dry-run",
        ]
    )

    summary = json.loads(
        (tmp_path / "demo" / "artifacts" / "multicard_parallel_suite_sync_suite.json").read_text()
    )
    assert rc == 0
    assert summary["status"] == "DRY_RUN"
    assert summary["world_size"] == 2
    assert summary["workers"][0]["status"] == "SKIPPED"
    assert (tmp_path / "demo" / "artifacts" / "multicard_parallel_suite_sync_suite_rank0.worker.json").exists()
