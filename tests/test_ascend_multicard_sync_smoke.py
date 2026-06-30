import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch


def load_sync_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "ascend_multicard_sync_smoke.py"
    spec = importlib.util.spec_from_file_location("ascend_multicard_sync_smoke", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_backend_uses_hccl_for_npu_and_gloo_for_cpu():
    sync = load_sync_module()

    assert sync.default_backend("npu") == "hccl"
    assert sync.default_backend("npu:0") == "hccl"
    assert sync.default_backend("cpu") == "gloo"


def test_parse_cards_returns_ints_and_rejects_empty_input():
    sync = load_sync_module()

    assert sync.parse_cards("0, 2,7") == [0, 2, 7]

    try:
        sync.parse_cards(" , ")
    except ValueError as exc:
        assert "at least one card" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_write_summary_marks_pass_when_all_ranks_pass(tmp_path):
    sync = load_sync_module()
    artifacts = tmp_path / "artifacts"
    reports = tmp_path / "reports"
    artifacts.mkdir()
    reports.mkdir()

    for rank in range(2):
        (artifacts / f"multicard_sync_rank{rank}.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "rank": rank,
                    "world_size": 2,
                    "device": "cpu",
                    "backend": "gloo",
                    "loss": 1.0 + rank,
                    "grad_checksum": 3.5,
                    "param_checksum": 4.5,
                    "max_param_delta": 0.0,
                }
            )
        )

    summary = sync.write_summary(artifacts_dir=artifacts, reports_dir=reports)

    assert summary["status"] == "PASS"
    assert len(summary["ranks"]) == 2
    assert (artifacts / "multicard_sync_summary.json").exists()
    md = (reports / "multicard-sync-summary.md").read_text()
    assert "Multi-Card Sync Smoke Summary" in md
    assert "| 0 | PASS |" in md


def test_apply_hccl_port_defaults_sets_auto_ranges(monkeypatch):
    sync = load_sync_module()
    monkeypatch.delenv("HCCL_HOST_SOCKET_PORT_RANGE", raising=False)
    monkeypatch.delenv("HCCL_NPU_SOCKET_PORT_RANGE", raising=False)

    sync.apply_hccl_port_defaults("npu")

    assert sync.os.environ["HCCL_HOST_SOCKET_PORT_RANGE"] == "auto"
    assert sync.os.environ["HCCL_NPU_SOCKET_PORT_RANGE"] == "auto"


def test_seeded_batch_is_created_on_cpu_then_moved_to_target_device():
    sync = load_sync_module()

    batch = sync.seeded_batch(
        batch_size=2,
        hidden_size=8,
        seed=123,
        device=torch.device("cpu"),
    )

    assert batch.shape == (2, 8)
    assert batch.device.type == "cpu"


def test_cpu_gloo_sync_smoke_runs_two_processes(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "ascend_multicard_sync_smoke.py"
    demo_root = tmp_path / "demo"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--demo-root",
            str(demo_root),
            "--device",
            "cpu",
            "--backend",
            "gloo",
            "--cards",
            "0,1",
            "--hidden-size",
            "8",
            "--batch-size",
            "2",
            "--steps",
            "1",
            "--timeout-seconds",
            "60",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=120,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads((demo_root / "artifacts" / "multicard_sync_summary.json").read_text())
    assert summary["status"] == "PASS"
    assert summary["world_size"] == 2
    assert all(rank["max_param_delta"] == 0.0 for rank in summary["ranks"])
