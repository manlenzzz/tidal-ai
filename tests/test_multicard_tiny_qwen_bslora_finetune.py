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
    script = Path(__file__).resolve().parents[1] / "scripts" / "multicard_tiny_qwen_bslora_finetune.py"
    spec = importlib.util.spec_from_file_location("multicard_tiny_qwen_bslora_finetune", script)
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
        "trainable_adapter_params": 512,
        "unique_adapter_tensors": 4,
        "target_module_count": 7,
        "shared_groups": {
            "attn_hidden_32x32": {"rank": 2, "modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
            "mlp_up_32x64": {"rank": 2, "modules": ["gate_proj", "up_proj"]},
        },
        "initial_loss": 4.4,
        "final_loss": 4.2 + rank * 0.01,
        "loss_delta": -0.2 + rank * 0.01,
        "adapter_checksum": 3.14,
        "max_adapter_delta": 0.0,
        "distributed_totals": {
            "initial_loss_sum": 8.8,
            "final_loss_sum": 8.41,
            "pass_count": 2,
            "adapter_sync_pass_count": 2,
        },
    }


def test_write_summary_aggregates_shared_lora_sync_metrics(tmp_path):
    bench = load_multicard_module()
    artifacts = tmp_path / "artifacts"
    reports = tmp_path / "reports"
    artifacts.mkdir()
    reports.mkdir()
    for rank in range(2):
        (artifacts / f"multicard_tiny_qwen_bslora_finetune_run_rank{rank}.json").write_text(
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
    assert summary["aggregate"]["pass_count"] == 2
    assert summary["aggregate"]["adapter_sync_consistent"] is True
    assert summary["aggregate"]["final_loss_avg"] == 4.205
    assert (artifacts / "multicard_tiny_qwen_bslora_finetune_run.json").exists()
    text = (reports / "multicard-tiny-qwen-bslora-finetune-run.md").read_text()
    assert "Multi-Card TinyQwen BSLoRA Fine-Tune" in text
    assert "all-reduce" in text
    assert "| 0 | PASS | cpu |" in text


def test_cpu_gloo_multicard_tiny_qwen_bslora_runs_two_processes(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)
    script = Path(__file__).resolve().parents[1] / "scripts" / "multicard_tiny_qwen_bslora_finetune.py"

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
            "--steps",
            "2",
            "--batch-size",
            "2",
            "--seq-len",
            "8",
            "--rank",
            "2",
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
    summary = json.loads(
        (demo_root / "artifacts" / "multicard_tiny_qwen_bslora_finetune_cpu_test.json").read_text()
    )
    assert summary["status"] == "PASS"
    assert summary["world_size"] == 2
    assert summary["aggregate"]["pass_count"] == 2
    assert summary["aggregate"]["adapter_sync_consistent"] is True
    assert summary["aggregate"]["final_loss_avg"] <= summary["aggregate"]["initial_loss_avg"] + 0.5
    assert all(row["max_adapter_delta"] <= 1e-5 for row in summary["ranks"])
