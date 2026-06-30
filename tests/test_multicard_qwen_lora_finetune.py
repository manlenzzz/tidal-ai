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
    script = Path(__file__).resolve().parents[1] / "scripts" / "multicard_qwen_lora_finetune.py"
    spec = importlib.util.spec_from_file_location("multicard_qwen_lora_finetune", script)
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
        "target_module_limit": 2,
        "target_module_count": 2,
        "targeted_modules_total": 7,
        "rank_allocation": {"model.layers.0.self_attn.q_proj": 1, "model.layers.0.self_attn.k_proj": 1},
        "trainable_adapter_params": 512,
        "initial_loss": 4.4,
        "final_loss": 4.2 + rank * 0.01,
        "loss_delta": -0.2 + rank * 0.01,
        "adapter_checksum": 3.14,
        "max_adapter_delta": 0.0,
        "peak_mem_mb": 123.0 + rank,
        "train_mode": "instruction",
        "train_samples": [
            {
                "prompt": "Explain CAP compression in one sentence.",
                "target": "CAP replaces selected dense layers with low-rank factors.",
            }
        ],
        "validation_samples": [
            {
                "prompt": "What does LoRA train?",
                "target": "LoRA trains adapter matrices while freezing base weights.",
            }
        ],
        "before_generate": "Explain CAP compression in one sentence. dense layers",
        "after_generate": "Explain CAP compression in one sentence. low rank",
        "validation_initial_loss": 4.8,
        "validation_final_loss": 4.4 + rank * 0.01,
        "validation_loss_delta": -0.4 + rank * 0.01,
        "distributed_totals": {
            "initial_loss_sum": 8.8,
            "final_loss_sum": 8.41,
            "validation_initial_loss_sum": 9.6,
            "validation_final_loss_sum": 8.81,
            "peak_mem_mb_sum": 247.0,
            "pass_count": 2,
            "adapter_sync_pass_count": 2,
        },
    }


def test_write_summary_aggregates_qwen_lora_sync_metrics(tmp_path):
    bench = load_multicard_module()
    artifacts = tmp_path / "artifacts"
    reports = tmp_path / "reports"
    artifacts.mkdir()
    reports.mkdir()
    for rank in range(2):
        (artifacts / f"multicard_qwen_lora_finetune_run_rank{rank}.json").write_text(
            json.dumps(rank_payload(rank))
        )

    summary = bench.write_summary(
        artifacts_dir=artifacts,
        reports_dir=reports,
        run_label="run",
        spawn_error=None,
    )

    assert summary["status"] == "PASS"
    assert summary["method"] == "rankadaptor_lora"
    assert summary["world_size"] == 2
    assert summary["target_module_limit"] == 2
    assert summary["target_module_count"] == 2
    assert summary["targeted_modules_total"] == 7
    assert summary["aggregate"]["pass_count"] == 2
    assert summary["aggregate"]["adapter_sync_consistent"] is True
    assert summary["aggregate"]["distributed_reduce_consistent"] is True
    assert summary["aggregate"]["final_loss_avg"] == 4.205
    assert summary["aggregate"]["validation_final_loss_avg"] == 4.405
    assert summary["aggregate"]["validation_loss_delta_avg"] == -0.395
    assert summary["before_generate"] == "Explain CAP compression in one sentence. dense layers"
    assert summary["after_generate"] == "Explain CAP compression in one sentence. low rank"
    assert summary["train_mode"] == "instruction"
    assert summary["aggregate"]["peak_mem_mb_total"] == 247.0
    assert (artifacts / "multicard_qwen_lora_finetune_run.json").exists()
    text = (reports / "multicard-qwen-lora-finetune-run.md").read_text()
    assert "Multi-Card Qwen3 RankAdaptor LoRA Fine-Tune" in text
    assert "Instruction Evaluation" in text
    assert "Explain CAP compression" in text
    assert "low rank" in text
    assert "all-reduce" in text
    assert "adapter checksum synchronized" in text
    assert "| 0 | PASS | cpu |" in text


def test_cpu_gloo_multicard_qwen_lora_runs_two_processes(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)
    script = Path(__file__).resolve().parents[1] / "scripts" / "multicard_qwen_lora_finetune.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--demo-root",
            str(demo_root),
            "--model-id",
            "Qwen/Qwen3-0.6B",
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
            "--budget",
            "4096",
            "--max-rank",
            "2",
            "--target-module-limit",
            "2",
            "--train-mode",
            "instruction",
            "--max-new-tokens",
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
    summary = json.loads((demo_root / "artifacts" / "multicard_qwen_lora_finetune_cpu_test.json").read_text())
    assert summary["status"] == "PASS"
    assert summary["world_size"] == 2
    assert summary["aggregate"]["pass_count"] == 2
    assert summary["aggregate"]["adapter_sync_consistent"] is True
    assert summary["aggregate"]["distributed_reduce_consistent"] is True
    assert summary["aggregate"]["final_loss_avg"] <= summary["aggregate"]["initial_loss_avg"] + 0.5
    assert summary["target_module_count"] == 2
    assert summary["trainable_adapter_params"] > 0
    assert summary["train_mode"] == "instruction"
    assert summary["before_generate"]
    assert summary["after_generate"]
    assert summary["aggregate"]["validation_final_loss_avg"] <= summary["aggregate"]["validation_initial_loss_avg"] + 0.5
    assert all(row["max_adapter_delta"] <= 1e-5 for row in summary["ranks"])
