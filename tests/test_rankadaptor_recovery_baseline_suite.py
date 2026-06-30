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


def load_suite_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "rankadaptor_recovery_baseline_suite.py"
    spec = importlib.util.spec_from_file_location("rankadaptor_recovery_baseline_suite", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_recovery_suite_summary_identifies_rankadaptor_paper_baselines():
    suite = load_suite_module()
    runs = [
        {
            "status": "PASS",
            "method": "no_recovery",
            "paper_baseline": "without recovery",
            "trainable_adapter_params": 0,
            "initial_loss": 4.0,
            "final_loss": 4.0,
            "loss_delta": 0.0,
            "peak_mem_mb": 100.0,
        },
        {
            "status": "PASS",
            "method": "lora",
            "paper_baseline": "LoRA",
            "trainable_adapter_params": 512,
            "initial_loss": 4.0,
            "final_loss": 3.8,
            "loss_delta": -0.2,
            "peak_mem_mb": 120.0,
        },
        {
            "status": "PASS",
            "method": "adalora_style",
            "paper_baseline": "AdaLoRA-style",
            "trainable_adapter_params": 384,
            "initial_loss": 4.0,
            "final_loss": 3.85,
            "loss_delta": -0.15,
            "peak_mem_mb": 118.0,
            "claim_scope": "controlled AdaLoRA-style approximation, not official AdaLoRA implementation",
        },
    ]

    summary = suite.build_summary(
        runs=runs,
        model_id="Qwen/Qwen3-0.6B",
        model_path="/models/qwen",
        device="npu:0",
        dtype="torch.float16",
        run_label="qwen3_06b_npu",
    )

    assert summary["status"] == "PASS"
    assert summary["method"] == "rankadaptor_recovery_baseline_suite"
    assert summary["paper_baselines"] == ["without recovery", "LoRA", "AdaLoRA-style"]
    assert summary["best_recovery_method"] == "lora"
    assert summary["no_recovery_loss_delta"] == 0.0
    assert summary["lora_loss_delta"] == -0.2
    assert summary["adalora_style_loss_delta"] == -0.15
    assert summary["all_methods_passed"] is True
    text = suite.markdown_report(summary)
    assert "RankAdaptor Recovery Baseline Suite" in text
    assert "without recovery" in text
    assert "AdaLoRA-style" in text
    assert "not official AdaLoRA" in text


def test_recovery_suite_runs_tiny_qwen_cpu(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)
    script = Path(__file__).resolve().parents[1] / "scripts" / "rankadaptor_recovery_baseline_suite.py"

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
            "--run-label",
            "cpu_test",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads((demo_root / "artifacts" / "rankadaptor_recovery_baseline_suite_cpu_test.json").read_text())
    assert payload["status"] == "PASS"
    assert payload["all_methods_passed"] is True
    assert payload["paper_baselines"] == ["without recovery", "LoRA", "AdaLoRA-style"]
    assert set(payload["methods"]) == {"no_recovery", "lora", "adalora_style"}
    assert payload["methods"]["no_recovery"]["trainable_adapter_params"] == 0
    assert payload["methods"]["lora"]["trainable_adapter_params"] > 0
    assert payload["methods"]["adalora_style"]["trainable_adapter_params"] > 0
    assert payload["methods"]["adalora_style"]["claim_scope"].startswith("controlled AdaLoRA-style")
    assert payload["methods"]["lora"]["final_loss"] <= payload["methods"]["lora"]["initial_loss"] + 0.5
    assert (demo_root / "reports" / "rankadaptor-recovery-baseline-suite-cpu_test.md").exists()
