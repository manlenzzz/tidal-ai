import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch


def load_fixture_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "create_tiny_qwen_fixture.py"
    spec = importlib.util.spec_from_file_location("create_tiny_qwen_fixture", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_finetune_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "tiny_qwen_lora_finetune.py"
    spec = importlib.util.spec_from_file_location("tiny_qwen_lora_finetune", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_qwen_lora_target_filter_selects_projection_layers(tmp_path):
    fixture = load_fixture_module()
    finetune = load_finetune_module()
    model_dir = tmp_path / "TinyQwen3"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)

    model = finetune.load_model(model_dir, torch.device("cpu"), torch.float32)
    allocation = finetune.rankadaptor_qwen_allocation(model, budget=4096, max_rank=2)

    assert allocation
    assert any(name.endswith("q_proj") for name in allocation)
    assert any(name.endswith("down_proj") for name in allocation)


def test_tiny_qwen_lora_finetune_runs_offline_cpu(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)
    script = Path(__file__).resolve().parents[1] / "scripts" / "tiny_qwen_lora_finetune.py"

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
        timeout=120,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads((demo_root / "artifacts" / "tiny_qwen_lora_finetune_cpu_test.json").read_text())
    assert payload["status"] == "PASS"
    assert payload["download_attempted"] is False
    assert payload["trainable_adapter_params"] > 0
    assert payload["final_loss"] <= payload["initial_loss"] + 0.5
    assert payload["before_generate"]
    assert payload["after_generate"]
