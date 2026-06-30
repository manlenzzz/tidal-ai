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


def load_bslora_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "tiny_qwen_bslora_finetune.py"
    spec = importlib.util.spec_from_file_location("tiny_qwen_bslora_finetune", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_shared_lora_reuses_adapter_tensors_for_matching_projection_groups(tmp_path):
    fixture = load_fixture_module()
    bslora = load_bslora_module()
    model_dir = tmp_path / "TinyQwen3"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)
    model = bslora.load_model(model_dir, torch.device("cpu"), torch.float32)

    allocation = bslora.qwen_shared_lora_allocation(model, rank=2)
    metadata = bslora.apply_shared_lora_adapters(model, allocation)
    modules = dict(model.named_modules())
    shared_groups = [group for group in metadata["shared_groups"].values() if len(group["modules"]) >= 2]

    assert shared_groups
    assert metadata["trainable_adapter_params"] < metadata["unshared_adapter_params"]
    for group in shared_groups:
        first = modules[group["modules"][0]]
        second = modules[group["modules"][1]]
        assert first.lora_a is second.lora_a
        assert first.lora_b is second.lora_b


def test_tiny_qwen_bslora_finetune_runs_offline_cpu(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)
    script = Path(__file__).resolve().parents[1] / "scripts" / "tiny_qwen_bslora_finetune.py"

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
            "--rank",
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
    payload = json.loads((demo_root / "artifacts" / "tiny_qwen_bslora_finetune_cpu_test.json").read_text())
    assert payload["status"] == "PASS"
    assert payload["method"] == "bslora_shared_lora"
    assert payload["download_attempted"] is False
    assert payload["trainable_adapter_params"] > 0
    assert payload["trainable_adapter_params"] < payload["unshared_adapter_params"]
    assert any(len(group["modules"]) >= 2 for group in payload["shared_groups"].values())
    assert payload["final_loss"] <= payload["initial_loss"] + 0.5
    assert payload["before_generate"]
    assert payload["after_generate"]
    assert (demo_root / "reports" / "tiny-qwen-bslora-finetune-cpu_test.md").exists()
