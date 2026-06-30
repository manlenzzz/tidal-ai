import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch


def load_lora_sync_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "rankadaptor_lora_sync_smoke.py"
    spec = importlib.util.spec_from_file_location("rankadaptor_lora_sync_smoke", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_rankadaptor_allocation_targets_tiny_lora_modules():
    smoke = load_lora_sync_module()
    model = smoke.TinyLoraCausalLM(vocab_size=32, hidden_size=8)

    allocation = smoke.rankadaptor_allocation(model, budget=128, max_rank=2)

    assert allocation
    assert all(rank > 0 for rank in allocation.values())
    assert any(name.endswith("q_proj") for name in allocation)


def test_apply_lora_adapters_freezes_base_weights_and_trains_adapters():
    smoke = load_lora_sync_module()
    model = smoke.TinyLoraCausalLM(vocab_size=32, hidden_size=8)
    allocation = {"q_proj": 2, "v_proj": 1}

    adapted = smoke.apply_lora_adapters(model, allocation)

    trainable = [name for name, param in adapted.named_parameters() if param.requires_grad]
    frozen = [name for name, param in adapted.named_parameters() if not param.requires_grad]
    assert trainable
    assert all(("lora_a" in name or "lora_b" in name) for name in trainable)
    assert any("base.weight" in name for name in frozen)


def test_cpu_gloo_rankadaptor_lora_sync_smoke_runs_two_processes(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "rankadaptor_lora_sync_smoke.py"
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
            "--vocab-size",
            "32",
            "--hidden-size",
            "8",
            "--batch-size",
            "2",
            "--seq-len",
            "6",
            "--steps",
            "2",
            "--timeout-seconds",
            "60",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=120,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads((demo_root / "artifacts" / "rankadaptor_lora_sync_summary.json").read_text())
    assert summary["status"] == "PASS"
    assert summary["world_size"] == 2
    assert summary["trainable_adapter_params"] > 0
    assert all(rank["max_adapter_delta"] == 0.0 for rank in summary["ranks"])
