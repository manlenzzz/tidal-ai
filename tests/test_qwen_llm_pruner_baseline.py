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


def load_llm_pruner_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "qwen_llm_pruner_baseline.py"
    assert script.exists(), f"missing LLM-Pruner baseline script: {script}"
    spec = importlib.util.spec_from_file_location("qwen_llm_pruner_baseline", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_llm_pruner_baseline_markdown_reports_memory_first_paper_baseline():
    bench = load_llm_pruner_module()

    text = bench.markdown_report(
        {
            "status": "PASS",
            "model_id": "Qwen/Qwen3-0.6B",
            "model_path": "/mnt/nvme/622/models/Qwen3-0.6B",
            "device": "npu",
            "dtype": "torch.float16",
            "paper_baseline": "LLM-Pruner",
            "baseline_variant": "llm_pruner_style_structured_channel_pruning",
            "claim_scope": "lightweight Ascend structural-pruning baseline evidence, not official full LLM-Pruner reproduction",
            "target_layer_pattern": "self_attn\\.(q_proj|k_proj)$",
            "target_layer_limit": 4,
            "targeted_layers_total": 196,
            "targeted_layer_names_sample": ["model.layers.0.self_attn.q_proj"],
            "baseline": {"status": "PASS", "loss": 8.0, "latency_ms": 20.0, "tokens_per_s": 100.0},
            "llm_pruner": {
                "status": "PASS",
                "loss": 8.25,
                "loss_delta": 0.25,
                "latency_ms": 18.0,
                "tokens_per_s": 111.111,
                "latency_speedup": 1.111,
                "structured_prune_ratio": 0.25,
                "targeted_param_reduction_pct": 25.0,
                "targeted_layers": 4,
                "compression_time_s": 0.5,
            },
            "peak_mem_mb": 1200.0,
        }
    )

    assert "Qwen LLM-Pruner Baseline" in text
    assert "LLM-Pruner-style structural channel pruning" in text
    assert "Targeted memory reduction" in text
    assert "25.000%" in text
    assert "not official full LLM-Pruner reproduction" in text
    assert "4 / 196" in text
    assert "model.layers.0.self_attn.q_proj" in text


def test_qwen_llm_pruner_baseline_runs_offline_cpu_with_layer_limit(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)
    script = Path(__file__).resolve().parents[1] / "scripts" / "qwen_llm_pruner_baseline.py"

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
            "--max-length",
            "32",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--target-layer-limit",
            "2",
            "--target-layer-pattern",
            "self_attn\\.(q_proj|k_proj)$",
            "--prune-ratio",
            "0.25",
            "--enable-inference-cache",
            "--run-label",
            "cpu_test",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads((demo_root / "artifacts" / "qwen_llm_pruner_baseline_cpu_test.json").read_text())
    assert payload["status"] == "PASS"
    assert payload["paper_baseline"] == "LLM-Pruner"
    assert payload["baseline_family"] == "LLM-Pruner"
    assert payload["baseline_variant"] == "llm_pruner_style_structured_channel_pruning"
    assert payload["claim_scope"] == "lightweight Ascend structural-pruning baseline evidence, not official full LLM-Pruner reproduction"
    assert payload["backend"] == "torch_forward"
    assert payload["target_layer_limit"] == 2
    assert payload["target_layer_pattern"] == "self_attn\\.(q_proj|k_proj)$"
    assert payload["baseline"]["status"] == "PASS"
    assert payload["llm_pruner"]["status"] == "PASS"
    assert payload["llm_pruner"]["targeted_layers"] == 2
    assert payload["llm_pruner"]["structured_prune_ratio"] == 0.25
    assert payload["llm_pruner"]["targeted_param_reduction_pct"] > 0
    assert payload["llm_pruner"]["targeted_params_retained"] < payload["llm_pruner"]["targeted_params_original"]
    assert payload["llm_pruner"]["compression_time_s"] >= 0
    assert "loss_delta" in payload["llm_pruner"]
    report = (demo_root / "reports" / "qwen-llm-pruner-baseline-cpu_test.md").read_text()
    assert "Qwen LLM-Pruner Baseline" in report
    assert "Targeted memory reduction" in report
    assert "self_attn\\.(q_proj|k_proj)$" in report
