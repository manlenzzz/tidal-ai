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


def load_sweep_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "qwen_qpruner_quality_memory_sweep.py"
    spec = importlib.util.spec_from_file_location("qwen_qpruner_quality_memory_sweep", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_qpruner_sweep_markdown_picks_best_memory_quality_point():
    sweep = load_sweep_module()
    report = {
        "status": "PASS",
        "model_id": "Qwen/Qwen3-0.6B",
        "target_layer_pattern": "self_attn\\.(q_proj|k_proj)$",
        "points": [
            {
                "run_label": "demo_layers4_bits4",
                "status": "PASS",
                "target_layer_limit": 4,
                "targeted_layers_total": 196,
                "qpruner_average_bits_request": 4.0,
                "qpruner": {
                    "status": "PASS",
                    "average_bits": 4.0,
                    "loss_delta": 5.2872490882873535,
                    "tokens_per_s": 505.468,
                    "compression_time_s": 0.4,
                    "bitwidths": {"a": 4, "b": 4, "c": 2, "d": 8},
                },
            },
            {
                "run_label": "demo_layers4_bits6",
                "status": "PASS",
                "target_layer_limit": 4,
                "targeted_layers_total": 196,
                "qpruner_average_bits_request": 6.0,
                "qpruner": {
                    "status": "PASS",
                    "average_bits": 5.333333333333333,
                    "loss_delta": -0.22416067123413086,
                    "tokens_per_s": 526.19,
                    "compression_time_s": 0.5,
                    "bitwidths": {"a": 4, "b": 4, "c": 8, "d": 8},
                },
            },
            {
                "run_label": "demo_layers4_bits8",
                "status": "PASS",
                "target_layer_limit": 4,
                "targeted_layers_total": 196,
                "qpruner_average_bits_request": 8.0,
                "qpruner": {
                    "status": "PASS",
                    "average_bits": 8.0,
                    "loss_delta": -0.009675979614257812,
                    "tokens_per_s": 582.228,
                    "compression_time_s": 0.6,
                    "bitwidths": {"a": 8, "b": 8, "c": 8, "d": 8},
                },
            },
        ],
    }

    text = sweep.aggregate_markdown(report, max_loss_delta=0.5)

    assert "# Qwen QPruner Quality-Memory Sweep" in text
    assert "| demo_layers4_bits4 | PASS | 4 / 196 | 4.000 | 4.000 | 75.000% | 5.287 | 505.468 | 4-bit:2, 2-bit:1, 8-bit:1 |" in text
    assert "| demo_layers4_bits6 | PASS | 4 / 196 | 6.000 | 5.333 | 66.667% | -0.224 | 526.190 | 4-bit:2, 8-bit:2 |" in text
    assert "Best memory-quality point under loss delta <= 0.500 is demo_layers4_bits6" in text
    assert "4-bit saves the most memory but exceeds the quality threshold" in text


def test_qpruner_sweep_runs_offline_cpu_and_writes_aggregate_artifacts(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)
    script = Path(__file__).resolve().parents[1] / "scripts" / "qwen_qpruner_quality_memory_sweep.py"

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
            "--target-layer-limits",
            "2",
            "--target-layer-pattern",
            "self_attn\\.(q_proj|k_proj)$",
            "--qpruner-average-bits",
            "4,8",
            "--enable-inference-cache",
            "--inplace-compression",
            "--run-label",
            "cpu_sweep",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(
        (demo_root / "artifacts" / "qwen_qpruner_quality_memory_sweep_cpu_sweep.json").read_text()
    )
    assert payload["status"] == "PASS"
    assert payload["backend"] == "torch_forward"
    assert payload["model_id"] == "Qwen/Qwen3-0.6B"
    assert payload["target_layer_limits"] == [2]
    assert payload["qpruner_average_bits"] == [4.0, 8.0]
    assert len(payload["points"]) == 2
    assert {point["run_label"] for point in payload["points"]} == {
        "cpu_sweep_layers2_bits4",
        "cpu_sweep_layers2_bits8",
    }
    first = payload["points"][0]
    assert first["baseline"]["status"] == "PASS"
    assert first["baseline"]["targeted_layers"] == 2
    assert first["cap"]["status"] == "SKIPPED"
    assert first["wanda"]["status"] == "SKIPPED"
    assert first["qpruner"]["status"] == "PASS"
    assert first["qpruner"]["targeted_layers"] == 2
    assert "loss_delta" in first["qpruner"]
    assert (demo_root / "artifacts" / "qwen_compression_quality_cpu_sweep_layers2_bits4.json").exists()
    assert (demo_root / "reports" / "qwen-compression-quality-cpu_sweep_layers2_bits8.md").exists()
    report = (demo_root / "reports" / "qwen-qpruner-quality-memory-sweep-cpu_sweep.md").read_text()
    assert "Qwen QPruner Quality-Memory Sweep" in report
    assert "cpu_sweep_layers2_bits4" in report
    assert "Best memory-quality point" in report
    canonical = (demo_root / "reports" / "qwen-qpruner-quality-memory-sweep.md").read_text()
    assert canonical == report
    csv_text = (demo_root / "reports" / "qwen-qpruner-quality-memory-sweep.csv").read_text()
    assert "label,status,target_layer_limit,targeted_layers_total,requested_bits,actual_bits,target_memory_reduction_pct,loss_delta,tokens_per_s,bitwidth_mix" in csv_text
    assert "cpu_sweep_layers2_bits4,PASS,2," in csv_text
