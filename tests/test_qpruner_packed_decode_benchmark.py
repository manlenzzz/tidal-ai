import importlib.util
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tidal.methods.qpruner.torch import QuantizedLinear


def load_benchmark_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "qpruner_packed_decode_benchmark.py"
    spec = importlib.util.spec_from_file_location("qpruner_packed_decode_benchmark", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_storage_summary_counts_packed_qpruner_payload():
    module = load_benchmark_module()
    linear = torch.nn.Linear(16, 8)
    quantized = QuantizedLinear.from_linear(linear, bits=4)

    summary = module.storage_summary(quantized, linear)

    assert summary["runtime_storage_format"] == "packed_nbit_weight_codes"
    assert summary["code_count"] == 128
    assert summary["packed_weight_code_bytes"] == 64
    assert summary["quantized_payload_bytes"] < summary["dense_payload_bytes"]
    assert summary["storage_reduction_pct"] > 80.0


def test_cpu_benchmark_writes_json_and_markdown(tmp_path):
    module = load_benchmark_module()
    demo_root = tmp_path / "demo"

    exit_code = module.main(
        [
            "--demo-root",
            str(demo_root),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--in-features",
            "16",
            "--out-features",
            "8",
            "--batch-size",
            "2",
            "--bits",
            "4",
            "--iters",
            "2",
            "--warmup",
            "0",
            "--run-label",
            "cpu_decode",
        ]
    )

    assert exit_code == 0
    payload_path = demo_root / "artifacts" / "qpruner_packed_decode_benchmark_cpu_decode.json"
    report_path = demo_root / "reports" / "qpruner-packed-decode-benchmark-cpu_decode.md"
    payload = json.loads(payload_path.read_text())
    report = report_path.read_text()

    assert payload["status"] == "PASS"
    assert payload["device"] == "cpu"
    assert payload["runtime_storage_format"] == "packed_nbit_weight_codes"
    assert payload["uncached"]["runtime_strategy"] == "dequantize_per_forward"
    assert payload["code_cached"]["runtime_strategy"] == "int8_code_cache_dequantize_on_device"
    assert payload["scaled_code_matmul"]["runtime_strategy"] == "scaled_int8_code_matmul"
    assert payload["cached"]["runtime_strategy"] == "dense_weight_cache"
    assert payload["dense"]["runtime_strategy"] == "dense_dequantized_linear"
    assert payload["packed_weight_code_bytes"] > 0
    assert payload["code_cache_bytes"] == payload["code_count"]
    assert payload["code_cache_payload_bytes"] > payload["quantized_payload_bytes"]
    assert payload["code_cache_payload_bytes"] < payload["dense_payload_bytes"]
    assert payload["code_cache_storage_reduction_pct"] > 0.0
    assert payload["quantized_payload_bytes"] < payload["dense_payload_bytes"]
    assert payload["storage_reduction_pct"] > 0.0
    assert payload["code_cached"]["speedup_vs_uncached"] is not None
    assert payload["scaled_code_matmul"]["speedup_vs_uncached"] is not None
    assert payload["cached"]["speedup_vs_uncached"] is not None
    assert payload["max_abs_diff_uncached_code_cached"] <= 1e-5
    assert payload["max_abs_diff_uncached_scaled_code_matmul"] <= 1e-5
    assert payload["max_abs_diff_uncached_cached"] <= 1e-5
    assert "code-cache" in payload["next_action"]

    assert "# QPruner Packed Decode Benchmark" in report
    assert "packed_nbit_weight_codes" in report
    assert "int8_code_cache_dequantize_on_device" in report
    assert "scaled_int8_code_matmul" in report
    assert "dense_weight_cache" in report
    assert "code-cache" in report


def test_cpu_benchmark_reports_grouped_scaled_code_projection_diagnostic(tmp_path):
    module = load_benchmark_module()
    demo_root = tmp_path / "demo"

    exit_code = module.main(
        [
            "--demo-root",
            str(demo_root),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--in-features",
            "16",
            "--out-features",
            "8",
            "--batch-size",
            "2",
            "--bits",
            "4",
            "--iters",
            "2",
            "--warmup",
            "0",
            "--grouped-modules",
            "3",
            "--run-label",
            "cpu_grouped_decode",
        ]
    )

    assert exit_code == 0
    payload_path = demo_root / "artifacts" / "qpruner_packed_decode_benchmark_cpu_grouped_decode.json"
    report_path = demo_root / "reports" / "qpruner-packed-decode-benchmark-cpu_grouped_decode.md"
    payload = json.loads(payload_path.read_text())
    report = report_path.read_text()

    grouped = payload["grouped_scaled_code_matmul"]
    sequential = payload["sequential_scaled_code_matmul_group"]

    assert sequential["runtime_strategy"] == "sequential_scaled_int8_code_matmul_group"
    assert grouped["runtime_strategy"] == "grouped_scaled_int8_code_bmm"
    assert sequential["module_count"] == 3
    assert grouped["module_count"] == 3
    assert grouped["speedup_vs_sequential_scaled_code"] is not None
    assert grouped["grouped_code_cache_bytes"] == payload["code_cache_bytes"] * 3
    assert grouped["grouped_aux_cache_bytes"] == (payload["scale_bytes"] + payload["quantized_bias_bytes"]) * 3
    assert payload["grouped_max_abs_diff_vs_sequential_scaled_code"] <= 1e-5
    assert "Grouped Projection Diagnostic" in report
    assert "grouped_scaled_int8_code_bmm" in report


def test_shape_sweep_profiles_multiple_projection_shapes(tmp_path):
    module = load_benchmark_module()
    demo_root = tmp_path / "demo"

    exit_code = module.main(
        [
            "--demo-root",
            str(demo_root),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--shape-sweep",
            "attn_qkv:2x16x16,mlp_gate_up:2x16x32",
            "--bits",
            "4",
            "--iters",
            "2",
            "--warmup",
            "0",
            "--run-label",
            "cpu_shape_sweep",
        ]
    )

    assert exit_code == 0
    payload_path = demo_root / "artifacts" / "qpruner_packed_decode_benchmark_cpu_shape_sweep.json"
    report_path = demo_root / "reports" / "qpruner-packed-decode-benchmark-cpu_shape_sweep.md"
    payload = json.loads(payload_path.read_text())
    report = report_path.read_text()

    assert payload["status"] == "PASS"
    assert payload["shape_preset"] is None
    assert [row["label"] for row in payload["shape_sweep"]] == ["attn_qkv", "mlp_gate_up"]
    assert payload["shape_sweep"][0]["shape"] == {"batch_size": 2, "in_features": 16, "out_features": 16}
    assert payload["shape_sweep"][1]["shape"] == {"batch_size": 2, "in_features": 16, "out_features": 32}
    assert payload["summary"]["shape_count"] == 2
    assert payload["summary"]["best_memory_preserving"]["strategy"] in {
        "int8_code_cache_dequantize_on_device",
        "scaled_int8_code_matmul",
    }
    assert payload["summary"]["best_memory_preserving"]["label"] in {"attn_qkv", "mlp_gate_up"}
    assert payload["summary"]["best_latency"]["strategy"] in {
        "dense_weight_cache",
        "dense_dequantized_linear",
        "int8_code_cache_dequantize_on_device",
        "scaled_int8_code_matmul",
        "dequantize_per_forward",
    }
    assert "Shape Sweep" in report
    assert "attn_qkv" in report
    assert "mlp_gate_up" in report
    assert "Best memory-preserving path" in report


def test_qwen3_06b_shape_preset_matches_projection_families():
    module = load_benchmark_module()

    shapes = module.shape_sweep_specs(shape_sweep=None, shape_preset="qwen3-0.6b")

    labels = [shape.label for shape in shapes]
    assert labels == [
        "qwen3_06b_attn_qkv",
        "qwen3_06b_attn_o",
        "qwen3_06b_mlp_gate_up",
        "qwen3_06b_mlp_down",
    ]
    assert shapes[0].batch_size == 8
    assert shapes[0].in_features == 1024
    assert shapes[0].out_features == 1024
    assert shapes[2].out_features == 2816


def test_qwen3_06b_actual_shape_preset_matches_profiled_projection_families():
    module = load_benchmark_module()

    shapes = module.shape_sweep_specs(shape_sweep=None, shape_preset="qwen3-0.6b-actual")

    assert [(shape.label, shape.batch_size, shape.in_features, shape.out_features) for shape in shapes] == [
        ("qwen3_06b_actual_q_proj", 8, 1024, 2048),
        ("qwen3_06b_actual_k_proj", 8, 1024, 1024),
        ("qwen3_06b_actual_v_proj", 8, 1024, 1024),
        ("qwen3_06b_actual_o_proj", 8, 2048, 1024),
        ("qwen3_06b_actual_gate_proj", 8, 1024, 3072),
        ("qwen3_06b_actual_up_proj", 8, 1024, 3072),
        ("qwen3_06b_actual_down_proj", 8, 3072, 1024),
    ]
