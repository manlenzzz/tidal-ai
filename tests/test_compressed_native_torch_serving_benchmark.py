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


def load_benchmark_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "compressed_native_torch_serving_benchmark.py"
    spec = importlib.util.spec_from_file_location("compressed_native_torch_serving_benchmark", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_markdown_report_separates_compressed_native_from_dense_export():
    bench = load_benchmark_module()

    text = bench.markdown_report(
        {
            "status": "PASS",
            "backend": "torch_generate_compressed_native",
            "model_id": "TinyQwen3-Offline",
            "device": "npu",
            "dtype": "torch.float16",
            "serving_dense_export": False,
            "inference_cache_enabled": False,
            "summary": {
                "best_method": "qpruner",
                "best_tokens_per_s": 140.0,
                "cap_vs_baseline_speedup": 1.1,
                "qpruner_vs_baseline_speedup": 1.4,
                "dense_export_erases_storage_savings": True,
            },
            "memory_reference": {
                "baseline_targeted_params": 4096,
                "target_layer_coverage_pct": 100.0,
                "cap_targeted_param_reduction_pct": 66.667,
                "qpruner_targeted_param_reduction_pct": 75.0,
                "baseline_targeted_storage_bytes": 8192,
                "cap_targeted_storage_bytes": 2731,
                "qpruner_targeted_storage_bytes": 2048,
                "cap_targeted_storage_reduction_pct": 66.663,
                "qpruner_targeted_storage_reduction_pct": 75.0,
                "storage_measurement": "compressed_payload_model_storage_bytes",
            },
            "baseline": {
                "status": "PASS",
                "latency_ms": 20.0,
                "tokens_per_s": 100.0,
                "peak_mem_mb": 40.0,
                "targeted_layers": 7,
                "targeted_storage_bytes": 8192,
            },
            "cap": {
                "status": "PASS",
                "latency_ms": 18.0,
                "tokens_per_s": 111.0,
                "peak_mem_mb": 39.0,
                "runtime_storage_format": "coordinate_sparse_residual",
                "runtime_strategy": "coordinate_sparse_residual",
                "dense_sparse_buffers": 0,
                "sparse_entries": 128,
                "packed_layers": 7,
                "targeted_compression_ratio": 3.0,
                "targeted_param_reduction_pct": 66.667,
                "targeted_storage_bytes": 2731,
                "targeted_storage_reduction_pct": 66.663,
                "cache_modules": 0,
                "exported_dense_linears": 0,
            },
            "qpruner": {
                "status": "PASS",
                "latency_ms": 14.0,
                "tokens_per_s": 140.0,
                "peak_mem_mb": 38.0,
                "runtime_storage_format": "packed_nbit_weight_codes",
                "runtime_strategy": "dequantize_per_forward",
                "quantized_layers": 7,
                "average_bits": 4.0,
                "targeted_param_reduction_pct": 75.0,
                "targeted_storage_bytes": 2048,
                "targeted_storage_reduction_pct": 75.0,
                "cache_modules": 0,
                "exported_dense_linears": 0,
            },
        }
    )

    assert "Compressed-Native Torch Serving Benchmark" in text
    assert "keeps CAPPackedLinear/QuantizedLinear modules live" in text
    assert "| Serving dense export | False | False | False |" in text
    assert "| Exported dense linears | - | 0 | 0 |" in text
    assert "| Targeted memory reduction | - | 66.667% | 75.0% |" in text
    assert "| Targeted storage bytes | 8192 | 2731 | 2048 |" in text
    assert "| Targeted storage reduction | - | 66.663% | 75.0% |" in text
    assert "| Runtime storage | dense fp16/fp32 weights | coordinate_sparse_residual | packed_nbit_weight_codes |" in text
    assert "| Runtime strategy | dense_linear | coordinate_sparse_residual | dequantize_per_forward |" in text
    assert "| Dense sparse buffers | - | 0 | - |" in text
    assert "| Sparse entries | - | 128 | - |" in text
    assert "Native compressed-module storage" in text
    assert "compressed_payload_model_storage_bytes" in text
    assert "dense_export_erases_storage_savings" in text


def test_cap_runtime_metadata_tracks_coordinate_storage_and_cache_strategy():
    import torch

    bench = load_benchmark_module()
    cap = bench.CAPPackedLinear(
        torch.zeros((3, 0)),
        torch.zeros((0, 4)),
        torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 3.0],
            ]
        ),
        None,
        parameter_count=3,
    )

    metadata = bench.cap_runtime_metadata(torch.nn.Sequential(cap))

    assert metadata["runtime_storage_format"] == "coordinate_sparse_residual"
    assert metadata["runtime_strategy"] == "coordinate_sparse_residual"
    assert metadata["packed_layers"] == 1
    assert metadata["sparse_entries"] == 3
    assert metadata["coordinate_value_entries"] == 3
    assert metadata["dense_sparse_buffers"] == 0
    assert metadata["cached_dense_weight_modules"] == 0

    cap.enable_inference_cache(dtype=torch.float32, device=torch.device("cpu"))
    cached_metadata = bench.cap_runtime_metadata(torch.nn.Sequential(cap))

    assert cached_metadata["runtime_storage_format"] == "coordinate_sparse_residual"
    assert cached_metadata["runtime_strategy"] == "dense_weight_cache"
    assert cached_metadata["dense_sparse_buffers"] == 0
    assert cached_metadata["cached_dense_weight_modules"] == 1


def test_qpruner_runtime_metadata_tracks_quantized_storage_and_cache_strategy():
    import torch

    bench = load_benchmark_module()
    qlinear = bench.QuantizedLinear.from_linear(torch.nn.Linear(4, 3, bias=False), bits=4)

    metadata = bench.qpruner_runtime_metadata(torch.nn.Sequential(qlinear))

    assert metadata["runtime_storage_format"] == "packed_nbit_weight_codes"
    assert metadata["runtime_strategy"] == "dequantize_per_forward"
    assert metadata["quantized_layers"] == 1
    assert metadata["weight_code_entries"] == 12
    assert metadata["packed_code_bytes"] == 6
    assert metadata["cached_dense_weight_modules"] == 0

    qlinear.enable_inference_cache(dtype=torch.float32, device=torch.device("cpu"))
    cached_metadata = bench.qpruner_runtime_metadata(torch.nn.Sequential(qlinear))

    assert cached_metadata["runtime_storage_format"] == "packed_nbit_weight_codes"
    assert cached_metadata["runtime_strategy"] == "dense_weight_cache"
    assert cached_metadata["packed_code_bytes"] == 6
    assert cached_metadata["cached_dense_weight_modules"] == 1

    qlinear.clear_inference_cache()
    qlinear.enable_code_cache(device=torch.device("cpu"))
    code_cached_metadata = bench.qpruner_runtime_metadata(torch.nn.Sequential(qlinear))

    assert code_cached_metadata["runtime_storage_format"] == "packed_nbit_weight_codes"
    assert code_cached_metadata["runtime_strategy"] == "int8_code_cache_dequantize_on_device"
    assert code_cached_metadata["packed_code_bytes"] == 6
    assert code_cached_metadata["cached_dense_weight_modules"] == 0
    assert code_cached_metadata["cached_code_modules"] == 1
    assert code_cached_metadata["cached_code_bytes"] == 12
    assert code_cached_metadata["cached_dense_weight_bytes"] == 0
    assert code_cached_metadata["scaled_code_matmul_modules"] == 0

    qlinear.clear_inference_cache()
    qlinear.enable_scaled_code_matmul(device=torch.device("cpu"))
    scaled_code_metadata = bench.qpruner_runtime_metadata(torch.nn.Sequential(qlinear))

    assert scaled_code_metadata["runtime_storage_format"] == "packed_nbit_weight_codes"
    assert scaled_code_metadata["runtime_strategy"] == "scaled_int8_code_matmul"
    assert scaled_code_metadata["packed_code_bytes"] == 6
    assert scaled_code_metadata["cached_dense_weight_modules"] == 0
    assert scaled_code_metadata["cached_code_modules"] == 1
    assert scaled_code_metadata["cached_code_bytes"] == 12
    assert scaled_code_metadata["cached_scaled_code_bytes"] == 0
    assert scaled_code_metadata["cached_dense_weight_bytes"] == 0
    assert scaled_code_metadata["scaled_code_matmul_modules"] == 1

    qlinear.clear_inference_cache()
    qlinear.enable_scaled_code_matmul(device=torch.device("cpu"), dtype=torch.float32)
    prebuilt_scaled_code_metadata = bench.qpruner_runtime_metadata(torch.nn.Sequential(qlinear))

    assert prebuilt_scaled_code_metadata["runtime_storage_format"] == "packed_nbit_weight_codes"
    assert prebuilt_scaled_code_metadata["runtime_strategy"] == "scaled_int8_code_matmul"
    assert prebuilt_scaled_code_metadata["cached_code_modules"] == 1
    assert prebuilt_scaled_code_metadata["cached_code_bytes"] == 12
    assert prebuilt_scaled_code_metadata["cached_scaled_code_bytes"] == 48
    assert prebuilt_scaled_code_metadata["cached_dense_weight_bytes"] == 0
    assert prebuilt_scaled_code_metadata["scaled_code_matmul_modules"] == 1


def test_qpruner_runtime_metadata_tracks_released_packed_code_storage():
    import torch

    bench = load_benchmark_module()
    qlinear = bench.QuantizedLinear.from_linear(torch.nn.Linear(4, 3, bias=False), bits=4)
    packed_bytes = qlinear.packed_weight_codes.numel() * qlinear.packed_weight_codes.element_size()

    qlinear.enable_code_cache(device=torch.device("cpu"), release_packed_codes=True)
    metadata = bench.qpruner_runtime_metadata(torch.nn.Sequential(qlinear))

    assert packed_bytes == 6
    assert metadata["runtime_strategy"] == "int8_code_cache_dequantize_on_device"
    assert metadata["packed_code_bytes"] == 0
    assert metadata["released_packed_code_bytes"] == packed_bytes
    assert metadata["cached_code_bytes"] == 12
    assert metadata["cached_code_modules"] == 1


def test_qpruner_runtime_storage_bytes_counts_dense_cache():
    import torch

    bench = load_benchmark_module()
    qlinear = bench.QuantizedLinear.from_linear(torch.nn.Linear(4, 3, bias=False), bits=4)
    packed_bytes = qlinear.packed_weight_codes.numel() * qlinear.packed_weight_codes.element_size()

    qlinear.enable_inference_cache(dtype=torch.float32, device=torch.device("cpu"))
    metadata = bench.qpruner_runtime_metadata(torch.nn.Sequential(qlinear))

    assert metadata["cached_dense_weight_bytes"] == 48
    assert bench.qpruner_runtime_storage_bytes(packed_bytes, metadata) == packed_bytes + 48


def test_qpruner_runtime_storage_bytes_counts_aux_cache():
    import torch

    bench = load_benchmark_module()
    qlinear = bench.QuantizedLinear.from_linear(torch.nn.Linear(4, 3, bias=True), bits=4)
    packed_bytes = qlinear.packed_weight_codes.numel() * qlinear.packed_weight_codes.element_size()

    qlinear.enable_code_cache(device=torch.device("cpu"))
    qlinear.enable_aux_cache(dtype=torch.float16, device=torch.device("cpu"))
    metadata = bench.qpruner_runtime_metadata(torch.nn.Sequential(qlinear))

    assert metadata["cached_aux_modules"] == 1
    assert metadata["cached_scale_bytes"] == 2
    assert metadata["cached_bias_bytes"] == 6
    assert metadata["cached_aux_bytes"] == 8
    assert bench.qpruner_runtime_storage_bytes(packed_bytes, metadata) == packed_bytes + 12 + 8


def test_qpruner_dense_cache_budget_replaces_smallest_code_cache_modules_first():
    import torch

    bench = load_benchmark_module()
    model = torch.nn.Module()
    model.q_proj = bench.QuantizedLinear.from_linear(torch.nn.Linear(4, 4, bias=False), bits=4)
    model.down_proj = bench.QuantizedLinear.from_linear(torch.nn.Linear(16, 16, bias=False), bits=4)
    bench.enable_qpruner_shape_aware_code_cache(
        model,
        device=torch.device("cpu"),
        release_packed_codes=True,
    )

    plan = bench.enable_qpruner_dense_cache_budget(
        model,
        device=torch.device("cpu"),
        dtype=torch.float32,
        budget_bytes=16 * 16 * 4,
    )
    metadata = bench.qpruner_runtime_metadata(model)
    shape_plan = metadata["shape_aware_plan"]

    assert plan["selected_modules"] == 1
    assert plan["selection_policy"] == "smallest_dense_cache_first"
    assert plan["selected_dense_cache_bytes"] == 4 * 4 * 4
    assert plan["remaining_budget_bytes"] == (16 * 16 * 4) - (4 * 4 * 4)
    assert model.q_proj.cached_weight_bytes == 4 * 4 * 4
    assert model.q_proj.cached_code_bytes == 0
    assert model.down_proj.cached_weight_bytes == 0
    assert model.down_proj.cached_code_bytes == 256
    assert metadata["runtime_strategy"] == "shape_aware_mixed_dense_int8_code_cache"
    assert metadata["cached_dense_weight_modules"] == 1
    assert metadata["cached_code_modules"] == 1
    assert metadata["cached_dense_weight_bytes"] == 4 * 4 * 4
    assert metadata["cached_code_bytes"] == 256
    assert bench.qpruner_runtime_storage_bytes(0, metadata) == 256 + (4 * 4 * 4)
    assert shape_plan["dense_cache_budget_bytes"] == 16 * 16 * 4
    assert shape_plan["dense_cache_selection_policy"] == "smallest_dense_cache_first"
    assert shape_plan["dense_cache_selected_modules"] == 1
    assert shape_plan["strategy_counts"] == {
        "dense_weight_cache": 1,
        "scaled_int8_code_matmul": 1,
    }
    assert {
        (entry["name"], entry["strategy"], entry.get("previous_strategy"))
        for entry in shape_plan["module_shapes"]
    } == {
        ("q_proj", "dense_weight_cache", "int8_code_cache_dequantize_on_device"),
        ("down_proj", "scaled_int8_code_matmul", None),
    }


def test_qpruner_dense_cache_budget_can_prioritize_qwen_projection_hotspots():
    import torch

    bench = load_benchmark_module()
    model = torch.nn.Module()
    model.self_attn = torch.nn.Module()
    model.self_attn.k_proj = bench.QuantizedLinear.from_linear(torch.nn.Linear(4, 4, bias=False), bits=4)
    model.self_attn.v_proj = bench.QuantizedLinear.from_linear(torch.nn.Linear(4, 4, bias=False), bits=4)
    model.self_attn.q_proj = bench.QuantizedLinear.from_linear(torch.nn.Linear(8, 4, bias=False), bits=4)
    model.self_attn.o_proj = bench.QuantizedLinear.from_linear(torch.nn.Linear(8, 4, bias=False), bits=4)
    model.mlp = torch.nn.Module()
    model.mlp.down_proj = bench.QuantizedLinear.from_linear(torch.nn.Linear(12, 4, bias=False), bits=4)
    bench.enable_qpruner_shape_aware_code_cache(
        model,
        device=torch.device("cpu"),
        release_packed_codes=True,
    )

    plan = bench.enable_qpruner_dense_cache_budget(
        model,
        device=torch.device("cpu"),
        dtype=torch.float32,
        budget_bytes=2 * 4 * 8 * 4,
        selection_policy="qwen_projection_hotspot_first",
    )
    metadata = bench.qpruner_runtime_metadata(model)
    shape_plan = metadata["shape_aware_plan"]

    assert plan["selection_policy"] == "qwen_projection_hotspot_first"
    assert plan["selected_modules"] == 2
    assert [row["name"] for row in plan["selected"]] == [
        "self_attn.q_proj",
        "self_attn.o_proj",
    ]
    assert model.self_attn.q_proj.cached_weight_bytes == 4 * 8 * 4
    assert model.self_attn.o_proj.cached_weight_bytes == 4 * 8 * 4
    assert model.self_attn.k_proj.cached_weight_bytes == 0
    assert model.self_attn.v_proj.cached_weight_bytes == 0
    assert model.mlp.down_proj.cached_weight_bytes == 0
    assert shape_plan["dense_cache_selection_policy"] == "qwen_projection_hotspot_first"
    assert shape_plan["strategy_counts"]["dense_weight_cache"] == 2


def test_qpruner_scaled_code_dtype_cache_budget_prebuilds_only_budgeted_modules():
    import torch

    bench = load_benchmark_module()
    model = torch.nn.Module()
    model.q_proj = bench.QuantizedLinear.from_linear(torch.nn.Linear(4, 4, bias=False), bits=4)
    model.down_proj = bench.QuantizedLinear.from_linear(torch.nn.Linear(16, 16, bias=False), bits=4)
    bench.enable_qpruner_scaled_code_matmul(model, device=torch.device("cpu"))

    plan = bench.enable_qpruner_scaled_code_dtype_cache_budget(
        model,
        device=torch.device("cpu"),
        dtype=torch.float32,
        budget_bytes=4 * 4 * 4,
    )
    metadata = bench.qpruner_runtime_metadata(model)

    assert plan["selected_modules"] == 1
    assert plan["selection_policy"] == "smallest_scaled_code_dtype_cache_first"
    assert plan["selected_scaled_code_dtype_cache_bytes"] == 4 * 4 * 4
    assert plan["remaining_budget_bytes"] == 0
    assert [row["name"] for row in plan["selected"]] == ["q_proj"]
    assert model.q_proj.cached_code_bytes == 4 * 4
    assert model.q_proj.cached_scaled_code_bytes == 4 * 4 * 4
    assert model.down_proj.cached_code_bytes == 16 * 16
    assert model.down_proj.cached_scaled_code_bytes == 0
    assert metadata["runtime_strategy"] == "scaled_int8_code_matmul"
    assert metadata["cached_code_modules"] == 2
    assert metadata["cached_code_bytes"] == (4 * 4) + (16 * 16)
    assert metadata["cached_scaled_code_bytes"] == 4 * 4 * 4
    assert bench.qpruner_runtime_storage_bytes(0, metadata) == (4 * 4) + (16 * 16) + (4 * 4 * 4)


def test_qpruner_scaled_code_dtype_cache_budget_can_prioritize_qwen_projection_hotspots():
    import torch

    bench = load_benchmark_module()
    model = torch.nn.Module()
    model.self_attn = torch.nn.Module()
    model.self_attn.k_proj = bench.QuantizedLinear.from_linear(torch.nn.Linear(4, 4, bias=False), bits=4)
    model.self_attn.v_proj = bench.QuantizedLinear.from_linear(torch.nn.Linear(4, 4, bias=False), bits=4)
    model.self_attn.q_proj = bench.QuantizedLinear.from_linear(torch.nn.Linear(8, 4, bias=False), bits=4)
    model.self_attn.o_proj = bench.QuantizedLinear.from_linear(torch.nn.Linear(8, 4, bias=False), bits=4)
    bench.enable_qpruner_scaled_code_matmul(model, device=torch.device("cpu"))

    plan = bench.enable_qpruner_scaled_code_dtype_cache_budget(
        model,
        device=torch.device("cpu"),
        dtype=torch.float32,
        budget_bytes=2 * 4 * 8 * 4,
        selection_policy="qwen_projection_hotspot_first",
    )

    assert plan["selection_policy"] == "qwen_projection_hotspot_first"
    assert plan["selected_modules"] == 2
    assert [row["name"] for row in plan["selected"]] == [
        "self_attn.q_proj",
        "self_attn.o_proj",
    ]
    assert model.self_attn.q_proj.cached_scaled_code_bytes == 4 * 8 * 4
    assert model.self_attn.o_proj.cached_scaled_code_bytes == 4 * 8 * 4
    assert model.self_attn.k_proj.cached_scaled_code_bytes == 0
    assert model.self_attn.v_proj.cached_scaled_code_bytes == 0


def test_profile_qpruner_runtime_aggregates_quantized_forward_paths():
    import torch

    bench = load_benchmark_module()
    model = torch.nn.Sequential(
        bench.QuantizedLinear.from_linear(torch.nn.Linear(4, 3, bias=False), bits=4),
        bench.QuantizedLinear.from_linear(torch.nn.Linear(3, 2, bias=False), bits=4),
    )
    model[0].enable_code_cache(device=torch.device("cpu"))
    model[1].enable_scaled_code_matmul(device=torch.device("cpu"))
    inputs = torch.randn(5, 4)

    profile = bench.profile_qpruner_runtime(
        model,
        lambda: [model(inputs), model(inputs)],
        device=torch.device("cpu"),
        full_latency_ms=10_000.0,
    )

    assert profile["status"] == "PASS"
    assert profile["quantized_layers"] == 2
    assert profile["total_forward_calls"] == 4
    assert profile["total_forward_time_ms"] > 0
    assert profile["estimated_forward_share_pct"] > 0
    assert profile["runtime_strategy"] == "scaled_int8_code_matmul"
    assert profile["by_strategy"]["int8_code_cache_dequantize_on_device"]["modules"] == 1
    assert profile["by_strategy"]["int8_code_cache_dequantize_on_device"]["calls"] == 2
    assert profile["by_strategy"]["scaled_int8_code_matmul"]["modules"] == 1
    assert profile["by_strategy"]["scaled_int8_code_matmul"]["calls"] == 2
    assert profile["top_modules"][0]["calls"] == 2
    assert {entry["name"] for entry in profile["top_modules"]} == {"0", "1"}


def test_shape_aware_qpruner_cache_uses_scaled_code_for_large_square_layers():
    import torch

    bench = load_benchmark_module()
    model = torch.nn.Module()
    model.q_proj = bench.QuantizedLinear.from_linear(torch.nn.Linear(4, 4, bias=False), bits=4)
    model.down_proj = bench.QuantizedLinear.from_linear(torch.nn.Linear(16, 16, bias=False), bits=4)

    modules = bench.enable_qpruner_shape_aware_code_cache(model, device=torch.device("cpu"))
    metadata = bench.qpruner_runtime_metadata(model)
    plan = metadata["shape_aware_plan"]

    assert modules == 2
    assert metadata["runtime_strategy"] == "shape_aware_mixed_int8_code_cache"
    assert metadata["cached_dense_weight_modules"] == 0
    assert metadata["cached_code_modules"] == 2
    assert metadata["shape_aware_cache_modules"] == 2
    assert metadata["shape_aware_scaled_code_modules"] == 1
    assert metadata["shape_aware_dense_cache_modules"] == 0
    assert model.q_proj.scaled_code_matmul_enabled is False
    assert model.down_proj.scaled_code_matmul_enabled is True
    assert plan["policy"] == "memory_preserving_code_cache_by_projection_shape"
    assert plan["strategy_counts"] == {
        "int8_code_cache_dequantize_on_device": 1,
        "scaled_int8_code_matmul": 1,
    }
    assert {
        (entry["name"], entry["strategy"], entry["shape_score"]) for entry in plan["module_shapes"]
    } == {
        ("q_proj", "int8_code_cache_dequantize_on_device", 16),
        ("down_proj", "scaled_int8_code_matmul", 256),
    }


def test_shape_aware_qpruner_cache_uses_shape_sweep_policy_when_available(tmp_path):
    import torch

    bench = load_benchmark_module()
    policy_path = tmp_path / "qpruner_packed_decode_shape_sweep.json"
    policy_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "backend": "torch_qpruner_packed_decode_shape_sweep",
                "shape_sweep": [
                    {
                        "label": "tiny_attn",
                        "shape": {"batch_size": 1, "in_features": 4, "out_features": 4},
                        "code_cached": {
                            "runtime_strategy": "int8_code_cache_dequantize_on_device",
                            "latency_ms": 2.0,
                            "speedup_vs_uncached": 1.2,
                        },
                        "scaled_code_matmul": {
                            "runtime_strategy": "scaled_int8_code_matmul",
                            "latency_ms": 1.0,
                            "speedup_vs_uncached": 2.4,
                        },
                    },
                    {
                        "label": "tiny_mlp",
                        "shape": {"batch_size": 1, "in_features": 16, "out_features": 16},
                        "code_cached": {
                            "runtime_strategy": "int8_code_cache_dequantize_on_device",
                            "latency_ms": 1.5,
                            "speedup_vs_uncached": 2.0,
                        },
                        "scaled_code_matmul": {
                            "runtime_strategy": "scaled_int8_code_matmul",
                            "latency_ms": 3.0,
                            "speedup_vs_uncached": 1.0,
                        },
                    },
                ],
            }
        )
    )
    policy = bench.load_qpruner_shape_policy(policy_path)
    model = torch.nn.Module()
    model.q_proj = bench.QuantizedLinear.from_linear(torch.nn.Linear(4, 4, bias=False), bits=4)
    model.down_proj = bench.QuantizedLinear.from_linear(torch.nn.Linear(16, 16, bias=False), bits=4)

    modules = bench.enable_qpruner_shape_aware_code_cache(model, device=torch.device("cpu"), shape_policy=policy)
    metadata = bench.qpruner_runtime_metadata(model)
    plan = metadata["shape_aware_plan"]

    assert modules == 2
    assert metadata["runtime_strategy"] == "shape_aware_mixed_int8_code_cache"
    assert model.q_proj.scaled_code_matmul_enabled is True
    assert model.down_proj.scaled_code_matmul_enabled is False
    assert plan["shape_policy_source"] == str(policy_path)
    assert plan["shape_policy_match_count"] == 2
    assert plan["strategy_counts"] == {
        "scaled_int8_code_matmul": 1,
        "int8_code_cache_dequantize_on_device": 1,
    }
    assert {
        (entry["name"], entry["strategy"], entry["strategy_source"], entry["shape_policy_label"])
        for entry in plan["module_shapes"]
    } == {
        ("q_proj", "scaled_int8_code_matmul", "shape_sweep_artifact", "tiny_attn"),
        ("down_proj", "int8_code_cache_dequantize_on_device", "shape_sweep_artifact", "tiny_mlp"),
    }


def test_shape_policy_loader_builds_family_fallback_strategies(tmp_path):
    bench = load_benchmark_module()
    policy_path = tmp_path / "qpruner_packed_decode_shape_sweep.json"
    policy_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "backend": "torch_qpruner_packed_decode_shape_sweep",
                "shape_sweep": [
                    {
                        "label": "qwen3_06b_q_proj_b1",
                        "family": "q_proj",
                        "shape": {"batch_size": 1, "in_features": 1024, "out_features": 1024},
                        "code_cached": {
                            "runtime_strategy": "int8_code_cache_dequantize_on_device",
                            "latency_ms": 2.0,
                            "speedup_vs_uncached": 1.1,
                        },
                        "scaled_code_matmul": {
                            "runtime_strategy": "scaled_int8_code_matmul",
                            "latency_ms": 1.0,
                            "speedup_vs_uncached": 2.2,
                        },
                    },
                    {
                        "label": "qwen3_06b_down_proj_b1",
                        "family": "down_proj",
                        "shape": {"batch_size": 1, "in_features": 2816, "out_features": 1024},
                        "code_cached": {
                            "runtime_strategy": "int8_code_cache_dequantize_on_device",
                            "latency_ms": 1.5,
                            "speedup_vs_uncached": 2.0,
                        },
                        "scaled_code_matmul": {
                            "runtime_strategy": "scaled_int8_code_matmul",
                            "latency_ms": 3.0,
                            "speedup_vs_uncached": 1.0,
                        },
                    },
                ],
            }
        )
    )

    policy = bench.load_qpruner_shape_policy(policy_path)

    assert policy["family_count"] == 2
    assert policy["family_strategies"]["q_proj"]["strategy"] == "scaled_int8_code_matmul"
    assert policy["family_strategies"]["q_proj"]["strategy_source"] == "shape_sweep_family_fallback"
    assert policy["family_strategies"]["down_proj"]["strategy"] == "int8_code_cache_dequantize_on_device"
    assert policy["family_strategies"]["down_proj"]["shape_policy_label"] == "qwen3_06b_down_proj_b1"


def test_shape_aware_qpruner_cache_falls_back_to_projection_family_policy(tmp_path):
    import torch

    bench = load_benchmark_module()
    policy_path = tmp_path / "qpruner_packed_decode_shape_sweep.json"
    policy_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "backend": "torch_qpruner_packed_decode_shape_sweep",
                "shape_sweep": [
                    {
                        "label": "qwen3_06b_q_proj_b1",
                        "family": "q_proj",
                        "shape": {"batch_size": 1, "in_features": 1024, "out_features": 1024},
                        "code_cached": {
                            "runtime_strategy": "int8_code_cache_dequantize_on_device",
                            "latency_ms": 2.0,
                            "speedup_vs_uncached": 1.1,
                        },
                        "scaled_code_matmul": {
                            "runtime_strategy": "scaled_int8_code_matmul",
                            "latency_ms": 1.0,
                            "speedup_vs_uncached": 2.2,
                        },
                    },
                    {
                        "label": "qwen3_06b_down_proj_b1",
                        "family": "down_proj",
                        "shape": {"batch_size": 1, "in_features": 2816, "out_features": 1024},
                        "code_cached": {
                            "runtime_strategy": "int8_code_cache_dequantize_on_device",
                            "latency_ms": 1.5,
                            "speedup_vs_uncached": 2.0,
                        },
                        "scaled_code_matmul": {
                            "runtime_strategy": "scaled_int8_code_matmul",
                            "latency_ms": 3.0,
                            "speedup_vs_uncached": 1.0,
                        },
                    },
                ],
            }
        )
    )
    policy = bench.load_qpruner_shape_policy(policy_path)
    model = torch.nn.Module()
    model.q_proj = bench.QuantizedLinear.from_linear(torch.nn.Linear(8, 4, bias=False), bits=4)
    model.down_proj = bench.QuantizedLinear.from_linear(torch.nn.Linear(12, 8, bias=False), bits=4)

    modules = bench.enable_qpruner_shape_aware_code_cache(model, device=torch.device("cpu"), shape_policy=policy)
    metadata = bench.qpruner_runtime_metadata(model)
    plan = metadata["shape_aware_plan"]

    assert modules == 2
    assert model.q_proj.scaled_code_matmul_enabled is True
    assert model.down_proj.scaled_code_matmul_enabled is False
    assert plan["shape_policy_match_count"] == 0
    assert plan["shape_policy_family_match_count"] == 2
    assert plan["strategy_source_counts"]["shape_sweep_family_fallback"] == 2
    assert {
        (
            entry["name"],
            entry["strategy"],
            entry["strategy_source"],
            entry["shape_policy_family"],
            entry["shape_policy_label"],
        )
        for entry in plan["module_shapes"]
    } == {
        (
            "q_proj",
            "scaled_int8_code_matmul",
            "shape_sweep_family_fallback",
            "q_proj",
            "qwen3_06b_q_proj_b1",
        ),
        (
            "down_proj",
            "int8_code_cache_dequantize_on_device",
            "shape_sweep_family_fallback",
            "down_proj",
            "qwen3_06b_down_proj_b1",
        ),
    }


def test_compressed_payload_storage_bytes_count_backend_tensors():
    import torch

    bench = load_benchmark_module()
    cap = bench.CAPPackedLinear(
        torch.ones((3, 1), dtype=torch.float32),
        torch.ones((1, 4), dtype=torch.float32),
        torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 3.0],
            ],
            dtype=torch.float32,
        ),
        torch.ones(3, dtype=torch.float32),
        parameter_count=10,
    )
    qlinear = bench.QuantizedLinear.from_linear(torch.nn.Linear(4, 3, bias=True), bits=4)

    cap_bytes = bench.compressed_payload_storage_bytes(torch.nn.Sequential(cap), ["0"])
    qpruner_bytes = bench.compressed_payload_storage_bytes(torch.nn.Sequential(qlinear), ["0"])

    expected_cap_bytes = (
        cap.low_rank_left.numel() * cap.low_rank_left.element_size()
        + cap.low_rank_right.numel() * cap.low_rank_right.element_size()
        + cap.sparse_indices.numel() * cap.sparse_indices.element_size()
        + cap.sparse_values.numel() * cap.sparse_values.element_size()
        + cap.bias.numel() * cap.bias.element_size()
    )
    expected_qpruner_bytes = (
        qlinear.packed_weight_codes.numel() * qlinear.packed_weight_codes.element_size()
        + qlinear.scale.numel() * qlinear.scale.element_size()
        + qlinear.bias.numel() * qlinear.bias.element_size()
    )

    assert cap_bytes == expected_cap_bytes
    assert qpruner_bytes == expected_qpruner_bytes
    assert cap_bytes != bench.bits_to_storage_bytes(cap.parameter_count, 16.0)
    assert qlinear.packed_weight_codes.numel() < qlinear.weight_codes.numel()
    assert qpruner_bytes == bench.bits_to_storage_bytes(qlinear.code_count, 4.0) + (
        qlinear.scale.numel() * qlinear.scale.element_size()
        + qlinear.bias.numel() * qlinear.bias.element_size()
    )


def test_compressed_native_torch_serving_benchmark_runs_offline_cpu(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)
    script = Path(__file__).resolve().parents[1] / "scripts" / "compressed_native_torch_serving_benchmark.py"

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
            "--max-new-tokens",
            "1",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--cap-budget",
            "4096",
            "--cap-max-iter",
            "1",
            "--cap-policy-steps",
            "1",
            "--cap-samples-per-step",
            "1",
            "--qpruner-average-bits",
            "4.0",
            "--run-label",
            "cpu_native",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads((demo_root / "artifacts" / "compressed_native_torch_serving_cpu_native.json").read_text())
    assert payload["status"] == "PASS"
    assert payload["backend"] == "torch_generate_compressed_native"
    assert payload["serving_dense_export"] is False
    assert payload["inference_cache_enabled"] is False
    assert payload["baseline"]["status"] == "PASS"
    assert payload["cap"]["status"] == "PASS"
    assert payload["cap"]["packed_layers"] > 0
    assert payload["cap"]["exported_dense_linears"] == 0
    assert payload["cap"]["runtime_storage_format"] == "coordinate_sparse_residual"
    assert payload["cap"]["runtime_strategy"] == "coordinate_sparse_residual"
    assert payload["cap"]["dense_sparse_buffers"] == 0
    assert payload["cap"]["sparse_entries"] >= 0
    assert payload["cap"]["targeted_storage_measurement"] == "compressed_payload_model_storage_bytes"
    assert payload["cap"]["targeted_storage_bytes"] == payload["cap"]["compressed_payload_storage_bytes"]
    assert payload["qpruner"]["status"] == "PASS"
    assert payload["qpruner"]["quantized_layers"] > 0
    assert payload["qpruner"]["exported_dense_linears"] == 0
    assert payload["qpruner"]["runtime_storage_format"] == "packed_nbit_weight_codes"
    assert payload["qpruner"]["runtime_strategy"] == "dequantize_per_forward"
    assert payload["qpruner"]["packed_code_bytes"] > 0
    assert payload["qpruner"]["packed_code_bytes"] < payload["qpruner"]["weight_code_entries"]
    assert payload["qpruner"]["targeted_storage_measurement"] == "compressed_payload_model_storage_bytes"
    assert payload["qpruner"]["targeted_storage_bytes"] == payload["qpruner"]["compressed_payload_storage_bytes"]
    assert payload["memory_reference"]["cap_targeted_param_reduction_pct"] is not None
    assert payload["memory_reference"]["qpruner_targeted_param_reduction_pct"] is not None
    assert (
        payload["memory_reference"]["storage_measurement"]
        == "compressed_payload_model_storage_bytes"
    )
    assert payload["memory_reference"]["baseline_targeted_storage_bytes"] > 0
    assert payload["memory_reference"]["cap_targeted_storage_bytes"] > 0
    assert payload["memory_reference"]["qpruner_targeted_storage_bytes"] > 0
    assert (
        payload["memory_reference"]["cap_targeted_storage_bytes"]
        < payload["memory_reference"]["baseline_targeted_storage_bytes"]
    )
    assert (
        payload["memory_reference"]["qpruner_targeted_storage_bytes"]
        < payload["memory_reference"]["baseline_targeted_storage_bytes"]
    )
    assert payload["memory_reference"]["cap_targeted_storage_reduction_pct"] is not None
    assert payload["memory_reference"]["qpruner_targeted_storage_reduction_pct"] is not None
    assert (
        payload["baseline"]["targeted_storage_bytes"]
        == payload["memory_reference"]["baseline_targeted_storage_bytes"]
    )
    assert payload["cap"]["targeted_storage_reduction_pct"] == payload["memory_reference"][
        "cap_targeted_storage_reduction_pct"
    ]
    assert payload["qpruner"]["targeted_storage_reduction_pct"] == payload["memory_reference"][
        "qpruner_targeted_storage_reduction_pct"
    ]
    assert payload["summary"]["cap_vs_baseline_speedup"] is not None
    assert payload["summary"]["qpruner_vs_baseline_speedup"] is not None
    report = (demo_root / "reports" / "compressed-native-torch-serving-cpu_native.md").read_text()
    assert "Compressed-Native Torch Serving Benchmark" in report
    assert "Native compressed-module storage" in report


def test_compressed_native_torch_serving_benchmark_runs_qpruner_code_cache_offline_cpu(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)
    script = Path(__file__).resolve().parents[1] / "scripts" / "compressed_native_torch_serving_benchmark.py"

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
            "--max-new-tokens",
            "1",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--cap-budget",
            "4096",
            "--cap-max-iter",
            "1",
            "--cap-policy-steps",
            "1",
            "--cap-samples-per-step",
            "1",
            "--qpruner-average-bits",
            "4.0",
            "--qpruner-cache-mode",
            "code",
            "--run-label",
            "cpu_native_code_cache",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(
        (demo_root / "artifacts" / "compressed_native_torch_serving_cpu_native_code_cache.json").read_text()
    )
    qpruner = payload["qpruner"]

    assert payload["status"] == "PASS"
    assert payload["serving_dense_export"] is False
    assert payload["inference_cache_enabled"] is False
    assert payload["qpruner_cache_mode"] == "code"
    assert qpruner["runtime_storage_format"] == "packed_nbit_weight_codes"
    assert qpruner["runtime_strategy"] == "int8_code_cache_dequantize_on_device"
    assert qpruner["cache_modules"] == qpruner["cached_code_modules"]
    assert qpruner["cached_code_modules"] > 0
    assert qpruner["cached_code_bytes"] > 0
    assert qpruner["cached_dense_weight_modules"] == 0
    assert qpruner["cached_dense_weight_bytes"] == 0
    assert qpruner["code_cache_storage_bytes"] == qpruner["compressed_payload_storage_bytes"] + qpruner["cached_code_bytes"]
    assert qpruner["code_cache_storage_reduction_pct"] is not None
    assert qpruner["code_cache_storage_reduction_pct"] > 0
    assert payload["memory_reference"]["qpruner_code_cache_storage_bytes"] == qpruner["code_cache_storage_bytes"]
    assert (
        payload["memory_reference"]["qpruner_code_cache_storage_reduction_pct"]
        == qpruner["code_cache_storage_reduction_pct"]
    )

    report = (demo_root / "reports" / "compressed-native-torch-serving-cpu_native_code_cache.md").read_text()
    assert "int8_code_cache_dequantize_on_device" in report
    assert "QPruner code-cache storage bytes" in report
    assert "QPruner code-cache storage reduction" in report


def test_compressed_native_torch_serving_benchmark_runs_shape_aware_qpruner_code_cache_offline_cpu(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)
    script = Path(__file__).resolve().parents[1] / "scripts" / "compressed_native_torch_serving_benchmark.py"

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
            "--max-new-tokens",
            "1",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--cap-budget",
            "4096",
            "--cap-max-iter",
            "1",
            "--cap-policy-steps",
            "1",
            "--cap-samples-per-step",
            "1",
            "--qpruner-average-bits",
            "4.0",
            "--qpruner-cache-mode",
            "shape-aware-code",
            "--profile-qpruner-runtime",
            "--run-label",
            "cpu_native_shape_aware_code",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(
        (demo_root / "artifacts" / "compressed_native_torch_serving_cpu_native_shape_aware_code.json").read_text()
    )
    qpruner = payload["qpruner"]

    assert payload["status"] == "PASS"
    assert payload["qpruner_cache_mode"] == "shape-aware-code"
    assert qpruner["runtime_storage_format"] == "packed_nbit_weight_codes"
    assert qpruner["runtime_strategy"] == "shape_aware_mixed_int8_code_cache"
    assert qpruner["shape_aware_cache_modules"] == qpruner["cached_code_modules"]
    assert qpruner["shape_aware_scaled_code_modules"] > 0
    assert qpruner["shape_aware_scaled_code_modules"] == qpruner["scaled_code_matmul_modules"]
    assert qpruner["shape_aware_dense_cache_modules"] == 0
    assert qpruner["cached_dense_weight_modules"] == 0
    assert qpruner["cached_dense_weight_bytes"] == 0
    assert qpruner["shape_aware_plan"]["policy"] == "memory_preserving_code_cache_by_projection_shape"
    assert qpruner["shape_aware_plan"]["module_count"] == qpruner["cached_code_modules"]
    assert qpruner["shape_aware_plan"]["strategy_counts"]["scaled_int8_code_matmul"] == qpruner[
        "shape_aware_scaled_code_modules"
    ]
    assert qpruner["shape_aware_plan"]["scaled_code_shape_score_threshold"] == 256
    assert qpruner["shape_aware_plan"]["families"]
    assert qpruner["shape_aware_plan"]["module_shapes"]
    assert qpruner["code_cache_storage_reduction_pct"] is not None
    assert qpruner["code_cache_storage_reduction_pct"] > 0
    assert qpruner["runtime_profile"]["status"] == "PASS"
    assert qpruner["runtime_profile"]["quantized_layers"] == qpruner["quantized_layers"]
    assert qpruner["runtime_profile"]["total_forward_calls"] > 0
    assert qpruner["runtime_profile"]["estimated_forward_share_pct"] > 0
    assert qpruner["runtime_profile"]["by_strategy"]
    assert qpruner["runtime_profile"]["top_modules"]

    report = (demo_root / "reports" / "compressed-native-torch-serving-cpu_native_shape_aware_code.md").read_text()
    assert "shape_aware_mixed_int8_code_cache" in report
    assert "Shape-aware QPruner policy" in report
    assert "memory_preserving_code_cache_by_projection_shape" in report
    assert "QPruner runtime profile" in report


def test_compressed_native_torch_serving_benchmark_runs_scaled_code_matmul_offline_cpu(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)
    script = Path(__file__).resolve().parents[1] / "scripts" / "compressed_native_torch_serving_benchmark.py"

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
            "--max-new-tokens",
            "1",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--cap-budget",
            "4096",
            "--cap-max-iter",
            "1",
            "--cap-policy-steps",
            "1",
            "--cap-samples-per-step",
            "1",
            "--qpruner-average-bits",
            "4.0",
            "--qpruner-cache-mode",
            "scaled-code-matmul",
            "--run-label",
            "cpu_native_scaled_code_matmul",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(
        (demo_root / "artifacts" / "compressed_native_torch_serving_cpu_native_scaled_code_matmul.json").read_text()
    )
    qpruner = payload["qpruner"]

    assert payload["status"] == "PASS"
    assert payload["qpruner_cache_mode"] == "scaled-code-matmul"
    assert qpruner["runtime_storage_format"] == "packed_nbit_weight_codes"
    assert qpruner["runtime_strategy"] == "scaled_int8_code_matmul"
    assert qpruner["cache_modules"] == qpruner["cached_code_modules"]
    assert qpruner["cached_code_modules"] > 0
    assert qpruner["scaled_code_matmul_modules"] == qpruner["cached_code_modules"]
    assert qpruner["cached_dense_weight_modules"] == 0
    assert qpruner["cached_dense_weight_bytes"] == 0
    assert qpruner["code_cache_storage_reduction_pct"] is not None
    assert qpruner["code_cache_storage_reduction_pct"] > 0

    report = (demo_root / "reports" / "compressed-native-torch-serving-cpu_native_scaled_code_matmul.md").read_text()
    assert "scaled_int8_code_matmul" in report
