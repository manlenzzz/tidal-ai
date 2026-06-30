import importlib.util
import json
from pathlib import Path


def load_report_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "write_inference_bottleneck_report.py"
    assert script.exists(), f"missing diagnosis script: {script}"
    spec = importlib.util.spec_from_file_location("write_inference_bottleneck_report", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def populate_bottleneck_artifacts(demo_root: Path) -> None:
    artifacts = demo_root / "artifacts"
    write_json(
        artifacts / "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu.json",
        {
            "status": "PASS",
            "run_label": "tiny_qwen3_serving_vllm_metadata_shim_npu",
            "backend": "vllm_ascend_generate",
            "dtype": "float16",
            "max_new_tokens": 1,
            "iters": 3,
            "warmup": 1,
            "gpu_memory_utilization": 0.05,
            "max_model_len": 128,
            "metadata_shim": {"status": "INSTALLED"},
            "summary": {
                "best_method": "baseline",
                "best_tokens_per_s": 13.394,
                "cap_vs_baseline_speedup": 0.996,
                "qpruner_vs_baseline_speedup": 0.926,
            },
            "exports": {
                "baseline": {"status": "PASS", "tokens_per_s": 13.394, "latency_ms": 298.64},
                "cap": {"status": "PASS", "tokens_per_s": 13.344, "latency_ms": 299.75},
                "qpruner": {"status": "PASS", "tokens_per_s": 12.399, "latency_ms": 322.61},
            },
        },
    )
    write_json(
        artifacts / "multicard_parallel_suite_vllm_metadata_sync_8card_npu_metrics.json",
        {
            "status": "PASS",
            "run_label": "vllm_metadata_sync_8card_npu_metrics",
            "world_size": 8,
            "cards": list(range(8)),
            "start_window_s": 0.005,
            "release_lag_window_s": 0.004,
            "launched_synchronously": True,
            "aggregate": {"pass_count": 8, "best_vllm_tokens_per_s": 12.526},
            "workers": [
                {
                    "status": "PASS",
                    "rank": 0,
                    "card": 0,
                    "task": "vllm_serving_benchmark",
                    "payload_status": "PASS",
                    "metrics": {
                        "method": "qpruner",
                        "tokens_per_s": 12.526,
                        "latency_ms": 319.336,
                        "metadata_shim_status": "ALREADY_INSTALLED",
                    },
                },
                {
                    "status": "PASS",
                    "rank": 1,
                    "card": 1,
                    "task": "vllm_serving_benchmark",
                    "payload_status": "PASS",
                    "metrics": {
                        "method": "cap",
                        "tokens_per_s": 11.704,
                        "latency_ms": 341.763,
                        "metadata_shim_status": "INSTALLED",
                    },
                },
                {
                    "status": "PASS",
                    "rank": 2,
                    "card": 2,
                    "task": "vllm_serving_benchmark",
                    "payload_status": "PASS",
                    "metrics": {
                        "method": "baseline",
                        "tokens_per_s": 11.814,
                        "latency_ms": 338.58,
                        "metadata_shim_status": "ALREADY_INSTALLED",
                    },
                },
            ],
        },
    )
    write_json(
        artifacts / "torch_serving_fallback_tiny_qwen3_serving_fallback_npu.json",
        {
            "status": "PASS",
            "run_label": "tiny_qwen3_serving_fallback_npu",
            "backend": "torch_generate",
            "device": "npu:0",
            "max_new_tokens": 4,
            "summary": {
                "best_method": "qpruner",
                "best_tokens_per_s": 175.185,
                "cap_vs_baseline_speedup": 1.027,
                "qpruner_vs_baseline_speedup": 1.048,
            },
            "exports": {
                "baseline": {"status": "PASS", "tokens_per_s": 167.142},
                "cap": {"status": "PASS", "tokens_per_s": 171.681},
                "qpruner": {"status": "PASS", "tokens_per_s": 175.185},
            },
        },
    )
    write_json(
        artifacts / "qwen_compression_generate_qwen3_06b_generate_npu.json",
        {
            "status": "PASS",
            "backend": "torch_generate",
            "inference_cache_enabled": True,
            "serving_dense_export": True,
            "target_layer_limit": 2,
            "targeted_layers_total": 196,
            "max_new_tokens": 4,
            "baseline": {"latency_ms": 50.0, "tokens_per_s": 80.0},
            "cap": {
                "latency_speedup": 3.205,
                "targeted_compression_ratio": 3.0,
                "cache_modules": 2,
                "exported_dense_linears": 2,
            },
            "qpruner": {
                "latency_speedup": 3.184,
                "average_bits": 4.0,
                "cache_modules": 2,
                "exported_dense_linears": 2,
            },
        },
    )
    write_json(
        artifacts / "qwen_compression_quality_qwen3_06b_quality_npu.json",
        {
            "status": "PASS",
            "target_layer_limit": 2,
            "targeted_layers_total": 196,
            "baseline": {"loss": 6.797},
            "cap": {"loss_delta": -0.134, "targeted_compression_ratio": 3.0},
            "qpruner": {"loss_delta": 0.276, "average_bits": 4.0},
        },
    )
    write_json(
        artifacts / "qwen_qpruner_native_profile_qwen3_06b_native_profile_npu.json",
        {
            "status": "PASS",
            "backend": "torch_generate_qwen_qpruner_native_profile",
            "model_id": "Qwen/Qwen3-0.6B",
            "target_layer_limit": 8,
            "targeted_layers_total": 196,
            "qpruner_cache_mode": "code",
            "serving_dense_export": False,
            "max_new_tokens": 4,
            "baseline": {
                "status": "PASS",
                "latency_ms": 120.0,
                "tokens_per_s": 90.0,
                "targeted_layers": 8,
            },
            "qpruner": {
                "status": "PASS",
                "latency_ms": 132.0,
                "tokens_per_s": 81.8,
                "latency_speedup": 0.909,
                "targeted_layers": 8,
                "targeted_storage_reduction_pct": 74.5,
                "code_cache_storage_reduction_pct": 48.9,
                "runtime_strategy": "int8_code_cache_dequantize_on_device",
                "runtime_profile": {
                    "status": "PASS",
                    "quantized_layers": 8,
                    "total_forward_calls": 64,
                    "total_forward_time_ms": 28.4,
                    "estimated_forward_share_pct": 21.515,
                    "runtime_strategy": "int8_code_cache_dequantize_on_device",
                    "by_strategy": {
                        "int8_code_cache_dequantize_on_device": {
                            "modules": 8,
                            "calls": 64,
                            "time_ms": 28.4,
                            "avg_ms_per_call": 0.444,
                        }
                    },
                    "top_modules": [
                        {
                            "name": "model.layers.0.self_attn.q_proj",
                            "strategy": "int8_code_cache_dequantize_on_device",
                            "calls": 8,
                            "time_ms": 5.5,
                            "avg_ms_per_call": 0.688,
                        }
                    ],
                },
            },
            "summary": {"qpruner_vs_baseline_speedup": 0.909},
        },
    )
    write_json(
        artifacts / "compressed_native_torch_serving_tiny_qwen3_native_serving_scaled_code_matmul_npu.json",
        {
            "status": "PASS",
            "backend": "torch_generate_compressed_native",
            "run_label": "tiny_qwen3_native_serving_scaled_code_matmul_npu",
            "max_new_tokens": 8,
            "serving_dense_export": False,
            "qpruner_cache_mode": "scaled-code-matmul",
            "summary": {
                "best_method": "baseline",
                "best_tokens_per_s": 391.56,
                "cap_vs_baseline_speedup": 0.669,
                "qpruner_vs_baseline_speedup": 0.876,
            },
            "baseline": {"status": "PASS", "tokens_per_s": 391.56, "latency_ms": 40.862, "peak_mem_mb": 20.4},
            "cap": {
                "status": "PASS",
                "tokens_per_s": 262.138,
                "latency_ms": 61.037,
                "targeted_param_reduction_pct": 94.737,
                "targeted_storage_reduction_pct": 88.816,
                "runtime_strategy": "coordinate_sparse_residual",
            },
            "qpruner": {
                "status": "PASS",
                "tokens_per_s": 343.144,
                "latency_ms": 46.628,
                "targeted_param_reduction_pct": 76.316,
                "targeted_storage_reduction_pct": 76.307,
                "code_cache_storage_reduction_pct": 26.307,
                "runtime_strategy": "scaled_int8_code_matmul",
            },
        },
    )
    write_json(
        artifacts / "compressed_native_torch_serving_tiny_qwen3_native_serving_npu.json",
        {
            "status": "PASS",
            "backend": "torch_generate_compressed_native",
            "run_label": "tiny_qwen3_native_serving_npu",
            "max_new_tokens": 16,
            "serving_dense_export": False,
            "summary": {
                "best_method": "qpruner",
                "best_tokens_per_s": 396.218,
                "cap_vs_baseline_speedup": 1.004,
                "qpruner_vs_baseline_speedup": 1.006,
            },
            "baseline": {"status": "PASS", "tokens_per_s": 393.819, "latency_ms": 81.256, "peak_mem_mb": 20.4},
            "cap": {
                "status": "PASS",
                "tokens_per_s": 395.551,
                "latency_ms": 80.900,
                "targeted_param_reduction_pct": 94.737,
                "targeted_storage_reduction_pct": 88.816,
                "runtime_strategy": "dense_weight_cache",
            },
            "qpruner": {
                "status": "PASS",
                "tokens_per_s": 396.218,
                "latency_ms": 80.764,
                "targeted_param_reduction_pct": 76.316,
                "targeted_storage_reduction_pct": 76.307,
                "cached_dense_weight_modules": 7,
                "runtime_strategy": "dense_weight_cache",
            },
        },
    )
    write_json(
        artifacts / "compressed_native_torch_serving_tiny_qwen3_native_serving_shape_policy_npu.json",
        {
            "status": "PASS",
            "backend": "torch_generate_compressed_native",
            "run_label": "tiny_qwen3_native_serving_shape_policy_npu",
            "max_new_tokens": 8,
            "serving_dense_export": False,
            "qpruner_cache_mode": "shape-aware-code",
            "summary": {
                "best_method": "baseline",
                "best_tokens_per_s": 391.56,
                "cap_vs_baseline_speedup": 0.669,
                "qpruner_vs_baseline_speedup": 0.836,
            },
            "baseline": {"status": "PASS", "tokens_per_s": 391.56, "latency_ms": 40.862, "peak_mem_mb": 20.4},
            "cap": {
                "status": "PASS",
                "tokens_per_s": 262.138,
                "latency_ms": 61.037,
                "targeted_param_reduction_pct": 94.737,
                "targeted_storage_reduction_pct": 88.816,
                "runtime_strategy": "coordinate_sparse_residual",
            },
            "qpruner": {
                "status": "PASS",
                "tokens_per_s": 327.396,
                "latency_ms": 48.870,
                "targeted_param_reduction_pct": 76.316,
                "targeted_storage_reduction_pct": 76.307,
                "code_cache_storage_reduction_pct": 26.307,
                "cached_dense_weight_modules": 0,
                "runtime_strategy": "shape_aware_int8_code_cache",
                "shape_aware_plan": {
                    "shape_policy_source": "artifacts/qpruner_packed_decode_benchmark_tiny_shape_policy.json",
                    "shape_policy_match_count": 7,
                    "strategy_source_counts": {"shape_sweep_artifact": 7},
                },
                "runtime_profile": {
                    "status": "PASS",
                    "quantized_layers": 7,
                    "total_forward_calls": 56,
                    "total_forward_time_ms": 15.4,
                    "estimated_forward_share_pct": 31.5,
                    "runtime_strategy": "shape_aware_int8_code_cache",
                    "by_strategy": {
                        "int8_code_cache_dequantize_on_device": {
                            "modules": 7,
                            "calls": 56,
                            "time_ms": 15.4,
                            "avg_ms_per_call": 0.275,
                        }
                    },
                    "top_modules": [
                        {
                            "name": "model.layers.0.mlp.down_proj",
                            "strategy": "int8_code_cache_dequantize_on_device",
                            "calls": 8,
                            "time_ms": 3.2,
                            "avg_ms_per_call": 0.4,
                            "in_features": 16,
                            "out_features": 16,
                        }
                    ],
                },
            },
        },
    )
    write_json(
        artifacts / "qpruner_packed_decode_benchmark_tiny_qwen3_qpruner_decode_npu.json",
        {
            "status": "PASS",
            "backend": "torch_qpruner_packed_decode_microbenchmark",
            "device": "npu",
            "bits": 4,
            "shape": {"batch_size": 8, "in_features": 512, "out_features": 512},
            "storage_reduction_pct": 74.658,
            "code_cache_storage_reduction_pct": 49.707,
            "uncached": {"runtime_strategy": "dequantize_per_forward", "latency_ms": 3.402},
            "code_cached": {
                "runtime_strategy": "int8_code_cache_dequantize_on_device",
                "latency_ms": 0.167,
                "speedup_vs_uncached": 20.361,
                "peak_mem_mb": 3.1,
            },
            "scaled_code_matmul": {
                "runtime_strategy": "scaled_int8_code_matmul",
                "latency_ms": 0.201,
                "speedup_vs_uncached": 16.988,
                "peak_mem_mb": 3.1,
            },
            "cached": {"runtime_strategy": "dense_weight_cache", "latency_ms": 0.086, "speedup_vs_uncached": 39.674},
            "dense": {"runtime_strategy": "dense_dequantized_linear", "latency_ms": 0.082},
            "sequential_scaled_code_matmul_group": {
                "runtime_strategy": "sequential_scaled_int8_code_matmul_group",
                "module_count": 4,
                "latency_ms": 0.690,
                "peak_mem_mb": 3.2,
            },
            "grouped_scaled_code_matmul": {
                "runtime_strategy": "grouped_scaled_int8_code_bmm",
                "module_count": 4,
                "latency_ms": 0.240,
                "peak_mem_mb": 3.4,
                "speedup_vs_sequential_scaled_code": 2.875,
                "grouped_code_cache_bytes": 1048576,
                "grouped_aux_cache_bytes": 8192,
                "memory_strategy": "int8_code_cache_plus_aux_tensors",
            },
            "grouped_max_abs_diff_vs_sequential_scaled_code": 0.0,
            "next_action": "Fuse packed/quantized decode kernels on NPU.",
        },
    )
    write_json(
        artifacts / "qpruner_packed_decode_benchmark_qwen3_06b_shape_sweep_npu.json",
        {
            "status": "PASS",
            "backend": "torch_qpruner_packed_decode_shape_sweep",
            "device": "npu",
            "dtype": "float16",
            "bits": 4,
            "shape_preset": "qwen3-0.6b",
            "summary": {
                "shape_count": 4,
                "best_memory_preserving": {
                    "label": "qwen3_06b_attn_qkv",
                    "path": "scaled_code_matmul",
                    "strategy": "scaled_int8_code_matmul",
                    "latency_ms": 0.231,
                    "speedup_vs_uncached": 18.224,
                    "storage_reduction_pct": 74.9,
                    "code_cache_storage_reduction_pct": 49.8,
                },
                "best_latency": {
                    "label": "qwen3_06b_attn_qkv",
                    "path": "dense",
                    "strategy": "dense_dequantized_linear",
                    "latency_ms": 0.082,
                    "speedup_vs_uncached": 51.341,
                },
            },
            "shape_sweep": [
                {
                    "label": "qwen3_06b_attn_qkv",
                    "shape": {"batch_size": 8, "in_features": 1024, "out_features": 1024},
                    "storage_reduction_pct": 74.9,
                    "code_cache_storage_reduction_pct": 49.8,
                    "code_cached": {
                        "runtime_strategy": "int8_code_cache_dequantize_on_device",
                        "latency_ms": 0.255,
                        "speedup_vs_uncached": 16.5,
                    },
                    "scaled_code_matmul": {
                        "runtime_strategy": "scaled_int8_code_matmul",
                        "latency_ms": 0.231,
                        "speedup_vs_uncached": 18.224,
                    },
                    "cached": {
                        "runtime_strategy": "dense_weight_cache",
                        "latency_ms": 0.103,
                        "speedup_vs_uncached": 40.0,
                    },
                    "dense": {
                        "runtime_strategy": "dense_dequantized_linear",
                        "latency_ms": 0.082,
                        "speedup_vs_uncached": 51.341,
                    },
                }
            ],
            "next_action": "Use shape-specific QPruner strategy selection before vLLM integration.",
        },
    )


def test_report_surfaces_multicard_sync_and_vllm_bottleneck(tmp_path):
    report_module = load_report_module()
    demo_root = tmp_path / "demo"
    populate_bottleneck_artifacts(demo_root)

    report = report_module.build_report(demo_root)
    text = report_module.markdown(report)

    assert report["status"] == "ACTIONABLE"
    assert report["engineering_baseline"] == {
        "artifact_method_key": "baseline",
        "display_label": "engineering reference",
        "role": "uncompressed serving/runtime reference",
        "not_paper_baseline": True,
    }
    assert report["paper_baseline_alignment"]["qpruner"]["primary"] == ["LLM-Pruner"]
    assert {"SparseGPT", "Wanda"} <= set(report["paper_baseline_alignment"]["cap"]["pruning_baselines"])
    assert {"LoRA", "AdaLoRA", "without recovery"} <= set(
        report["paper_baseline_alignment"]["rankadaptor"]["recovery_baselines"]
    )
    assert report["vllm_parallel_sync"]["artifact"] == (
        "multicard_parallel_suite_vllm_metadata_sync_8card_npu_metrics.json"
    )
    assert report["vllm_parallel_sync"]["world_size"] == 8
    assert report["vllm_parallel_sync"]["method_bests"]["qpruner"] == 12.526
    assert report["sequential_vllm"]["speedups"]["qpruner_vs_baseline"] == 0.926
    assert report["compressed_native"]["artifact"] == (
        "compressed_native_torch_serving_tiny_qwen3_native_serving_shape_policy_npu.json"
    )
    assert report["compressed_native"]["speedups"]["qpruner_vs_baseline"] == 0.836
    assert report["compressed_native"]["qpruner_runtime_strategy"] == "shape_aware_int8_code_cache"
    assert report["compressed_native"]["qpruner_storage_reduction_pct"] == 76.307
    assert report["compressed_native"]["qpruner_shape_policy_match_count"] == 7
    assert report["compressed_native"]["qpruner_shape_strategy_source_counts"] == {"shape_sweep_artifact": 7}
    assert report["compressed_native"]["qpruner_runtime_profile"]["total_forward_calls"] == 56
    assert report["compressed_native"]["qpruner_runtime_profile"]["estimated_forward_share_pct"] == 31.5
    assert report["compressed_native_fastest"]["artifact"] == (
        "compressed_native_torch_serving_tiny_qwen3_native_serving_npu.json"
    )
    assert report["compressed_native_fastest"]["speedups"]["qpruner_vs_baseline"] == 1.006
    assert report["compressed_native_fastest"]["qpruner_runtime_strategy"] == "dense_weight_cache"
    assert report["qpruner_packed_decode"]["artifact"] == (
        "qpruner_packed_decode_benchmark_tiny_qwen3_qpruner_decode_npu.json"
    )
    assert report["qpruner_packed_decode"]["scaled_code_speedup_vs_uncached"] == 16.988
    assert report["qpruner_packed_decode"]["code_cache_storage_reduction_pct"] == 49.707
    assert report["qpruner_packed_decode"]["grouped_scaled_code_speedup_vs_sequential"] == 2.875
    assert report["qpruner_packed_decode"]["grouped_code_cache_bytes"] == 1048576
    assert report["qpruner_packed_decode"]["grouped_aux_cache_bytes"] == 8192
    assert report["qpruner_packed_decode_shape_sweep"]["artifact"] == (
        "qpruner_packed_decode_benchmark_qwen3_06b_shape_sweep_npu.json"
    )
    assert report["qpruner_packed_decode_shape_sweep"]["shape_preset"] == "qwen3-0.6b"
    assert report["qpruner_packed_decode_shape_sweep"]["shape_count"] == 4
    assert report["qpruner_packed_decode_shape_sweep"]["best_memory_preserving"]["label"] == "qwen3_06b_attn_qkv"
    assert report["qpruner_packed_decode_shape_sweep"]["best_memory_preserving"]["speedup_vs_uncached"] == 18.224
    assert report["qwen_qpruner_native_profile"]["artifact"] == (
        "qwen_qpruner_native_profile_qwen3_06b_native_profile_npu.json"
    )
    assert report["qwen_qpruner_native_profile"]["target_layer_limit"] == 8
    assert report["qwen_qpruner_native_profile"]["qpruner_runtime_profile"]["total_forward_calls"] == 64
    assert {item["id"] for item in report["diagnoses"]} >= {
        "vllm_compressed_not_faster_than_baseline",
        "short_decode_hides_steady_state",
        "serving_export_dense_or_cache_overhead",
        "torch_fallback_qpruner_beats_vllm_qpruner",
        "multi_card_sync_ready_not_speedup",
        "compressed_native_qpruner_end_to_end_not_faster",
        "qpruner_runtime_profile_forward_overhead",
        "compressed_native_fastest_uses_dense_weight_cache",
        "qpruner_packed_decode_kernel_positive",
        "qpruner_grouped_projection_kernel_positive",
        "qpruner_qwen_shape_sweep_memory_path_positive",
        "qwen_qpruner_native_profile_confirms_model_overhead",
    }
    assert "Inference Bottleneck Diagnosis" in text
    assert "Engineering reference: `baseline` means uncompressed serving/runtime reference, not a paper baseline." in text
    assert "Paper baselines: QPruner -> LLM-Pruner; CAP -> SparseGPT, Wanda; RankAdaptor -> LoRA, AdaLoRA, without recovery." in text
    assert "8-card vLLM synchronized slice PASS" in text
    assert "QPruner vLLM 12.526 tokens/s" in text
    assert "vLLM compressed path is not faster than baseline" in text
    assert "short decode hides steady-state throughput" in text
    assert "torch_npu fallback qpruner_vs_baseline=1.048x" in text
    assert "compressed-native QPruner qpruner_vs_baseline=0.836x" in text
    assert "shape_policy_matches=7" in text
    assert "QPruner runtime profile: 56 QuantizedLinear forwards" in text
    assert "top module model.layers.0.mlp.down_proj" in text
    assert "Qwen3 QPruner native profile: target layers 8/196" in text
    assert "64 QuantizedLinear forwards" in text
    assert "shape_sweep_artifact" in text
    assert "compressed-native fastest QPruner qpruner_vs_baseline=1.006x" in text
    assert "dense_weight_cache" in text
    assert "QPruner packed decode scaled-code speedup 16.988x" in text
    assert "QPruner grouped projection diagnostic speedup 2.875x" in text
    assert "grouped code-cache bytes=1048576.000" in text
    assert "QPruner Qwen3 shape sweep best memory-preserving scaled_int8_code_matmul" in text
    assert "qwen3_06b_attn_qkv" in text
    assert "scaled_int8_code_matmul" in text
    assert "Next optimization target" in text
    assert "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu.json" in text
    assert "multicard_parallel_suite_vllm_metadata_sync_8card_npu_metrics.json" in text


def test_qwen_native_profile_selector_prefers_faster_full_target_memory_native_profile(tmp_path):
    report_module = load_report_module()
    artifacts = tmp_path / "artifacts"
    faster = artifacts / "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_code_npu.json"
    slower = artifacts / "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_scaled_npu.json"
    write_json(
        faster,
        {
            "status": "PASS",
            "target_layer_limit": 196,
            "targeted_layers_total": 196,
            "max_new_tokens": 4,
            "summary": {"qpruner_vs_baseline_speedup": 0.92},
            "qpruner": {
                "status": "PASS",
                "targeted_layers": 196,
                "latency_speedup": 0.92,
                "runtime_strategy": "int8_code_cache_dequantize_on_device",
                "code_cache_storage_reduction_pct": 0.0,
                "runtime_profile": {"total_forward_calls": 784},
            },
        },
    )
    write_json(
        slower,
        {
            "status": "PASS",
            "target_layer_limit": 196,
            "targeted_layers_total": 196,
            "max_new_tokens": 4,
            "summary": {"qpruner_vs_baseline_speedup": 0.85},
            "qpruner": {
                "status": "PASS",
                "targeted_layers": 196,
                "latency_speedup": 0.85,
                "runtime_strategy": "scaled_int8_code_matmul",
                "code_cache_storage_reduction_pct": 0.0,
                "runtime_profile": {"total_forward_calls": 784},
            },
        },
    )

    name, payload = report_module.best_qwen_qpruner_native_profile_artifact(artifacts)

    assert name == faster.name
    assert payload["summary"]["qpruner_vs_baseline_speedup"] == 0.92


def test_report_tracks_memory_first_qwen_native_profile_separately(tmp_path):
    report_module = load_report_module()
    demo_root = tmp_path / "demo"
    populate_bottleneck_artifacts(demo_root)
    artifacts = demo_root / "artifacts"
    fastest = artifacts / "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_shapeaware_npu.json"
    memory_first = artifacts / (
        "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_code_releasepacked_npu.json"
    )
    write_json(
        fastest,
        {
            "status": "PASS",
            "target_layer_limit": 196,
            "targeted_layers_total": 196,
            "max_new_tokens": 4,
            "summary": {"qpruner_vs_baseline_speedup": 0.8},
            "qpruner": {
                "status": "PASS",
                "targeted_layers": 196,
                "latency_speedup": 0.8,
                "runtime_strategy": "shape_aware_mixed_int8_code_cache",
                "targeted_storage_reduction_pct": 50.0,
                "code_cache_storage_reduction_pct": -0.0,
                "runtime_profile": {"status": "PASS", "total_forward_calls": 784},
            },
        },
    )
    write_json(
        memory_first,
        {
            "status": "PASS",
            "target_layer_limit": 196,
            "targeted_layers_total": 196,
            "max_new_tokens": 4,
            "qpruner_release_packed_after_cache": True,
            "summary": {"qpruner_vs_baseline_speedup": 0.75},
            "qpruner": {
                "status": "PASS",
                "targeted_layers": 196,
                "latency_speedup": 0.75,
                "runtime_strategy": "int8_code_cache_dequantize_on_device",
                "targeted_storage_reduction_pct": 50.0,
                "code_cache_storage_reduction_pct": 50.0,
                "release_packed_after_cache": True,
                "packed_code_bytes": 0,
                "released_packed_code_bytes": 440401920,
                "live_compressed_payload_storage_bytes": 784,
                "code_cache_storage_bytes": 440402704,
                "runtime_profile": {"status": "PASS", "total_forward_calls": 784},
            },
        },
    )

    report = report_module.build_report(demo_root)
    text = report_module.markdown(report)

    assert report["qwen_qpruner_native_profile"]["artifact"] == fastest.name
    assert report["qwen_qpruner_native_memory_profile"]["artifact"] == memory_first.name
    assert report["qwen_qpruner_native_memory_profile"]["qpruner_code_cache_storage_reduction_pct"] == 50.0
    assert report["qwen_qpruner_native_memory_profile"]["qpruner_release_packed_after_cache"] is True
    assert "Qwen3 QPruner memory-first native profile" in text
    assert "code-cache storage reduction=50.000%" in text
    assert memory_first.name in text


def test_report_memory_first_qwen_native_profile_breaks_memory_ties_by_speed(tmp_path):
    report_module = load_report_module()
    demo_root = tmp_path / "demo"
    populate_bottleneck_artifacts(demo_root)
    artifacts = demo_root / "artifacts"
    faster_memory = artifacts / (
        "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_"
        "shapeaware_releasepacked_long16_npu.json"
    )
    slower_newer_memory = artifacts / (
        "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_"
        "shapepolicy_releasepacked_long16_npu.json"
    )
    base_payload = {
        "status": "PASS",
        "target_layer_limit": 196,
        "targeted_layers_total": 196,
        "max_new_tokens": 16,
        "qpruner_release_packed_after_cache": True,
    }
    write_json(
        faster_memory,
        {
            **base_payload,
            "summary": {"qpruner_vs_baseline_speedup": 0.913},
            "qpruner": {
                "status": "PASS",
                "targeted_layers": 196,
                "latency_speedup": 0.913,
                "runtime_strategy": "shape_aware_mixed_int8_code_cache",
                "targeted_storage_reduction_pct": 50.0,
                "code_cache_storage_reduction_pct": 50.0,
                "release_packed_after_cache": True,
                "released_packed_code_bytes": 440401920,
                "live_compressed_payload_storage_bytes": 784,
            },
        },
    )
    write_json(
        slower_newer_memory,
        {
            **base_payload,
            "summary": {"qpruner_vs_baseline_speedup": 0.792},
            "qpruner": {
                "status": "PASS",
                "targeted_layers": 196,
                "latency_speedup": 0.792,
                "runtime_strategy": "shape_aware_mixed_int8_code_cache",
                "targeted_storage_reduction_pct": 50.0,
                "code_cache_storage_reduction_pct": 50.0,
                "release_packed_after_cache": True,
                "released_packed_code_bytes": 440401920,
                "live_compressed_payload_storage_bytes": 784,
            },
        },
    )

    report = report_module.build_report(demo_root)

    assert report["qwen_qpruner_native_memory_profile"]["artifact"] == faster_memory.name
    assert report["qwen_qpruner_native_memory_profile"]["speedups"]["qpruner_vs_baseline"] == 0.913


def test_report_prefers_stable_qwen_native_profile_over_single_shot_speedup(tmp_path):
    report_module = load_report_module()
    demo_root = tmp_path / "demo"
    populate_bottleneck_artifacts(demo_root)
    artifacts = demo_root / "artifacts"
    stable_memory = artifacts / (
        "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_"
        "shapeaware_releasepacked_long16_npu.json"
    )
    exploratory_memory = artifacts / (
        "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_"
        "actualshapepolicy_releasepacked_long16_npu.json"
    )
    base_payload = {
        "status": "PASS",
        "target_layer_limit": 196,
        "targeted_layers_total": 196,
        "max_new_tokens": 16,
        "qpruner_release_packed_after_cache": True,
    }
    write_json(
        stable_memory,
        {
            **base_payload,
            "iters": 3,
            "warmup": 1,
            "summary": {"qpruner_vs_baseline_speedup": 0.913},
            "qpruner": {
                "status": "PASS",
                "targeted_layers": 196,
                "latency_speedup": 0.913,
                "runtime_strategy": "shape_aware_mixed_int8_code_cache",
                "targeted_storage_reduction_pct": 50.0,
                "code_cache_storage_reduction_pct": 50.0,
                "release_packed_after_cache": True,
                "released_packed_code_bytes": 440401920,
                "live_compressed_payload_storage_bytes": 784,
                "runtime_profile": {"status": "PASS", "total_forward_calls": 3136},
            },
        },
    )
    write_json(
        exploratory_memory,
        {
            **base_payload,
            "iters": 1,
            "warmup": 0,
            "summary": {"qpruner_vs_baseline_speedup": 1.259},
            "qpruner": {
                "status": "PASS",
                "targeted_layers": 196,
                "latency_speedup": 1.259,
                "runtime_strategy": "shape_aware_int8_code_cache",
                "targeted_storage_reduction_pct": 50.0,
                "code_cache_storage_reduction_pct": 50.0,
                "release_packed_after_cache": True,
                "released_packed_code_bytes": 440401920,
                "live_compressed_payload_storage_bytes": 784,
                "runtime_profile": {"status": "PASS", "total_forward_calls": 3136},
            },
        },
    )

    report = report_module.build_report(demo_root)
    text = report_module.markdown(report)

    assert report["qwen_qpruner_native_profile"]["artifact"] == stable_memory.name
    assert report["qwen_qpruner_native_profile"]["iters"] == 3
    assert report["qwen_qpruner_native_profile"]["warmup"] == 1
    assert report["qwen_qpruner_native_profile"]["speedups"]["qpruner_vs_baseline"] == 0.913
    assert report["qwen_qpruner_native_memory_profile"]["artifact"] == stable_memory.name
    assert "iters=3; warmup=1" in text


def test_report_prefers_paired_qwen_native_profile_over_sequential_stable_speedup(tmp_path):
    report_module = load_report_module()
    demo_root = tmp_path / "demo"
    populate_bottleneck_artifacts(demo_root)
    artifacts = demo_root / "artifacts"
    sequential_stable = artifacts / (
        "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_"
        "shapeaware_releasepacked_stable_long16_npu.json"
    )
    paired_stable = artifacts / (
        "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_"
        "shapeaware_releasepacked_paired3_long16_npu.json"
    )
    base_payload = {
        "status": "PASS",
        "target_layer_limit": 196,
        "targeted_layers_total": 196,
        "max_new_tokens": 16,
        "qpruner_release_packed_after_cache": True,
    }
    write_json(
        sequential_stable,
        {
            **base_payload,
            "iters": 3,
            "warmup": 1,
            "summary": {
                "qpruner_vs_baseline_speedup": 0.913,
                "measurement_basis": "single_sequential_run",
            },
            "qpruner": {
                "status": "PASS",
                "targeted_layers": 196,
                "latency_speedup": 0.913,
                "runtime_strategy": "shape_aware_mixed_int8_code_cache",
                "targeted_storage_reduction_pct": 50.0,
                "code_cache_storage_reduction_pct": 50.0,
                "release_packed_after_cache": True,
                "released_packed_code_bytes": 440401920,
                "live_compressed_payload_storage_bytes": 784,
                "runtime_profile": {"status": "PASS", "total_forward_calls": 3136},
            },
        },
    )
    write_json(
        paired_stable,
        {
            **base_payload,
            "iters": 1,
            "warmup": 0,
            "paired_rounds": 3,
            "paired_summary": {
                "rounds": 3,
                "samples": 6,
                "qpruner_vs_baseline_latency_median_speedup": 0.797,
                "qpruner_vs_baseline_tokens_median_ratio": 0.797,
            },
            "summary": {
                "qpruner_vs_baseline_speedup": 0.797,
                "qpruner_tokens_per_s_ratio": 0.797,
                "measurement_basis": "paired_interleaved_median",
            },
            "qpruner": {
                "status": "PASS",
                "targeted_layers": 196,
                "latency_speedup": 0.797,
                "runtime_strategy": "shape_aware_mixed_int8_code_cache",
                "targeted_storage_reduction_pct": 50.0,
                "code_cache_storage_reduction_pct": 50.0,
                "release_packed_after_cache": True,
                "released_packed_code_bytes": 440401920,
                "live_compressed_payload_storage_bytes": 784,
                "runtime_profile": {"status": "PASS", "total_forward_calls": 3136},
            },
        },
    )

    report = report_module.build_report(demo_root)
    text = report_module.markdown(report)

    assert report["qwen_qpruner_native_profile"]["artifact"] == paired_stable.name
    assert report["qwen_qpruner_native_profile"]["paired_rounds"] == 3
    assert report["qwen_qpruner_native_profile"]["measurement_basis"] == "paired_interleaved_median"
    assert report["qwen_qpruner_native_profile"]["speedups"]["qpruner_vs_baseline"] == 0.797
    assert report["qwen_qpruner_native_memory_profile"]["artifact"] == paired_stable.name
    assert "paired_rounds=3" in text
    assert "measurement=paired_interleaved_median" in text


def test_report_surfaces_qwen_native_long_decode_context(tmp_path):
    report_module = load_report_module()
    demo_root = tmp_path / "demo"
    populate_bottleneck_artifacts(demo_root)
    artifacts = demo_root / "artifacts"
    long_decode = artifacts / (
        "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_"
        "shapeaware_releasepacked_long16_npu.json"
    )
    write_json(
        long_decode,
        {
            "status": "PASS",
            "target_layer_limit": 196,
            "targeted_layers_total": 196,
            "max_new_tokens": 16,
            "qpruner_release_packed_after_cache": True,
            "summary": {"qpruner_vs_baseline_speedup": 0.913},
            "qpruner": {
                "status": "PASS",
                "targeted_layers": 196,
                "latency_speedup": 0.913,
                "runtime_strategy": "shape_aware_mixed_int8_code_cache",
                "targeted_storage_reduction_pct": 50.0,
                "code_cache_storage_reduction_pct": 50.0,
                "release_packed_after_cache": True,
                "packed_code_bytes": 0,
                "released_packed_code_bytes": 440401920,
                "live_compressed_payload_storage_bytes": 784,
                "code_cache_storage_bytes": 440402704,
                "runtime_profile": {
                    "status": "PASS",
                    "total_forward_calls": 3136,
                    "total_forward_time_ms": 660.866,
                    "estimated_forward_share_pct": 48.549,
                    "runtime_strategy": "shape_aware_mixed_int8_code_cache",
                    "top_modules": [
                        {
                            "name": "model.layers.0.mlp.down_proj",
                            "strategy": "scaled_int8_code_matmul",
                            "calls": 16,
                            "time_ms": 8.5,
                        }
                    ],
                },
            },
        },
    )

    report = report_module.build_report(demo_root)
    text = report_module.markdown(report)

    assert report["qwen_qpruner_native_profile"]["artifact"] == long_decode.name
    assert report["qwen_qpruner_native_profile"]["max_new_tokens"] == 16
    assert report["qwen_qpruner_native_profile"]["speedups"]["qpruner_vs_baseline"] == 0.913
    assert report["qwen_qpruner_native_memory_profile"]["artifact"] == long_decode.name
    assert report["qwen_qpruner_native_memory_profile"]["max_new_tokens"] == 16
    qwen_diagnosis = next(
        item for item in report["diagnoses"] if item["id"] == "qwen_qpruner_native_profile_confirms_model_overhead"
    )
    memory_diagnosis = next(
        item for item in report["diagnoses"] if item["id"] == "qwen_qpruner_native_release_packed_memory_positive"
    )
    assert "max_new_tokens=16" in qwen_diagnosis["evidence"]
    assert "max_new_tokens=16" in memory_diagnosis["evidence"]
    assert "Qwen3 QPruner native profile" in text
    assert "qpruner_vs_baseline=0.913x" in text
    assert "max_new_tokens=16" in text
    assert "3136 QuantizedLinear forwards" in text
    assert long_decode.name in text


def test_report_surfaces_qwen_native_speed_first_profile_separately_from_memory_first(tmp_path):
    report_module = load_report_module()
    demo_root = tmp_path / "demo"
    populate_bottleneck_artifacts(demo_root)
    artifacts = demo_root / "artifacts"
    memory_first = artifacts / (
        "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_"
        "shapeaware_releasepacked_long16_npu.json"
    )
    speed_first = artifacts / (
        "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_"
        "scaled_prebuilt_long16_npu.json"
    )
    write_json(
        memory_first,
        {
            "status": "PASS",
            "target_layer_limit": 196,
            "targeted_layers_total": 196,
            "max_new_tokens": 16,
            "iters": 3,
            "warmup": 1,
            "qpruner_release_packed_after_cache": True,
            "summary": {"qpruner_vs_baseline_speedup": 0.913},
            "qpruner": {
                "status": "PASS",
                "targeted_layers": 196,
                "latency_speedup": 0.913,
                "runtime_strategy": "shape_aware_mixed_int8_code_cache",
                "targeted_storage_reduction_pct": 50.0,
                "code_cache_storage_reduction_pct": 50.0,
                "release_packed_after_cache": True,
                "packed_code_bytes": 0,
                "released_packed_code_bytes": 440401920,
                "live_compressed_payload_storage_bytes": 784,
                "code_cache_storage_bytes": 440402704,
                "runtime_profile": {"status": "PASS", "total_forward_calls": 3136},
            },
        },
    )
    write_json(
        speed_first,
        {
            "status": "PASS",
            "target_layer_limit": 196,
            "targeted_layers_total": 196,
            "max_new_tokens": 16,
            "iters": 1,
            "warmup": 0,
            "qpruner_release_packed_after_cache": True,
            "qpruner_prebuild_scaled_code_dtype_cache": True,
            "summary": {"qpruner_vs_baseline_speedup": 1.392},
            "baseline": {"tokens_per_s": 17.523, "latency_ms": 1826.0},
            "qpruner": {
                "status": "PASS",
                "targeted_layers": 196,
                "tokens_per_s": 24.4,
                "latency_speedup": 1.392,
                "runtime_strategy": "scaled_int8_code_matmul",
                "targeted_storage_reduction_pct": 50.0,
                "code_cache_storage_reduction_pct": -50.0,
                "cached_scaled_code_bytes": 880803840,
                "release_packed_after_cache": True,
                "packed_code_bytes": 0,
                "released_packed_code_bytes": 440401920,
                "live_compressed_payload_storage_bytes": 784,
                "code_cache_storage_bytes": 1321206544,
                "runtime_profile": {
                    "status": "PASS",
                    "total_forward_calls": 3136,
                    "total_forward_time_ms": 624.598,
                    "estimated_forward_share_pct": 47.625,
                },
            },
        },
    )

    report = report_module.build_report(demo_root)
    text = report_module.markdown(report)

    assert report["qwen_qpruner_native_profile"]["artifact"] == memory_first.name
    assert report["qwen_qpruner_native_memory_profile"]["artifact"] == memory_first.name
    assert report["qwen_qpruner_native_speed_first_profile"]["artifact"] == speed_first.name
    assert report["qwen_qpruner_native_speed_first_profile"]["speedups"]["qpruner_vs_baseline"] == 1.392
    assert report["qwen_qpruner_native_speed_first_profile"]["qpruner_code_cache_storage_reduction_pct"] == -50.0
    assert report["qwen_qpruner_native_speed_first_profile"]["qpruner_cached_scaled_code_bytes"] == 880803840
    assert report["qwen_qpruner_native_speed_first_profile"]["qpruner_prebuild_scaled_code_dtype_cache"] is True
    assert "Qwen3 QPruner speed-first native profile" in text
    assert "qpruner_vs_baseline=1.392x" in text
    assert "code-cache storage reduction=-50.000%" in text
    assert "scaled-code dtype-cache bytes=880803840.000" in text


def test_report_surfaces_qwen_native_aux_cache_bytes(tmp_path):
    report_module = load_report_module()
    demo_root = tmp_path / "demo"
    populate_bottleneck_artifacts(demo_root)
    artifacts = demo_root / "artifacts"
    aux_cache = artifacts / (
        "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_"
        "shapeaware_releasepacked_auxcache_paired3_long16_npu.json"
    )
    write_json(
        aux_cache,
        {
            "status": "PASS",
            "target_layer_limit": 196,
            "targeted_layers_total": 196,
            "max_new_tokens": 16,
            "paired_rounds": 3,
            "qpruner_release_packed_after_cache": True,
            "qpruner_cache_aux_tensors": True,
            "paired_summary": {"rounds": 3, "samples": 6},
            "summary": {
                "qpruner_vs_baseline_speedup": 0.84,
                "measurement_basis": "paired_interleaved_median",
            },
            "qpruner": {
                "status": "PASS",
                "targeted_layers": 196,
                "latency_speedup": 0.84,
                "runtime_strategy": "shape_aware_mixed_int8_code_cache",
                "targeted_storage_reduction_pct": 50.0,
                "code_cache_storage_reduction_pct": 50.0,
                "release_packed_after_cache": True,
                "cached_aux_modules": 196,
                "cached_aux_bytes": 392,
                "cached_scale_bytes": 392,
                "cached_bias_bytes": 0,
                "packed_code_bytes": 0,
                "released_packed_code_bytes": 440401920,
                "live_compressed_payload_storage_bytes": 784,
                "code_cache_storage_bytes": 440403096,
                "runtime_profile": {"status": "PASS", "total_forward_calls": 3136},
            },
        },
    )

    report = report_module.build_report(demo_root)
    text = report_module.markdown(report)

    assert report["qwen_qpruner_native_profile"]["artifact"] == aux_cache.name
    assert report["qwen_qpruner_native_profile"]["qpruner_cache_aux_tensors"] is True
    assert report["qwen_qpruner_native_profile"]["qpruner_cached_aux_modules"] == 196
    assert report["qwen_qpruner_native_profile"]["qpruner_cached_aux_bytes"] == 392
    assert report["qwen_qpruner_native_memory_profile"]["artifact"] == aux_cache.name
    assert "aux-cache bytes=392.000" in text
    assert "aux-cache modules=196.000" in text


def test_report_selects_qwen_native_dense_budget_as_speed_first_tradeoff(tmp_path):
    report_module = load_report_module()
    demo_root = tmp_path / "demo"
    artifacts = demo_root / "artifacts"
    artifacts.mkdir(parents=True)
    memory_first = artifacts / (
        "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_"
        "shapeaware_releasepacked_long16_npu.json"
    )
    dense_budget = artifacts / (
        "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_"
        "densebudget64mb_long16_npu.json"
    )
    write_json(
        memory_first,
        {
            "status": "PASS",
            "target_layer_limit": 196,
            "targeted_layers_total": 196,
            "max_new_tokens": 16,
            "iters": 3,
            "warmup": 1,
            "qpruner_release_packed_after_cache": True,
            "summary": {"qpruner_vs_baseline_speedup": 0.913},
            "qpruner": {
                "status": "PASS",
                "targeted_layers": 196,
                "latency_speedup": 0.913,
                "runtime_strategy": "shape_aware_int8_code_cache",
                "targeted_storage_reduction_pct": 50.0,
                "code_cache_storage_reduction_pct": 50.0,
                "release_packed_after_cache": True,
                "packed_code_bytes": 0,
                "released_packed_code_bytes": 440401920,
                "live_compressed_payload_storage_bytes": 784,
                "code_cache_storage_bytes": 440402704,
            },
        },
    )
    write_json(
        dense_budget,
        {
            "status": "PASS",
            "target_layer_limit": 196,
            "targeted_layers_total": 196,
            "max_new_tokens": 16,
            "iters": 1,
            "warmup": 0,
            "qpruner_release_packed_after_cache": True,
            "qpruner_dense_cache_budget_bytes": 67108864,
            "summary": {"qpruner_vs_baseline_speedup": 1.05},
            "qpruner": {
                "status": "PASS",
                "targeted_layers": 196,
                "latency_speedup": 1.05,
                "runtime_strategy": "shape_aware_mixed_dense_int8_code_cache",
                "targeted_storage_reduction_pct": 50.0,
                "code_cache_storage_reduction_pct": 42.381,
                "cached_dense_weight_bytes": 67108864,
                "cached_scaled_code_bytes": 0,
                "release_packed_after_cache": True,
                "packed_code_bytes": 0,
                "released_packed_code_bytes": 440401920,
                "live_compressed_payload_storage_bytes": 784,
                "code_cache_storage_bytes": 507511568,
                "dense_cache_budget_plan": {
                    "budget_bytes": 67108864,
                    "selected_modules": 32,
                    "selected_dense_cache_bytes": 67108864,
                    "remaining_budget_bytes": 0,
                },
            },
        },
    )

    report = report_module.build_report(demo_root)

    assert report["qwen_qpruner_native_memory_profile"]["artifact"] == memory_first.name
    assert report["qwen_qpruner_native_speed_first_profile"]["artifact"] == dense_budget.name
    assert report["qwen_qpruner_native_speed_first_profile"]["qpruner_dense_cache_budget_bytes"] == 67108864
    assert report["qwen_qpruner_native_speed_first_profile"]["qpruner_cached_dense_weight_bytes"] == 67108864


def test_report_lists_qwen_native_bits_memory_speed_tradeoff(tmp_path):
    report_module = load_report_module()
    demo_root = tmp_path / "demo"
    populate_bottleneck_artifacts(demo_root)
    artifacts = demo_root / "artifacts"
    base_payload = {
        "status": "PASS",
        "target_layer_limit": 196,
        "targeted_layers_total": 196,
        "max_new_tokens": 16,
        "iters": 3,
        "warmup": 1,
        "qpruner_release_packed_after_cache": True,
    }
    rows = [
        (4, 3.99, 0.784, 75.059, 50.0, 219676672),
        (6, 6.0, 0.803, 62.5, 50.0, 330301440),
        (8, 8.0, 0.913, 50.0, 50.0, 440401920),
    ]
    for bits, avg_bits, speedup, storage, code_storage, released in rows:
        name = (
            "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_"
            f"bits{bits}_code_releasepacked_long16_npu.json"
        )
        write_json(
            artifacts / name,
            {
                **base_payload,
                "qpruner": {
                    "status": "PASS",
                    "average_bits": avg_bits,
                    "targeted_layers": 196,
                    "latency_speedup": speedup,
                    "runtime_strategy": "int8_code_cache_dequantize_on_device",
                    "targeted_storage_reduction_pct": storage,
                    "code_cache_storage_reduction_pct": code_storage,
                    "release_packed_after_cache": True,
                    "released_packed_code_bytes": released,
                    "live_compressed_payload_storage_bytes": 784,
                    "runtime_profile": {"status": "PASS", "total_forward_calls": 3136},
                },
                "summary": {"qpruner_vs_baseline_speedup": speedup},
            },
        )

    report = report_module.build_report(demo_root)
    text = report_module.markdown(report)
    tradeoff = report["qwen_qpruner_native_bits_tradeoff"]

    assert [row["bits_label"] for row in tradeoff] == ["bits4", "bits6", "bits8"]
    assert tradeoff[0]["qpruner_storage_reduction_pct"] == 75.059
    assert tradeoff[0]["qpruner_code_cache_storage_reduction_pct"] == 50.0
    assert tradeoff[2]["speedups"]["qpruner_vs_baseline"] == 0.913
    assert "Qwen3 QPruner native bits tradeoff" in text
    assert "bits4" in text
    assert "storage reduction=75.059%" in text
    assert "code-cache storage reduction=50.000%" in text


def test_report_prefers_latest_qpruner_shape_sweep_artifact(tmp_path):
    report_module = load_report_module()
    demo_root = tmp_path / "demo"
    populate_bottleneck_artifacts(demo_root)
    artifacts = demo_root / "artifacts"
    latest_artifact = "qpruner_packed_decode_benchmark_qwen3_06b_shape_sweep_dtype_preserving_npu.json"
    write_json(
        artifacts / latest_artifact,
        {
            "status": "PASS",
            "backend": "torch_qpruner_packed_decode_shape_sweep",
            "device": "npu",
            "dtype": "float16",
            "bits": 4,
            "shape_preset": "qwen3-0.6b",
            "summary": {
                "shape_count": 4,
                "best_memory_preserving": {
                    "label": "qwen3_06b_mlp_gate_up",
                    "path": "code_cached",
                    "strategy": "int8_code_cache_dequantize_on_device",
                    "latency_ms": 0.181678,
                    "speedup_vs_uncached": 208.597,
                    "storage_reduction_pct": 74.829,
                    "code_cache_storage_reduction_pct": 49.854,
                },
                "best_latency": {
                    "label": "qwen3_06b_mlp_gate_up",
                    "path": "dense",
                    "strategy": "dense_dequantized_linear",
                    "latency_ms": 0.04586,
                    "speedup_vs_uncached": 826.372,
                },
            },
            "next_action": "Keep code-cache as the memory-preserving path; fused kernels remain the speed route.",
        },
    )

    report = report_module.build_report(demo_root)
    text = report_module.markdown(report)

    shape_sweep = report["qpruner_packed_decode_shape_sweep"]
    assert shape_sweep["artifact"] == latest_artifact
    assert shape_sweep["best_memory_preserving"]["path"] == "code_cached"
    assert shape_sweep["best_memory_preserving"]["strategy"] == "int8_code_cache_dequantize_on_device"
    assert shape_sweep["best_memory_preserving"]["label"] == "qwen3_06b_mlp_gate_up"
    assert shape_sweep["best_memory_preserving"]["speedup_vs_uncached"] == 208.597
    assert latest_artifact in text
    assert "QPruner Qwen3 shape sweep best memory-preserving int8_code_cache_dequantize_on_device" in text
    assert "qwen3_06b_mlp_gate_up" in text


def test_report_prefers_actual_qwen_projection_bits8_shape_sweep(tmp_path):
    report_module = load_report_module()
    demo_root = tmp_path / "demo"
    populate_bottleneck_artifacts(demo_root)
    artifacts = demo_root / "artifacts"
    actual_artifact = "qpruner_packed_decode_benchmark_qwen3_06b_actual_projection_bits8_shape_sweep_npu.json"
    write_json(
        artifacts / actual_artifact,
        {
            "status": "PASS",
            "backend": "torch_qpruner_packed_decode_shape_sweep",
            "device": "npu",
            "dtype": "float16",
            "bits": 8,
            "shape_preset": None,
            "shape_sweep": [{"label": f"shape_{index}"} for index in range(14)],
            "summary": {
                "best_memory_preserving": {
                    "label": "gate_proj_b2",
                    "path": "code_cached",
                    "strategy": "int8_code_cache_dequantize_on_device",
                    "latency_ms": 0.146129,
                    "speedup_vs_uncached": 249.151,
                    "storage_reduction_pct": 74.854,
                    "code_cache_storage_reduction_pct": 49.854,
                },
                "best_latency": {
                    "label": "q_proj_b1",
                    "path": "dense",
                    "strategy": "dense_dequantized_linear",
                    "latency_ms": 0.039357,
                    "speedup_vs_uncached": 925.114,
                },
            },
            "next_action": "Use actual Qwen3 projection shapes when selecting full-model QPruner policy.",
        },
    )

    report = report_module.build_report(demo_root)
    text = report_module.markdown(report)

    shape_sweep = report["qpruner_packed_decode_shape_sweep"]
    assert shape_sweep["artifact"] == actual_artifact
    assert shape_sweep["bits"] == 8
    assert shape_sweep["shape_count"] == 14
    assert shape_sweep["best_memory_preserving"]["label"] == "gate_proj_b2"
    assert shape_sweep["best_memory_preserving"]["path"] == "code_cached"
    assert actual_artifact in text
    assert "gate_proj_b2" in text


def test_report_summarizes_best_grouped_qpruner_shape_sweep(tmp_path):
    report_module = load_report_module()
    demo_root = tmp_path / "demo"
    populate_bottleneck_artifacts(demo_root)
    artifacts = demo_root / "artifacts"
    grouped_artifact = "qpruner_packed_decode_benchmark_qwen3_06b_shape_sweep_grouped8_bits8_npu.json"
    write_json(
        artifacts / grouped_artifact,
        {
            "status": "PASS",
            "backend": "torch_qpruner_packed_decode_shape_sweep",
            "device": "npu",
            "dtype": "float16",
            "bits": 8,
            "shape_preset": "qwen3-0.6b",
            "shape_sweep": [
                {
                    "label": "qwen3_06b_attn_qkv",
                    "shape": {"batch_size": 8, "in_features": 1024, "out_features": 1024},
                    "sequential_scaled_code_matmul_group": {
                        "runtime_strategy": "sequential_scaled_int8_code_matmul_group",
                        "module_count": 8,
                        "latency_ms": 1.300988,
                    },
                    "grouped_scaled_code_matmul": {
                        "runtime_strategy": "grouped_scaled_int8_code_bmm",
                        "module_count": 8,
                        "latency_ms": 0.755710,
                        "speedup_vs_sequential_scaled_code": 1.722,
                        "grouped_code_cache_bytes": 8388608,
                        "grouped_aux_cache_bytes": 16400,
                    },
                    "grouped_max_abs_diff_vs_sequential_scaled_code": 0.0,
                },
                {
                    "label": "qwen3_06b_mlp_down",
                    "shape": {"batch_size": 8, "in_features": 2816, "out_features": 1024},
                    "sequential_scaled_code_matmul_group": {
                        "runtime_strategy": "sequential_scaled_int8_code_matmul_group",
                        "module_count": 8,
                        "latency_ms": 1.277143,
                    },
                    "grouped_scaled_code_matmul": {
                        "runtime_strategy": "grouped_scaled_int8_code_bmm",
                        "module_count": 8,
                        "latency_ms": 0.711919,
                        "speedup_vs_sequential_scaled_code": 1.794,
                        "grouped_code_cache_bytes": 23068672,
                        "grouped_aux_cache_bytes": 16400,
                    },
                    "grouped_max_abs_diff_vs_sequential_scaled_code": 0.00195312,
                },
            ],
            "summary": {"shape_count": 2},
            "next_action": "Fuse same-shape QPruner projection groups on NPU.",
        },
    )

    report = report_module.build_report(demo_root)
    text = report_module.markdown(report)

    grouped = report["qpruner_grouped_shape_sweep"]
    assert grouped["artifact"] == grouped_artifact
    assert grouped["best_grouped_projection"]["label"] == "qwen3_06b_mlp_down"
    assert grouped["best_grouped_projection"]["module_count"] == 8
    assert grouped["best_grouped_projection"]["speedup_vs_sequential_scaled_code"] == 1.794
    assert grouped["best_grouped_projection"]["grouped_code_cache_bytes"] == 23068672
    assert "QPruner Qwen3 grouped shape sweep best grouped projection qwen3_06b_mlp_down" in text
    assert "grouped speedup 1.794x" in text
    assert "grouped code-cache bytes=23068672.000" in text
    assert "qpruner_grouped_shape_sweep_positive" in {item["id"] for item in report["diagnoses"]}


def test_report_builds_full_model_grouped_projection_plan_from_actual_qwen_shapes(tmp_path):
    report_module = load_report_module()
    demo_root = tmp_path / "demo"
    populate_bottleneck_artifacts(demo_root)
    artifacts = demo_root / "artifacts"
    native_artifact = (
        "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_"
        "shapeaware_releasepacked_auxcache_paired3_long16_npu.json"
    )
    grouped_artifact = "qpruner_packed_decode_benchmark_qwen3_06b_actual_shape_sweep_grouped8_bits8_npu.json"
    write_json(
        artifacts / native_artifact,
        {
            "status": "PASS",
            "backend": "torch_generate_qwen_qpruner_native_profile",
            "model_id": "Qwen/Qwen3-0.6B",
            "target_layer_limit": 196,
            "targeted_layers_total": 196,
            "target_layer_coverage_pct": 100.0,
            "qpruner_cache_mode": "shape-aware-code",
            "qpruner_cache_aux_tensors": True,
            "qpruner_release_packed_after_cache": True,
            "max_new_tokens": 16,
            "iters": 1,
            "warmup": 1,
            "paired_rounds": 3,
            "paired_summary": {"samples": 6},
            "baseline": {"latency_ms": 1877.5297298561782, "tokens_per_s": 17.044},
            "qpruner": {
                "status": "PASS",
                "latency_ms": 1480.39775993675,
                "tokens_per_s": 21.616,
                "latency_speedup": 1.268,
                "targeted_layers": 196,
                "average_bits": 8.0,
                "targeted_storage_reduction_pct": 50.0,
                "code_cache_storage_reduction_pct": 50.0,
                "code_cache_storage_bytes": 440403096,
                "cached_aux_modules": 196,
                "cached_aux_bytes": 392,
                "cached_scaled_code_bytes": 0,
                "released_packed_code_bytes": 440401920,
                "live_compressed_payload_storage_bytes": 784,
                "release_packed_after_cache": True,
                "runtime_strategy": "shape_aware_mixed_int8_code_cache",
                "runtime_profile": {
                    "status": "PASS",
                    "measurement": "synchronized_forward_hooks",
                    "quantized_layers": 196,
                    "runtime_strategy": "shape_aware_mixed_int8_code_cache",
                    "synchronizes_per_forward": True,
                    "total_forward_calls": 3136,
                    "total_forward_time_ms": 687.682869,
                    "estimated_forward_share_pct": 46.453,
                    "by_strategy": {
                        "scaled_int8_code_matmul": {
                            "modules": 196,
                            "calls": 3136,
                            "time_ms": 687.682869,
                            "avg_ms_per_call": 0.219287,
                        }
                    },
                    "top_modules": [
                        {
                            "name": "model.layers.0.self_attn.q_proj",
                            "strategy": "scaled_int8_code_matmul",
                            "calls": 16,
                            "time_ms": 4.11031,
                            "avg_ms_per_call": 0.256894,
                            "in_features": 1024,
                            "out_features": 2048,
                        },
                        {
                            "name": "model.layers.0.mlp.down_proj",
                            "strategy": "scaled_int8_code_matmul",
                            "calls": 16,
                            "time_ms": 3.63489,
                            "avg_ms_per_call": 0.227181,
                            "in_features": 3072,
                            "out_features": 1024,
                        },
                    ],
                },
            },
            "summary": {
                "qpruner_vs_baseline_speedup": 1.268,
                "measurement_basis": "paired_interleaved_median",
            },
        },
    )
    write_json(
        artifacts / grouped_artifact,
        {
            "status": "PASS",
            "backend": "torch_qpruner_packed_decode_shape_sweep",
            "device": "npu",
            "dtype": "float16",
            "bits": 8,
            "shape_preset": "qwen3-0.6b-actual",
            "shape_sweep": [
                {
                    "label": "qwen3_06b_actual_q_proj",
                    "shape": {"batch_size": 8, "in_features": 1024, "out_features": 2048},
                    "sequential_scaled_code_matmul_group": {"module_count": 8, "latency_ms": 1.41},
                    "grouped_scaled_code_matmul": {
                        "runtime_strategy": "grouped_scaled_int8_code_bmm",
                        "module_count": 8,
                        "latency_ms": 0.80,
                        "speedup_vs_sequential_scaled_code": 1.762,
                        "grouped_code_cache_bytes": 16777216,
                        "grouped_aux_cache_bytes": 32784,
                    },
                    "grouped_max_abs_diff_vs_sequential_scaled_code": 0.00195312,
                },
                {
                    "label": "qwen3_06b_actual_down_proj",
                    "shape": {"batch_size": 8, "in_features": 3072, "out_features": 1024},
                    "sequential_scaled_code_matmul_group": {"module_count": 8, "latency_ms": 1.50},
                    "grouped_scaled_code_matmul": {
                        "runtime_strategy": "grouped_scaled_int8_code_bmm",
                        "module_count": 8,
                        "latency_ms": 0.72,
                        "speedup_vs_sequential_scaled_code": 2.083,
                        "grouped_code_cache_bytes": 25165824,
                        "grouped_aux_cache_bytes": 16400,
                    },
                    "grouped_max_abs_diff_vs_sequential_scaled_code": 0.00195312,
                },
            ],
            "summary": {"shape_count": 2},
        },
    )

    report = report_module.build_report(demo_root)
    text = report_module.markdown(report)

    plan = report["qpruner_grouped_projection_plan"]
    assert plan["status"] == "ACTIONABLE"
    assert plan["profile_artifact"] == native_artifact
    assert plan["shape_artifact"] == grouped_artifact
    assert plan["shape_preset"] == "qwen3-0.6b-actual"
    assert plan["full_model"]["target_layers"] == 196
    assert plan["full_model"]["total_forward_calls"] == 3136
    assert plan["full_model"]["forward_share_pct"] == 46.453
    assert plan["full_model"]["code_cache_storage_reduction_pct"] == 50.0
    assert plan["full_model"]["cached_scaled_code_bytes"] == 0
    assert plan["best_candidate"]["label"] == "qwen3_06b_actual_down_proj"
    assert plan["best_candidate"]["speedup_vs_sequential_scaled_code"] == 2.083
    assert plan["best_candidate"]["profile_top_module_hits"] == 1
    assert plan["candidate_count"] == 2
    assert "qpruner_full_model_grouped_projection_plan" in {item["id"] for item in report["diagnoses"]}
    assert "QPruner full-model grouped projection plan" in text
    assert "qwen3_06b_actual_down_proj" in text
    assert "code-cache storage reduction 50.000%" in text


def test_report_surfaces_real_qwen_qpruner_grouped_replay(tmp_path):
    report_module = load_report_module()
    demo_root = tmp_path / "demo"
    populate_bottleneck_artifacts(demo_root)
    artifacts = demo_root / "artifacts"
    replay_artifact = "qwen_qpruner_grouped_replay_qwen3_06b_real_module_grouped_replay_npu.json"
    write_json(
        artifacts / replay_artifact,
        {
            "status": "PASS",
            "backend": "qwen_qpruner_real_quantizedlinear_grouped_replay",
            "model_id": "Qwen/Qwen3-0.6B",
            "device": "npu",
            "dtype": "torch.float16",
            "target_layer_limit": 196,
            "targeted_layers_total": 196,
            "quantized_layers": 196,
            "cache_modules": 196,
            "aux_cache_modules": 196,
            "qpruner_release_packed_after_cache": True,
            "qpruner_prebuild_scaled_code_dtype_cache": False,
            "targeted_storage_reduction_pct": 50.0,
            "code_cache_storage_reduction_pct": 50.0,
            "cached_scaled_code_bytes": 0,
            "cached_aux_bytes": 392,
            "group_count": 7,
            "available_group_count": 7,
            "best_group": {
                "role": "o_proj",
                "shape": {"batch_size": 8, "in_features": 2048, "out_features": 1024},
                "module_count": 8,
                "available_module_count": 28,
                "sample_module_names": [
                    "model.layers.0.self_attn.o_proj",
                    "model.layers.1.self_attn.o_proj",
                ],
                "sequential_scaled_code_matmul_group": {
                    "runtime_strategy": "sequential_scaled_int8_code_matmul_group",
                    "latency_ms": 1.234,
                },
                "grouped_scaled_code_matmul": {
                    "runtime_strategy": "grouped_scaled_int8_code_bmm",
                    "latency_ms": 0.676,
                    "speedup_vs_sequential_scaled_code": 1.825,
                    "memory_strategy": "int8_code_cache_plus_aux_tensors",
                },
                "grouped_code_cache_bytes": 16777216,
                "grouped_aux_cache_bytes": 16392,
                "grouped_scaled_code_cache_bytes": 0,
                "max_abs_diff_vs_sequential_scaled_code": 0.0,
            },
        },
    )

    report = report_module.build_report(demo_root)
    text = report_module.markdown(report)

    replay = report["qwen_qpruner_grouped_replay"]
    assert replay["artifact"] == replay_artifact
    assert replay["status"] == "PASS"
    assert replay["target_layer_limit"] == 196
    assert replay["quantized_layers"] == 196
    assert replay["group_count"] == 7
    assert replay["best_group"]["role"] == "o_proj"
    assert replay["best_group"]["speedup_vs_sequential_scaled_code"] == 1.825
    assert replay["best_group"]["grouped_code_cache_bytes"] == 16777216
    assert replay["best_group"]["grouped_scaled_code_cache_bytes"] == 0
    assert "qwen_qpruner_real_grouped_replay_positive" in {item["id"] for item in report["diagnoses"]}
    assert "Qwen3 QPruner real grouped replay" in text
    assert "best role=o_proj" in text
    assert "grouped speedup=1.825x" in text
    assert replay_artifact in text


def test_cli_writes_json_and_markdown(tmp_path):
    report_module = load_report_module()
    demo_root = tmp_path / "demo"
    populate_bottleneck_artifacts(demo_root)

    rc = report_module.main(["--demo-root", str(demo_root)])

    assert rc == 0
    payload = json.loads((demo_root / "artifacts" / "inference_bottleneck_report.json").read_text())
    markdown = (demo_root / "reports" / "inference-bottleneck-report.md").read_text()
    assert payload["status"] == "ACTIONABLE"
    assert "Inference Bottleneck Diagnosis" in markdown


def test_report_prefers_longer_decode_vllm_artifact(tmp_path):
    report_module = load_report_module()
    demo_root = tmp_path / "demo"
    populate_bottleneck_artifacts(demo_root)
    write_json(
        demo_root / "artifacts" / "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu_long_decode.json",
        {
            "status": "PASS",
            "run_label": "tiny_qwen3_serving_vllm_metadata_shim_npu_long_decode",
            "backend": "vllm_ascend_generate",
            "max_new_tokens": 16,
            "iters": 3,
            "warmup": 1,
            "summary": {
                "best_method": "qpruner",
                "best_tokens_per_s": 25.0,
                "cap_vs_baseline_speedup": 1.05,
                "qpruner_vs_baseline_speedup": 1.25,
            },
            "exports": {
                "baseline": {"status": "PASS", "tokens_per_s": 20.0},
                "cap": {"status": "PASS", "tokens_per_s": 21.0},
                "qpruner": {"status": "PASS", "tokens_per_s": 25.0},
            },
        },
    )

    report = report_module.build_report(demo_root)
    text = report_module.markdown(report)

    assert report["sequential_vllm"]["artifact"] == (
        "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu_long_decode.json"
    )
    assert report["sequential_vllm"]["max_new_tokens"] == 16
    assert report["sequential_vllm"]["speedups"]["qpruner_vs_baseline"] == 1.25
    assert "max_new_tokens=16" in text
    assert "short_decode_hides_steady_state" not in {item["id"] for item in report["diagnoses"]}
    assert report["next_optimization_target"]["title"] == "Optimize compressed serving export and kernels"
    assert "Long-decode vLLM serving benchmark already covers max_new_tokens=16" in report["next_optimization_target"]["steps"][0]


def test_report_surfaces_current_vllm_hbm_parallel_failure_without_hiding_pass_baseline(tmp_path):
    report_module = load_report_module()
    demo_root = tmp_path / "demo"
    populate_bottleneck_artifacts(demo_root)
    write_json(
        demo_root / "artifacts" / "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu_long_decode.json",
        {
            "status": "PASS",
            "run_label": "tiny_qwen3_serving_vllm_metadata_shim_npu_long_decode",
            "backend": "vllm_ascend_generate",
            "max_new_tokens": 16,
            "iters": 3,
            "warmup": 1,
            "summary": {
                "best_method": "baseline",
                "best_tokens_per_s": 385.044,
                "cap_vs_baseline_speedup": 0.986,
                "qpruner_vs_baseline_speedup": 0.975,
            },
            "exports": {
                "baseline": {"status": "PASS", "tokens_per_s": 385.044},
                "cap": {"status": "PASS", "tokens_per_s": 379.673},
                "qpruner": {"status": "PASS", "tokens_per_s": 375.386},
            },
        },
    )
    write_json(
        demo_root
        / "artifacts"
        / "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu_long_decode_hbm_parallel.json",
        {
            "status": "FAIL",
            "run_label": "tiny_qwen3_serving_vllm_metadata_shim_npu_long_decode_hbm_parallel",
            "backend": "vllm_ascend_generate",
            "parallel_methods": True,
            "method_cards": [0, 1, 2],
            "max_new_tokens": 16,
            "exports": {
                "baseline": {
                    "status": "FAIL",
                    "visible_card": 0,
                    "error_type": "RuntimeError",
                    "stderr": "from acl.rt import memcpy\nModuleNotFoundError: No module named 'acl'",
                },
                "cap": {
                    "status": "FAIL",
                    "visible_card": 1,
                    "error_type": "RuntimeError",
                    "stderr": "from acl.rt import memcpy\nModuleNotFoundError: No module named 'acl'",
                },
                "qpruner": {
                    "status": "FAIL",
                    "visible_card": 2,
                    "error_type": "RuntimeError",
                    "stderr": "from acl.rt import memcpy\nModuleNotFoundError: No module named 'acl'",
                },
            },
        },
    )

    report = report_module.build_report(demo_root)
    text = report_module.markdown(report)

    assert report["sequential_vllm"]["artifact"] == (
        "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu_long_decode.json"
    )
    current = report["current_vllm_rerun"]
    assert current["artifact"] == (
        "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu_long_decode_hbm_parallel.json"
    )
    assert current["status"] == "FAIL"
    assert current["parallel_methods"] is True
    assert current["method_cards"] == [0, 1, 2]
    assert current["failing_methods"] == ["baseline", "cap", "qpruner"]
    assert "acl.rt" in current["root_cause"]
    assert "vllm_current_hbm_parallel_rerun_failed" in {item["id"] for item in report["diagnoses"]}
    assert "Current vLLM HBM rerun" in text
    assert "acl.rt" in text


def test_report_prefers_passing_current_vllm_hbm_rerun_over_longer_failed_attempt(tmp_path):
    report_module = load_report_module()
    demo_root = tmp_path / "demo"
    populate_bottleneck_artifacts(demo_root)
    write_json(
        demo_root
        / "artifacts"
        / "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu_long_decode_hbm_parallel.json",
        {
            "status": "FAIL",
            "run_label": "tiny_qwen3_serving_vllm_metadata_shim_npu_long_decode_hbm_parallel",
            "backend": "vllm_ascend_generate",
            "parallel_methods": True,
            "method_cards": [0, 1, 2],
            "max_new_tokens": 32,
            "exports": {
                method: {
                    "status": "FAIL",
                    "visible_card": index,
                    "error_type": "RuntimeError",
                    "stdout_tail": "from acl.rt import memcpy\nModuleNotFoundError: No module named 'acl'",
                }
                for index, method in enumerate(("baseline", "cap", "qpruner"))
            },
        },
    )
    write_json(
        demo_root
        / "artifacts"
        / "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu_hbm_parallel_aclpath.json",
        {
            "status": "PASS",
            "run_label": "tiny_qwen3_serving_vllm_metadata_shim_npu_hbm_parallel_aclpath",
            "backend": "vllm_ascend_generate",
            "parallel_methods": True,
            "method_cards": [0, 1, 2],
            "max_new_tokens": 4,
            "summary": {
                "best_method": "cap",
                "best_tokens_per_s": 270.483,
                "cap_vs_baseline_speedup": 1.046,
                "qpruner_vs_baseline_speedup": 0.919,
            },
            "exports": {
                "baseline": {"status": "PASS", "visible_card": 0, "tokens_per_s": 258.645},
                "cap": {"status": "PASS", "visible_card": 1, "tokens_per_s": 270.483},
                "qpruner": {"status": "PASS", "visible_card": 2, "tokens_per_s": 237.719},
            },
        },
    )

    report = report_module.build_report(demo_root)

    current = report["current_vllm_rerun"]
    assert current["artifact"] == (
        "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu_hbm_parallel_aclpath.json"
    )
    assert current["status"] == "PASS"
    assert current["failing_methods"] == []
    assert current["best_method"] == "cap"
    assert current["best_tokens_per_s"] == 270.483
    assert current["speedups"]["qpruner_vs_baseline"] == 0.919
    assert "vllm_current_hbm_parallel_rerun_failed" not in {item["id"] for item in report["diagnoses"]}


def test_report_surfaces_external_npu_occupancy_blocking_long_decode(tmp_path):
    report_module = load_report_module()
    demo_root = tmp_path / "demo"
    populate_bottleneck_artifacts(demo_root)
    write_json(
        demo_root / "artifacts" / "current_npu_occupancy.json",
        {
            "status": "BLOCKED",
            "reason": "npu_memory_occupied_by_external_vllm_service",
            "occupied_cards": 8,
            "total_cards": 8,
            "container": {
                "id": "0f0d94d365a0",
                "name": "tdqs_qwen3-30b-a3b",
                "image": "quay.io/ascend/vllm-ascend:v0.20.2rc1",
            },
            "service": {
                "model": "Qwen3-30B-A3B",
                "command": "vllm serve /root/.cache --served-model-name Qwen3-30B-A3B",
                "tensor_parallel_size": 8,
            },
            "policy": "do_not_kill_unrelated_container",
            "next_action": "Wait for the external vLLM service to release cards, then rerun long-decode serving benchmarks.",
        },
    )

    report = report_module.build_report(demo_root)
    text = report_module.markdown(report)

    assert report["resource_blocker"]["status"] == "BLOCKED"
    assert report["resource_blocker"]["occupied_cards"] == 8
    assert report["resource_blocker"]["container_name"] == "tdqs_qwen3-30b-a3b"
    assert "long_decode_waiting_for_free_cards" in {item["id"] for item in report["diagnoses"]}
    assert "External NPU occupancy" in text
    assert "tdqs_qwen3-30b-a3b" in text
    assert "tensor_parallel_size=8" in text
    assert "do_not_kill_unrelated_container" in text
