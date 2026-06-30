import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def load_readiness_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "write_demo_readiness_report.py"
    spec = importlib.util.spec_from_file_location("write_demo_readiness_report", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def populate_ready_demo(artifacts: Path) -> None:
    demo_root = artifacts.parent
    for path in (
        demo_root / "reports" / "ascend-910b-demo-progress.md",
        demo_root / "reports" / "demo-storyboard.md",
        demo_root / "reports" / "qwen-compression-quality-memory-sweep.md",
        demo_root / "reports" / "qwen-qpruner-quality-memory-sweep.md",
        demo_root / "reports" / "qwen-qpruner-scale-quality-summary.md",
        demo_root / "reports" / "qwen-qpruner-scale-quality-summary.csv",
        demo_root / "reports" / "qwen-compression-choice-accuracy-qwen3_06b_choice_accuracy_npu.md",
        demo_root / "reports" / "qwen-compression-choice-accuracy-qwen3_06b_choice_accuracy_npu.csv",
        demo_root
        / "reports"
        / "qwen-compression-choice-accuracy-qwen3_06b_multitask_choice_fulltarget_qpruner_npu.md",
        demo_root
        / "reports"
        / "qwen-compression-choice-accuracy-qwen3_06b_multitask_choice_fulltarget_qpruner_npu.csv",
        demo_root / "reports" / "qwen-qpruner-quality-memory-frontier.svg",
        demo_root / "reports" / "qwen-qpruner-quality-memory-frontier.md",
        demo_root / "reports" / "inference-acceleration-summary.md",
        demo_root / "reports" / "inference-acceleration-summary.csv",
        demo_root / "reports" / "inference-acceleration-summary.svg",
        demo_root / "reports" / "compressed-native-cache-long-decode-sweep.md",
        demo_root / "reports" / "compressed-native-cache-long-decode-sweep.csv",
        demo_root / "reports" / "compressed-native-cache-long-decode-sweep.svg",
        demo_root / "reports" / "qpruner-packed-decode-benchmark-tiny_qwen3_qpruner_decode_npu.md",
        demo_root / "reports" / "multicard-compression-sync-summary.md",
        demo_root / "reports" / "multicard-compression-sync-summary.csv",
        demo_root / "reports" / "multicard-compression-sync-summary.svg",
        demo_root / "reports" / "objective-coverage-audit.md",
        demo_root / "recordings" / "ascend-910b-demo-playback-latest.typescript",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "qwen-compression-quality-memory-sweep.md":
            path.write_text(
                "Best memory-quality point under loss delta <= 0.500 is "
                "qwen3_06b_quality_pattern_bits6: 66.667% targeted memory reduction, "
                "loss delta -0.224, average bits 5.333.\n"
            )
        elif path.name == "qwen-qpruner-quality-memory-sweep.md":
            path.write_text(
                "Best memory-quality point under loss delta <= 0.500 is "
                "qwen3_06b_qpruner_sweep_layers4_bits6: 66.667% targeted memory reduction, "
                "loss delta -0.224, average bits 5.333.\n"
            )
        elif path.name == "qwen-qpruner-scale-quality-summary.md":
            path.write_text(
                "QPruner scale-quality sweep reaches 16/196 target layers (8.163% coverage); "
                "best under loss delta <= 0.500 is qwen3_06b_qpruner_sweep_layers16_bits6 "
                "with 62.500% targeted memory reduction and loss-derived approximate PPL "
                "403.429 -> 518.013 (28.403%).\n"
            )
        elif path.name == "qwen-qpruner-scale-quality-summary.csv":
            path.write_text(
                "run_label,status,target_layer_limit,targeted_layers_total,coverage_pct,average_bits\n"
            )
        elif path.name == "qwen-compression-choice-accuracy-qwen3_06b_choice_accuracy_npu.md":
            path.write_text(
                "# Qwen Compression Choice Accuracy\n"
                "| baseline | PASS | 75.000% | 3 / 4 |\n"
                "| cap | PASS | 50.000% | 2 / 4 |\n"
                "| qpruner | PASS | 75.000% | 3 / 4 |\n"
            )
        elif path.name == "qwen-compression-choice-accuracy-qwen3_06b_choice_accuracy_npu.csv":
            path.write_text("method,status,accuracy,correct,total\n")
        elif path.name == "qwen-compression-choice-accuracy-qwen3_06b_multitask_choice_fulltarget_qpruner_npu.md":
            path.write_text(
                "# Qwen Compression Choice Accuracy\n"
                "| baseline | __overall__ | PASS | 75.000% | 6 / 8 |\n"
                "| qpruner | __overall__ | PASS | 87.500% | 7 / 8 |\n"
            )
        elif path.name == "qwen-compression-choice-accuracy-qwen3_06b_multitask_choice_fulltarget_qpruner_npu.csv":
            path.write_text("method,task,status,accuracy,correct,total\n")
        elif path.suffix == ".svg":
            path.write_text("<svg></svg>\n")
        elif path.name == "qwen-qpruner-quality-memory-frontier.md":
            path.write_text("![QPruner quality-memory frontier](qwen-qpruner-quality-memory-frontier.svg)\n")
        elif path.name == "inference-acceleration-summary.md":
            path.write_text(
                "# Inference Acceleration Summary\n"
                "Best throughput point: `torch fallback / QPruner` at 456.000 tokens/s.\n"
                "compressed-native preserves quantized/pruned modules with dense export `False`.\n"
            )
        elif path.name == "inference-acceleration-summary.csv":
            path.write_text("track,method,tokens_per_s\n")
        elif path.name == "inference-acceleration-summary.svg":
            path.write_text("<svg>Inference acceleration</svg>\n")
        elif path.name == "compressed-native-cache-long-decode-sweep.md":
            path.write_text(
                "# Compressed-Native Cache/Long-Decode Sweep\n"
                "Cached long-decode compressed-native path is latency-positive: "
                "QPruner reaches 1.200x baseline at max_new_tokens=16 while retaining "
                "75.000% targeted storage reduction.\n"
            )
        elif path.name == "compressed-native-cache-long-decode-sweep.csv":
            path.write_text("run_label,max_new_tokens,inference_cache_enabled\n")
        elif path.name == "compressed-native-cache-long-decode-sweep.svg":
            path.write_text("<svg>Compressed-native cache/long-decode sweep</svg>\n")
        elif path.name == "qpruner-packed-decode-benchmark-tiny_qwen3_qpruner_decode_npu.md":
            path.write_text(
                "# QPruner Packed Decode Benchmark\n"
                "Packed QPruner storage saves memory; dense_weight_cache isolates decode overhead.\n"
            )
        elif path.name == "multicard-compression-sync-summary.md":
            path.write_text(
                "# Multi-Card Compression Sync Summary\n"
                "8-card synchronized compression path is ready: QPruner 240.000 tokens/s total "
                "(1.250x baseline), all-reduce consistent True, while compressed-native keeps "
                "76.316% QPruner storage reduction.\n"
            )
        elif path.name == "multicard-compression-sync-summary.csv":
            path.write_text("metric,value\n")
        elif path.name == "multicard-compression-sync-summary.svg":
            path.write_text("<svg>Multi-card compression sync</svg>\n")
        elif path.name == "objective-coverage-audit.md":
            path.write_text("# TIDAL-AI Ascend Objective Coverage Audit\n")
        else:
            path.write_text("fixture\n")
    write_json(
        artifacts / "inference_acceleration_summary.json",
        {
            "status": "ACTIONABLE",
            "summary": {
                "best_throughput": {
                    "track": "torch fallback",
                    "method": "qpruner",
                    "method_label": "QPruner",
                    "tokens_per_s": 456.0,
                },
                "best_native_memory": {
                    "method": "qpruner",
                    "method_label": "QPruner",
                    "memory_reduction_pct": 75.0,
                    "storage_reduction_pct": 75.0,
                },
            },
            "vllm_bottleneck": {"title": "vLLM compressed path is not faster than baseline"},
        },
    )
    write_json(
        artifacts / "compressed_native_sweep_summary.json",
        {
            "status": "PASS",
            "best": {
                "label": "tiny_long_cache_on",
                "method": "qpruner",
                "method_label": "QPruner",
                "speedup": 1.2,
                "max_new_tokens": 16,
                "inference_cache_enabled": True,
                "storage_reduction_pct": 75.0,
            },
            "cache_effect": {"max_new_tokens": 1, "qpruner_speedup_delta": 0.08},
            "long_decode_effect": {
                "from_max_new_tokens": 1,
                "to_max_new_tokens": 16,
                "qpruner_speedup_delta": 0.22,
            },
            "memory": {"best_qpruner_storage_reduction_pct": 75.0},
            "memory_preserving_qpruner_best": {
                "label": "tiny_short_scaled_code",
                "qpruner_cache_mode": "scaled-code-matmul",
                "qpruner_runtime_strategy": "scaled_int8_code_matmul",
                "qpruner_vs_baseline_speedup": 1.18,
                "qpruner_code_cache_storage_reduction_pct": 49.5,
            },
            "runtime_diagnosis": {
                "baseline_decode_scaling": 2.0,
                "qpruner_decode_scaling": 2.449,
                "qpruner_scaling_gap_vs_baseline": 0.449,
                "qpruner_cache_peak_mem_delta_mb": 1.0,
                "qpruner_cache_vs_baseline_peak_mem_delta_mb": 2.0,
                "cache_tradeoff": (
                    "QPruner cache adds 1.000 MB over uncached QPruner peak at max_new_tokens=1 "
                    "for 0.080x speedup delta; +2.000 MB vs uncached baseline peak."
                ),
                "next_action": (
                    "Keep compressed weights live, then fuse packed/quantized decode kernels on NPU "
                    "for long generation."
                ),
            },
            "readout": (
                "Cached long-decode compressed-native path is latency-positive: "
                "QPruner reaches 1.200x baseline at max_new_tokens=16 while retaining "
                "75.000% targeted storage reduction."
            ),
        },
    )
    write_json(
        artifacts / "qpruner_packed_decode_benchmark_tiny_qwen3_qpruner_decode_npu.json",
        {
            "status": "PASS",
            "device": "npu",
            "runtime_storage_format": "packed_nbit_weight_codes",
            "packed_weight_code_bytes": 2048,
            "code_cache_bytes": 4096,
            "quantized_payload_bytes": 2084,
            "code_cache_payload_bytes": 4132,
            "dense_payload_bytes": 8192,
            "storage_reduction_pct": 74.561,
            "code_cache_storage_reduction_pct": 49.561,
            "uncached": {"runtime_strategy": "dequantize_per_forward", "latency_ms": 2.0},
            "code_cached": {
                "runtime_strategy": "int8_code_cache_dequantize_on_device",
                "latency_ms": 1.2,
                "speedup_vs_uncached": 1.667,
                "peak_mem_mb": 4.2,
            },
            "scaled_code_matmul": {
                "runtime_strategy": "scaled_int8_code_matmul",
                "latency_ms": 1.1,
                "speedup_vs_uncached": 1.818,
                "peak_mem_mb": 2.5,
            },
            "cached": {
                "runtime_strategy": "dense_weight_cache",
                "latency_ms": 1.0,
                "speedup_vs_uncached": 2.0,
            },
            "dense": {
                "runtime_strategy": "dense_dequantized_linear",
                "latency_ms": 0.9,
                "speedup_vs_uncached": 2.222,
            },
            "decode_overhead_vs_cached_pct": 100.0,
            "next_action": "Fuse packed/quantized decode kernels on NPU.",
        },
    )
    write_json(
        artifacts / "multicard_compression_sync_summary.json",
        {
            "status": "PASS",
            "readout": (
                "8-card synchronized compression path is ready: QPruner 240.000 tokens/s total "
                "(1.250x baseline), all-reduce consistent True, while compressed-native keeps "
                "76.316% QPruner storage reduction."
            ),
            "sync": {"world_size": 8, "backend": "hccl"},
            "qwen_compression_generate": {
                "distributed_reduce_consistent": True,
                "baseline_tokens_per_s_total": 192.0,
                "cap_tokens_per_s_total": 224.0,
                "qpruner_tokens_per_s_total": 240.0,
                "qpruner_speedup_vs_baseline": 1.25,
                "pass_count": 8,
            },
            "compressed_native_memory": {
                "qpruner_storage_reduction_pct": 76.316,
                "qpruner_cache_peak_mem_delta_mb": 0.3,
                "next_action": (
                    "Keep compressed weights live, then fuse packed/quantized decode kernels on NPU "
                    "for long generation."
                ),
            },
            "parallel_suite": {"start_window_s": 0.005, "release_lag_window_s": 0.001},
        },
    )
    write_json(
        artifacts / "multicard_sync_summary.json",
        {"status": "PASS", "world_size": 8, "backend": "hccl"},
    )
    write_json(
        artifacts / "objective_coverage_audit.json",
        {
            "status": "PARTIAL_READY",
            "sections": [
                {
                    "title": "Model compression",
                    "status": "PARTIAL_READY",
                    "evidence_summary": "QPruner storage reduction 76.316%",
                    "remaining_gap": "Scale target-layer coverage",
                }
            ],
        },
    )
    write_json(
        artifacts / "qwen_qpruner_scale_quality_summary.json",
        {
            "status": "PASS",
            "readout": (
                "QPruner scale-quality sweep reaches 16/196 target layers (8.163% coverage); "
                "best under loss delta <= 0.500 is qwen3_06b_qpruner_sweep_layers16_bits6 "
                "with 62.500% targeted memory reduction and loss-derived approximate PPL "
                "403.429 -> 518.013 (28.403%)."
            ),
        },
    )
    write_json(
        artifacts / "qwen_compression_choice_accuracy_qwen3_06b_choice_accuracy_npu.json",
        {
            "status": "PASS",
            "backend": "choice_log_likelihood",
            "model_id": "Qwen/Qwen3-0.6B",
            "target_layer_limit": 64,
            "targeted_layers_total": 196,
            "methods": {
                "baseline": {"status": "PASS", "accuracy": 0.75, "correct": 3, "total": 4},
                "cap": {"status": "PASS", "accuracy": 0.5, "correct": 2, "total": 4},
                "qpruner": {"status": "PASS", "accuracy": 0.75, "correct": 3, "total": 4},
            },
        },
    )
    write_json(
        artifacts / "qwen_compression_choice_accuracy_qwen3_06b_multitask_choice_fulltarget_qpruner_npu.json",
        {
            "status": "PASS",
            "backend": "choice_log_likelihood",
            "model_id": "Qwen/Qwen3-0.6B",
            "target_layer_limit": 0,
            "targeted_layers_total": 196,
            "task_count": 4,
            "methods": {
                "baseline": {"status": "PASS", "accuracy": 0.75, "correct": 6, "total": 8, "targeted_layers": 196},
                "qpruner": {
                    "status": "PASS",
                    "accuracy": 0.875,
                    "correct": 7,
                    "total": 8,
                    "targeted_layers": 196,
                    "compression_time_s": 13.111,
                },
            },
        },
    )
    write_json(
        artifacts / "rankadaptor_lora_sync_summary.json",
        {"status": "PASS", "world_size": 8, "trainable_adapter_params": 512},
    )
    write_json(
        artifacts / "compatibility_issues.json",
        {
            "status": "ISSUES_FOUND",
            "issues": [{"id": "vllm_attention_selector_api_mismatch"}],
            "ready": [{"id": "qwen3_torch_npu_ready"}, {"id": "torch_serving_fallback_ready"}],
        },
    )
    write_json(
        artifacts / "model_snapshot_qwen3_06b.json",
        {"status": "DOWNLOADED", "provider": "modelscope", "parameter_mb": 1433.659},
    )
    write_json(
        artifacts / "ascend_inference_qwen3_06b_torch_npu.json",
        {"status": "PASS", "tokens_per_s": 11.923, "latency_ms": 1341.941, "peak_mem_mb": 1159.8},
    )
    write_json(
        artifacts / "torch_serving_fallback_tiny_qwen3_serving_fallback_npu.json",
        {
            "status": "PASS",
            "backend": "torch_generate",
            "summary": {
                "best_method": "qpruner",
                "best_tokens_per_s": 175.185,
                "cap_vs_baseline_speedup": 1.027,
                "qpruner_vs_baseline_speedup": 1.048,
            },
            "exports": {
                "baseline": {"tokens_per_s": 167.142, "peak_mem_mb": 16.6},
                "cap": {"tokens_per_s": 171.681, "peak_mem_mb": 16.6},
                "qpruner": {"tokens_per_s": 175.185, "peak_mem_mb": 16.6},
            },
        },
    )
    write_json(
        artifacts / "compressed_native_torch_serving_tiny_qwen3_native_serving_npu.json",
        {
            "status": "PASS",
            "backend": "torch_generate_compressed_native",
            "serving_dense_export": False,
            "inference_cache_enabled": False,
            "summary": {
                "best_method": "qpruner",
                "best_tokens_per_s": 210.0,
                "cap_vs_baseline_speedup": 1.067,
                "qpruner_vs_baseline_speedup": 1.312,
                "dense_export_erases_storage_savings": True,
            },
            "memory_reference": {
                "baseline_targeted_params": 4096,
                "targeted_layers": 7,
                "target_layer_coverage_pct": 100.0,
                "cap_targeted_param_reduction_pct": 66.667,
                "qpruner_targeted_param_reduction_pct": 75.0,
                "baseline_targeted_storage_bytes": 8192,
                "cap_targeted_storage_bytes": 2731,
                "qpruner_targeted_storage_bytes": 2048,
                "cap_targeted_storage_reduction_pct": 66.663,
                "qpruner_targeted_storage_reduction_pct": 75.0,
            },
            "baseline": {"tokens_per_s": 160.0, "latency_ms": 25.0, "peak_mem_mb": 40.0},
            "cap": {
                "tokens_per_s": 170.0,
                "latency_ms": 23.4,
                "peak_mem_mb": 39.0,
                "packed_layers": 7,
                "exported_dense_linears": 0,
                "targeted_param_reduction_pct": 66.667,
            },
            "qpruner": {
                "tokens_per_s": 210.0,
                "latency_ms": 19.0,
                "peak_mem_mb": 38.0,
                "quantized_layers": 7,
                "exported_dense_linears": 0,
                "targeted_param_reduction_pct": 75.0,
            },
        },
    )
    write_json(
        artifacts / "multicard_parallel_suite_demo_parallel_suite_npu.json",
        {
            "status": "PASS",
            "world_size": 8,
            "cards": list(range(8)),
            "sync_start_target_ts": 2000.0,
            "start_window_s": 0.004,
            "release_lag_window_s": 0.003,
            "task_counts": {
                "workflow_cap": 2,
                "workflow_qpruner": 2,
                "rankadaptor": 2,
                "torch_serving_fallback": 2,
            },
        },
    )
    write_json(
        artifacts / "multicard_qwen_lora_finetune_qwen3_06b_lora_sync_npu.json",
        {
            "status": "PASS",
            "world_size": 8,
            "aggregate": {
                "distributed_reduce_consistent": True,
                "adapter_sync_consistent": True,
                "validation_loss_delta_avg": -0.008,
                "pass_count": 8,
            },
        },
    )
    write_json(
        artifacts / "multicard_qwen_lora_finetune_qwen3_06b_lora_instruction_sync_8card_steps6_20260625.json",
        {
            "status": "PASS",
            "world_size": 8,
            "train_mode": "instruction",
            "aggregate": {
                "distributed_reduce_consistent": True,
                "adapter_sync_consistent": True,
                "validation_loss_delta_avg": -0.132813,
                "pass_count": 8,
            },
        },
    )
    write_json(
        artifacts / "multicard_tiny_qwen_bslora_finetune_tiny_qwen3_bslora_sync_npu.json",
        {
            "status": "PASS",
            "world_size": 8,
            "aggregate": {
                "distributed_reduce_consistent": True,
                "adapter_sync_consistent": True,
                "loss_delta_avg": -0.039,
            },
        },
    )
    write_json(
        artifacts / "tiny_qwen_bslora_finetune_qwen3_06b_bslora_npu.json",
        {
            "status": "PASS",
            "method": "bslora_shared_lora",
            "model_id": "Qwen/Qwen3-0.6B",
            "target_module_count": 196,
            "trainable_adapter_params": 32768,
            "unshared_adapter_params": 1261568,
            "initial_loss": 14.383,
            "final_loss": 14.094,
            "loss_delta": -0.289,
            "tokens_trained": 96,
            "peak_mem_mb": 1430.9,
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
        artifacts / "qwen_compression_generate_qwen3_06b_generate_npu.json",
        {
            "status": "PASS",
            "target_layer_limit": 2,
            "targeted_layers_total": 196,
            "cap": {"latency_speedup": 3.205, "targeted_compression_ratio": 3.0},
            "qpruner": {"latency_speedup": 3.184, "average_bits": 4.0},
        },
    )
    write_json(
        artifacts / "multicard_qwen_compression_quality_qwen3_06b_quality_pattern_2card_npu.json",
        {
            "status": "PASS",
            "world_size": 2,
            "backend": "hccl",
            "target_layer_pattern": r"self_attn\.(q_proj|k_proj)$",
            "target_layer_limit": 4,
            "targeted_layers_total": 196,
            "aggregate": {
                "distributed_reduce_consistent": True,
                "baseline_loss_avg": 6.797,
                "cap_loss_delta_avg": 0.157,
                "wanda_loss_delta_avg": 0.089,
                "sparsegpt_loss_delta_avg": 0.118,
                "qpruner_loss_delta_avg": -0.224,
                "pass_count": 2,
                "peak_mem_mb_total": 7111.0,
            },
        },
    )
    write_json(
        artifacts / "multicard_qwen_compression_generate_qwen3_06b_generate_pattern_2card_npu.json",
        {
            "status": "PASS",
            "world_size": 2,
            "backend": "hccl",
            "target_layer_pattern": r"self_attn\.(q_proj|k_proj)$",
            "target_layer_limit": 4,
            "targeted_layers_total": 196,
            "aggregate": {
                "distributed_reduce_consistent": True,
                "baseline_tokens_per_s_total": 18.594,
                "cap_tokens_per_s_total": 57.175,
                "qpruner_tokens_per_s_total": 54.419,
                "pass_count": 2,
                "peak_mem_mb_total": 9300.6,
            },
        },
    )
    write_json(
        artifacts / "compression_memory_report.json",
        {
            "status": "ACTIONABLE",
            "paper_baselines": {
                "qpruner": {"primary": ["LLM-Pruner"], "recovery_baselines": ["LoRA", "LoftQ"]},
                "cap": {"pruning_baselines": ["SparseGPT", "Wanda"]},
                "rankadaptor": {"recovery_baselines": ["LoRA", "AdaLoRA"]},
            },
            "engineering_reference": {
                "serving_baseline_role": "uncompressed_serving_export",
                "not_paper_baseline": True,
            },
            "baseline_alignment_matrix": [
                {
                    "method": "QPruner",
                    "paper_baseline_role": "pruning baseline",
                    "paper_baselines": ["LLM-Pruner"],
                    "demo_reference_role": "compressed-vs-uncompressed Ascend runtime reference",
                    "current_status": "NPU algorithm evidence exists; paper-baseline rerun is not claimed.",
                },
                {
                    "method": "CAP",
                    "paper_baseline_role": "pruning / joint-compression / SVD baselines",
                    "paper_baselines": ["SparseGPT", "Wanda"],
                    "demo_reference_role": "compressed-vs-uncompressed Ascend runtime reference",
                    "current_status": "NPU algorithm evidence exists; paper-baseline rerun is not claimed.",
                },
                {
                    "method": "RankAdaptor",
                    "paper_baseline_role": "recovery baselines",
                    "paper_baselines": ["LoRA", "AdaLoRA", "without recovery"],
                    "demo_reference_role": "LoRA/BSLoRA recovery evidence on Ascend",
                    "current_status": "NPU recovery evidence exists; full paper-baseline rerun remains future work.",
                },
                {
                    "method": "Engineering runtime baseline",
                    "paper_baseline_role": "not a paper baseline",
                    "paper_baselines": ["uncompressed serving export"],
                    "demo_reference_role": "runtime sanity, latency, throughput, and memory reference",
                    "current_status": "Used only for runtime ratios and not claimed as a paper baseline.",
                },
            ],
            "paper_baseline_evidence": [
                {
                    "method": "CAP",
                    "paper_baseline": "Wanda",
                    "status": "RUN_ON_ASCEND",
                    "evidence_role": "actual paper pruning baseline",
                    "artifacts": ["artifacts/qwen_compression_quality_qwen3_06b_quality_npu.json"],
                    "metrics": {
                        "loss_delta": 0.089,
                        "targeted_param_reduction_pct": 50.0,
                        "latency_speedup": 1.125,
                        "targeted_layers": 2,
                    },
                },
                {
                    "method": "QPruner",
                    "paper_baseline": "LLM-Pruner",
                    "status": "PENDING_NOT_RUN_ON_ASCEND",
                    "artifacts": [],
                    "metrics": {},
                },
            ],
            "compression_memory": {
                "cap": {"targeted_param_reduction_pct": 66.667},
                "qpruner": {"targeted_param_reduction_pct": 75.0},
                "target_layer_limit": 2,
                "targeted_layers_total": 196,
                "target_layer_coverage_pct": 1.02,
            },
            "memory_gap": {
                "serving_dense_export": True,
                "dense_export_erases_storage_savings": True,
            },
            "next_action": "Keep compression in pruned/quantized form inside serving runtime so memory savings survive inference.",
        },
    )
    write_json(
        artifacts / "paper_baseline_coverage_audit.json",
        {
            "status": "PASS",
            "engineering_baseline_note": "`baseline` in serving artifacts is dense/uncompressed runtime reference, not a paper baseline.",
            "video_readout": [
                "Baseline critique resolved by separating paper baselines from engineering runtime references.",
                "QPruner paper baseline now has 196/196 LLM-Pruner-style Ascend evidence: 25% targeted parameter pruning, loss delta 10.007, latency speedup 1.007x.",
                "CAP paper baselines now have 196/196 Wanda and SparseGPT Ascend evidence: Wanda loss delta 0.662, SparseGPT loss delta 11.018, both at 50% targeted sparsity.",
                "RankAdaptor recovery baselines now have controlled Ascend evidence: no-recovery delta 0.000, LoRA delta -0.484, AdaLoRA-style delta 0.102; best method lora.",
            ],
            "evidence": {
                "qpruner_llm_pruner_full196": {
                    "status": "PASS",
                    "targeted_layers": 196,
                    "targeted_layers_total": 196,
                    "targeted_param_reduction_pct": 25.0,
                    "loss_delta": 10.007,
                    "latency_speedup": 1.007,
                },
                "cap_wanda_sparsegpt_full196": {
                    "status": "PASS",
                    "targeted_layers": 196,
                    "targeted_layers_total": 196,
                    "wanda_targeted_param_reduction_pct": 50.0,
                    "wanda_loss_delta": 0.662,
                    "wanda_latency_speedup": 0.976,
                    "sparsegpt_targeted_param_reduction_pct": 50.0,
                    "sparsegpt_loss_delta": 11.018,
                    "sparsegpt_latency_speedup": 0.929,
                },
                "rankadaptor_recovery_controlled": {
                    "status": "PASS",
                    "all_methods_passed": True,
                    "best_recovery_method": "lora",
                    "no_recovery_loss_delta": 0.0,
                    "lora_loss_delta": -0.484,
                    "adalora_style_loss_delta": 0.102,
                },
            },
        },
    )
    write_json(
        artifacts / "vllm_serving_probe_tiny_qwen3_serving_export_npu.json",
        {
            "status": "FAIL",
            "issue_classes": [{"id": "vllm_attention_selector_api_mismatch"}],
            "api_diagnostics": {
                "status": "MISMATCH",
                "missing_patch_get_attn_backend_parameters": ["use_per_head_quant_scales", "num_heads"],
            },
        },
    )
    write_json(
        artifacts / "inference_bottleneck_report.json",
        {
            "status": "ACTIONABLE",
            "resource_blocker": {
                "status": "BLOCKED",
                "occupied_cards": 8,
                "total_cards": 8,
                "container_name": "tdqs_qwen3-30b-a3b",
                "service_model": "Qwen3-30B-A3B",
                "tensor_parallel_size": 8,
                "policy": "do_not_kill_unrelated_container",
            },
            "diagnoses": [
                {
                    "id": "vllm_compressed_not_faster_than_baseline",
                    "title": "vLLM compressed path is not faster than baseline",
                    "evidence": "baseline 13.394 tokens/s, QPruner 12.399 tokens/s.",
                    "recommendation": "Measure longer decode and optimize the compressed serving path.",
                },
                {
                    "id": "compressed_native_qpruner_end_to_end_not_faster",
                    "title": "compressed-native QPruner preserves memory but is not end-to-end faster",
                    "evidence": (
                        "compressed-native QPruner qpruner_vs_baseline=0.876x with 76.307% "
                        "targeted storage reduction."
                    ),
                    "recommendation": "Move the packed decode speedup into full-model kernels.",
                },
                {
                    "id": "qpruner_packed_decode_kernel_positive",
                    "title": "QPruner packed decode kernel microbenchmark is latency-positive",
                    "evidence": "QPruner packed decode scaled-code speedup 16.988x.",
                    "recommendation": "Use this as the kernel target for integration work.",
                }
            ],
            "next_optimization_target": {
                "title": "Reduce dense export/cache overhead and measure longer decode",
                "steps": ["Run longer decode slices."],
            },
        },
    )


def test_readiness_report_maps_user_objective_to_evidence(tmp_path):
    readiness = load_readiness_module()
    demo_root = tmp_path / "demo"
    populate_ready_demo(demo_root / "artifacts")

    report = readiness.build_report(demo_root)
    text = readiness.markdown(report)

    assert report["status"] == "PARTIAL_READY"
    assert [section["id"] for section in report["sections"]] == [
        "ascend_910b_baseline",
        "inference_acceleration",
        "finetuning_effect",
        "model_compression",
        "demo_materials",
    ]
    assert "## 1. Ascend 910B Baseline" in text
    assert "CAP/QPruner/RankAdaptor workflows have NPU evidence" in text
    assert "8-card HCCL sync PASS, world_size=8" in text
    assert "sync gate 2000.000, release lag window 0.003s" in text
    assert "compatibility_issues.json" in text
    assert "## 2. Inference Acceleration" in text
    assert (
        "Engineering reference: `baseline` means uncompressed serving/runtime reference, "
        "not a paper baseline."
    ) in text
    assert (
        "Paper baselines: QPruner -> LLM-Pruner; CAP -> SparseGPT, Wanda; "
        "RankAdaptor -> LoRA, AdaLoRA, without recovery."
    ) in text
    assert "baseline 167.142 tokens/s" in text
    assert "QPruner 175.185 tokens/s" in text
    assert "qpruner_vs_baseline=1.048x" in text
    assert "compressed-native torch_npu baseline 160.000 tokens/s" in text
    assert "inference summary best torch fallback/QPruner 456.000 tokens/s" in text
    assert "inference summary native memory QPruner 75.000%, storage 75.000%" in text
    assert "CAP 170.000 tokens/s" in text
    assert "QPruner 210.000 tokens/s" in text
    assert "packed 7, quantized 7, exported dense 0/0" in text
    assert "native memory reductions CAP 66.667%, QPruner 75.000%" in text
    assert "native storage bytes baseline 8192, CAP 2731, QPruner 2048" in text
    assert "native storage reductions CAP 66.663%, QPruner 75.000%" in text
    assert "compressed_native_torch_serving_tiny_qwen3_native_serving_npu.json" in text
    assert "compressed-native-torch-serving-tiny_qwen3_native_serving_npu.md" in text
    assert "vLLM blocked at vllm_attention_selector_api_mismatch" in text
    assert "bottleneck diagnosis ACTIONABLE" in text
    assert "vLLM compressed path is not faster than baseline" in text
    assert "compressed-native QPruner preserves memory but is not end-to-end faster" in text
    assert "compressed-native QPruner qpruner_vs_baseline=0.876x" in text
    assert "QPruner packed decode kernel microbenchmark is latency-positive" in text
    assert "QPruner packed decode scaled-code speedup 16.988x" in text
    assert "resource blocker BLOCKED" in text
    assert "tdqs_qwen3-30b-a3b" in text
    assert "Qwen3-30B-A3B" in text
    assert "do_not_kill_unrelated_container" in text
    assert "inference_bottleneck_report.json" in text
    assert "inference-bottleneck-report.md" in text
    assert "inference_acceleration_summary.json" in text
    assert "inference-acceleration-summary.md" in text
    assert "inference-acceleration-summary.csv" in text
    assert "inference-acceleration-summary.svg" in text
    assert "compressed-native sweep PASS" in text
    assert "Cached long-decode compressed-native path is latency-positive" in text
    assert "QPruner reaches 1.200x baseline at max_new_tokens=16" in text
    assert "cache qpruner delta 0.080x" in text
    assert "long decode qpruner delta 0.220x" in text
    assert "memory-preserving QPruner tiny_short_scaled_code" in text
    assert "strategy scaled_int8_code_matmul, cache mode scaled-code-matmul" in text
    assert "code-cache storage reduction 49.500%" in text
    assert "compressed-native sweep runtime diagnosis QPruner scaling gap 0.449x" in text
    assert "QPruner cache peak +1.000 MB" in text
    assert "fuse packed/quantized decode kernels on NPU" in text
    assert "compressed_native_sweep_summary.json" in text
    assert "compressed-native-cache-long-decode-sweep.md" in text
    assert "compressed-native-cache-long-decode-sweep.csv" in text
    assert "compressed-native-cache-long-decode-sweep.svg" in text
    assert "QPruner packed decode PASS on npu" in text
    assert "runtime packed_nbit_weight_codes" in text
    assert "storage 74.561%" in text
    assert "code-cache storage 49.561%" in text
    assert "uncached 2.000 ms, cache 1.000 ms, dense 0.900 ms" in text
    assert "code-cache 1.200 ms" in text
    assert "code-cache speedup 1.667x" in text
    assert "scaled-code matmul 1.100 ms" in text
    assert "scaled-code speedup 1.818x" in text
    assert "scaled-code peak 2.500 MB" in text
    assert "cache speedup 2.000x" in text
    assert "qpruner_packed_decode_benchmark_tiny_qwen3_qpruner_decode_npu.json" in text
    assert "qpruner-packed-decode-benchmark-tiny_qwen3_qpruner_decode_npu.md" in text
    assert "qpruner_packed_decode_benchmark_qwen3_06b_shape_sweep_npu.json" in text
    assert "qpruner-packed-decode-benchmark-qwen3_06b_shape_sweep_npu.md" in text
    assert "qpruner_packed_decode_benchmark_qwen3_06b_shape_sweep_dtype_preserving_npu.json" in text
    assert "qpruner-packed-decode-benchmark-qwen3_06b_shape_sweep_dtype_preserving_npu.md" in text
    assert "## 3. Fine-Tuning Effect" in text
    assert "Qwen3 LoRA validation delta -0.133" in text
    assert "multicard_qwen_lora_finetune_qwen3_06b_lora_instruction_sync_8card_steps6_20260625.json" in text
    assert "BSLoRA loss delta -0.039" in text
    assert "Qwen3 BSLoRA loss delta -0.289" in text
    assert "Qwen3 BSLoRA sharing 32768 vs 1261568 adapter params" in text
    assert "tiny_qwen_bslora_finetune_qwen3_06b_bslora_npu.json" in text
    assert "## 4. Model Compression" in text
    assert "CAP loss delta -0.134" in text
    assert "QPruner average bits 4.000" in text
    assert "target layers 2/196" in text
    assert "patterned Qwen3 quality sync PASS, world_size=2, pattern self_attn\\.(q_proj|k_proj)$, target layers 4/196" in text
    assert "WANDA loss delta 0.089" in text
    assert "SparseGPT loss delta 0.118" in text
    assert "patterned Qwen3 generate sync PASS, CAP 57.175 tokens/s, QPruner 54.419 tokens/s" in text
    assert "Best memory-quality point under loss delta <= 0.500 is qwen3_06b_quality_pattern_bits6" in text
    assert "qwen-compression-quality-memory-sweep.md" in text
    assert "qwen3_06b_qpruner_sweep_layers4_bits6" in text
    assert "qwen-qpruner-quality-memory-sweep.md" in text
    assert "QPruner scale-quality sweep reaches 16/196 target layers" in text
    assert "loss-derived approximate PPL 403.429 -> 518.013" in text
    assert "full-target QPruner choice accuracy baseline 75.000%, QPruner 87.500%" in text
    assert "choice accuracy target layers 196/196" in text
    assert "QPruner compression time 13.111s" in text
    assert "qwen_compression_choice_accuracy_qwen3_06b_choice_accuracy_npu.json" in text
    assert "qwen_compression_choice_accuracy_qwen3_06b_multitask_choice_fulltarget_qpruner_npu.json" in text
    assert "qwen-compression-choice-accuracy-qwen3_06b_choice_accuracy_npu.md" in text
    assert "qwen-compression-choice-accuracy-qwen3_06b_multitask_choice_fulltarget_qpruner_npu.md" in text
    assert "qwen_llm_pruner_baseline_qwen3_06b_llm_pruner_npu.json" in text
    assert "qwen-llm-pruner-baseline-qwen3_06b_llm_pruner_npu.md" in text
    assert "qwen_qpruner_scale_quality_summary.json" in text
    assert "qwen-qpruner-scale-quality-summary.md" in text
    assert "qwen-qpruner-quality-memory-frontier.svg" in text
    assert "qwen-qpruner-quality-memory-frontier.md" in text
    assert "multicard_qwen_compression_quality_qwen3_06b_quality_pattern_2card_npu.json" in text
    assert "multicard_qwen_compression_generate_qwen3_06b_generate_pattern_2card_npu.json" in text
    assert "multi-card compression sync PASS" in text
    assert "8-card synchronized compression path is ready" in text
    assert "QPruner 240.000 tokens/s total" in text
    assert "QPruner storage reduction 76.316%" in text
    assert "multicard_compression_sync_summary.json" in text
    assert "multicard-compression-sync-summary.md" in text
    assert "multicard-compression-sync-summary.csv" in text
    assert "multicard-compression-sync-summary.svg" in text
    assert "objective_coverage_audit.json" in text
    assert "objective-coverage-audit.md" in text
    assert "compression_memory_report.json" in text
    assert "paper_baseline_coverage_audit.json" in text
    assert "Paper baseline audit PASS" in text
    assert "QPruner LLM-Pruner-style PASS, target layers 196/196" in text
    assert "CAP SparseGPT PASS, target layers 196/196" in text
    assert "RankAdaptor recovery baselines PASS, best lora" in text
    assert (
        "Memory-first compression readout: CAP targeted memory -66.667%, "
        "QPruner targeted memory -75.000%, target layers 2/196"
    ) in text
    assert "QPruner paper baseline LLM-Pruner" in text
    assert "CAP paper baselines SparseGPT, Wanda" in text
    assert "baseline alignment QPruner: pruning baseline LLM-Pruner -> compressed-vs-uncompressed Ascend runtime reference" in text
    assert "baseline alignment CAP: pruning / joint-compression / SVD baselines SparseGPT, Wanda -> compressed-vs-uncompressed Ascend runtime reference" in text
    assert "baseline alignment RankAdaptor: recovery baselines LoRA, AdaLoRA, without recovery -> LoRA/BSLoRA recovery evidence on Ascend" in text
    assert "baseline alignment Engineering runtime baseline: not a paper baseline uncompressed serving export -> runtime sanity, latency, throughput, and memory reference" in text
    assert "memory reductions CAP 66.667%, QPruner 75.000%" in text
    assert "dense export erases storage savings True" in text
    assert "## 5. Demo Materials" in text
    assert "demo-storyboard.md" in text
    assert "ascend-910b-demo-playback-latest.typescript" in text
    assert "Next action" in text


def test_readiness_report_treats_selector_shim_pass_as_inference_ready(tmp_path):
    readiness = load_readiness_module()
    demo_root = tmp_path / "demo"
    populate_ready_demo(demo_root / "artifacts")
    write_json(
        demo_root / "artifacts" / "vllm_serving_probe_tiny_qwen3_serving_export_npu_selector_shim.json",
        {
            "status": "PASS",
            "preload_vllm_ascend_patch": True,
            "preload_vllm_ascend_selector_shim": True,
            "export_run_label": "tiny_qwen3_serving_export_npu",
            "exports": {
                "cap": {"exists": True, "transformers_load": "PASS", "vllm_load": "PASS"},
                "qpruner": {"exists": True, "transformers_load": "PASS", "vllm_load": "PASS"},
            },
            "issue_classes": [],
        },
    )

    report = readiness.build_report(demo_root)
    inference = next(section for section in report["sections"] if section["id"] == "inference_acceleration")
    text = readiness.markdown(report)

    assert report["status"] == "PARTIAL_READY"
    assert inference["status"] == "READY_WITH_BACKEND_GAP"
    assert "selector shim PASS" in text
    assert "Run vLLM latency/throughput benchmark" in text
    assert "vLLM blocked at" not in text
    assert "vllm_serving_probe_tiny_qwen3_serving_export_npu_selector_shim.json" in text


def test_readiness_report_formats_missing_resource_blocker_cleanly(tmp_path):
    readiness = load_readiness_module()
    blocker_text = readiness.resource_blocker_metric({"resource_blocker": {"status": "MISSING"}})

    assert blocker_text == "resource blocker none"


def test_readiness_report_surfaces_failed_vllm_benchmark_root_cause(tmp_path):
    readiness = load_readiness_module()
    demo_root = tmp_path / "demo"
    populate_ready_demo(demo_root / "artifacts")
    write_json(
        demo_root / "artifacts" / "vllm_serving_probe_tiny_qwen3_serving_export_npu_selector_shim.json",
        {
            "status": "PASS",
            "preload_vllm_ascend_patch": True,
            "preload_vllm_ascend_selector_shim": True,
            "export_run_label": "tiny_qwen3_serving_export_npu",
            "exports": {
                "cap": {"exists": True, "transformers_load": "PASS", "vllm_load": "PASS"},
                "qpruner": {"exists": True, "transformers_load": "PASS", "vllm_load": "PASS"},
            },
            "issue_classes": [],
        },
    )
    write_json(
        demo_root / "artifacts" / "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu.json",
        {
            "status": "FAIL",
            "backend": "vllm_ascend_generate",
            "summary": {},
            "exports": {
                "baseline": {
                    "status": "FAIL",
                    "vllm_load": "FAIL",
                    "error_type": "EngineDeadError",
                    "stdout_tail": (
                        "ERROR attention_v1.py line 897\n"
                        "AttributeError: 'list' object has no attribute 'slot_mapping'\n"
                    ),
                },
                "cap": {
                    "status": "FAIL",
                    "vllm_load": "FAIL",
                    "error_type": "EngineDeadError",
                    "stdout_tail": (
                        "ERROR attention_v1.py line 897\n"
                        "AttributeError: 'list' object has no attribute 'slot_mapping'\n"
                    ),
                },
                "qpruner": {
                    "status": "FAIL",
                    "vllm_load": "FAIL",
                    "error_type": "EngineDeadError",
                    "stdout_tail": (
                        "ERROR attention_v1.py line 897\n"
                        "AttributeError: 'list' object has no attribute 'slot_mapping'\n"
                    ),
                },
            },
        },
    )

    report = readiness.build_report(demo_root)
    inference = next(section for section in report["sections"] if section["id"] == "inference_acceleration")
    text = readiness.markdown(report)

    assert report["status"] == "PARTIAL_READY"
    assert inference["status"] == "READY_WITH_BACKEND_GAP"
    assert "vLLM benchmark failed: baseline, CAP, QPruner: EngineDeadError" in text
    assert "AttributeError: 'list' object has no attribute 'slot_mapping'" in text


def test_readiness_report_treats_vllm_benchmark_pass_as_inference_ready(tmp_path):
    readiness = load_readiness_module()
    demo_root = tmp_path / "demo"
    populate_ready_demo(demo_root / "artifacts")
    write_json(
        demo_root / "artifacts" / "vllm_serving_probe_tiny_qwen3_serving_export_npu_selector_shim.json",
        {
            "status": "PASS",
            "preload_vllm_ascend_patch": True,
            "preload_vllm_ascend_selector_shim": True,
            "export_run_label": "tiny_qwen3_serving_export_npu",
            "exports": {
                "cap": {"exists": True, "transformers_load": "PASS", "vllm_load": "PASS"},
                "qpruner": {"exists": True, "transformers_load": "PASS", "vllm_load": "PASS"},
            },
            "issue_classes": [],
        },
    )
    write_json(
        demo_root / "artifacts" / "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu.json",
        {
            "status": "PASS",
            "backend": "vllm_ascend_generate",
            "summary": {
                "best_method": "qpruner",
                "best_tokens_per_s": 222.0,
                "cap_vs_baseline_speedup": 1.05,
                "qpruner_vs_baseline_speedup": 1.11,
            },
            "exports": {
                "baseline": {"tokens_per_s": 200.0, "peak_mem_mb": 512.0},
                "cap": {"tokens_per_s": 210.0, "peak_mem_mb": 512.0},
                "qpruner": {"tokens_per_s": 222.0, "peak_mem_mb": 512.0},
            },
        },
    )

    report = readiness.build_report(demo_root)
    inference = next(section for section in report["sections"] if section["id"] == "inference_acceleration")
    text = readiness.markdown(report)

    assert report["status"] == "READY"
    assert inference["status"] == "READY"
    assert "vLLM benchmark best qpruner 222.000 tokens/s" in text
    assert "qpruner_vs_baseline=1.110x" in text
    assert "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu.json" in text


def test_readiness_report_surfaces_synchronized_vllm_slice(tmp_path):
    readiness = load_readiness_module()
    demo_root = tmp_path / "demo"
    populate_ready_demo(demo_root / "artifacts")
    write_json(
        demo_root / "artifacts" / "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu.json",
        {
            "status": "PASS",
            "backend": "vllm_ascend_generate",
            "summary": {
                "best_method": "qpruner",
                "best_tokens_per_s": 222.0,
                "qpruner_vs_baseline_speedup": 1.11,
            },
            "exports": {
                "baseline": {"tokens_per_s": 200.0},
                "cap": {"tokens_per_s": 210.0},
                "qpruner": {"tokens_per_s": 222.0},
            },
        },
    )
    write_json(
        demo_root / "artifacts" / "multicard_parallel_suite_vllm_metadata_sync_npu_metrics.json",
        {
            "status": "PASS",
            "world_size": 3,
            "cards": [0, 1, 2],
            "start_window_s": 0.001,
            "release_lag_window_s": 0.0,
            "task_counts": {"vllm_serving_benchmark": 3},
            "aggregate": {"best_vllm_tokens_per_s": 13.039, "pass_count": 3, "fail_count": 0},
            "workers": [
                {"rank": 0, "card": 0, "task": "vllm_serving_benchmark", "status": "PASS"},
                {"rank": 1, "card": 1, "task": "vllm_serving_benchmark", "status": "PASS"},
                {"rank": 2, "card": 2, "task": "vllm_serving_benchmark", "status": "PASS"},
            ],
        },
    )
    write_json(
        demo_root / "artifacts" / "multicard_parallel_suite_vllm_metadata_sync_8card_npu_metrics.json",
        {
            "status": "PASS",
            "world_size": 8,
            "cards": list(range(8)),
            "start_window_s": 0.006,
            "release_lag_window_s": 0.001,
            "task_counts": {"vllm_serving_benchmark": 8},
            "aggregate": {"best_vllm_tokens_per_s": 18.25, "pass_count": 8, "fail_count": 0},
            "workers": [
                {"rank": rank, "card": rank, "task": "vllm_serving_benchmark", "status": "PASS"}
                for rank in range(8)
            ],
        },
    )

    report = readiness.build_report(demo_root)
    inference = next(section for section in report["sections"] if section["id"] == "inference_acceleration")
    text = readiness.markdown(report)

    assert report["status"] == "READY"
    assert inference["status"] == "READY"
    assert "vLLM synchronized slice PASS, workers 8, start window 0.006s, best_vllm_tokens_per_s=18.250" in text
    assert "multicard_parallel_suite_vllm_metadata_sync_8card_npu_metrics.json" in text


def test_readiness_report_surfaces_grouped_replay_from_inference_summary(tmp_path):
    readiness = load_readiness_module()
    demo_root = tmp_path / "demo"
    populate_ready_demo(demo_root / "artifacts")
    replay_artifact = (
        "artifacts/multicard_qwen_qpruner_grouped_replay_"
        "qwen3_06b_real_grouped_replay_8card_bits8_releasepacked_npu_b64_i2000_20260625.json"
    )
    monitor_artifact = (
        "artifacts/npu_monitor_"
        "qwen3_06b_real_grouped_replay_8card_bits8_releasepacked_npu_b64_i2000_utilmon_20260625.json"
    )
    write_json(
        demo_root / "artifacts" / "inference_acceleration_summary.json",
        {
            "status": "ACTIONABLE",
            "summary": {
                "best_throughput": {
                    "track": "torch fallback",
                    "method": "qpruner",
                    "method_label": "QPruner",
                    "tokens_per_s": 456.0,
                },
                "best_native_memory": {
                    "method": "qpruner",
                    "method_label": "QPruner",
                    "memory_reduction_pct": 75.0,
                    "storage_reduction_pct": 75.0,
                },
                "grouped_replay": {
                    "status": "PASS",
                    "artifact": replay_artifact,
                    "world_size": 8,
                    "pass_count": 8,
                    "memory_native_worker_count": 8,
                    "code_cache_storage_reduction_pct": 50.0,
                    "best_grouped_speedup": 1.202,
                    "mean_grouped_speedup": 1.147,
                    "released_packed_code_bytes_total": 3523215360.0,
                    "live_compressed_payload_storage_bytes_total": 6272.0,
                    "monitor": {
                        "status": "PASS",
                        "artifact": monitor_artifact,
                        "max_active_process_cards": 8,
                        "samples_with_8_nonzero_aicore": 55,
                    },
                },
            },
            "vllm_bottleneck": {"title": "vLLM compressed path is not faster than baseline"},
        },
    )

    report = readiness.build_report(demo_root)
    inference = next(section for section in report["sections"] if section["id"] == "inference_acceleration")
    text = readiness.markdown(report)

    assert replay_artifact in inference["evidence"]
    assert monitor_artifact in inference["evidence"]
    assert "inference summary grouped replay PASS, world 8, memory-native workers 8/8" in text
    assert "code-cache storage 50.000%" in text
    assert "best grouped speedup 1.202x" in text
    assert "55 all-8 AICore samples" in text
    assert "released packed-code bytes 3523215360.000" in text
    assert "live compressed payload bytes 6272.000" in text


def test_readiness_report_surfaces_8card_qwen_native_memory_profile_summary(tmp_path):
    readiness = load_readiness_module()
    demo_root = tmp_path / "demo"
    populate_ready_demo(demo_root / "artifacts")
    summary_name = "qwen_qpruner_native_profile_8card_bits8_code_releasepacked_long16_summary_20260625.json"
    report_name = "qwen-qpruner-native-profile-8card-bits8-code-releasepacked-long16-summary-20260625.md"
    monitor_name = "npu_monitor_qwen3_06b_native_profile_8card_bits8_code_releasepacked_long16_utilmon_20260625.json"
    write_json(
        demo_root / "artifacts" / summary_name,
        {
            "status": "PASS",
            "world_size": 8,
            "pass_count": 8,
            "memory_native_worker_count": 8,
            "target_layer_limit": 196,
            "targeted_layers_total": 196,
            "max_new_tokens": 16,
            "min_code_cache_storage_reduction_pct": 50.0,
            "released_packed_code_bytes_total": 3523215360,
            "live_compressed_payload_storage_bytes_total": 6272,
            "monitor": {
                "status": "PASS",
                "artifact": f"artifacts/{monitor_name}",
                "samples_with_8_active_process_cards": 22,
                "samples_with_8_nonzero_aicore": 47,
                "max_active_process_cards": 8,
            },
        },
    )
    (demo_root / "reports" / report_name).write_text("# 8-card memory-native profile\n")

    report = readiness.build_report(demo_root)
    inference = next(section for section in report["sections"] if section["id"] == "inference_acceleration")
    text = readiness.markdown(report)

    assert f"artifacts/{summary_name}" in inference["evidence"]
    assert f"artifacts/{monitor_name}" in inference["evidence"]
    assert f"reports/{report_name}" in inference["evidence"]
    assert "Qwen3 8-card memory-native native profile PASS" in text
    assert "workers 8/8" in text
    assert "target layers 196/196" in text
    assert "max_new_tokens 16" in text
    assert "code-cache storage 50.000% reduction" in text
    assert "released packed-code bytes 3523215360.000" in text
    assert "live compressed payload bytes 6272.000" in text
    assert "22 all-8 process samples" in text
    assert "47 all-8 AICore samples" in text


def test_readiness_report_does_not_require_its_own_markdown_before_generation(tmp_path):
    readiness = load_readiness_module()
    demo_root = tmp_path / "demo"
    populate_ready_demo(demo_root / "artifacts")
    (demo_root / "reports" / "demo-readiness-report.md").unlink(missing_ok=True)

    report = readiness.build_report(demo_root)
    materials = next(section for section in report["sections"] if section["id"] == "demo_materials")

    assert materials["status"] == "READY"
    assert "reports/demo-readiness-report.md" in materials["evidence"]
    assert not any("demo-readiness-report.md" in gap for gap in materials["gaps"])


def test_missing_readiness_evidence_is_called_out(tmp_path):
    readiness = load_readiness_module()
    demo_root = tmp_path / "demo"
    write_json(demo_root / "artifacts" / "multicard_sync_summary.json", {"status": "PASS", "world_size": 8})

    report = readiness.build_report(demo_root)
    text = readiness.markdown(report)

    assert report["status"] == "INCOMPLETE"
    assert "Missing evidence" in text
    assert "torch_serving_fallback_tiny_qwen3_serving_fallback_npu.json" in text
    assert "qwen_compression_quality_qwen3_06b_quality_npu.json" in text


def test_write_demo_readiness_cli_writes_json_and_markdown(tmp_path):
    demo_root = tmp_path / "demo"
    populate_ready_demo(demo_root / "artifacts")
    script = Path(__file__).resolve().parents[1] / "scripts" / "write_demo_readiness_report.py"

    proc = subprocess.run(
        [sys.executable, str(script), "--demo-root", str(demo_root)],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "DEMO_READINESS_REPORT " in proc.stdout
    assert (demo_root / "artifacts" / "demo_readiness_report.json").exists()
    assert (demo_root / "reports" / "demo-readiness-report.md").exists()


def test_readiness_lists_current_vllm_hbm_rerun_from_bottleneck_report(tmp_path):
    readiness = load_readiness_module()
    demo_root = tmp_path / "demo"
    artifact = "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu_hbm_parallel_aclpath.json"
    write_json(
        demo_root / "artifacts" / "inference_bottleneck_report.json",
        {
            "status": "ACTIONABLE",
            "current_vllm_rerun": {
                "artifact": artifact,
                "status": "PASS",
                "parallel_methods": True,
                "method_cards": [0, 1, 2],
                "max_new_tokens": 32,
                "best_method": "qpruner",
                "best_tokens_per_s": 395.891,
                "speedups": {"qpruner_vs_baseline": 1.05},
            },
        },
    )

    report = readiness.build_report(demo_root)
    text = readiness.markdown(report)

    assert "current vLLM HBM rerun PASS" in text
    assert f"artifacts/{artifact}" in text
    assert "cards 0,1,2" in text
    assert "best qpruner 395.891 tokens/s" in text
    assert "qpruner_vs_baseline 1.050x" in text


def test_readiness_lists_selected_actual_qwen_shape_sweep(tmp_path):
    readiness = load_readiness_module()
    demo_root = tmp_path / "demo"
    actual_artifact = "qpruner_packed_decode_benchmark_qwen3_06b_actual_projection_bits8_shape_sweep_npu.json"
    actual_report = "qpruner-packed-decode-benchmark-qwen3_06b_actual_projection_bits8_shape_sweep_npu.md"
    native_artifact = (
        "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_"
        "shapeaware_releasepacked_long16_npu.json"
    )
    native_report = "qwen-qpruner-native-profile-qwen3_06b_native_profile_fulltarget_bits8_shapeaware_releasepacked_long16_npu.md"
    write_json(
        demo_root / "artifacts" / "inference_bottleneck_report.json",
        {
            "status": "ACTIONABLE",
            "qpruner_packed_decode_shape_sweep": {"artifact": actual_artifact},
            "qwen_qpruner_native_profile": {
                "artifact": native_artifact,
                "paired_rounds": 3,
                "measurement_basis": "paired_interleaved_median",
                "speedups": {"qpruner_vs_baseline": 0.797},
                "qpruner_code_cache_storage_reduction_pct": 50.0,
            },
        },
    )
    (demo_root / "reports" / actual_report).parent.mkdir(parents=True, exist_ok=True)
    (demo_root / "reports" / actual_report).write_text("# actual shape sweep\n")
    (demo_root / "reports" / native_report).write_text("# stable native profile\n")

    report = readiness.build_report(demo_root)
    text = readiness.markdown(report)

    assert f"artifacts/{actual_artifact}" in text
    assert f"reports/{actual_report}" in text
    assert f"artifacts/{native_artifact}" in text
    assert f"reports/{native_report}" in text
    assert (
        "Qwen3 paired native profile: 0.797x baseline, measurement paired_interleaved_median, "
        "paired_rounds 3, code-cache storage 50.000%"
    ) in text


def test_readiness_lists_qwen_speed_first_native_profile_tradeoff(tmp_path):
    readiness = load_readiness_module()
    demo_root = tmp_path / "demo"
    speed_artifact = (
        "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_"
        "scaled_prebuilt_long16_npu.json"
    )
    speed_report = "qwen-qpruner-native-profile-qwen3_06b_native_profile_fulltarget_bits8_scaled_prebuilt_long16_npu.md"
    write_json(
        demo_root / "artifacts" / "inference_bottleneck_report.json",
        {
            "status": "ACTIONABLE",
            "qwen_qpruner_native_speed_first_profile": {
                "artifact": speed_artifact,
                "speedups": {"qpruner_vs_baseline": 1.392},
                "qpruner_code_cache_storage_reduction_pct": -50.0,
                "qpruner_cached_scaled_code_bytes": 880803840.0,
                "qpruner_prebuild_scaled_code_dtype_cache": True,
            },
        },
    )
    (demo_root / "reports").mkdir(parents=True, exist_ok=True)
    (demo_root / "reports" / speed_report).write_text("# speed first\n")

    report = readiness.build_report(demo_root)
    text = readiness.markdown(report)

    assert f"artifacts/{speed_artifact}" in text
    assert f"reports/{speed_report}" in text
    assert (
        "Qwen3 speed-first native profile: 1.392x baseline, measurement missing, "
        "paired_rounds missing, code-cache storage -50.000%, scaled-code dtype-cache bytes "
        "880803840.000, dense-cache bytes missing, dense-cache budget bytes missing, "
        "prebuild dtype cache True"
    ) in text


def test_readiness_surfaces_objective_audit_inference_actionable_gap(tmp_path):
    readiness = load_readiness_module()
    demo_root = tmp_path / "demo"
    populate_ready_demo(demo_root / "artifacts")
    write_json(
        demo_root / "artifacts" / "objective_coverage_audit.json",
        {
            "status": "PARTIAL_READY",
            "sections": [
                {
                    "id": "ascend_910b_baseline",
                    "title": "Ascend 910B baseline",
                    "status": "READY",
                    "evidence_summary": "baseline ready",
                    "remaining_gap": "No baseline blocker found",
                },
                {
                    "id": "inference_acceleration",
                    "title": "Inference acceleration",
                    "status": "ACTIONABLE",
                    "evidence_summary": (
                        "Qwen3 QPruner memory-first native profile 0.797x baseline, "
                        "code-cache storage 50.000% reduction"
                    ),
                    "remaining_gap": "vLLM compressed path is not faster than baseline",
                },
                {
                    "id": "finetuning_effect",
                    "title": "Fine-tuning effect",
                    "status": "READY",
                    "evidence_summary": "fine-tuning ready",
                    "remaining_gap": "Longer validation and generation-quality rubric are still needed",
                },
                {
                    "id": "model_compression",
                    "title": "Model compression",
                    "status": "READY",
                    "evidence_summary": "compression ready",
                    "remaining_gap": "Full target-layer coverage is present",
                },
                {
                    "id": "demo_materials",
                    "title": "Demo materials",
                    "status": "READY",
                    "evidence_summary": "materials ready",
                    "remaining_gap": "No material gap found",
                },
            ],
        },
    )

    report = readiness.build_report(demo_root)
    inference = next(section for section in report["sections"] if section["id"] == "inference_acceleration")
    text = readiness.markdown(report)

    assert report["status"] == "PARTIAL_READY"
    assert inference["status"] == "READY_WITH_ACTIONABLE_GAP"
    assert (
        "objective audit inference acceleration ACTIONABLE: "
        "vLLM compressed path is not faster than baseline"
    ) in inference["gaps"]
    assert "- Status: `READY_WITH_ACTIONABLE_GAP`" in text
    assert "objective audit inference acceleration ACTIONABLE" in text


def test_readiness_lists_qwen_native_bits_tradeoff(tmp_path):
    readiness = load_readiness_module()
    demo_root = tmp_path / "demo"
    bits4 = (
        "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits4_"
        "code_releasepacked_long16_npu.json"
    )
    bits6 = (
        "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits6_"
        "code_releasepacked_long16_npu.json"
    )
    bits8 = (
        "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_"
        "shapeaware_releasepacked_long16_npu.json"
    )
    write_json(
        demo_root / "artifacts" / "inference_bottleneck_report.json",
        {
            "status": "ACTIONABLE",
            "qwen_qpruner_native_bits_tradeoff": [
                {
                    "bits_label": "bits4",
                    "artifact": bits4,
                    "speedups": {"qpruner_vs_baseline": 0.784},
                    "qpruner_storage_reduction_pct": 75.059,
                    "qpruner_code_cache_storage_reduction_pct": 50.0,
                },
                {
                    "bits_label": "bits6",
                    "artifact": bits6,
                    "speedups": {"qpruner_vs_baseline": 0.803},
                    "qpruner_storage_reduction_pct": 62.5,
                    "qpruner_code_cache_storage_reduction_pct": 50.0,
                },
                {
                    "bits_label": "bits8",
                    "artifact": bits8,
                    "speedups": {"qpruner_vs_baseline": 0.913},
                    "qpruner_storage_reduction_pct": 50.0,
                    "qpruner_code_cache_storage_reduction_pct": 50.0,
                },
            ],
        },
    )

    report = readiness.build_report(demo_root)
    text = readiness.markdown(report)

    assert (
        "Qwen3 native bits tradeoff: bits4 0.784x/75.059% packed storage/50.000% code-cache storage; "
        "bits6 0.803x/62.500% packed storage/50.000% code-cache storage; "
        "bits8 0.913x/50.000% packed storage/50.000% code-cache storage"
    ) in text
    for name in (bits4, bits6, bits8):
        assert f"artifacts/{name}" in text
        assert f"reports/{readiness.qwen_native_profile_report_name(name)}" in text
