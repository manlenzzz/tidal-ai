import importlib.util
import json
from pathlib import Path


def load_report_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "write_demo_progress_report.py"
    spec = importlib.util.spec_from_file_location("write_demo_progress_report", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_compressed_native_sweep_fixture(demo_root: Path) -> None:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    (reports / "compressed-native-cache-long-decode-sweep.md").write_text(
        "# Compressed-Native Cache/Long-Decode Sweep\n"
        "Cached long-decode compressed-native path is latency-positive: "
        "QPruner reaches 1.200x baseline at max_new_tokens=16 while retaining "
        "75.000% targeted storage reduction.\n"
    )
    (reports / "compressed-native-cache-long-decode-sweep.csv").write_text(
        "run_label,max_new_tokens,inference_cache_enabled\n"
    )
    (reports / "compressed-native-cache-long-decode-sweep.svg").write_text(
        "<svg>Compressed-native cache/long-decode sweep</svg>\n"
    )
    (artifacts / "compressed_native_sweep_summary.json").write_text(
        json.dumps(
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
            }
        )
    )


def write_multicard_compression_sync_fixture(demo_root: Path) -> None:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    (reports / "multicard-compression-sync-summary.md").write_text(
        "# Multi-Card Compression Sync Summary\n"
        "8-card synchronized compression path is ready: QPruner 240.000 tokens/s total "
        "(1.250x baseline), all-reduce consistent True, while compressed-native keeps "
        "76.316% QPruner storage reduction.\n"
    )
    (reports / "multicard-compression-sync-summary.csv").write_text("metric,value\n")
    (reports / "multicard-compression-sync-summary.svg").write_text("<svg>Multi-card compression sync</svg>\n")
    (artifacts / "multicard_compression_sync_summary.json").write_text(
        json.dumps(
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
            }
        )
    )


def write_multicard_qwen_qpruner_grouped_replay_fixture(demo_root: Path) -> None:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    (reports / "multicard-qwen-qpruner-grouped-replay-qwen3_06b_real_grouped_replay_2card_npu.md").write_text(
        "# Synchronized Qwen QPruner Grouped Replay\n"
        "| Best role | down_proj |\n"
        "| Best grouped speedup | 1.180x |\n"
        "| Memory-native workers | 2 / 2 |\n"
    )
    (artifacts / "multicard_qwen_qpruner_grouped_replay_qwen3_06b_real_grouped_replay_2card_npu.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "run_label": "qwen3_06b_real_grouped_replay_2card_npu",
                "world_size": 2,
                "cards": [0, 1],
                "launched_synchronously": True,
                "release_lag_window_s": 0.01,
                "aggregate": {
                    "pass_count": 2,
                    "fail_count": 0,
                    "best_role": "down_proj",
                    "best_grouped_speedup": 1.18,
                    "mean_grouped_speedup": 1.165,
                    "min_grouped_speedup": 1.15,
                    "max_group_count": 7,
                    "target_layer_limit": 196,
                    "targeted_layers_total": 196,
                    "quantized_layers": 196,
                    "all_workers_memory_native": True,
                    "memory_native_worker_count": 2,
                    "min_code_cache_storage_reduction_pct": 50.0,
                    "max_grouped_code_cache_bytes": 25165824,
                    "max_grouped_scaled_code_cache_bytes": 0,
                    "max_abs_diff": 0.00195312,
                },
                "workers": [
                    {
                        "rank": 0,
                        "card": 0,
                        "status": "PASS",
                        "payload_status": "PASS",
                        "runtime_s": 90.0,
                        "metrics": {
                            "best_role": "q_proj",
                            "best_grouped_speedup": 1.15,
                            "code_cache_storage_reduction_pct": 50.0,
                        },
                    },
                    {
                        "rank": 1,
                        "card": 1,
                        "status": "PASS",
                        "payload_status": "PASS",
                        "runtime_s": 91.0,
                        "metrics": {
                            "best_role": "down_proj",
                            "best_grouped_speedup": 1.18,
                            "code_cache_storage_reduction_pct": 50.0,
                        },
                    },
                ],
            }
        )
    )


def write_npu_monitor_fixture(demo_root: Path) -> None:
    artifacts = demo_root / "artifacts"
    logs = demo_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "npu-smi-qwen3_06b_real_grouped_replay_8card_utilmon.log").write_text(
        "--- SAMPLE 001 2026-06-24T22:57:06+08:00 ---\n"
    )
    (artifacts / "npu_monitor_qwen3_06b_real_grouped_replay_8card_utilmon.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "run_label": "qwen3_06b_real_grouped_replay_8card_utilmon",
                "monitor_log": str(logs / "npu-smi-qwen3_06b_real_grouped_replay_8card_utilmon.log"),
                "samples": 57,
                "max_active_process_cards": 8,
                "samples_with_8_active_process_cards": 21,
                "samples_with_8_nonzero_aicore": 4,
                "max_aicore_by_card": {
                    "0": 8,
                    "1": 9,
                    "2": 8,
                    "3": 8,
                    "4": 9,
                    "5": 8,
                    "6": 9,
                    "7": 8,
                },
            }
        )
    )


def write_qpruner_packed_decode_fixture(demo_root: Path) -> None:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    (reports / "qpruner-packed-decode-benchmark-tiny_qwen3_qpruner_decode_npu.md").write_text(
        "# QPruner Packed Decode Benchmark\n"
        "Packed QPruner storage saves memory; dense_weight_cache isolates decode overhead.\n"
    )
    (artifacts / "qpruner_packed_decode_benchmark_tiny_qwen3_qpruner_decode_npu.json").write_text(
        json.dumps(
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
                "sequential_scaled_code_matmul_group": {
                    "runtime_strategy": "sequential_scaled_int8_code_matmul_group",
                    "module_count": 4,
                    "latency_ms": 4.8,
                    "peak_mem_mb": 2.8,
                },
                "grouped_scaled_code_matmul": {
                    "runtime_strategy": "grouped_scaled_int8_code_bmm",
                    "module_count": 4,
                    "latency_ms": 2.0,
                    "peak_mem_mb": 2.9,
                    "speedup_vs_sequential_scaled_code": 2.4,
                    "grouped_code_cache_bytes": 16384,
                    "grouped_aux_cache_bytes": 512,
                },
                "grouped_max_abs_diff_vs_sequential_scaled_code": 0.0,
                "next_action": "Fuse packed/quantized decode kernels on NPU.",
            }
        )
    )


def write_choice_accuracy_fixture(demo_root: Path) -> None:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    (reports / "qwen-compression-choice-accuracy-qwen3_06b_choice_accuracy_npu.md").write_text(
        "# Qwen Compression Choice Accuracy\n"
        "| baseline | PASS | 75.000% | 3 / 4 |\n"
        "| cap | PASS | 50.000% | 2 / 4 |\n"
        "| qpruner | PASS | 75.000% | 3 / 4 |\n"
    )
    (reports / "qwen-compression-choice-accuracy-qwen3_06b_choice_accuracy_npu.csv").write_text(
        "method,status,accuracy,correct,total\n"
    )
    (artifacts / "qwen_compression_choice_accuracy_qwen3_06b_choice_accuracy_npu.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "backend": "choice_log_likelihood",
                "target_layer_limit": 64,
                "targeted_layers_total": 196,
                "methods": {
                    "baseline": {"status": "PASS", "accuracy": 0.75, "correct": 3, "total": 4},
                    "cap": {"status": "PASS", "accuracy": 0.5, "correct": 2, "total": 4},
                    "qpruner": {"status": "PASS", "accuracy": 0.75, "correct": 3, "total": 4},
                },
            }
        )
    )
    (reports / "qwen-compression-choice-accuracy-qwen3_06b_multitask_choice_fulltarget_qpruner_npu.md").write_text(
        "# Qwen Compression Choice Accuracy\n"
        "| baseline | __overall__ | PASS | 75.000% | 6 / 8 |\n"
        "| qpruner | __overall__ | PASS | 87.500% | 7 / 8 |\n"
    )
    (reports / "qwen-compression-choice-accuracy-qwen3_06b_multitask_choice_fulltarget_qpruner_npu.csv").write_text(
        "method,task,status,accuracy,correct,total\n"
    )
    (artifacts / "qwen_compression_choice_accuracy_qwen3_06b_multitask_choice_fulltarget_qpruner_npu.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "backend": "choice_log_likelihood",
                "target_layer_limit": 0,
                "targeted_layers_total": 196,
                "task_count": 4,
                "methods": {
                    "baseline": {
                        "status": "PASS",
                        "accuracy": 0.75,
                        "correct": 6,
                        "total": 8,
                        "targeted_layers": 196,
                    },
                    "qpruner": {
                        "status": "PASS",
                        "accuracy": 0.875,
                        "correct": 7,
                        "total": 8,
                        "targeted_layers": 196,
                        "compression_time_s": 13.111,
                    },
                },
            }
        )
    )


def test_write_demo_progress_report_summarizes_inference_and_sync(tmp_path):
    report = load_report_module()
    demo_root = tmp_path / "demo"
    artifacts = demo_root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports = demo_root / "reports"
    reports.mkdir()
    write_multicard_compression_sync_fixture(demo_root)
    write_multicard_qwen_qpruner_grouped_replay_fixture(demo_root)
    write_npu_monitor_fixture(demo_root)
    write_qpruner_packed_decode_fixture(demo_root)
    write_choice_accuracy_fixture(demo_root)
    (reports / "qwen-qpruner-quality-memory-sweep.md").write_text(
        "Best memory-quality point under loss delta <= 0.500 is "
        "qwen3_06b_qpruner_sweep_layers4_8_16_npu_layers4_bits6: "
        "66.667% targeted memory reduction, loss delta -0.224, average bits 5.333.\n"
    )
    (reports / "qwen-qpruner-scale-quality-summary.md").write_text(
        "QPruner scale-quality sweep reaches 16/196 target layers (8.163% coverage); "
        "best under loss delta <= 0.500 is qwen3_06b_qpruner_sweep_layers16_bits6 "
        "with 62.500% targeted memory reduction and loss-derived approximate PPL "
        "403.429 -> 518.013 (28.403%).\n"
    )
    (reports / "qwen-qpruner-scale-quality-summary.csv").write_text(
        "run_label,status,target_layer_limit,targeted_layers_total,coverage_pct,average_bits\n"
    )
    (artifacts / "qwen_qpruner_scale_quality_summary.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "readout": (
                    "QPruner scale-quality sweep reaches 16/196 target layers (8.163% coverage); "
                    "best under loss delta <= 0.500 is qwen3_06b_qpruner_sweep_layers16_bits6 "
                    "with 62.500% targeted memory reduction and loss-derived approximate PPL "
                    "403.429 -> 518.013 (28.403%)."
                ),
                "best_under_loss_delta": {
                    "run_label": "qwen3_06b_qpruner_sweep_layers16_bits6",
                    "target_layer_limit": 16,
                    "targeted_layers_total": 196,
                    "coverage_pct": 8.163,
                    "memory_reduction_pct": 62.5,
                    "baseline_ppl": 403.429,
                    "qpruner_ppl": 518.013,
                    "ppl_delta_pct": 28.403,
                },
            }
        )
    )
    (reports / "objective-coverage-audit.md").write_text(
        "# TIDAL-AI Ascend Objective Coverage Audit\n"
        "Overall status: `PARTIAL_READY`\n"
        "| Model compression | PARTIAL_READY | QPruner storage reduction 76.316% | Scale target-layer coverage |\n"
    )
    (artifacts / "objective_coverage_audit.json").write_text(
        json.dumps(
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
            }
        )
    )
    (reports / "qwen-qpruner-quality-memory-frontier.svg").write_text("<svg></svg>\n")
    (reports / "qwen-qpruner-quality-memory-frontier.md").write_text(
        "![QPruner quality-memory frontier](qwen-qpruner-quality-memory-frontier.svg)\n"
    )
    (reports / "inference-acceleration-summary.md").write_text(
        "# Inference Acceleration Summary\n"
        "Best throughput point: `torch fallback / QPruner` at 456.000 tokens/s.\n"
        "compressed-native preserves quantized/pruned modules with dense export `False`.\n"
    )
    (reports / "inference-acceleration-summary.csv").write_text("track,method,tokens_per_s\n")
    (reports / "inference-acceleration-summary.svg").write_text("<svg>Inference acceleration</svg>\n")
    (artifacts / "inference_acceleration_summary.json").write_text(
        json.dumps(
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
                        "world_size": 8,
                        "pass_count": 8,
                        "memory_native_worker_count": 8,
                        "code_cache_storage_reduction_pct": 50.0,
                        "best_grouped_speedup": 1.189,
                        "released_packed_code_bytes_total": 3523215360.0,
                        "live_compressed_payload_storage_bytes_total": 6272.0,
                        "monitor": {
                            "max_active_process_cards": 8,
                            "samples_with_8_nonzero_aicore": 55,
                        },
                    },
                },
                "vllm_bottleneck": {"title": "vLLM compressed path is not faster than baseline"},
            }
        )
    )
    write_compressed_native_sweep_fixture(demo_root)
    (artifacts / "ascend_inference_tiny_qwen3_torch_npu.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "backend": "torch",
                "device": "npu",
                "dtype": "torch.float16",
                "latency_ms": 243.1,
                "tokens_per_s": 65.8,
                "peak_mem_mb": 16.6,
                "model_id": "TinyQwen3-Offline",
            }
        )
    )
    (artifacts / "multicard_sync_summary.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "world_size": 8,
                "backend": "hccl",
                "ranks": [
                    {
                        "rank": rank,
                        "status": "PASS",
                        "device": f"npu:{rank}",
                        "max_grad_delta": 0.0,
                        "max_param_delta": 0.0,
                    }
                    for rank in range(8)
                ],
            }
        )
    )
    (artifacts / "rankadaptor_lora_sync_summary.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "world_size": 8,
                "backend": "hccl",
                "trainable_adapter_params": 256,
                "rank_allocation": {"q_proj": 1, "v_proj": 2},
                "ranks": [
                    {
                        "rank": rank,
                        "status": "PASS",
                        "device": f"npu:{rank}",
                        "initial_loss": 4.17,
                        "final_loss": 4.15,
                        "max_adapter_delta": 0.0,
                    }
                    for rank in range(8)
                ],
            }
        )
    )
    (artifacts / "tiny_qwen_lora_finetune_tiny_qwen3_lora_npu.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "model_id": "TinyQwen3-Offline",
                "device": "npu",
                "dtype": "torch.float16",
                "trainable_adapter_params": 5760,
                "initial_loss": 5.56,
                "final_loss": 5.39,
                "loss_delta": -0.17,
                "peak_mem_mb": 21.1,
                "before_generate": "hello world tok_1",
                "after_generate": "hello world tok_2",
            }
        )
    )
    (artifacts / "tiny_qwen_bslora_finetune_tiny_qwen3_bslora_npu.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "method": "bslora_shared_lora",
                "model_id": "TinyQwen3-Offline",
                "device": "npu",
                "dtype": "torch.float16",
                "trainable_adapter_params": 4096,
                "unshared_adapter_params": 7168,
                "unique_adapter_tensors": 6,
                "target_module_count": 7,
                "initial_loss": 5.52,
                "final_loss": 5.41,
                "loss_delta": -0.11,
                "peak_mem_mb": 21.5,
                "before_generate": "hello world tok_3",
                "after_generate": "hello world tok_4",
                "shared_groups": {
                    "attn_hidden_64x64": {"modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
                    "mlp_up_64x128": {"modules": ["gate_proj", "up_proj"]},
                },
            }
        )
    )
    (artifacts / "multicard_tiny_qwen_bslora_finetune_tiny_qwen3_bslora_sync_npu.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "method": "bslora_shared_lora",
                "world_size": 8,
                "backend": "hccl",
                "trainable_adapter_params": 4096,
                "unshared_adapter_params": 7168,
                "aggregate": {
                    "distributed_reduce_consistent": True,
                    "adapter_sync_consistent": True,
                    "initial_loss_avg": 5.52,
                    "final_loss_avg": 5.40,
                    "loss_delta_avg": -0.12,
                    "pass_count": 8,
                    "adapter_sync_pass_count": 8,
                },
                "ranks": [
                    {
                        "rank": rank,
                        "status": "PASS",
                        "device": f"npu:{rank}",
                        "initial_loss": 5.52,
                        "final_loss": 5.40,
                        "max_adapter_delta": 0.0,
                    }
                    for rank in range(8)
                ],
            }
        )
    )
    (artifacts / "model_inventory.json").write_text(
        json.dumps(
            {
                "status": "TARGET_MODEL_FOUND",
                "recommended_next_action": "Place a modern local model under /mnt/nvme/622/models.",
                "target_candidates": [
                    {
                        "model_id": "Qwen/Qwen3.5-0.8B",
                        "status": "FOUND",
                        "path": "/mnt/nvme/622/models/Qwen3.5-0.8B",
                        "parameter_mb": 1666.014,
                    },
                    {"model_id": "Qwen/Qwen3-0.6B", "status": "MISSING", "path": None},
                    {"model_id": "Qwen/Qwen2.5-0.5B-Instruct", "status": "MISSING", "path": None},
                ],
                "tiny_fixtures": [
                    {
                        "name": "TinyQwen3-Offline",
                        "status": "FOUND",
                        "path": "/mnt/nvme/622/models/TinyQwen3-Offline",
                        "parameter_mb": 0.72,
                    }
                ],
            }
        )
    )
    (artifacts / "model_snapshot_qwen35_08b.json").write_text(
        json.dumps(
            {
                "status": "FOUND",
                "model_id": "Qwen/Qwen3.5-0.8B",
                "model_path": "/mnt/nvme/622/models/Qwen3.5-0.8B",
                "parameter_mb": 1666.014,
                "download_attempted": False,
            }
        )
    )
    (artifacts / "ascend_inference_qwen35_08b_torch_npu.json").write_text(
        json.dumps(
            {
                "status": "BACKEND_ERROR",
                "model_id": "Qwen/Qwen3.5-0.8B",
                "model_path": "/mnt/nvme/622/models/Qwen3.5-0.8B",
                "backend": "torch",
                "error_type": "ValueError",
                "error": "The checkpoint you are trying to load has model type `qwen3_5` but Transformers does not recognize this architecture.",
            }
        )
    )
    (artifacts / "model_snapshot_qwen3_06b.json").write_text(
        json.dumps(
            {
                "status": "DOWNLOADED",
                "provider": "modelscope",
                "model_id": "Qwen/Qwen3-0.6B",
                "model_path": "/mnt/nvme/622/models/Qwen3-0.6B",
                "parameter_mb": 1433.659,
                "download_attempted": True,
            }
        )
    )
    (artifacts / "ascend_inference_qwen3_06b_torch_npu.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "model_id": "Qwen/Qwen3-0.6B",
                "model_path": "/mnt/nvme/622/models/Qwen3-0.6B",
                "backend": "torch",
                "device": "npu",
                "dtype": "torch.float16",
                "batch_size": 2,
                "prompt_tokens": 28,
                "generated_tokens": 16,
                "latency_ms": 1341.941,
                "tokens_per_s": 11.923,
                "peak_mem_mb": 1159.8,
            }
        )
    )
    (artifacts / "tiny_qwen_compression_tiny_qwen3_compression_npu.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "model_id": "TinyQwen3-Offline",
                "device": "npu",
                "dtype": "torch.float16",
                "baseline": {"latency_ms": 10.0, "tokens_per_s": 1600.0, "loss": 5.5},
                "cap": {
                    "status": "PASS",
                    "latency_ms": 9.0,
                    "tokens_per_s": 1777.8,
                    "loss_delta": 0.02,
                    "targeted_compression_ratio": 2.5,
                    "targeted_param_reduction_pct": 60.0,
                },
                "qpruner": {
                    "status": "PASS",
                    "latency_ms": 11.0,
                    "tokens_per_s": 1454.5,
                    "loss_delta": 0.01,
                    "average_bits": 3.75,
                    "quantized_layers": 7,
                },
                "peak_mem_mb": 22.0,
            }
        )
    )
    (artifacts / "tiny_qwen_compression_generate_tiny_qwen3_generate_serving_export_npu.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "backend": "torch_generate",
                "model_id": "TinyQwen3-Offline",
                "device": "npu",
                "dtype": "torch.float16",
                "serving_dense_export": True,
                "baseline": {"latency_ms": 22.3, "tokens_per_s": 358.0},
                "cap": {
                    "status": "PASS",
                    "latency_ms": 22.1,
                    "tokens_per_s": 362.0,
                    "latency_speedup": 1.011,
                    "targeted_compression_ratio": 19.0,
                    "cache_modules": 7,
                    "exported_dense_linears": 7,
                },
                "qpruner": {
                    "status": "PASS",
                    "latency_ms": 21.5,
                    "tokens_per_s": 372.4,
                    "latency_speedup": 1.04,
                    "average_bits": 3.789,
                    "cache_modules": 7,
                    "exported_dense_linears": 7,
                },
                "peak_mem_mb": 22.4,
            }
        )
    )
    (artifacts / "multicard_tiny_qwen_generate_tiny_qwen3_multicard_generate_npu.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "world_size": 8,
                "backend": "hccl",
                "aggregate": {
                    "distributed_reduce_consistent": True,
                    "baseline_tokens_per_s_total": 2800.0,
                    "cap_tokens_per_s_total": 2920.0,
                    "qpruner_tokens_per_s_total": 3000.0,
                    "pass_count": 8,
                },
                "ranks": [
                    {
                        "rank": rank,
                        "status": "PASS",
                        "device": f"npu:{rank}",
                        "benchmark": {
                            "baseline": {"tokens_per_s": 350.0},
                            "cap": {"tokens_per_s": 365.0},
                            "qpruner": {"tokens_per_s": 375.0},
                        },
                    }
                    for rank in range(8)
                ],
            }
        )
    )
    (artifacts / "multicard_qwen_inference_qwen3_06b_npu.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "model_id": "Qwen/Qwen3-0.6B",
                "model_path": "/mnt/nvme/622/models/Qwen3-0.6B",
                "world_size": 8,
                "backend": "hccl",
                "device_request": "npu",
                "aggregate": {
                    "distributed_reduce_consistent": True,
                    "tokens_per_s_total": 95.384,
                    "generated_tokens_total": 128,
                    "peak_mem_mb_total": 9278.4,
                    "pass_count": 8,
                },
                "ranks": [
                    {
                        "rank": rank,
                        "status": "PASS",
                        "device": f"npu:{rank}",
                        "benchmark": {
                            "latency_ms": 1341.941,
                            "tokens_per_s": 11.923,
                            "generated_tokens": 16,
                            "peak_mem_mb": 1159.8,
                        },
                    }
                    for rank in range(8)
                ],
            }
        )
    )
    (artifacts / "qwen_compression_generate_qwen3_06b_generate_npu.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "backend": "torch_generate",
                "model_id": "Qwen/Qwen3-0.6B",
                "model_path": "/mnt/nvme/622/models/Qwen3-0.6B",
                "device": "npu",
                "dtype": "torch.float16",
                "target_layer_limit": 2,
                "targeted_layers_total": 196,
                "serving_dense_export": True,
                "inference_cache_enabled": True,
                "baseline": {"status": "PASS", "latency_ms": 120.0, "tokens_per_s": 66.667, "targeted_layers": 2},
                "cap": {
                    "status": "PASS",
                    "latency_ms": 110.0,
                    "tokens_per_s": 72.727,
                    "latency_speedup": 1.091,
                    "targeted_layers": 2,
                    "targeted_compression_ratio": 24.0,
                    "cache_modules": 2,
                    "exported_dense_linears": 2,
                },
                "qpruner": {
                    "status": "PASS",
                    "latency_ms": 108.0,
                    "tokens_per_s": 74.074,
                    "latency_speedup": 1.111,
                    "targeted_layers": 2,
                    "average_bits": 4.0,
                    "cache_modules": 2,
                    "exported_dense_linears": 2,
                },
                "peak_mem_mb": 1200.0,
            }
        )
    )
    (artifacts / "qwen_compression_quality_qwen3_06b_quality_npu.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "backend": "torch_forward",
                "model_id": "Qwen/Qwen3-0.6B",
                "model_path": "/mnt/nvme/622/models/Qwen3-0.6B",
                "device": "npu",
                "dtype": "torch.float16",
                "target_layer_limit": 2,
                "targeted_layers_total": 196,
                "baseline": {
                    "status": "PASS",
                    "loss": 11.5,
                    "latency_ms": 120.0,
                    "tokens_per_s": 66.667,
                    "targeted_layers": 2,
                },
                "cap": {
                    "status": "PASS",
                    "loss": 11.625,
                    "loss_delta": 0.125,
                    "latency_ms": 110.0,
                    "tokens_per_s": 72.727,
                    "latency_speedup": 1.091,
                    "targeted_layers": 2,
                    "targeted_compression_ratio": 24.0,
                    "compression_time_s": 1.2,
                },
                "wanda": {
                    "status": "PASS",
                    "loss": 11.45,
                    "loss_delta": -0.05,
                    "latency_ms": 112.0,
                    "tokens_per_s": 71.429,
                    "latency_speedup": 1.071,
                    "targeted_layers": 2,
                    "targeted_sparsity": 0.5,
                    "targeted_param_reduction_pct": 50.0,
                    "compression_time_s": 0.2,
                },
                "sparsegpt": {
                    "status": "PASS",
                    "loss": 11.611,
                    "loss_delta": 0.111,
                    "latency_ms": 114.0,
                    "tokens_per_s": 70.175,
                    "latency_speedup": 1.053,
                    "targeted_layers": 2,
                    "targeted_sparsity": 0.5,
                    "targeted_param_reduction_pct": 50.0,
                    "compression_time_s": 0.3,
                },
                "qpruner": {
                    "status": "PASS",
                    "loss": 11.5625,
                    "loss_delta": 0.0625,
                    "latency_ms": 108.0,
                    "tokens_per_s": 74.074,
                    "latency_speedup": 1.111,
                    "targeted_layers": 2,
                    "average_bits": 4.0,
                    "compression_time_s": 0.4,
                },
                "peak_mem_mb": 1200.0,
            }
        )
    )
    (artifacts / "multicard_qwen_compression_quality_qwen3_06b_quality_sync_npu.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "model_id": "Qwen/Qwen3-0.6B",
                "model_path": "/mnt/nvme/622/models/Qwen3-0.6B",
                "world_size": 8,
                "backend": "hccl",
                "device_request": "npu",
                "target_layer_limit": 2,
                "targeted_layers_total": 196,
                "aggregate": {
                    "distributed_reduce_consistent": True,
                    "baseline_loss_avg": 11.5,
                    "cap_loss_delta_avg": 0.125,
                    "wanda_loss_delta_avg": -0.05,
                    "sparsegpt_loss_delta_avg": 0.111,
                    "qpruner_loss_delta_avg": 0.0625,
                    "baseline_tokens_per_s_total": 70.0,
                    "cap_tokens_per_s_total": 210.0,
                    "wanda_tokens_per_s_total": 214.0,
                    "sparsegpt_tokens_per_s_total": 212.0,
                    "qpruner_tokens_per_s_total": 224.0,
                    "peak_mem_mb_total": 27914.4,
                    "pass_count": 8,
                },
                "ranks": [
                    {
                        "rank": rank,
                        "status": "PASS",
                        "device": f"npu:{rank}",
                        "benchmark": {
                            "baseline": {"loss": 11.5, "tokens_per_s": 8.75},
                            "cap": {"loss_delta": 0.125, "tokens_per_s": 26.25},
                            "wanda": {"loss_delta": -0.05, "tokens_per_s": 26.75},
                            "sparsegpt": {"loss_delta": 0.111, "tokens_per_s": 26.5},
                            "qpruner": {"loss_delta": 0.0625, "tokens_per_s": 28.0},
                            "peak_mem_mb": 3489.3,
                        },
                    }
                    for rank in range(8)
                ],
            }
        )
    )
    (artifacts / "multicard_qwen_lora_finetune_qwen3_06b_lora_sync_npu.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "method": "rankadaptor_lora",
                "model_id": "Qwen/Qwen3-0.6B",
                "model_path": "/mnt/nvme/622/models/Qwen3-0.6B",
                "world_size": 8,
                "backend": "hccl",
                "device_request": "npu",
                "target_module_limit": 2,
                "target_module_count": 2,
                "targeted_modules_total": 196,
                "trainable_adapter_params": 4096,
                "train_mode": "instruction",
                "validation_samples": [
                    {
                        "prompt": "What does LoRA train?",
                        "target": "LoRA freezes the base model and trains small adapter matrices.",
                    }
                ],
                "before_generate": "What does LoRA train? dense layers",
                "after_generate": "What does LoRA train? adapter matrices",
                "aggregate": {
                    "distributed_reduce_consistent": True,
                    "adapter_sync_consistent": True,
                    "initial_loss_avg": 12.5,
                    "final_loss_avg": 12.1,
                    "loss_delta_avg": -0.4,
                    "validation_initial_loss_avg": 13.0,
                    "validation_final_loss_avg": 12.2,
                    "validation_loss_delta_avg": -0.8,
                    "peak_mem_mb_total": 9600.0,
                    "pass_count": 8,
                    "adapter_sync_pass_count": 8,
                },
                "ranks": [
                    {
                        "rank": rank,
                        "status": "PASS",
                        "device": f"npu:{rank}",
                        "initial_loss": 12.5,
                        "final_loss": 12.1,
                        "max_adapter_delta": 0.0,
                        "peak_mem_mb": 1200.0,
                        "trainable_adapter_params": 4096,
                    }
                    for rank in range(8)
                ],
            }
        )
    )
    (artifacts / "multicard_qwen_compression_generate_qwen3_06b_compression_generate_npu.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "model_id": "Qwen/Qwen3-0.6B",
                "model_path": "/mnt/nvme/622/models/Qwen3-0.6B",
                "world_size": 8,
                "backend": "hccl",
                "device_request": "npu",
                "target_layer_limit": 2,
                "targeted_layers_total": 196,
                "aggregate": {
                    "distributed_reduce_consistent": True,
                    "baseline_tokens_per_s_total": 70.0,
                    "cap_tokens_per_s_total": 210.0,
                    "qpruner_tokens_per_s_total": 224.0,
                    "peak_mem_mb_total": 27914.4,
                    "pass_count": 8,
                },
                "ranks": [
                    {
                        "rank": rank,
                        "status": "PASS",
                        "device": f"npu:{rank}",
                        "benchmark": {
                            "baseline": {"tokens_per_s": 8.75},
                            "cap": {"tokens_per_s": 26.25},
                            "qpruner": {"tokens_per_s": 28.0},
                            "peak_mem_mb": 3489.3,
                        },
                    }
                    for rank in range(8)
                ],
            }
        )
    )
    (artifacts / "vllm_serving_probe_tiny_qwen3_serving_export_npu.json").write_text(
        json.dumps(
            {
                "status": "FAIL",
                "vllm_available": True,
                "gpu_memory_utilization": 0.05,
                "preload_vllm_ascend_patch": False,
                "export_run_label": "tiny_qwen3_serving_export_npu",
                "exports": {
                    "cap": {"exists": True, "transformers_load": "PASS", "vllm_load": "FAIL"},
                    "qpruner": {"exists": True, "transformers_load": "PASS", "vllm_load": "FAIL"},
                },
                "issue_classes": [{"id": "vllm_attention_selector_api_mismatch"}],
                "acl_diagnostics": {
                    "import_acl": "FOUND",
                    "acl_origin": "/usr/local/Ascend/cann-8.5.1/python/site-packages/acl.so",
                },
                "package_diagnostics": {
                    "vllm": {"version": "0.18.0+empty"},
                    "vllm-ascend": {"version": "0.1.dev2924+ge5f7e2f43"},
                },
                "api_diagnostics": {
                    "status": "MISMATCH",
                    "missing_patch_config_fields": ["use_per_head_quant_scales"],
                    "missing_patch_get_attn_backend_parameters": ["use_per_head_quant_scales", "num_heads"],
                },
            }
        )
    )
    (artifacts / "vllm_serving_probe_tiny_qwen3_serving_export_npu_preload_patch.json").write_text(
        json.dumps(
            {
                "status": "FAIL",
                "vllm_available": True,
                "gpu_memory_utilization": 0.05,
                "preload_vllm_ascend_patch": True,
                "export_run_label": "tiny_qwen3_serving_export_npu",
                "exports": {
                    "cap": {"exists": True, "transformers_load": "PASS", "vllm_load": "FAIL"},
                    "qpruner": {"exists": True, "transformers_load": "PASS", "vllm_load": "FAIL"},
                },
                "issue_classes": [{"id": "vllm_attention_selector_api_mismatch"}],
                "acl_diagnostics": {"import_acl": "FOUND"},
                "api_diagnostics": {
                    "status": "MISMATCH",
                    "missing_patch_config_fields": ["use_per_head_quant_scales"],
                    "missing_patch_get_attn_backend_parameters": ["use_per_head_quant_scales", "num_heads"],
                },
            }
        )
    )
    (artifacts / "vllm_serving_probe_tiny_qwen3_serving_export_npu_selector_shim.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "vllm_available": True,
                "gpu_memory_utilization": 0.05,
                "preload_vllm_ascend_patch": True,
                "preload_vllm_ascend_selector_shim": True,
                "preload_vllm_ascend_metadata_shim": True,
                "metadata_shim": {"status": "INSTALLED"},
                "export_run_label": "tiny_qwen3_serving_export_npu",
                "exports": {
                    "cap": {"exists": True, "transformers_load": "PASS", "vllm_load": "PASS"},
                    "qpruner": {"exists": True, "transformers_load": "PASS", "vllm_load": "PASS"},
                },
                "issue_classes": [],
                "acl_diagnostics": {"import_acl": "FOUND"},
                "api_diagnostics": {
                    "status": "MISMATCH",
                    "missing_patch_config_fields": ["use_per_head_quant_scales"],
                    "missing_patch_get_attn_backend_parameters": ["use_per_head_quant_scales", "num_heads"],
                },
            }
        )
    )
    (artifacts / "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "backend": "vllm_ascend_generate",
                "dtype": "float16",
                "gpu_memory_utilization": 0.05,
                "preload_vllm_ascend_patch": True,
                "preload_vllm_ascend_selector_shim": True,
                "export_run_label": "tiny_qwen3_serving_export_npu",
                "summary": {
                    "best_method": "qpruner",
                    "best_tokens_per_s": 222.0,
                    "cap_vs_baseline_speedup": 1.05,
                    "qpruner_vs_baseline_speedup": 1.11,
                    "cap_vs_qpruner_speedup": 0.946,
                },
                "exports": {
                    "baseline": {
                        "status": "PASS",
                        "exists": True,
                        "vllm_load": "PASS",
                        "load_time_s": 1.2,
                        "latency_ms": 30.0,
                        "tokens_per_s": 200.0,
                        "peak_mem_mb": 512.0,
                    },
                    "cap": {
                        "status": "PASS",
                        "exists": True,
                        "vllm_load": "PASS",
                        "load_time_s": 1.1,
                        "latency_ms": 28.0,
                        "tokens_per_s": 210.0,
                        "peak_mem_mb": 512.0,
                    },
                    "qpruner": {
                        "status": "PASS",
                        "exists": True,
                        "vllm_load": "PASS",
                        "load_time_s": 1.0,
                        "latency_ms": 27.0,
                        "tokens_per_s": 222.0,
                        "peak_mem_mb": 512.0,
                    },
                },
            }
        )
    )
    (artifacts / "torch_serving_fallback_tiny_qwen3_serving_fallback_npu.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "backend": "torch_generate",
                "device": "npu",
                "export_run_label": "tiny_qwen3_serving_export_npu",
                "summary": {
                    "best_method": "qpruner",
                    "best_tokens_per_s": 456.0,
                    "cap_vs_baseline_speedup": 1.08,
                    "qpruner_vs_baseline_speedup": 1.14,
                    "cap_vs_qpruner_speedup": 0.95,
                },
                "exports": {
                    "baseline": {
                        "status": "PASS",
                        "exists": True,
                        "transformers_load": "PASS",
                        "latency_ms": 24.0,
                        "tokens_per_s": 400.0,
                        "peak_mem_mb": 39.0,
                    },
                    "cap": {
                        "status": "PASS",
                        "exists": True,
                        "transformers_load": "PASS",
                        "latency_ms": 20.0,
                        "tokens_per_s": 432.0,
                        "peak_mem_mb": 40.0,
                    },
                    "qpruner": {
                        "status": "PASS",
                        "exists": True,
                        "transformers_load": "PASS",
                        "latency_ms": 18.0,
                        "tokens_per_s": 456.0,
                        "peak_mem_mb": 42.0,
                    },
                },
            }
        )
    )
    (artifacts / "multicard_parallel_suite_demo_parallel_suite_npu.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "run_label": "demo_parallel_suite_npu",
                "world_size": 4,
                "cards": [0, 1, 2, 3],
                "launched_synchronously": True,
                "start_window_s": 0.084,
                "task_counts": {
                    "workflow_cap": 1,
                    "workflow_qpruner": 1,
                    "rankadaptor": 1,
                    "torch_serving_fallback": 1,
                },
                "aggregate": {
                    "pass_count": 4,
                    "fail_count": 0,
                    "best_tokens_per_s": 456.0,
                },
                "workers": [
                    {
                        "rank": 0,
                        "card": 0,
                        "task": "workflow_cap",
                        "status": "PASS",
                        "payload_status": "PASS",
                        "runtime_s": 1.0,
                        "payload_json": "artifacts/workflow_cap.json",
                        "log_path": "logs/workflow_cap.log",
                    },
                    {
                        "rank": 1,
                        "card": 1,
                        "task": "workflow_qpruner",
                        "status": "PASS",
                        "payload_status": "PASS",
                        "runtime_s": 1.1,
                        "payload_json": "artifacts/workflow_qpruner.json",
                        "log_path": "logs/workflow_qpruner.log",
                    },
                    {
                        "rank": 2,
                        "card": 2,
                        "task": "rankadaptor",
                        "status": "PASS",
                        "payload_status": "PASS",
                        "runtime_s": 1.2,
                        "payload_json": "artifacts/rankadaptor.json",
                        "log_path": "logs/rankadaptor.log",
                    },
                    {
                        "rank": 3,
                        "card": 3,
                        "task": "torch_serving_fallback",
                        "status": "PASS",
                        "payload_status": "PASS",
                        "runtime_s": 1.3,
                        "payload_json": "artifacts/serving_fallback.json",
                        "log_path": "logs/serving_fallback.log",
                        "metrics": {"best_tokens_per_s": 456.0},
                    },
                ],
            }
        )
    )
    (artifacts / "multicard_parallel_suite_vllm_metadata_sync_npu_metrics.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "run_label": "vllm_metadata_sync_npu_metrics",
                "world_size": 3,
                "cards": [0, 1, 2],
                "launched_synchronously": True,
                "sync_start_target_ts": 3000.0,
                "start_window_s": 0.001,
                "release_lag_window_s": 0.0,
                "task_counts": {"vllm_serving_benchmark": 3},
                "aggregate": {
                    "pass_count": 3,
                    "fail_count": 0,
                    "best_tokens_per_s": 13.039,
                    "best_vllm_tokens_per_s": 13.039,
                },
                "workers": [
                    {
                        "rank": 0,
                        "card": 0,
                        "task": "vllm_serving_benchmark",
                        "status": "PASS",
                        "payload_status": "PASS",
                        "runtime_s": 55.862,
                        "payload_json": "artifacts/vllm_serving_benchmark_vllm_metadata_sync_npu_metrics_rank0_qpruner_npu0.json",
                        "metrics": {
                            "method": "qpruner",
                            "tokens_per_s": 13.039,
                            "latency_ms": 153.387,
                            "metadata_shim_status": "ALREADY_INSTALLED",
                            "metadata_backend_forward_status": "INSTALLED",
                        },
                    },
                    {
                        "rank": 1,
                        "card": 1,
                        "task": "vllm_serving_benchmark",
                        "status": "PASS",
                        "payload_status": "PASS",
                        "runtime_s": 59.073,
                        "payload_json": "artifacts/vllm_serving_benchmark_vllm_metadata_sync_npu_metrics_rank1_cap_npu1.json",
                        "metrics": {
                            "method": "cap",
                            "tokens_per_s": 11.478,
                            "latency_ms": 174.243,
                            "metadata_shim_status": "ALREADY_INSTALLED",
                            "metadata_backend_forward_status": "INSTALLED",
                        },
                    },
                    {
                        "rank": 2,
                        "card": 2,
                        "task": "vllm_serving_benchmark",
                        "status": "PASS",
                        "payload_status": "PASS",
                        "runtime_s": 59.073,
                        "payload_json": "artifacts/vllm_serving_benchmark_vllm_metadata_sync_npu_metrics_rank2_baseline_npu2.json",
                        "metrics": {
                            "method": "baseline",
                            "tokens_per_s": 12.184,
                            "latency_ms": 164.149,
                            "metadata_shim_status": "ALREADY_INSTALLED",
                            "metadata_backend_forward_status": "INSTALLED",
                        },
                    },
                ],
            }
        )
    )
    (artifacts / "multicard_parallel_suite_vllm_metadata_sync_8card_npu_metrics.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "run_label": "vllm_metadata_sync_8card_npu_metrics",
                "world_size": 8,
                "cards": list(range(8)),
                "launched_synchronously": True,
                "sync_start_target_ts": 4000.0,
                "start_window_s": 0.006,
                "release_lag_window_s": 0.001,
                "task_counts": {"vllm_serving_benchmark": 8},
                "aggregate": {
                    "pass_count": 8,
                    "fail_count": 0,
                    "best_tokens_per_s": 18.25,
                    "best_vllm_tokens_per_s": 18.25,
                },
                "workers": [
                    {
                        "rank": 0,
                        "card": 0,
                        "task": "vllm_serving_benchmark",
                        "status": "PASS",
                        "payload_status": "PASS",
                        "runtime_s": 60.0,
                        "metrics": {
                            "method": "qpruner",
                            "tokens_per_s": 18.25,
                            "latency_ms": 109.589,
                            "metadata_shim_status": "ALREADY_INSTALLED",
                            "metadata_backend_forward_status": "INSTALLED",
                        },
                    },
                    {
                        "rank": 1,
                        "card": 1,
                        "task": "vllm_serving_benchmark",
                        "status": "PASS",
                        "payload_status": "PASS",
                        "runtime_s": 60.0,
                        "metrics": {
                            "method": "cap",
                            "tokens_per_s": 17.5,
                            "latency_ms": 114.286,
                            "metadata_shim_status": "ALREADY_INSTALLED",
                            "metadata_backend_forward_status": "INSTALLED",
                        },
                    },
                    {
                        "rank": 2,
                        "card": 2,
                        "task": "vllm_serving_benchmark",
                        "status": "PASS",
                        "payload_status": "PASS",
                        "runtime_s": 60.0,
                        "metrics": {
                            "method": "baseline",
                            "tokens_per_s": 16.0,
                            "latency_ms": 125.0,
                            "metadata_shim_status": "ALREADY_INSTALLED",
                            "metadata_backend_forward_status": "INSTALLED",
                        },
                    },
                ],
            }
        )
    )
    (artifacts / "compatibility_issues.json").write_text(
        json.dumps(
            {
                "status": "ISSUES_FOUND",
                "issues": [
                    {"id": "vllm_attention_selector_api_mismatch"},
                    {"id": "qwen35_transformers_unsupported"},
                ],
                "ready": [{"id": "hccl_sync_ready"}, {"id": "multicard_generate_ready"}],
                "notes": [{"id": "serving_dense_export_storage_tradeoff"}],
            }
        )
    )
    (artifacts / "inference_bottleneck_report.json").write_text(
        json.dumps(
            {
                "status": "ACTIONABLE",
                "diagnoses": [
                    {
                        "id": "vllm_compressed_not_faster_than_baseline",
                        "title": "vLLM compressed path is not faster than baseline",
                        "evidence": "baseline 200.000 tokens/s, CAP 210.000 tokens/s, QPruner 222.000 tokens/s.",
                        "recommendation": "Use the synchronized harness as baseline evidence.",
                    },
                    {
                        "id": "compressed_native_qpruner_end_to_end_not_faster",
                        "title": "compressed-native QPruner preserves memory but is not end-to-end faster",
                        "evidence": (
                            "compressed-native QPruner qpruner_vs_baseline=0.876x with 76.307% "
                            "targeted storage reduction, runtime=scaled_int8_code_matmul, "
                            "cache_mode=scaled-code-matmul."
                        ),
                        "recommendation": "Move the packed decode speedup into full-model kernels.",
                    },
                    {
                        "id": "qpruner_packed_decode_kernel_positive",
                        "title": "QPruner packed decode kernel microbenchmark is latency-positive",
                        "evidence": (
                            "QPruner packed decode scaled-code speedup 16.988x, code-cache speedup "
                            "20.361x, code-cache storage reduction 49.707%, scaled-code peak 3.100 MB."
                        ),
                        "recommendation": "Use this as the kernel target for integration work.",
                    }
                ],
                "next_optimization_target": {
                    "title": "Reduce dense export/cache overhead and measure longer decode",
                    "steps": ["Run longer decode slices."],
                },
            }
        )
    )
    (artifacts / "compression_memory_report.json").write_text(
        json.dumps(
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
                    "artifacts": ["artifacts/qwen_compression_quality_qwen3_06b_quality_npu.json"],
                    "metrics": {
                        "loss_delta": 0.089,
                        "targeted_param_reduction_pct": 50.0,
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
            }
        )
    )
    (artifacts / "paper_baseline_coverage_audit.json").write_text(
        json.dumps(
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
            }
        )
    )

    output = report.write_demo_progress_report(demo_root)

    text = output.read_text()
    assert "Ascend 910B Demo Progress" in text
    assert "Multi-card synchronization is the primary demo evidence" in text
    assert "TinyQwen3-Offline" in text
    assert "Modern Model Track" in text
    assert "Qwen/Qwen3.5-0.8B" in text
    assert "1666.014" in text
    assert "qwen3_5" in text
    assert "reports/model-snapshot-qwen35_08b.md" in text
    assert "reports/ascend-inference-qwen35_08b_torch_npu.md" in text
    assert "Qwen3-0.6B Runnable Track" in text
    assert "modelscope" in text
    assert "1433.659" in text
    assert "1341.941" in text
    assert "11.923" in text
    assert "Multi-Card Qwen3-0.6B Inference" in text
    assert "95.384" in text
    assert "9278.400" in text
    assert "multicard_qwen_inference_qwen3_06b_npu.json" in text
    assert "reports/multicard-qwen-inference-qwen3_06b_npu.md" in text
    assert "Qwen3-0.6B CAP/QPruner Compression Quality" in text
    assert "| Metric | Baseline | CAP | WANDA | SparseGPT | QPruner |" in text
    assert "Multi-Card Qwen3-0.6B CAP/QPruner Compression Quality" in text
    assert "Baseline loss avg" in text
    assert "WANDA loss delta avg" in text
    assert "SparseGPT loss delta avg" in text
    assert "SparseGPT tokens/s total" in text
    assert "0.125" in text
    assert "-0.050" in text
    assert "0.111" in text
    assert "0.062" in text
    assert "qwen_compression_quality_qwen3_06b_quality_pattern_npu.json" in text
    assert "qwen_compression_quality_qwen3_06b_quality_npu.json" in text
    assert "multicard_qwen_compression_quality_qwen3_06b_quality_pattern_2card_npu.json" in text
    assert "multicard_qwen_compression_quality_qwen3_06b_quality_sync_npu.json" in text
    assert "reports/qwen-compression-quality-qwen3_06b_quality_pattern_npu.md" in text
    assert "reports/qwen-compression-quality-qwen3_06b_quality_npu.md" in text
    assert "artifacts/qwen_llm_pruner_baseline_qwen3_06b_llm_pruner_npu.json" in text
    assert "reports/qwen-llm-pruner-baseline-qwen3_06b_llm_pruner_npu.md" in text
    assert "reports/qwen-compression-quality-memory-sweep.md" in text
    assert "reports/qwen-qpruner-quality-memory-sweep.md" in text
    assert "qwen3_06b_qpruner_sweep_layers4_8_16_npu_layers4_bits6" in text
    assert "QPruner scale-quality sweep reaches 16/196 target layers" in text
    assert "loss-derived approximate PPL 403.429 -> 518.013" in text
    assert "Qwen3-0.6B Compression Choice Accuracy" in text
    assert "full-target QPruner choice accuracy baseline 75.000%, QPruner 87.500%" in text
    assert "choice accuracy target layers 196/196" in text
    assert "QPruner compression time 13.111s" in text
    assert "reports/qwen-compression-choice-accuracy-qwen3_06b_choice_accuracy_npu.md" in text
    assert "artifacts/qwen_compression_choice_accuracy_qwen3_06b_choice_accuracy_npu.json" in text
    assert (
        "reports/qwen-compression-choice-accuracy-qwen3_06b_multitask_choice_fulltarget_qpruner_npu.md"
        in text
    )
    assert "artifacts/qwen_compression_choice_accuracy_qwen3_06b_multitask_choice_fulltarget_qpruner_npu.json" in text
    assert "reports/qwen-qpruner-scale-quality-summary.md" in text
    assert "artifacts/qwen_qpruner_scale_quality_summary.json" in text
    assert "reports/qwen-qpruner-quality-memory-frontier.svg" in text
    assert "reports/qwen-qpruner-quality-memory-frontier.md" in text
    assert "reports/multicard-qwen-compression-quality-qwen3_06b_quality_pattern_2card_npu.md" in text
    assert "reports/multicard-qwen-compression-quality-qwen3_06b_quality_sync_npu.md" in text
    assert "Qwen3-0.6B CAP/QPruner Generate" in text
    assert "24.000" in text
    assert "4.000" in text
    assert "qwen_compression_generate_qwen3_06b_generate_npu.json" in text
    assert "reports/qwen-compression-generate-qwen3_06b_generate_npu.md" in text
    assert "reports/qwen-compression-generate-sweep.md" in text
    assert "Multi-Card Qwen3-0.6B RankAdaptor LoRA Fine-Tune" in text
    assert "adapter checksum synchronized" in text
    assert "-0.400" in text
    assert "Validation loss delta avg" in text
    assert "-0.800" in text
    assert "What does LoRA train?" in text
    assert "adapter matrices" in text
    assert "multicard_qwen_lora_finetune_qwen3_06b_lora_sync_npu.json" in text
    assert "reports/multicard-qwen-lora-finetune-qwen3_06b_lora_sync_npu.md" in text
    assert "Multi-Card Qwen3-0.6B CAP/QPruner Generate" in text
    assert "210.000" in text
    assert "224.000" in text
    assert "multicard_qwen_compression_generate_qwen3_06b_compression_generate_npu.json" in text
    assert "reports/multicard-qwen-compression-generate-qwen3_06b_compression_generate_npu.md" in text
    assert "reports/model-snapshot-qwen3_06b.md" in text
    assert "reports/ascend-inference-qwen3_06b_torch_npu.md" in text
    assert "Model Inventory" in text
    assert "TARGET_MODEL_FOUND" in text
    assert "Qwen/Qwen3-0.6B" in text
    assert "65.8" in text
    assert "8-card HCCL sync" in text
    assert "not extrapolated from one card" in text
    assert "HCCL_HOST_SOCKET_PORT_RANGE=auto" in text
    assert "RankAdaptor LoRA 8-card fine-tuning" in text
    assert "256" in text
    assert "4.150" in text
    assert "TinyQwen RankAdaptor LoRA fine-tune" in text
    assert "-0.170" in text
    assert "hello world tok_2" in text
    assert "TinyQwen BSLoRA Shared-LoRA Fine-Tune" in text
    assert "4096" in text
    assert "7168" in text
    assert "hello world tok_4" in text
    assert "Multi-Card TinyQwen BSLoRA Fine-Tune" in text
    assert "adapter checksum synchronized" in text
    assert "tiny_qwen3_bslora_sync_npu.json" in text
    assert "TinyQwen Compression Quality / Forward" in text
    assert "TinyQwen Compression Generate" in text
    assert "torch_generate" in text
    assert "Serving dense export" in text
    assert "0.020" in text
    assert "0.010" in text
    assert "19.000" in text
    assert "3.789" in text
    assert "7 dense / 7 cached" in text
    assert "Multi-Card TinyQwen Compression Generate" in text
    assert "all-reduce consistent" in text
    assert "3000.000" in text
    assert "vLLM Serving Compatibility Boundary" in text
    assert "Torch Serving Fallback Benchmark" in text
    assert "Baseline" in text
    assert "400.000" in text
    assert "CAP vs baseline speedup" in text
    assert "QPruner vs baseline speedup" in text
    assert "1.080" in text
    assert "1.140" in text
    assert "qpruner" in text
    assert "456.000" in text
    assert "Synchronized Multi-Card Parallel Suite" in text
    assert "start window" in text
    assert "workflow_cap" in text
    assert "torch_serving_fallback" in text
    assert "0.084" in text
    assert "multicard_parallel_suite_demo_parallel_suite_npu.json" in text
    assert "reports/multicard-parallel-suite-demo_parallel_suite_npu.md" in text
    assert "Synchronized vLLM Multi-Card Serving Slice" in text
    assert "vllm_metadata_sync_8card_npu_metrics" in text
    assert "best_vllm_tokens_per_s" in text
    assert "18.250" in text
    assert "17.500" in text
    assert "16.000" in text
    assert "ALREADY_INSTALLED / INSTALLED" in text
    assert "multicard_parallel_suite_vllm_metadata_sync_8card_npu_metrics.json" in text
    assert "reports/multicard-parallel-suite-vllm_metadata_sync_8card_npu_metrics.md" in text
    assert "Default vLLM probe" in text
    assert "Patch-preload vLLM probe" in text
    assert "Selector-shim vLLM probe" in text
    assert "vLLM-Ascend Serving Benchmark" in text
    assert "vllm_ascend_generate" in text
    assert "preload_vllm_ascend_metadata_shim" in text
    assert "metadata_shim_status" in text
    assert "222.000" in text
    assert "qpruner_vs_baseline=1.110x" in text
    assert "Root cause | no failing methods |" in text
    assert "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu.json" in text
    assert "reports/vllm-serving-benchmark-tiny_qwen3_serving_vllm_metadata_shim_npu.md" in text
    assert "Paper Baseline Alignment" in text
    assert "Engineering reference baseline" in text
    assert (
        "Memory-first compression readout: CAP targeted memory -66.667%, "
        "QPruner targeted memory -75.000%, target layers 2/196"
    ) in text
    assert "uncompressed_serving_export" in text
    assert "not paper baseline" in text
    assert "QPruner paper baseline" in text
    assert "LLM-Pruner" in text
    assert "CAP paper baselines" in text
    assert "SparseGPT, Wanda" in text
    assert "RankAdaptor paper recovery baselines" in text
    assert "LoRA, AdaLoRA" in text
    assert "Baseline alignment QPruner" in text
    assert "pruning baseline LLM-Pruner -> compressed-vs-uncompressed Ascend runtime reference" in text
    assert "Baseline alignment CAP" in text
    assert "Baseline alignment RankAdaptor" in text
    assert "Baseline alignment Engineering runtime baseline" in text
    assert "not a paper baseline uncompressed serving export -> runtime sanity, latency, throughput, and memory reference" in text
    assert "compression_memory_report.json" in text
    assert "paper_baseline_coverage_audit.json" in text
    assert "Paper baseline audit PASS" in text
    assert "QPruner LLM-Pruner-style PASS, target layers 196/196" in text
    assert "CAP SparseGPT PASS, target layers 196/196" in text
    assert "reports/compression-memory-report.md" in text
    assert "reports/paper-baseline-coverage-audit.md" in text
    assert "qwen_llm_pruner_baseline_qwen3_06b_llm_pruner_npu.json" in text
    assert "reports/qwen-llm-pruner-baseline-qwen3_06b_llm_pruner_npu.md" in text
    assert "Inference Bottleneck Diagnosis" in text
    assert "Inference Acceleration Summary" in text
    assert (
        "Engineering reference: `baseline` means uncompressed serving/runtime reference, "
        "not a paper baseline."
    ) in text
    assert (
        "Paper baselines: QPruner -> LLM-Pruner; CAP -> SparseGPT, Wanda; "
        "RankAdaptor -> LoRA, AdaLoRA, without recovery."
    ) in text
    assert "Best throughput point" in text
    assert "torch fallback / QPruner" in text
    assert "compressed-native preserves quantized/pruned modules" in text
    assert "8-card grouped replay" in text
    assert "memory-native workers 8/8" in text
    assert "55 samples with all 8 AICore nonzero" in text
    assert "released packed-code bytes 3523215360.000" in text
    assert "reports/inference-acceleration-summary.md" in text
    assert "reports/inference-acceleration-summary.csv" in text
    assert "reports/inference-acceleration-summary.svg" in text
    assert "artifacts/inference_acceleration_summary.json" in text
    assert "Compressed-Native Cache/Long-Decode Sweep" in text
    assert "Cached long-decode compressed-native path is latency-positive" in text
    assert "QPruner reaches 1.200x baseline at max_new_tokens=16" in text
    assert "cache qpruner delta 0.080x" in text
    assert "long decode qpruner delta 0.220x" in text
    assert "Runtime diagnosis" in text
    assert "QPruner scaling gap 0.449x" in text
    assert "QPruner cache peak +1.000 MB" in text
    assert "fuse packed/quantized decode kernels on NPU" in text
    assert "artifacts/compressed_native_sweep_summary.json" in text
    assert "reports/compressed-native-cache-long-decode-sweep.md" in text
    assert "reports/compressed-native-cache-long-decode-sweep.csv" in text
    assert "reports/compressed-native-cache-long-decode-sweep.svg" in text
    assert "QPruner Packed Decode Benchmark" in text
    assert "PASS on npu" in text
    assert "runtime packed_nbit_weight_codes" in text
    assert "storage 74.561%" in text
    assert "code-cache storage 49.561%" in text
    assert "uncached 2.000 ms, cache 1.000 ms, dense 0.900 ms" in text
    assert "code-cache 1.200 ms" in text
    assert "code-cache speedup 1.667x" in text
    assert "scaled-code matmul 1.100 ms" in text
    assert "scaled-code speedup 1.818x" in text
    assert "scaled-code peak 2.500 MB" in text
    assert "grouped scaled-code 2.000 ms vs sequential 4.800 ms" in text
    assert "grouped speedup 2.400x" in text
    assert "grouped code-cache bytes 16384" in text
    assert "cache speedup 2.000x" in text
    assert "artifacts/qpruner_packed_decode_benchmark_tiny_qwen3_qpruner_decode_npu.json" in text
    assert "reports/qpruner-packed-decode-benchmark-tiny_qwen3_qpruner_decode_npu.md" in text
    assert "artifacts/qpruner_packed_decode_benchmark_qwen3_06b_shape_sweep_npu.json" in text
    assert "reports/qpruner-packed-decode-benchmark-qwen3_06b_shape_sweep_npu.md" in text
    assert "artifacts/qpruner_packed_decode_benchmark_qwen3_06b_shape_sweep_dtype_preserving_npu.json" in text
    assert "reports/qpruner-packed-decode-benchmark-qwen3_06b_shape_sweep_dtype_preserving_npu.md" in text
    assert "Multi-Card Compression Sync Summary" in text
    assert "8-card synchronized compression path is ready" in text
    assert "QPruner 240.000 tokens/s total" in text
    assert "QPruner storage reduction 76.316%" in text
    assert "all-reduce consistent True" in text
    assert "artifacts/multicard_compression_sync_summary.json" in text
    assert "reports/multicard-compression-sync-summary.md" in text
    assert "reports/multicard-compression-sync-summary.csv" in text
    assert "reports/multicard-compression-sync-summary.svg" in text
    assert "Multi-card Qwen3 real grouped replay status: `PASS` with world size `2`" in text
    assert "Multi-card Qwen3 real grouped replay: best role down_proj, 1.180x best grouped speedup" in text
    assert "NPU utilization monitor status: `PASS`, max active process cards `8`." in text
    assert "artifacts/npu_monitor_qwen3_06b_real_grouped_replay_8card_utilmon.json" in text
    assert "logs/npu-smi-qwen3_06b_real_grouped_replay_8card_utilmon.log" in text
    assert (
        "artifacts/multicard_qwen_qpruner_grouped_replay_qwen3_06b_real_grouped_replay_2card_npu.json"
        in text
    )
    assert (
        "reports/multicard-qwen-qpruner-grouped-replay-qwen3_06b_real_grouped_replay_2card_npu.md"
        in text
    )
    assert "Objective Coverage Audit" in text
    assert "PARTIAL_READY" in text
    assert "Scale target-layer coverage" in text
    assert "artifacts/objective_coverage_audit.json" in text
    assert "reports/objective-coverage-audit.md" in text
    assert "vLLM compressed path is not faster than baseline" in text
    assert "compressed-native QPruner preserves memory but is not end-to-end faster" in text
    assert "compressed-native QPruner qpruner_vs_baseline=0.876x" in text
    assert "QPruner packed decode kernel microbenchmark is latency-positive" in text
    assert "QPruner packed decode scaled-code speedup 16.988x" in text
    assert "Next optimization target" in text
    assert "inference_bottleneck_report.json" in text
    assert "reports/inference-bottleneck-report.md" in text
    assert "vllm_attention_selector_api_mismatch" in text
    assert "Attention selector API status" in text
    assert "use_per_head_quant_scales" in text
    assert "num_heads" in text
    assert "CAP PASS / vLLM FAIL" in text
    assert "QPruner PASS / vLLM FAIL" in text
    assert "CAP PASS / vLLM PASS" in text
    assert "QPruner PASS / vLLM PASS" in text
    assert "0.18.0+empty" in text
    assert "0.1.dev2924" in text
    assert "vllm_serving_probe_tiny_qwen3_serving_export_npu_preload_patch.json" in text
    assert "vllm_serving_probe_tiny_qwen3_serving_export_npu_selector_shim.json" in text
    assert "torch_serving_fallback_tiny_qwen3_serving_fallback_npu.json" in text
    assert "multicard_tiny_qwen_generate_tiny_qwen3_multicard_generate_npu.json" in text
    assert "model_inventory.json" in text
    assert "Compatibility Issues" in text
    assert "ISSUES_FOUND" in text
    assert "compatibility-issues.md" in text


def test_progress_report_prefers_largest_passing_vllm_parallel_suite(tmp_path):
    report = load_report_module()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    for world_size in (3, 8):
        (artifacts / f"multicard_parallel_suite_vllm_metadata_sync_{world_size}card_npu_metrics.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "run_label": f"vllm_metadata_sync_{world_size}card_npu_metrics",
                    "world_size": world_size,
                    "aggregate": {"best_vllm_tokens_per_s": float(world_size)},
                }
            )
        )

    selected = report.best_vllm_parallel_suite(artifacts)
    report_name = report.parallel_suite_report_name(selected, "multicard-parallel-suite-vllm_metadata_sync_npu_metrics.md")

    assert selected["world_size"] == 8
    assert report_name == "multicard-parallel-suite-vllm_metadata_sync_8card_npu_metrics.md"


def test_progress_report_surfaces_vllm_benchmark_root_cause(tmp_path):
    report = load_report_module()
    demo_root = tmp_path / "demo"
    artifacts = demo_root / "artifacts"
    artifacts.mkdir(parents=True)
    benchmark = {
        "status": "FAIL",
        "backend": "vllm_ascend_generate",
        "dtype": "float16",
        "gpu_memory_utilization": 0.05,
        "preload_vllm_ascend_patch": True,
        "preload_vllm_ascend_selector_shim": True,
        "export_run_label": "tiny_qwen3_serving_export_npu",
        "summary": {},
        "exports": {
            "baseline": {
                "status": "FAIL",
                "vllm_load": "FAIL",
                "error_type": "EngineDeadError",
                "error": "EngineCore encountered an issue.",
                "stdout_tail": (
                    "ERROR attention_v1.py line 897\n"
                    "AttributeError: 'list' object has no attribute 'slot_mapping'\n"
                ),
            },
            "cap": {
                "status": "FAIL",
                "vllm_load": "FAIL",
                "error_type": "EngineDeadError",
                "error": "EngineCore encountered an issue.",
                "stdout_tail": (
                    "ERROR attention_v1.py line 897\n"
                    "AttributeError: 'list' object has no attribute 'slot_mapping'\n"
                ),
            },
            "qpruner": {
                "status": "FAIL",
                "vllm_load": "FAIL",
                "error_type": "EngineDeadError",
                "error": "EngineCore encountered an issue.",
                "stdout_tail": (
                    "ERROR attention_v1.py line 897\n"
                    "AttributeError: 'list' object has no attribute 'slot_mapping'\n"
                ),
            },
        },
    }
    (artifacts / "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu.json").write_text(
        json.dumps(benchmark)
    )

    text = report.markdown(
        demo_root=demo_root,
        model_inventory=None,
        inference=None,
        sync=None,
        lora=None,
        qwen_lora=None,
        qwen_bslora=None,
        multicard_bslora=None,
        qwen_compression=None,
        qwen_generate=None,
        multicard_generate=None,
        multicard_qwen_compression_generate=None,
        torch_serving_fallback=None,
        multicard_parallel_suite=None,
        vllm_parallel_suite=None,
        vllm_probe=None,
        vllm_patch_probe=None,
        vllm_selector_shim_probe=None,
        vllm_serving_benchmark=benchmark,
        qwen35_snapshot=None,
        qwen35_inference=None,
        qwen3_snapshot=None,
        qwen3_inference=None,
        multicard_qwen_inference=None,
        qwen_compression_quality=None,
        multicard_qwen_compression_quality=None,
        qwen_compression_generate=None,
        multicard_qwen_lora_finetune=None,
        compatibility=None,
    )

    assert "Root cause | baseline, CAP, QPruner: EngineDeadError" in text
    assert "AttributeError: 'list' object has no attribute 'slot_mapping'" in text


def test_progress_report_lists_selected_actual_qwen_shape_sweep(tmp_path):
    report = load_report_module()
    demo_root = tmp_path / "demo"
    actual_artifact = "qpruner_packed_decode_benchmark_qwen3_06b_actual_projection_bits8_shape_sweep_npu.json"
    actual_report = "qpruner-packed-decode-benchmark-qwen3_06b_actual_projection_bits8_shape_sweep_npu.md"
    native_artifact = (
        "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_"
        "shapeaware_releasepacked_long16_npu.json"
    )
    native_report = "qwen-qpruner-native-profile-qwen3_06b_native_profile_fulltarget_bits8_shapeaware_releasepacked_long16_npu.md"
    (demo_root / "reports").mkdir(parents=True)
    (demo_root / "reports" / actual_report).write_text("# actual shape sweep\n")
    (demo_root / "reports" / native_report).write_text("# stable native profile\n")
    bottleneck = {
        "status": "ACTIONABLE",
        "qpruner_packed_decode_shape_sweep": {"artifact": actual_artifact},
        "qwen_qpruner_native_profile": {
            "artifact": native_artifact,
            "paired_rounds": 3,
            "measurement_basis": "paired_interleaved_median",
            "speedups": {"qpruner_vs_baseline": 0.797},
            "qpruner_code_cache_storage_reduction_pct": 50.0,
        },
    }

    text = report.markdown(
        demo_root=demo_root,
        model_inventory=None,
        inference=None,
        sync=None,
        lora=None,
        qwen_lora=None,
        qwen_bslora=None,
        multicard_bslora=None,
        qwen_compression=None,
        qwen_generate=None,
        multicard_generate=None,
        multicard_qwen_compression_generate=None,
        torch_serving_fallback=None,
        multicard_parallel_suite=None,
        vllm_parallel_suite=None,
        vllm_probe=None,
        vllm_patch_probe=None,
        vllm_selector_shim_probe=None,
        vllm_serving_benchmark=None,
        qwen35_snapshot=None,
        qwen35_inference=None,
        qwen3_snapshot=None,
        qwen3_inference=None,
        multicard_qwen_inference=None,
        qwen_compression_quality=None,
        multicard_qwen_compression_quality=None,
        qwen_compression_generate=None,
        multicard_qwen_lora_finetune=None,
        compatibility=None,
        inference_bottleneck_report=bottleneck,
    )

    assert f"artifacts/{actual_artifact}" in text
    assert f"reports/{actual_report}" in text
    assert f"artifacts/{native_artifact}" in text
    assert f"reports/{native_report}" in text
    assert (
        "Qwen3 paired native profile: 0.797x baseline, measurement paired_interleaved_median, "
        "paired_rounds 3, code-cache storage 50.000%"
    ) in text


def test_progress_report_lists_current_vllm_hbm_rerun_from_bottleneck_report(tmp_path):
    report = load_report_module()
    demo_root = tmp_path / "demo"
    artifact = "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu_hbm_parallel_aclpath.json"
    bottleneck = {
        "status": "ACTIONABLE",
        "current_vllm_rerun": {
            "artifact": artifact,
            "status": "PASS",
            "parallel_methods": True,
            "method_cards": [0, 1, 2],
            "max_new_tokens": 4,
            "run_label": "tiny_qwen3_serving_vllm_metadata_shim_npu_hbm_parallel_aclpath",
            "best_method": "qpruner",
            "best_tokens_per_s": 395.891,
            "speedups": {"qpruner_vs_baseline": 1.05},
        },
    }

    text = report.markdown(
        demo_root=demo_root,
        model_inventory=None,
        inference=None,
        sync=None,
        lora=None,
        qwen_lora=None,
        qwen_bslora=None,
        multicard_bslora=None,
        qwen_compression=None,
        qwen_generate=None,
        multicard_generate=None,
        multicard_qwen_compression_generate=None,
        torch_serving_fallback=None,
        multicard_parallel_suite=None,
        vllm_parallel_suite=None,
        vllm_probe=None,
        vllm_patch_probe=None,
        vllm_selector_shim_probe=None,
        vllm_serving_benchmark=None,
        qwen35_snapshot=None,
        qwen35_inference=None,
        qwen3_snapshot=None,
        qwen3_inference=None,
        multicard_qwen_inference=None,
        qwen_compression_quality=None,
        multicard_qwen_compression_quality=None,
        qwen_compression_generate=None,
        multicard_qwen_lora_finetune=None,
        compatibility=None,
        inference_bottleneck_report=bottleneck,
    )

    assert "Current vLLM HBM rerun: PASS" in text
    assert f"artifacts/{artifact}" in text
    assert "cards 0,1,2" in text
    assert "best qpruner 395.891 tokens/s" in text
    assert "qpruner_vs_baseline 1.050x" in text


def test_progress_report_prefers_stronger_multicard_qwen_compression_generate(tmp_path):
    report = load_report_module()
    demo_root = tmp_path / "demo"
    artifacts = demo_root / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "multicard_qwen_compression_generate_qwen3_06b_compression_generate_npu.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "model_id": "Qwen/Qwen3-0.6B",
                "model_path": "/models/qwen",
                "world_size": 8,
                "backend": "hccl",
                "target_layer_limit": 2,
                "targeted_layers_total": 196,
                "aggregate": {
                    "distributed_reduce_consistent": True,
                    "baseline_tokens_per_s_total": 70.0,
                    "cap_tokens_per_s_total": 140.0,
                    "qpruner_tokens_per_s_total": 150.0,
                    "peak_mem_mb_total": 28000.0,
                    "pass_count": 8,
                },
            }
        )
    )
    (artifacts / "multicard_qwen_compression_generate_qwen3_06b_compression_generate_8card_target8_long8_npu.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "model_id": "Qwen/Qwen3-0.6B",
                "model_path": "/models/qwen",
                "world_size": 8,
                "backend": "hccl",
                "target_layer_limit": 8,
                "targeted_layers_total": 196,
                "aggregate": {
                    "distributed_reduce_consistent": True,
                    "baseline_tokens_per_s_total": 104.491,
                    "cap_tokens_per_s_total": 208.332,
                    "qpruner_tokens_per_s_total": 211.526,
                    "peak_mem_mb_total": 30172.0,
                    "pass_count": 8,
                },
                "ranks": [
                    {
                        "rank": 0,
                        "status": "PASS",
                        "device": "npu:0",
                        "benchmark": {
                            "baseline": {"tokens_per_s": 13.061},
                            "cap": {"tokens_per_s": 26.041},
                            "qpruner": {"tokens_per_s": 26.441},
                        },
                    }
                ],
            }
        )
    )

    output = report.write_demo_progress_report(demo_root)
    text = output.read_text()

    assert "Target layers | 8 / 196" in text
    assert "211.526" in text
    assert "multicard_qwen_compression_generate_qwen3_06b_compression_generate_8card_target8_long8_npu.json" in text
    assert "reports/multicard-qwen-compression-generate-qwen3_06b_compression_generate_8card_target8_long8_npu.md" in text
    assert "multicard_qwen_compression_generate_qwen3_06b_compression_generate_npu.json" not in text


def test_progress_report_lists_qwen_speed_first_native_profile_tradeoff(tmp_path):
    report = load_report_module()
    demo_root = tmp_path / "demo"
    speed_artifact = (
        "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_"
        "scaled_prebuilt_long16_npu.json"
    )
    speed_report = "qwen-qpruner-native-profile-qwen3_06b_native_profile_fulltarget_bits8_scaled_prebuilt_long16_npu.md"
    (demo_root / "reports").mkdir(parents=True)
    (demo_root / "reports" / speed_report).write_text("# speed first\n")
    bottleneck = {
        "status": "ACTIONABLE",
        "qwen_qpruner_native_speed_first_profile": {
            "artifact": speed_artifact,
            "speedups": {"qpruner_vs_baseline": 1.392},
            "qpruner_code_cache_storage_reduction_pct": -50.0,
            "qpruner_cached_scaled_code_bytes": 880803840.0,
            "qpruner_prebuild_scaled_code_dtype_cache": True,
        },
    }

    text = report.markdown(
        demo_root=demo_root,
        model_inventory=None,
        inference=None,
        sync=None,
        lora=None,
        qwen_lora=None,
        qwen_bslora=None,
        multicard_bslora=None,
        qwen_compression=None,
        qwen_generate=None,
        multicard_generate=None,
        multicard_qwen_compression_generate=None,
        torch_serving_fallback=None,
        multicard_parallel_suite=None,
        vllm_parallel_suite=None,
        vllm_probe=None,
        vllm_patch_probe=None,
        vllm_selector_shim_probe=None,
        vllm_serving_benchmark=None,
        qwen35_snapshot=None,
        qwen35_inference=None,
        qwen3_snapshot=None,
        qwen3_inference=None,
        multicard_qwen_inference=None,
        qwen_compression_quality=None,
        multicard_qwen_compression_quality=None,
        qwen_compression_generate=None,
        multicard_qwen_lora_finetune=None,
        compatibility=None,
        inference_bottleneck_report=bottleneck,
    )

    assert f"artifacts/{speed_artifact}" in text
    assert f"reports/{speed_report}" in text
    assert (
        "Qwen3 speed-first native profile: 1.392x baseline, measurement missing, "
        "paired_rounds missing, code-cache storage -50.000%, scaled-code dtype-cache bytes "
        "880803840.000, dense-cache bytes missing, dense-cache budget bytes missing, "
        "prebuild dtype cache True"
    ) in text


def test_progress_report_lists_qpruner_grouped_projection_plan(tmp_path):
    report = load_report_module()
    demo_root = tmp_path / "demo"
    shape_artifact = "qpruner_packed_decode_benchmark_qwen3_06b_actual_shape_sweep_grouped8_bits8_npu.json"
    bottleneck = {
        "status": "ACTIONABLE",
        "qpruner_grouped_projection_plan": {
            "status": "ACTIONABLE",
            "shape_artifact": shape_artifact,
            "shape_preset": "qwen3-0.6b-actual",
            "candidate_count": 7,
            "best_candidate": {
                "label": "qwen3_06b_actual_o_proj",
                "speedup_vs_sequential_scaled_code": 1.824,
                "profile_top_module_hits": 3,
            },
            "full_model": {
                "total_forward_calls": 3136,
                "forward_share_pct": 46.453,
                "code_cache_storage_reduction_pct": 50.0,
                "cached_scaled_code_bytes": 0.0,
            },
        },
    }

    text = report.markdown(
        demo_root=demo_root,
        model_inventory=None,
        inference=None,
        sync=None,
        lora=None,
        qwen_lora=None,
        qwen_bslora=None,
        multicard_bslora=None,
        qwen_compression=None,
        qwen_generate=None,
        multicard_generate=None,
        multicard_qwen_compression_generate=None,
        torch_serving_fallback=None,
        multicard_parallel_suite=None,
        vllm_parallel_suite=None,
        vllm_probe=None,
        vllm_patch_probe=None,
        vllm_selector_shim_probe=None,
        vllm_serving_benchmark=None,
        qwen35_snapshot=None,
        qwen35_inference=None,
        qwen3_snapshot=None,
        qwen3_inference=None,
        multicard_qwen_inference=None,
        qwen_compression_quality=None,
        multicard_qwen_compression_quality=None,
        qwen_compression_generate=None,
        multicard_qwen_lora_finetune=None,
        compatibility=None,
        inference_bottleneck_report=bottleneck,
    )

    assert f"artifacts/{shape_artifact}" in text
    assert (
        "QPruner grouped projection plan: qwen3_06b_actual_o_proj 1.824x grouped speedup, "
        "preset qwen3-0.6b-actual, candidates 7, top-module hits 3, full-model forwards 3136, "
        "forward share 46.453%, code-cache storage 50.000%, cached scaled-code bytes 0.000"
    ) in text


def test_progress_report_lists_qwen_qpruner_real_grouped_replay(tmp_path):
    report = load_report_module()
    demo_root = tmp_path / "demo"
    replay_artifact = "qwen_qpruner_grouped_replay_qwen3_06b_real_module_grouped_replay_npu.json"
    bottleneck = {
        "status": "ACTIONABLE",
        "qwen_qpruner_grouped_replay": {
            "artifact": replay_artifact,
            "status": "PASS",
            "target_layer_limit": 196,
            "targeted_layers_total": 196,
            "quantized_layers": 196,
            "group_count": 7,
            "code_cache_storage_reduction_pct": 50.0,
            "best_group": {
                "role": "o_proj",
                "module_count": 8,
                "available_module_count": 28,
                "sequential_latency_ms": 1.234,
                "grouped_latency_ms": 0.676,
                "speedup_vs_sequential_scaled_code": 1.825,
                "grouped_code_cache_bytes": 16777216,
                "grouped_scaled_code_cache_bytes": 0,
                "max_abs_diff_vs_sequential_scaled_code": 0.0,
            },
        },
    }

    text = report.markdown(
        demo_root=demo_root,
        model_inventory=None,
        inference=None,
        sync=None,
        lora=None,
        qwen_lora=None,
        qwen_bslora=None,
        multicard_bslora=None,
        qwen_compression=None,
        qwen_generate=None,
        multicard_generate=None,
        multicard_qwen_compression_generate=None,
        torch_serving_fallback=None,
        multicard_parallel_suite=None,
        vllm_parallel_suite=None,
        vllm_probe=None,
        vllm_patch_probe=None,
        vllm_selector_shim_probe=None,
        vllm_serving_benchmark=None,
        qwen35_snapshot=None,
        qwen35_inference=None,
        qwen3_snapshot=None,
        qwen3_inference=None,
        multicard_qwen_inference=None,
        qwen_compression_quality=None,
        multicard_qwen_compression_quality=None,
        qwen_compression_generate=None,
        multicard_qwen_lora_finetune=None,
        compatibility=None,
        inference_bottleneck_report=bottleneck,
    )

    assert f"artifacts/{replay_artifact}" in text
    assert (
        "Qwen3 real grouped replay: role o_proj, 1.825x grouped speedup, modules 8/28, "
        "groups 7, target layers 196/196, quantized layers 196, sequential 1.234 ms, "
        "grouped 0.676 ms, code-cache storage 50.000%, grouped code bytes 16777216.000, "
        "grouped scaled-code bytes 0.000, max_abs_diff 0.000"
    ) in text


def test_progress_report_lists_multicard_qwen_qpruner_grouped_replay(tmp_path):
    report = load_report_module()
    demo_root = tmp_path / "demo"
    artifact = "multicard_qwen_qpruner_grouped_replay_qwen3_06b_real_grouped_replay_2card_npu.json"
    multicard_replay = {
        "_artifact_name": artifact,
        "status": "PASS",
        "run_label": "qwen3_06b_real_grouped_replay_2card_npu",
        "world_size": 2,
        "cards": [0, 1],
        "launched_synchronously": True,
        "release_lag_window_s": 0.01,
        "aggregate": {
            "pass_count": 2,
            "fail_count": 0,
            "best_role": "down_proj",
            "best_grouped_speedup": 1.18,
            "mean_grouped_speedup": 1.165,
            "min_grouped_speedup": 1.15,
            "max_group_count": 7,
            "target_layer_limit": 196,
            "targeted_layers_total": 196,
            "quantized_layers": 196,
            "all_workers_memory_native": True,
            "memory_native_worker_count": 2,
            "min_code_cache_storage_reduction_pct": 50.0,
            "max_grouped_code_cache_bytes": 25165824,
            "max_grouped_scaled_code_cache_bytes": 0,
            "released_packed_code_bytes_total": 3523215360,
            "live_compressed_payload_storage_bytes_total": 6272,
            "max_abs_diff": 0.00195312,
        },
    }

    text = report.markdown(
        demo_root=demo_root,
        model_inventory=None,
        inference=None,
        sync=None,
        lora=None,
        qwen_lora=None,
        qwen_bslora=None,
        multicard_bslora=None,
        qwen_compression=None,
        qwen_generate=None,
        multicard_generate=None,
        multicard_qwen_compression_generate=None,
        torch_serving_fallback=None,
        multicard_parallel_suite=None,
        vllm_parallel_suite=None,
        vllm_probe=None,
        vllm_patch_probe=None,
        vllm_selector_shim_probe=None,
        vllm_serving_benchmark=None,
        qwen35_snapshot=None,
        qwen35_inference=None,
        qwen3_snapshot=None,
        qwen3_inference=None,
        multicard_qwen_inference=None,
        qwen_compression_quality=None,
        multicard_qwen_compression_quality=None,
        qwen_compression_generate=None,
        multicard_qwen_lora_finetune=None,
        compatibility=None,
        multicard_qwen_qpruner_grouped_replay=multicard_replay,
    )

    assert "Multi-card Qwen3 real grouped replay status: `PASS` with world size `2`" in text
    assert (
        "Multi-card Qwen3 real grouped replay: best role down_proj, 1.180x best grouped speedup, "
        "mean 1.165x, min 1.150x, groups 7, target layers 196/196, quantized layers 196, "
        "memory-native workers 2/2, code-cache storage 50.000%, grouped code bytes 25165824.000, "
        "grouped scaled-code bytes 0.000, released packed-code bytes 3523215360.000, "
        "live compressed payload bytes 6272.000, max_abs_diff 0.002"
    ) in text
    assert f"artifacts/{artifact}" in text
    assert "reports/multicard-qwen-qpruner-grouped-replay-qwen3_06b_real_grouped_replay_2card_npu.md" in text


def test_best_multicard_qwen_qpruner_grouped_replay_prefers_memory_native(tmp_path):
    report = load_report_module()
    artifacts = tmp_path / "demo" / "artifacts"
    artifacts.mkdir(parents=True)
    memory_native_artifact = (
        artifacts
        / "multicard_qwen_qpruner_grouped_replay_qwen3_06b_real_grouped_replay_8card_memory_native.json"
    )
    speed_first_artifact = (
        artifacts
        / "multicard_qwen_qpruner_grouped_replay_qwen3_06b_real_grouped_replay_8card_speed_first.json"
    )
    memory_native_artifact.write_text(
        json.dumps(
            {
                "status": "PASS",
                "run_label": "qwen3_06b_real_grouped_replay_8card_memory_native",
                "world_size": 8,
                "aggregate": {
                    "pass_count": 8,
                    "all_workers_memory_native": True,
                    "memory_native_worker_count": 8,
                    "min_code_cache_storage_reduction_pct": 50.0,
                    "best_grouped_speedup": 1.218,
                },
            }
        )
    )
    speed_first_artifact.write_text(
        json.dumps(
            {
                "status": "PASS",
                "run_label": "qwen3_06b_real_grouped_replay_8card_speed_first",
                "world_size": 8,
                "aggregate": {
                    "pass_count": 8,
                    "all_workers_memory_native": False,
                    "memory_native_worker_count": 0,
                    "min_code_cache_storage_reduction_pct": -50.0,
                    "best_grouped_speedup": 1.312,
                },
            }
        )
    )

    selected = report.best_multicard_qwen_qpruner_grouped_replay(artifacts)

    assert (
        selected["_artifact_name"]
        == "multicard_qwen_qpruner_grouped_replay_qwen3_06b_real_grouped_replay_8card_memory_native.json"
    )


def test_progress_report_prefers_grouped_replay_with_matching_monitor_over_higher_unmonitored_speedup(tmp_path):
    report = load_report_module()
    artifacts = tmp_path / "demo" / "artifacts"
    artifacts.mkdir(parents=True)
    common_aggregate = {
        "pass_count": 8,
        "all_workers_memory_native": True,
        "memory_native_worker_count": 8,
        "min_code_cache_storage_reduction_pct": 50.0,
    }
    (artifacts / "multicard_qwen_qpruner_grouped_replay_qwen_old_higher_speedup.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "run_label": "qwen_old_higher_speedup",
                "world_size": 8,
                "aggregate": {**common_aggregate, "best_grouped_speedup": 1.218},
                "workers": [{"command": ["python", "worker.py", "--batch-size", "32", "--iters", "1000"]}],
            }
        )
    )
    (artifacts / "multicard_qwen_qpruner_grouped_replay_qwen_fast_monitored.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "run_label": "qwen_fast_monitored",
                "world_size": 8,
                "aggregate": {**common_aggregate, "best_grouped_speedup": 1.189},
                "workers": [{"command": ["python", "worker.py", "--batch-size", "32", "--iters", "1000"]}],
            }
        )
    )
    (artifacts / "npu_monitor_qwen_fast_monitored_utilmon.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "run_label": "qwen_fast_monitored_utilmon",
                "command_text": "python scripts/multicard_qwen_qpruner_grouped_replay.py --run-label qwen_fast_monitored",
                "max_active_process_cards": 8,
                "samples_with_8_nonzero_aicore": 55,
            }
        )
    )

    selected = report.best_multicard_qwen_qpruner_grouped_replay(artifacts)

    assert selected["_artifact_name"] == "multicard_qwen_qpruner_grouped_replay_qwen_fast_monitored.json"


def test_progress_report_lists_npu_utilization_monitor(tmp_path):
    report = load_report_module()
    demo_root = tmp_path / "demo"
    artifact = "npu_monitor_qwen3_06b_real_grouped_replay_8card_utilmon.json"
    monitor = {
        "_artifact_name": artifact,
        "status": "PASS",
        "run_label": "qwen3_06b_real_grouped_replay_8card_utilmon",
        "monitor_log": str(demo_root / "logs" / "npu-smi-qwen3_06b_real_grouped_replay_8card_utilmon.log"),
        "samples": 57,
        "max_active_process_cards": 8,
        "samples_with_8_active_process_cards": 21,
        "samples_with_8_nonzero_aicore": 4,
        "max_aicore_by_card": {
            "0": 8,
            "1": 9,
            "2": 8,
            "3": 8,
            "4": 9,
            "5": 8,
            "6": 9,
            "7": 8,
        },
    }

    text = report.markdown(
        demo_root=demo_root,
        model_inventory=None,
        inference=None,
        sync=None,
        lora=None,
        qwen_lora=None,
        qwen_bslora=None,
        multicard_bslora=None,
        qwen_compression=None,
        qwen_generate=None,
        multicard_generate=None,
        multicard_qwen_compression_generate=None,
        torch_serving_fallback=None,
        multicard_parallel_suite=None,
        vllm_parallel_suite=None,
        vllm_probe=None,
        vllm_patch_probe=None,
        vllm_selector_shim_probe=None,
        vllm_serving_benchmark=None,
        qwen35_snapshot=None,
        qwen35_inference=None,
        qwen3_snapshot=None,
        qwen3_inference=None,
        multicard_qwen_inference=None,
        qwen_compression_quality=None,
        multicard_qwen_compression_quality=None,
        qwen_compression_generate=None,
        multicard_qwen_lora_finetune=None,
        compatibility=None,
        npu_utilization_monitor=monitor,
    )

    assert "NPU utilization monitor status: `PASS`, max active process cards `8`." in text
    assert (
        "NPU utilization monitor: 57 samples, max active process cards 8, "
        "21 samples with all 8 process cards, 4 samples with all 8 AICore nonzero, "
        "per-card AICore peaks 0=8, 1=9, 2=8, 3=8, 4=9, 5=8, 6=9, 7=8"
    ) in text
    assert f"artifacts/{artifact}" in text
    assert "logs/npu-smi-qwen3_06b_real_grouped_replay_8card_utilmon.log" in text


def test_progress_report_prefers_monitor_with_more_all_card_aicore_samples(tmp_path):
    report = load_report_module()
    artifacts = tmp_path / "demo" / "artifacts"
    artifacts.mkdir(parents=True)
    old = {
        "status": "PASS",
        "run_label": "old_more_process_samples",
        "samples": 57,
        "max_active_process_cards": 8,
        "samples_with_8_active_process_cards": 21,
        "samples_with_8_nonzero_aicore": 4,
        "max_aicore_by_card": {str(card): 9 for card in range(8)},
    }
    strong = {
        "status": "PASS",
        "run_label": "stronger_aicore_samples",
        "samples": 32,
        "max_active_process_cards": 8,
        "samples_with_8_active_process_cards": 15,
        "samples_with_8_nonzero_aicore": 32,
        "max_aicore_by_card": {str(card): 37 for card in range(8)},
    }
    (artifacts / "npu_monitor_old_more_process_samples.json").write_text(json.dumps(old))
    (artifacts / "npu_monitor_stronger_aicore_samples.json").write_text(json.dumps(strong))

    selected = report.best_npu_utilization_monitor(artifacts)

    assert selected["_artifact_name"] == "npu_monitor_stronger_aicore_samples.json"


def test_progress_report_prefers_grouped_replay_monitor_over_unrelated_longer_monitor(tmp_path):
    report = load_report_module()
    artifacts = tmp_path / "demo" / "artifacts"
    artifacts.mkdir(parents=True)
    unrelated = {
        "status": "PASS",
        "run_label": "qwen3_06b_compression_generate_8card_target8_utilmon",
        "command_text": "python scripts/multicard_qwen_compression_generate_benchmark.py",
        "samples": 180,
        "max_active_process_cards": 8,
        "samples_with_8_active_process_cards": 171,
        "samples_with_8_nonzero_aicore": 180,
        "max_aicore_by_card": {str(card): 37 for card in range(8)},
    }
    grouped = {
        "status": "PASS",
        "run_label": "qwen3_06b_real_grouped_replay_8card_b32_i1000_utilmon",
        "command_text": "python scripts/multicard_qwen_qpruner_grouped_replay.py --cards 0,1,2,3,4,5,6,7",
        "samples": 55,
        "max_active_process_cards": 8,
        "samples_with_8_active_process_cards": 27,
        "samples_with_8_nonzero_aicore": 55,
        "max_aicore_by_card": {str(card): 36 for card in range(8)},
    }
    (artifacts / "npu_monitor_qwen3_06b_compression_generate_8card_target8_utilmon.json").write_text(
        json.dumps(unrelated)
    )
    (artifacts / "npu_monitor_qwen3_06b_real_grouped_replay_8card_b32_i1000_utilmon.json").write_text(
        json.dumps(grouped)
    )

    selected = report.best_npu_utilization_monitor(artifacts)

    assert selected["_artifact_name"] == "npu_monitor_qwen3_06b_real_grouped_replay_8card_b32_i1000_utilmon.json"


def test_progress_report_lists_qwen_native_bits_tradeoff(tmp_path):
    report = load_report_module()
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
    bottleneck = {
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
    }

    text = report.markdown(
        demo_root=demo_root,
        model_inventory=None,
        inference=None,
        sync=None,
        lora=None,
        qwen_lora=None,
        qwen_bslora=None,
        multicard_bslora=None,
        qwen_compression=None,
        qwen_generate=None,
        multicard_generate=None,
        multicard_qwen_compression_generate=None,
        torch_serving_fallback=None,
        multicard_parallel_suite=None,
        vllm_parallel_suite=None,
        vllm_probe=None,
        vllm_patch_probe=None,
        vllm_selector_shim_probe=None,
        vllm_serving_benchmark=None,
        qwen35_snapshot=None,
        qwen35_inference=None,
        qwen3_snapshot=None,
        qwen3_inference=None,
        multicard_qwen_inference=None,
        qwen_compression_quality=None,
        multicard_qwen_compression_quality=None,
        qwen_compression_generate=None,
        multicard_qwen_lora_finetune=None,
        compatibility=None,
        inference_bottleneck_report=bottleneck,
    )

    assert (
        "Qwen3 native bits tradeoff: bits4 0.784x/75.059% packed storage/50.000% code-cache storage; "
        "bits6 0.803x/62.500% packed storage/50.000% code-cache storage; "
        "bits8 0.913x/50.000% packed storage/50.000% code-cache storage"
    ) in text
    for name in (bits4, bits6, bits8):
        assert f"artifacts/{name}" in text
        assert f"reports/{report.qwen_native_profile_report_name(name)}" in text
