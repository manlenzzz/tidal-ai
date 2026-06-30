import importlib.util
import json
import subprocess
from pathlib import Path


def load_playback_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "play_ascend_910b_demo.py"
    spec = importlib.util.spec_from_file_location("play_ascend_910b_demo", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def write_compressed_native_sweep_fixture(demo_root: Path) -> None:
    reports = demo_root / "reports"
    artifacts = demo_root / "artifacts"
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
                "cap_runtime_tradeoff": (
                    "CAP stores sparse residuals as coordinates with 0 dense sparse buffers; "
                    "uncached coordinate path reaches 0.920x baseline at 41.000 MB peak, "
                    "while cache switches runtime strategy to dense_weight_cache for +0.070x speedup delta."
                ),
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


def write_qpruner_packed_decode_fixture(demo_root: Path) -> None:
    reports = demo_root / "reports"
    artifacts = demo_root / "artifacts"
    (reports / "qpruner-packed-decode-benchmark-tiny_qwen3_qpruner_decode_npu.md").write_text(
        "# QPruner Packed Decode Benchmark\n"
        "Packed QPruner codes remain memory-efficient, while dense_weight_cache isolates "
        "the decode overhead that should move into an NPU kernel.\n"
    )
    write_json(
        artifacts / "qpruner_packed_decode_benchmark_tiny_qwen3_qpruner_decode_npu.json",
        {
            "status": "PASS",
            "device": "npu",
            "dtype": "torch.float16",
            "bits": 4,
            "runtime_storage_format": "packed_nbit_weight_codes",
            "packed_weight_code_bytes": 2048,
            "code_cache_bytes": 4096,
            "quantized_payload_bytes": 2084,
            "code_cache_payload_bytes": 4132,
            "dense_payload_bytes": 8192,
            "storage_reduction_pct": 74.561,
            "code_cache_storage_reduction_pct": 49.561,
            "uncached": {
                "runtime_strategy": "dequantize_per_forward",
                "latency_ms": 2.0,
            },
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


def write_multicard_compression_sync_fixture(demo_root: Path) -> None:
    reports = demo_root / "reports"
    artifacts = demo_root / "artifacts"
    (reports / "multicard-compression-sync-summary.md").write_text(
        "# Multi-Card Compression Sync Summary\n"
        "8-card synchronized compression path is ready: QPruner 240.000 tokens/s total "
        "(1.250x baseline), all-reduce consistent True, while compressed-native keeps "
        "76.316% QPruner storage reduction.\n"
    )
    (reports / "multicard-compression-sync-summary.csv").write_text("metric,value\n")
    (reports / "multicard-compression-sync-summary.svg").write_text("<svg>Multi-card compression sync</svg>\n")
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


def write_npu_monitor_fixture(demo_root: Path) -> None:
    artifacts = demo_root / "artifacts"
    logs = demo_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "npu-smi-qwen3_06b_real_grouped_replay_8card_utilmon.log").write_text(
        "NPU 0 AICore 8\nNPU 7 AICore 8\n"
    )
    write_json(
        artifacts / "npu_monitor_qwen3_06b_real_grouped_replay_8card_utilmon.json",
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
        },
    )


def write_multicard_qwen_qpruner_grouped_replay_fixture(demo_root: Path) -> None:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "multicard-qwen-qpruner-grouped-replay-qwen3_06b_real_grouped_replay_8card_memory_native.md").write_text(
        "# Synchronized Qwen QPruner Grouped Replay\n"
        "| Best role | down_proj |\n"
        "| Memory-native workers | 8 / 8 |\n"
        "| Code-cache storage reduction | 50.000% |\n"
    )
    write_json(
        artifacts / "multicard_qwen_qpruner_grouped_replay_qwen3_06b_real_grouped_replay_8card_memory_native.json",
        {
            "status": "PASS",
            "run_label": "qwen3_06b_real_grouped_replay_8card_memory_native",
            "world_size": 8,
            "cards": list(range(8)),
            "launched_synchronously": True,
            "release_lag_window_s": 0.0,
            "aggregate": {
                "pass_count": 8,
                "best_role": "down_proj",
                "best_grouped_speedup": 1.218,
                "mean_grouped_speedup": 1.205,
                "min_grouped_speedup": 1.18,
                "max_group_count": 7,
                "target_layer_limit": 196,
                "targeted_layers_total": 196,
                "quantized_layers": 196,
                "all_workers_memory_native": True,
                "memory_native_worker_count": 8,
                "min_code_cache_storage_reduction_pct": 50.0,
                "max_grouped_code_cache_bytes": 25165824,
                "max_grouped_scaled_code_cache_bytes": 0,
                "released_packed_code_bytes_total": 3523215360,
                "live_compressed_payload_storage_bytes_total": 6272,
                "max_abs_diff": 0.00195312,
            },
        },
    )
    write_json(
        artifacts / "multicard_qwen_qpruner_grouped_replay_qwen3_06b_real_grouped_replay_8card_speed_first.json",
        {
            "status": "PASS",
            "run_label": "qwen3_06b_real_grouped_replay_8card_speed_first",
            "world_size": 8,
            "cards": list(range(8)),
            "aggregate": {
                "pass_count": 8,
                "best_role": "down_proj",
                "best_grouped_speedup": 1.312,
                "all_workers_memory_native": False,
                "memory_native_worker_count": 0,
                "min_code_cache_storage_reduction_pct": -50.0,
            },
        },
    )


def write_choice_accuracy_fixture(demo_root: Path) -> None:
    reports = demo_root / "reports"
    artifacts = demo_root / "artifacts"
    (reports / "qwen-compression-choice-accuracy-qwen3_06b_choice_accuracy_npu.md").write_text(
        "# Qwen Compression Choice Accuracy\n"
        "| baseline | PASS | 75.000% | 3 / 4 |\n"
        "| cap | PASS | 50.000% | 2 / 4 |\n"
        "| qpruner | PASS | 75.000% | 3 / 4 |\n"
    )
    (reports / "qwen-compression-choice-accuracy-qwen3_06b_choice_accuracy_npu.csv").write_text(
        "method,status,accuracy,correct,total\n"
    )
    write_json(
        artifacts / "qwen_compression_choice_accuracy_qwen3_06b_choice_accuracy_npu.json",
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
        },
    )


def write_multitask_choice_accuracy_fixture(demo_root: Path) -> None:
    reports = demo_root / "reports"
    artifacts = demo_root / "artifacts"
    (reports / "qwen-compression-choice-accuracy-qwen3_06b_multitask_choice_npu.md").write_text(
        "# Qwen Compression Choice Accuracy\n"
        "| baseline | __overall__ | PASS | 75.000% | 6 / 8 |\n"
    )
    (reports / "qwen-compression-choice-accuracy-qwen3_06b_multitask_choice_npu.csv").write_text(
        "method,task,status,accuracy,correct,total\n"
    )
    write_json(
        artifacts / "qwen_compression_choice_accuracy_qwen3_06b_multitask_choice_npu.json",
        {
            "status": "PASS",
            "backend": "choice_log_likelihood",
            "sample_source": "/mnt/nvme/622/tidal-demo/eval/qwen3_06b_multitask_choice.jsonl",
            "task_count": 4,
            "target_layer_limit": 196,
            "targeted_layers_total": 196,
            "methods": {
                "baseline": {"status": "PASS", "accuracy": 0.75, "correct": 6, "total": 8},
                "cap": {"status": "PASS", "accuracy": 0.625, "correct": 5, "total": 8},
                "qpruner": {"status": "PASS", "accuracy": 0.75, "correct": 6, "total": 8},
            },
        },
    )


def write_fulltarget_qpruner_choice_accuracy_fixture(demo_root: Path) -> None:
    reports = demo_root / "reports"
    artifacts = demo_root / "artifacts"
    (reports / "qwen-compression-choice-accuracy-qwen3_06b_multitask_choice_fulltarget_qpruner_npu.md").write_text(
        "# Qwen Compression Choice Accuracy\n"
        "| baseline | __overall__ | PASS | 75.000% | 6 / 8 |\n"
        "| qpruner | __overall__ | PASS | 87.500% | 7 / 8 |\n"
    )
    (reports / "qwen-compression-choice-accuracy-qwen3_06b_multitask_choice_fulltarget_qpruner_npu.csv").write_text(
        "method,task,status,accuracy,correct,total\n"
    )
    write_json(
        artifacts / "qwen_compression_choice_accuracy_qwen3_06b_multitask_choice_fulltarget_qpruner_npu.json",
        {
            "status": "PASS",
            "backend": "choice_log_likelihood",
            "sample_source": "/mnt/nvme/622/tidal-demo/eval/qwen3_06b_multitask_choice.jsonl",
            "task_count": 4,
            "target_layer_limit": 0,
            "targeted_layers_total": 196,
            "methods": {
                "baseline": {"status": "PASS", "accuracy": 0.75, "correct": 6, "total": 8, "targeted_layers": 196},
                "qpruner": {
                    "status": "PASS",
                    "accuracy": 0.875,
                    "correct": 7,
                    "total": 8,
                    "targeted_layers": 196,
                    "compression_time_s": 13.111,
                    "average_bits": 8.0,
                },
            },
        },
    )


def test_demo_storyline_summarizes_key_artifacts(tmp_path):
    playback = load_playback_module()
    demo_root = tmp_path / "demo"
    artifacts = demo_root / "artifacts"
    (demo_root / "reports").mkdir(parents=True)
    write_multicard_compression_sync_fixture(demo_root)
    write_npu_monitor_fixture(demo_root)
    write_multicard_qwen_qpruner_grouped_replay_fixture(demo_root)
    write_choice_accuracy_fixture(demo_root)
    (demo_root / "reports" / "qwen-compression-quality-memory-sweep.md").write_text(
        "Best memory-quality point under loss delta <= 0.500 is "
        "qwen3_06b_quality_pattern_bits6: 66.667% targeted memory reduction, "
        "loss delta -0.224, average bits 5.333.\n"
    )
    (demo_root / "reports" / "qwen-qpruner-quality-memory-sweep.md").write_text(
        "Best memory-quality point under loss delta <= 0.500 is "
        "qwen3_06b_qpruner_sweep_layers4_bits6: 66.667% targeted memory reduction, "
        "loss delta -0.224, average bits 5.333.\n"
    )
    (demo_root / "reports" / "qwen-qpruner-scale-quality-summary.md").write_text(
        "QPruner scale-quality sweep reaches 16/196 target layers (8.163% coverage); "
        "best under loss delta <= 0.500 is qwen3_06b_qpruner_sweep_layers16_bits6 "
        "with 62.500% targeted memory reduction and loss-derived approximate PPL "
        "403.429 -> 518.013 (28.403%).\n"
    )
    (demo_root / "reports" / "qwen-qpruner-scale-quality-summary.csv").write_text(
        "run_label,status,target_layer_limit,targeted_layers_total,coverage_pct,average_bits\n"
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
    (demo_root / "reports" / "qwen-qpruner-quality-memory-frontier.svg").write_text("<svg></svg>\n")
    (demo_root / "reports" / "qwen-qpruner-quality-memory-frontier.md").write_text(
        "![QPruner quality-memory frontier](qwen-qpruner-quality-memory-frontier.svg)\n"
    )
    write_json(
        artifacts / "objective_coverage_audit.json",
        {
            "status": "PARTIAL_READY",
            "sections": [
                {
                    "title": "Inference acceleration",
                    "status": "ACTIONABLE",
                    "evidence_summary": "QPruner 456.000 tokens/s",
                    "remaining_gap": "vLLM compressed path is not faster than baseline",
                }
            ],
        },
    )
    write_json(
        artifacts / "demo_readiness_report.json",
        {
            "status": "PARTIAL_READY",
            "sections": [
                {
                    "id": "ascend_910b_baseline",
                    "title": "1. Ascend 910B Baseline",
                    "status": "READY",
                    "gaps": [],
                    "next_action": "Keep compatibility current.",
                },
                {
                    "id": "inference_acceleration",
                    "title": "2. Inference Acceleration",
                    "status": "READY_WITH_ACTIONABLE_GAP",
                    "gaps": [
                        "objective audit inference acceleration ACTIONABLE: "
                        "vLLM compressed path is not faster than baseline"
                    ],
                    "next_action": "Start fused/batched QuantizedLinear work.",
                },
            ],
        },
    )
    (demo_root / "reports" / "inference-acceleration-summary.md").write_text(
        "# Inference Acceleration Summary\n"
        "Best throughput point: `torch fallback / QPruner` at 456.000 tokens/s.\n"
        "compressed-native preserves quantized/pruned modules with dense export `False`.\n"
    )
    (demo_root / "reports" / "inference-acceleration-summary.csv").write_text("track,method,tokens_per_s\n")
    (demo_root / "reports" / "inference-acceleration-summary.svg").write_text("<svg>Inference acceleration</svg>\n")
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
    write_compressed_native_sweep_fixture(demo_root)
    write_qpruner_packed_decode_fixture(demo_root)
    write_json(
        artifacts / "ascend_910b_demo_manifest.json",
        {
            "status": "PASS",
            "stages": [
                {"name": "multicard_sync", "status": "PASS", "seconds": 2.0},
                {"name": "compression_generate", "status": "PASS", "seconds": 3.5},
            ],
        },
    )
    write_json(artifacts / "multicard_sync_summary.json", {"status": "PASS", "world_size": 8, "backend": "hccl"})
    write_json(
        artifacts / "model_inventory.json",
        {
            "status": "TARGET_MODEL_FOUND",
            "target_candidates": [
                {"model_id": "Qwen/Qwen3-0.6B", "status": "FOUND"},
                {"model_id": "Qwen/Qwen2.5-0.5B-Instruct", "status": "MISSING"},
            ],
            "tiny_fixtures": [{"name": "TinyQwen3-Offline", "status": "FOUND", "parameter_mb": 0.722}],
        },
    )
    write_json(
        artifacts / "model_snapshot_qwen3_06b.json",
        {
            "status": "DOWNLOADED",
            "provider": "modelscope",
            "model_id": "Qwen/Qwen3-0.6B",
            "parameter_mb": 1433.659,
        },
    )
    write_json(
        artifacts / "ascend_inference_qwen3_06b_torch_npu.json",
        {
            "status": "PASS",
            "model_id": "Qwen/Qwen3-0.6B",
            "backend": "torch",
            "device": "npu",
            "latency_ms": 1341.941,
            "tokens_per_s": 11.923,
            "peak_mem_mb": 1159.8,
        },
    )
    write_json(
        artifacts / "rankadaptor_lora_sync_summary.json",
        {"status": "PASS", "world_size": 8, "trainable_adapter_params": 512},
    )
    write_json(
        artifacts / "tiny_qwen_lora_finetune_tiny_qwen3_lora_npu.json",
        {"status": "PASS", "initial_loss": 5.5, "final_loss": 5.25, "loss_delta": -0.25},
    )
    write_json(
        artifacts / "tiny_qwen_bslora_finetune_tiny_qwen3_bslora_npu.json",
        {
            "status": "PASS",
            "initial_loss": 5.54,
            "final_loss": 5.47,
            "loss_delta": -0.07,
            "trainable_adapter_params": 3072,
            "unshared_adapter_params": 5760,
        },
    )
    write_json(
        artifacts / "multicard_tiny_qwen_bslora_finetune_tiny_qwen3_bslora_sync_npu.json",
        {
            "status": "PASS",
            "world_size": 8,
            "backend": "hccl",
            "aggregate": {
                "distributed_reduce_consistent": True,
                "adapter_sync_consistent": True,
                "initial_loss_avg": 5.54,
                "final_loss_avg": 5.49,
                "loss_delta_avg": -0.05,
            },
        },
    )
    write_json(
        artifacts / "tiny_qwen_compression_generate_tiny_qwen3_generate_serving_export_npu.json",
        {
            "status": "PASS",
            "backend": "torch_generate",
            "serving_dense_export": True,
            "baseline": {"latency_ms": 22.0, "tokens_per_s": 360.0},
            "cap": {"latency_speedup": 1.01, "targeted_compression_ratio": 19.0},
            "qpruner": {"latency_speedup": 1.04, "average_bits": 3.789},
            "peak_mem_mb": 22.4,
        },
    )
    write_json(
        artifacts / "multicard_tiny_qwen_generate_tiny_qwen3_multicard_generate_npu.json",
        {
            "status": "PASS",
            "world_size": 8,
            "backend": "hccl",
            "aggregate": {
                "distributed_reduce_consistent": True,
                "baseline_tokens_per_s_total": 2800.0,
                "cap_tokens_per_s_total": 2920.0,
                "qpruner_tokens_per_s_total": 3000.0,
            },
        },
    )
    write_json(
        artifacts / "multicard_qwen_inference_qwen3_06b_npu.json",
        {
            "status": "PASS",
            "model_id": "Qwen/Qwen3-0.6B",
            "world_size": 8,
            "backend": "hccl",
            "aggregate": {
                "distributed_reduce_consistent": True,
                "tokens_per_s_total": 95.384,
                "generated_tokens_total": 128,
                "peak_mem_mb_total": 9278.4,
                "pass_count": 8,
            },
        },
    )
    write_json(
        artifacts / "qwen_compression_generate_qwen3_06b_generate_npu.json",
        {
            "status": "PASS",
            "model_id": "Qwen/Qwen3-0.6B",
            "backend": "torch_generate",
            "target_layer_limit": 2,
            "targeted_layers_total": 196,
            "baseline": {"tokens_per_s": 66.667},
            "cap": {"latency_speedup": 1.091, "targeted_compression_ratio": 24.0},
            "qpruner": {"latency_speedup": 1.111, "average_bits": 4.0},
            "peak_mem_mb": 1200.0,
        },
    )
    write_json(
        artifacts / "qwen_compression_quality_qwen3_06b_quality_npu.json",
        {
            "status": "PASS",
            "model_id": "Qwen/Qwen3-0.6B",
            "backend": "torch_forward",
            "target_layer_limit": 2,
            "targeted_layers_total": 196,
            "baseline": {"loss": 11.5, "tokens_per_s": 66.667},
            "cap": {"loss_delta": 0.125, "targeted_compression_ratio": 24.0},
            "qpruner": {"loss_delta": 0.0625, "average_bits": 4.0},
            "peak_mem_mb": 1200.0,
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
                },
                {
                    "method": "CAP",
                    "paper_baseline_role": "pruning / joint-compression / SVD baselines",
                    "paper_baselines": ["SparseGPT", "Wanda"],
                    "demo_reference_role": "compressed-vs-uncompressed Ascend runtime reference",
                },
                {
                    "method": "RankAdaptor",
                    "paper_baseline_role": "recovery baselines",
                    "paper_baselines": ["LoRA", "AdaLoRA", "without recovery"],
                    "demo_reference_role": "LoRA/BSLoRA recovery evidence on Ascend",
                },
                {
                    "method": "Engineering runtime baseline",
                    "paper_baseline_role": "not a paper baseline",
                    "paper_baselines": ["uncompressed serving export"],
                    "demo_reference_role": "runtime sanity, latency, throughput, and memory reference",
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
                "RankAdaptor has 8-card LoRA/BSLoRA training evidence, but AdaLoRA/no-recovery controlled baselines remain next work.",
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
                "rankadaptor_existing": {
                    "status": "PASS",
                    "qwen3_lora_world_size": 8,
                    "qwen3_lora_validation_loss_delta_avg": -0.008,
                    "qwen3_bslora_target_module_count": 196,
                    "qwen3_bslora_loss_delta": -0.289,
                },
                "rankadaptor_recovery_controlled": {
                    "status": "PASS",
                    "all_methods_passed": True,
                    "best_recovery_method": "lora",
                    "no_recovery_loss_delta": 0.0,
                    "lora_loss_delta": -0.484,
                    "adalora_style_loss_delta": 0.102,
                },
                "memory_first_existing": {
                    "status": "ACTIONABLE",
                    "cap_targeted_memory_reduction_pct": 66.667,
                    "qpruner_targeted_memory_reduction_pct": 50.0,
                    "target_layer_coverage_pct": 100.0,
                },
            },
            "coverage_decisions": [
                {
                    "area": "QPruner",
                    "decision": "Use the new 196/196 LLM-Pruner-style artifact as the strongest current paper-baseline compatibility evidence.",
                    "remaining_gap": "Still not an official full LLM-Pruner codebase reproduction; report it as LLM-Pruner-style structural pruning until official port is run.",
                },
                {
                    "area": "CAP",
                    "decision": "Use the new 196/196 Wanda/SparseGPT artifact as full-target lightweight paper-baseline evidence on Ascend.",
                    "remaining_gap": "Need official upstream reproductions and remaining DSNoT/OATS/OWL/AlphaPruning/SLiM/JSQ/L2QER/LPAF/SVD baselines before claiming full paper reproduction.",
                },
                {
                    "area": "RankAdaptor",
                    "decision": "Use LoRA/BSLoRA training sync as Ascend fine-tuning evidence, not as a complete paper baseline table.",
                    "remaining_gap": "Need controlled LoRA vs AdaLoRA vs without-recovery runs after pruning-stage alignment.",
                },
            ],
            "next_runs": [
                "Port or approximate AdaLoRA and no-recovery controlled RankAdaptor baselines on the same train/eval prompts.",
                "If time allows, replace LLM-Pruner-style with official LLM-Pruner implementation on Ascend or mark exact incompatibilities.",
            ],
            "paper_baselines": {
                "qpruner": {"primary": ["LLM-Pruner"], "recovery": ["LoRA", "LoftQ"]},
                "cap": {"pruning": ["SparseGPT", "Wanda", "DSNoT", "OATS", "OWL", "AlphaPruning"]},
                "rankadaptor": {"recovery": ["LoRA", "AdaLoRA", "without recovery"]},
            },
        },
    )
    write_json(
        artifacts / "multicard_qwen_compression_quality_qwen3_06b_quality_sync_npu.json",
        {
            "status": "PASS",
            "model_id": "Qwen/Qwen3-0.6B",
            "world_size": 8,
            "backend": "hccl",
            "target_layer_limit": 2,
            "targeted_layers_total": 196,
            "aggregate": {
                "distributed_reduce_consistent": True,
                "baseline_loss_avg": 11.5,
                "cap_loss_delta_avg": 0.125,
                "qpruner_loss_delta_avg": 0.0625,
                "pass_count": 8,
            },
        },
    )
    write_json(
        artifacts / "multicard_qwen_compression_quality_qwen3_06b_quality_pattern_2card_npu.json",
        {
            "status": "PASS",
            "model_id": "Qwen/Qwen3-0.6B",
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
        artifacts / "multicard_qwen_lora_finetune_qwen3_06b_lora_sync_npu.json",
        {
            "status": "PASS",
            "model_id": "Qwen/Qwen3-0.6B",
            "world_size": 8,
            "backend": "hccl",
            "target_module_count": 2,
            "targeted_modules_total": 196,
            "train_mode": "instruction",
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
        artifacts / "multicard_qwen_compression_generate_qwen3_06b_compression_generate_npu.json",
        {
            "status": "PASS",
            "model_id": "Qwen/Qwen3-0.6B",
            "world_size": 8,
            "backend": "hccl",
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
        },
    )
    write_json(
        artifacts / "multicard_qwen_compression_generate_qwen3_06b_generate_pattern_2card_npu.json",
        {
            "status": "PASS",
            "model_id": "Qwen/Qwen3-0.6B",
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
        artifacts / "vllm_serving_probe_tiny_qwen3_serving_export_npu.json",
        {
            "status": "FAIL",
            "vllm_available": True,
            "gpu_memory_utilization": 0.05,
            "preload_vllm_ascend_patch": False,
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
        },
    )
    write_json(
        artifacts / "vllm_serving_probe_tiny_qwen3_serving_export_npu_preload_patch.json",
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
        },
    )
    write_json(
        artifacts / "vllm_serving_probe_tiny_qwen3_serving_export_npu_selector_shim.json",
        {
            "status": "PASS",
            "vllm_available": True,
            "gpu_memory_utilization": 0.05,
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
        artifacts / "torch_serving_fallback_tiny_qwen3_serving_fallback_npu.json",
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
                "baseline": {"status": "PASS", "tokens_per_s": 400.0, "latency_ms": 24.0, "peak_mem_mb": 39.0},
                "cap": {"status": "PASS", "tokens_per_s": 432.0, "latency_ms": 20.0, "peak_mem_mb": 40.0},
                "qpruner": {"status": "PASS", "tokens_per_s": 456.0, "latency_ms": 18.0, "peak_mem_mb": 42.0},
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
            "baseline": {"status": "PASS", "tokens_per_s": 160.0, "latency_ms": 25.0, "peak_mem_mb": 40.0},
            "cap": {
                "status": "PASS",
                "tokens_per_s": 170.0,
                "latency_ms": 23.4,
                "peak_mem_mb": 39.0,
                "packed_layers": 7,
                "exported_dense_linears": 0,
                "targeted_param_reduction_pct": 66.667,
            },
            "qpruner": {
                "status": "PASS",
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
        artifacts / "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu.json",
        {
            "status": "PASS",
            "backend": "vllm_ascend_generate",
            "preload_vllm_ascend_metadata_shim": True,
            "metadata_shim": {"status": "INSTALLED"},
            "summary": {
                "best_method": "qpruner",
                "best_tokens_per_s": 222.0,
                "cap_vs_baseline_speedup": 1.05,
                "qpruner_vs_baseline_speedup": 1.11,
            },
            "exports": {
                "baseline": {"tokens_per_s": 200.0, "latency_ms": 30.0, "peak_mem_mb": 512.0},
                "cap": {"tokens_per_s": 210.0, "latency_ms": 28.0, "peak_mem_mb": 512.0},
                "qpruner": {"tokens_per_s": 222.0, "latency_ms": 27.0, "peak_mem_mb": 512.0},
            },
        },
    )
    write_json(
        artifacts / "multicard_parallel_suite_demo_parallel_suite_npu.json",
        {
            "status": "PASS",
            "run_label": "demo_parallel_suite_npu",
            "world_size": 4,
            "cards": [0, 1, 2, 3],
            "launched_synchronously": True,
            "sync_start_target_ts": 2000.0,
            "start_window_s": 0.084,
            "release_lag_window_s": 0.012,
            "task_counts": {
                "workflow_cap": 1,
                "workflow_qpruner": 1,
                "rankadaptor": 1,
                "torch_serving_fallback": 1,
            },
            "aggregate": {"pass_count": 4, "fail_count": 0, "best_tokens_per_s": 456.0},
            "workers": [
                {"rank": 0, "card": 0, "task": "workflow_cap", "status": "PASS", "payload_status": "PASS"},
                {"rank": 1, "card": 1, "task": "workflow_qpruner", "status": "PASS", "payload_status": "PASS"},
                {"rank": 2, "card": 2, "task": "rankadaptor", "status": "PASS", "payload_status": "PASS"},
                {
                    "rank": 3,
                    "card": 3,
                    "task": "torch_serving_fallback",
                    "status": "PASS",
                    "payload_status": "PASS",
                    "metrics": {"best_tokens_per_s": 456.0},
                },
            ],
        },
    )
    write_json(
        artifacts / "multicard_parallel_suite_vllm_metadata_sync_npu_metrics.json",
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
                    "metrics": {
                        "method": "baseline",
                        "tokens_per_s": 12.184,
                        "latency_ms": 164.149,
                        "metadata_shim_status": "ALREADY_INSTALLED",
                        "metadata_backend_forward_status": "INSTALLED",
                    },
                },
            ],
        },
    )
    write_json(
        artifacts / "multicard_parallel_suite_vllm_metadata_sync_8card_npu_metrics.json",
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
                    "metrics": {
                        "method": "baseline",
                        "tokens_per_s": 16.0,
                        "latency_ms": 125.0,
                        "metadata_shim_status": "ALREADY_INSTALLED",
                        "metadata_backend_forward_status": "INSTALLED",
                    },
                },
            ],
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
                    "evidence": "baseline 200.000 tokens/s, CAP 210.000 tokens/s, QPruner 222.000 tokens/s.",
                    "recommendation": "Use synchronized evidence while optimizing the compressed serving path.",
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
        },
    )

    text = playback.demo_storyline(demo_root)

    assert "Ascend 910B TIDAL-AI Demo Playback" in text
    assert "Model inventory: TARGET_MODEL_FOUND" in text
    assert "Target model Qwen/Qwen3-0.6B: FOUND" in text
    assert "Qwen3-0.6B torch_npu inference: PASS, backend torch, device npu" in text
    assert "Qwen3-0.6B latency: 1341.941 ms, 11.923 tokens/s, peak 1159.800 MB" in text
    assert "Qwen3-0.6B snapshot: DOWNLOADED via modelscope, 1433.659 MB" in text
    assert "Qwen3-0.6B 8-card inference: PASS, world size 8, backend hccl" in text
    assert "Qwen3-0.6B all-reduce totals: True, 95.384 tokens/s, peak 9278.400 MB" in text
    assert "Qwen3-0.6B CAP/QPruner generate: PASS, target layers 2/196" in text
    assert "Qwen3-0.6B CAP/QPruner quality: PASS, target layers 2/196" in text
    assert "Qwen3 quality loss: baseline 11.500, CAP delta 0.125, QPruner delta 0.062" in text
    assert "Qwen3-0.6B 8-card CAP/QPruner quality: PASS, world size 8, backend hccl" in text
    assert "Qwen3 quality all-reduce: True, baseline loss 11.500, CAP delta 0.125, QPruner delta 0.062" in text
    assert "Qwen3 patterned quality sync: PASS, world size 2, backend hccl, pattern self_attn\\.(q_proj|k_proj)$, target layers 4/196" in text
    assert "Qwen3 patterned quality all-reduce: True, baseline loss 6.797, CAP delta 0.157, WANDA delta 0.089, SparseGPT delta 0.118, QPruner delta -0.224" in text
    assert "Qwen3 quality-memory sweep: Best memory-quality point under loss delta <= 0.500 is qwen3_06b_quality_pattern_bits6" in text
    assert "Qwen3 QPruner-only quality-memory sweep: Best memory-quality point under loss delta <= 0.500 is qwen3_06b_qpruner_sweep_layers4_bits6" in text
    assert "Qwen3 QPruner scale-quality summary: QPruner scale-quality sweep reaches 16/196 target layers" in text
    assert "loss-derived approximate PPL 403.429 -> 518.013" in text
    assert "Qwen3 compression choice accuracy: PASS" in text
    assert "choice accuracy baseline 75.000%, CAP 50.000%, QPruner 75.000%" in text
    assert "choice accuracy target layers 64/196" in text
    assert "Qwen3 QPruner frontier chart: reports/qwen-qpruner-quality-memory-frontier.svg" in text
    assert "Inference acceleration summary: ACTIONABLE" in text
    assert (
        "Engineering reference: `baseline` means uncompressed serving/runtime reference, "
        "not a paper baseline."
    ) in text
    assert (
        "Paper baselines: QPruner -> LLM-Pruner; CAP -> SparseGPT, Wanda; "
        "RankAdaptor -> LoRA, AdaLoRA, without recovery."
    ) in text
    assert "Inference summary best throughput: torch fallback / QPruner 456.000 tokens/s" in text
    assert "Inference summary native memory: QPruner saves 75.000% targeted memory, 75.000% targeted storage" in text
    assert "Qwen3 CAP speedup 1.091, ratio 24.000x; QPruner speedup 1.111, bits 4.000" in text
    assert "Compression memory report: ACTIONABLE" in text
    assert (
        "Memory-first compression readout: CAP targeted memory -66.667%, "
        "QPruner targeted memory -75.000%, target layers 2/196"
    ) in text
    assert "QPruner paper baseline: LLM-Pruner" in text
    assert "CAP paper baselines: SparseGPT, Wanda" in text
    assert "Paper baseline evidence: CAP Wanda RUN_ON_ASCEND" not in text
    assert "Paper baseline evidence: QPruner LLM-Pruner PENDING_NOT_RUN_ON_ASCEND" not in text
    assert "RankAdaptor paper recovery baselines: LoRA, AdaLoRA" in text
    assert "Engineering reference baseline: uncompressed_serving_export, not paper baseline True" in text
    assert "Baseline alignment: QPruner pruning baseline LLM-Pruner -> compressed-vs-uncompressed Ascend runtime reference" in text
    assert "Baseline alignment: Engineering runtime baseline not a paper baseline uncompressed serving export -> runtime sanity, latency, throughput, and memory reference" in text
    assert "Compression memory reductions: CAP 66.667%, QPruner 75.000%, target coverage 1.020%" in text
    assert "Serving memory gap: dense export True, erases storage savings True" in text
    assert "Paper baseline coverage audit: PASS" in text
    assert (
        "Paper baseline audit note: `baseline` in serving artifacts is dense/uncompressed runtime reference, "
        "not a paper baseline."
    ) in text
    assert (
        "Paper baseline audit readout: QPruner paper baseline now has 196/196 LLM-Pruner-style Ascend evidence: "
        "25% targeted parameter pruning, loss delta 10.007, latency speedup 1.007x."
    ) in text
    assert (
        "Paper baseline audit evidence: QPruner LLM-Pruner-style PASS, target layers 196/196; "
        "prune 25.000%, loss delta 10.007, speedup 1.007x"
    ) in text
    assert (
        "Paper baseline audit evidence: CAP Wanda PASS, target layers 196/196; "
        "sparsity 50.000%, loss delta 0.662, speedup 0.976x"
    ) in text
    assert (
        "Paper baseline audit evidence: CAP SparseGPT PASS, target layers 196/196; "
        "sparsity 50.000%, loss delta 11.018, speedup 0.929x"
    ) in text
    assert (
        "Paper baseline audit evidence: RankAdaptor LoRA/BSLoRA PASS, target modules 196; "
        "LoRA world 8, val delta -0.008, BSLoRA loss delta -0.289"
    ) in text
    assert (
        "Paper baseline audit evidence: RankAdaptor recovery baselines PASS, best lora; "
        "no-recovery delta 0.000, LoRA delta -0.484, AdaLoRA-style delta 0.102"
    ) in text
    assert (
        "Paper baseline audit memory-first: CAP targeted memory -66.667%, "
        "QPruner targeted memory -50.000%, coverage 100.000%"
    ) in text
    assert (
        "Paper baseline audit gap: RankAdaptor: Need controlled LoRA vs AdaLoRA vs without-recovery runs "
        "after pruning-stage alignment."
    ) in text
    assert (
        "Paper baseline audit next run: Port or approximate AdaLoRA and no-recovery controlled RankAdaptor "
        "baselines on the same train/eval prompts."
    ) in text


    assert "Qwen3-0.6B 8-card LoRA fine-tune: PASS, world size 8, backend hccl" in text
    assert "Qwen3 LoRA all-reduce loss: True, 12.500 -> 12.100, delta -0.400" in text
    assert "Qwen3 LoRA validation loss: 13.000 -> 12.200, delta -0.800" in text
    assert "Qwen3 LoRA generation after: What does LoRA train? adapter matrices" in text
    assert "Qwen3 LoRA adapter sync: True, peak 9600.000 MB, passing ranks 8" in text
    assert "Qwen3-0.6B 8-card CAP/QPruner generate: PASS, world size 8, backend hccl" in text
    assert "Qwen3 compressed all-reduce: True, CAP 210.000 tokens/s, QPruner 224.000 tokens/s" in text
    assert "Qwen3 patterned generate sync: PASS, world size 2, backend hccl, pattern self_attn\\.(q_proj|k_proj)$, target layers 4/196" in text
    assert "Qwen3 patterned generate all-reduce: True, baseline 18.594 tokens/s, CAP 57.175 tokens/s, QPruner 54.419 tokens/s" in text
    assert "Tiny fixture TinyQwen3-Offline: FOUND, 0.722 MB" in text
    assert "8-card HCCL sync: PASS, world size 8, backend hccl" in text
    assert "RankAdaptor LoRA sync: PASS, world size 8, trainable adapter params 512" in text
    assert "TinyQwen LoRA fine-tune: PASS, loss 5.500 -> 5.250, delta -0.250" in text
    assert "TinyQwen BSLoRA fine-tune: PASS, loss 5.540 -> 5.470, delta -0.070" in text
    assert "BSLoRA sharing: trainable 3072 vs unshared 5760 adapter params" in text
    assert "Qwen3 BSLoRA fine-tune: PASS, loss 14.383 -> 14.094, delta -0.289" in text
    assert "Qwen3 BSLoRA sharing: target modules 196, trainable 32768 vs unshared 1261568 adapter params" in text
    assert "Qwen3 BSLoRA tokens trained 96, peak 1430.900 MB" in text
    assert "8-card BSLoRA sync: PASS, world size 8, backend hccl" in text
    assert "BSLoRA all-reduce: True, adapter checksum sync True" in text
    assert "CAP compression: 19.000x targeted, speedup 1.010" in text
    assert "QPruner compression: 3.789 avg bits, speedup 1.040" in text
    assert "Multi-card compressed generate: PASS, world size 8, backend hccl" in text
    assert "all-reduce throughput totals consistent: True" in text
    assert "QPruner total: 3000.000 tokens/s" in text
    assert "vLLM serving probe: FAIL, classes vllm_attention_selector_api_mismatch" in text
    assert "vLLM API diagnostics: MISMATCH, missing patch config use_per_head_quant_scales" in text
    assert "vLLM patch get_attn_backend missing: use_per_head_quant_scales, num_heads" in text
    assert "Torch serving fallback: PASS, backend torch_generate, best qpruner 456.000 tokens/s" in text
    assert "Fallback baseline 400.000 tokens/s, CAP 432.000 tokens/s, QPruner 456.000 tokens/s" in text
    assert "Fallback speedups vs baseline: CAP 1.080x, QPruner 1.140x" in text
    assert "Compressed-native torch serving: PASS, backend torch_generate_compressed_native, dense export False" in text
    assert "Native baseline 160.000 tokens/s, CAP 170.000 tokens/s, QPruner 210.000 tokens/s" in text
    assert "Native compressed modules: CAP packed 7, QPruner quantized 7, exported dense 0/0" in text
    assert "Native memory reductions: CAP 66.667%, QPruner 75.000%, dense export erases savings True" in text
    assert "Native storage footprint: baseline 8192 bytes, CAP 2731 bytes, QPruner 2048 bytes" in text
    assert "Native storage reductions: CAP 66.663%, QPruner 75.000%" in text
    assert "QPruner packed decode: PASS, device npu, storage 74.561% reduction, runtime packed_nbit_weight_codes" in text
    assert "Packed decode latency: uncached 2.000 ms, cached 1.000 ms, dense 0.900 ms" in text
    assert "Int8 code-cache path: code-cache 1.200 ms, code-cache speedup 1.667x, code-cache storage 49.561% reduction" in text
    assert "Scaled code matmul path: scaled-code 1.100 ms, scaled-code speedup 1.818x, peak 2.500 MB" in text
    assert "Decode cache speedup: 2.000x vs uncached; dense equivalent 2.222x" in text
    assert "Packed storage: codes 2048 bytes, payload 2084 bytes vs dense 8192 bytes" in text
    assert "Next kernel target: Fuse packed/quantized decode kernels on NPU." in text
    assert "Parallel suite: PASS, workers 4, cards 0,1,2,3" in text
    assert "Parallel suite start window: 0.084s, synchronized True" in text
    assert "Parallel suite sync gate: 2000.000, release lag window 0.012s" in text
    assert "Parallel suite tasks: rankadaptor=1, torch_serving_fallback=1, workflow_cap=1, workflow_qpruner=1" in text
    assert "Parallel suite best serving fallback: 456.000 tokens/s" in text
    assert "Synchronized vLLM slice: PASS, workers 8, cards 0,1,2,3,4,5,6,7" in text
    assert "Synchronized vLLM slice start window: 0.006s, synchronized True" in text
    assert "Synchronized vLLM slice tasks: vllm_serving_benchmark=8, best vLLM 18.250 tokens/s" in text
    assert "Synchronized vLLM methods: QPruner 18.250 tokens/s, CAP 17.500 tokens/s, baseline 16.000 tokens/s" in text
    assert "Synchronized vLLM metadata shim: QPruner ALREADY_INSTALLED/INSTALLED, CAP ALREADY_INSTALLED/INSTALLED, baseline ALREADY_INSTALLED/INSTALLED" in text
    assert "vLLM patch-preload probe: FAIL, classes vllm_attention_selector_api_mismatch" in text
    assert "vLLM selector-shim probe: PASS, classes none" in text
    assert "Selector shim flag: True, export run label tiny_qwen3_serving_export_npu" in text
    assert "vLLM load: CAP PASS, QPruner PASS" in text
    assert "Transformers load through both exports: CAP PASS, QPruner PASS" in text
    assert "vLLM metadata-shim benchmark: PASS, backend vllm_ascend_generate, best qpruner 222.000 tokens/s" in text
    assert "vLLM metadata shim: True, status INSTALLED" in text
    assert "vLLM baseline 200.000 tokens/s, CAP 210.000 tokens/s, QPruner 222.000 tokens/s" in text
    assert "vLLM speedups vs baseline: CAP 1.050x, QPruner 1.110x" in text
    assert "Inference bottleneck diagnosis: ACTIONABLE" in text
    assert "Bottleneck: vLLM compressed path is not faster than baseline" in text
    assert "Bottleneck detail: compressed-native QPruner preserves memory but is not end-to-end faster" in text
    assert "compressed-native QPruner qpruner_vs_baseline=0.876x" in text
    assert "Bottleneck detail: QPruner packed decode kernel microbenchmark is latency-positive" in text
    assert "QPruner packed decode scaled-code speedup 16.988x" in text
    assert "Bottleneck: vLLM compressed path is not faster than baseline" in text
    assert "Resource blocker: BLOCKED, occupied cards 8/8" in text
    assert "Resource blocker service: tdqs_qwen3-30b-a3b, Qwen3-30B-A3B, tensor_parallel_size=8" in text
    assert "Resource blocker policy: do_not_kill_unrelated_container" in text
    assert "Next optimization target: Reduce dense export/cache overhead and measure longer decode" in text
    assert "reports/ascend-910b-demo-progress.md" in text
    assert "reports/model-inventory.md" in text
    assert "reports/ascend-inference-qwen3_06b_torch_npu.md" in text
    assert "reports/model-snapshot-qwen3_06b.md" in text
    assert "reports/multicard-qwen-inference-qwen3_06b_npu.md" in text
    assert "reports/qwen-compression-quality-qwen3_06b_quality_npu.md" in text
    assert "reports/qwen-llm-pruner-baseline-qwen3_06b_llm_pruner_npu.md" in text
    assert "reports/compression-memory-report.md" in text
    assert "reports/multicard-qwen-compression-quality-qwen3_06b_quality_sync_npu.md" in text
    assert "reports/qwen-compression-generate-qwen3_06b_generate_npu.md" in text
    assert "reports/qwen-compression-generate-sweep.md" in text
    assert "reports/qwen-compression-quality-memory-sweep.md" in text
    assert "reports/qwen-qpruner-quality-memory-sweep.md" in text
    assert "reports/qwen-qpruner-scale-quality-summary.md" in text
    assert "reports/qwen-compression-choice-accuracy-qwen3_06b_choice_accuracy_npu.md" in text
    assert "artifacts/qwen_compression_choice_accuracy_qwen3_06b_choice_accuracy_npu.json" in text
    assert "reports/qwen-qpruner-quality-memory-frontier.svg" in text
    assert "reports/qwen-qpruner-quality-memory-frontier.md" in text
    assert "reports/multicard-qwen-lora-finetune-qwen3_06b_lora_sync_npu.md" in text
    assert "reports/multicard-qwen-compression-generate-qwen3_06b_compression_generate_npu.md" in text
    assert "reports/vllm-serving-probe-tiny_qwen3_serving_export_npu_preload_patch.md" in text
    assert "reports/vllm-serving-probe-tiny_qwen3_serving_export_npu_selector_shim.md" in text
    assert "reports/vllm-serving-benchmark-tiny_qwen3_serving_vllm_metadata_shim_npu.md" in text
    assert "reports/inference-bottleneck-report.md" in text
    assert "reports/inference-acceleration-summary.md" in text
    assert "reports/inference-acceleration-summary.csv" in text
    assert "reports/inference-acceleration-summary.svg" in text
    assert "artifacts/inference_acceleration_summary.json" in text
    assert "Compressed-native sweep: PASS" in text
    assert "Cached long-decode compressed-native path is latency-positive" in text
    assert "QPruner reaches 1.200x baseline at max_new_tokens=16" in text
    assert "Cache effect: max_new_tokens 1, QPruner speedup delta 0.080x" in text
    assert "Long decode effect: 1 -> 16 tokens, QPruner speedup delta 0.220x" in text
    assert "CAP runtime tradeoff: CAP stores sparse residuals as coordinates with 0 dense sparse buffers" in text
    assert "dense_weight_cache for +0.070x speedup delta" in text
    assert "Memory-preserving QPruner runtime: tiny_short_scaled_code" in text
    assert "strategy scaled_int8_code_matmul, cache mode scaled-code-matmul" in text
    assert "code-cache storage reduction 49.500%" in text
    assert "Runtime diagnosis: QPruner scaling gap 0.449x" in text
    assert "QPruner cache peak +1.000 MB" in text
    assert "fuse packed/quantized decode kernels on NPU" in text
    assert "Multi-card compression sync: PASS" in text
    assert "8-card synchronized compression path is ready" in text
    assert "QPruner 240.000 tokens/s total" in text
    assert "QPruner storage reduction 76.316%" in text
    assert "reports/multicard-compression-sync-summary.md" in text
    assert "artifacts/multicard_compression_sync_summary.json" in text
    assert "8-card grouped replay: PASS" in text
    assert "Qwen3 QPruner grouped replay memory-native: PASS" in text
    assert "memory-native workers 8/8" in text
    assert "code-cache storage 50.000%" in text
    assert "released packed-code bytes 3523215360.000" in text
    assert "live compressed payload bytes 6272.000" in text
    assert "reports/multicard-qwen-qpruner-grouped-replay-qwen3_06b_real_grouped_replay_8card_memory_native.md" in text
    assert (
        "artifacts/multicard_qwen_qpruner_grouped_replay_qwen3_06b_real_grouped_replay_8card_memory_native.json"
        in text
    )
    assert "multicard_qwen_qpruner_grouped_replay_qwen3_06b_real_grouped_replay_8card_speed_first.json" not in text
    assert "## 13. 8-card utilization proof" in text
    assert (
        "NPU utilization monitor: PASS, 57 samples, max active process cards 8, "
        "21 samples with all 8 process cards, 4 samples with all 8 AICore nonzero, "
        "per-card AICore peaks 0=8, 1=9, 2=8, 3=8, 4=9, 5=8, 6=9, 7=8"
    ) in text
    assert "Monitor artifact: artifacts/npu_monitor_qwen3_06b_real_grouped_replay_8card_utilmon.json" in text
    assert "Monitor log: logs/npu-smi-qwen3_06b_real_grouped_replay_8card_utilmon.log" in text
    assert "Demo readiness: PARTIAL_READY" in text
    assert "Readiness section: 2. Inference Acceleration READY_WITH_ACTIONABLE_GAP" in text
    assert (
        "Readiness gap: objective audit inference acceleration ACTIONABLE: "
        "vLLM compressed path is not faster than baseline"
    ) in text
    assert "Objective coverage audit: PARTIAL_READY" in text
    assert "Objective coverage section: Inference acceleration ACTIONABLE" in text
    assert "reports/objective-coverage-audit.md" in text
    assert "artifacts/objective_coverage_audit.json" in text
    assert "reports/compressed-native-cache-long-decode-sweep.md" in text
    assert "reports/compressed-native-cache-long-decode-sweep.csv" in text
    assert "reports/compressed-native-cache-long-decode-sweep.svg" in text
    assert "reports/qpruner-packed-decode-benchmark-tiny_qwen3_qpruner_decode_npu.md" in text
    assert "reports/qpruner-packed-decode-benchmark-qwen3_06b_shape_sweep_npu.md" in text
    assert "reports/qpruner-packed-decode-benchmark-qwen3_06b_shape_sweep_dtype_preserving_npu.md" in text
    assert "reports/qwen-qpruner-native-profile-qwen3_06b_native_profile_npu.md" in text
    assert "artifacts/qpruner_packed_decode_benchmark_tiny_qwen3_qpruner_decode_npu.json" in text
    assert "artifacts/qpruner_packed_decode_benchmark_qwen3_06b_shape_sweep_npu.json" in text
    assert "artifacts/qpruner_packed_decode_benchmark_qwen3_06b_shape_sweep_dtype_preserving_npu.json" in text
    assert "artifacts/qwen_qpruner_native_profile_qwen3_06b_native_profile_npu.json" in text
    assert "artifacts/compressed_native_sweep_summary.json" in text
    assert "reports/torch-serving-fallback-tiny_qwen3_serving_fallback_npu.md" in text
    assert "reports/compressed-native-torch-serving-tiny_qwen3_native_serving_npu.md" in text
    assert "reports/multicard-parallel-suite-demo_parallel_suite_npu.md" in text
    assert "reports/multicard-parallel-suite-vllm_metadata_sync_8card_npu_metrics.md" in text
    assert "reports/demo-storyboard.md" in text
    assert "reports/demo-readiness-report.md" in text


def test_demo_storyline_lists_full_target_qwen_native_profile_when_present(tmp_path):
    playback = load_playback_module()
    demo_root = tmp_path / "demo"
    (demo_root / "reports").mkdir(parents=True)
    (demo_root / "artifacts").mkdir(parents=True)
    write_json(
        demo_root / "artifacts" / "qwen_qpruner_native_profile_qwen3_06b_native_profile_npu.json",
        {"status": "PASS", "target_layer_limit": 8, "targeted_layers_total": 196},
    )
    (demo_root / "reports" / "qwen-qpruner-native-profile-qwen3_06b_native_profile_npu.md").write_text("old\n")
    write_json(
        demo_root / "artifacts" / "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_npu.json",
        {"status": "PASS", "target_layer_limit": 196, "targeted_layers_total": 196, "max_new_tokens": 4},
    )
    (demo_root / "reports" / "qwen-qpruner-native-profile-qwen3_06b_native_profile_fulltarget_bits8_npu.md").write_text(
        "full\n"
    )

    text = playback.demo_storyline(demo_root)

    assert "reports/qwen-qpruner-native-profile-qwen3_06b_native_profile_fulltarget_bits8_npu.md" in text
    assert "artifacts/qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_npu.json" in text


def test_demo_storyline_lists_selected_actual_qwen_shape_sweep_from_bottleneck_report(tmp_path):
    playback = load_playback_module()
    demo_root = tmp_path / "demo"
    (demo_root / "reports").mkdir(parents=True)
    (demo_root / "artifacts").mkdir(parents=True)
    actual_artifact = "qpruner_packed_decode_benchmark_qwen3_06b_actual_projection_bits8_shape_sweep_npu.json"
    actual_report = "qpruner-packed-decode-benchmark-qwen3_06b_actual_projection_bits8_shape_sweep_npu.md"
    write_json(
        demo_root / "artifacts" / "inference_bottleneck_report.json",
        {
            "status": "ACTIONABLE",
            "qpruner_packed_decode_shape_sweep": {
                "artifact": actual_artifact,
                "bits": 8,
                "shape_count": 14,
                "best_memory_preserving": {"label": "gate_proj_b2", "path": "code_cached"},
            },
        },
    )
    (demo_root / "reports" / actual_report).write_text("# actual shape sweep\n")

    text = playback.demo_storyline(demo_root)

    assert f"artifacts/{actual_artifact}" in text
    assert f"reports/{actual_report}" in text


def test_demo_storyline_lists_memory_first_qwen_native_profile_when_present(tmp_path):
    playback = load_playback_module()
    demo_root = tmp_path / "demo"
    (demo_root / "reports").mkdir(parents=True)
    (demo_root / "artifacts").mkdir(parents=True)
    write_json(
        demo_root
        / "artifacts"
        / "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_shapeaware_npu.json",
        {
            "status": "PASS",
            "target_layer_limit": 196,
            "targeted_layers_total": 196,
            "max_new_tokens": 4,
            "summary": {"qpruner_vs_baseline_speedup": 0.8},
            "qpruner": {
                "targeted_layers": 196,
                "latency_speedup": 0.8,
                "code_cache_storage_reduction_pct": -0.0,
            },
        },
    )
    write_json(
        demo_root
        / "artifacts"
        / "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_code_releasepacked_npu.json",
        {
            "status": "PASS",
            "target_layer_limit": 196,
            "targeted_layers_total": 196,
            "max_new_tokens": 4,
            "summary": {"qpruner_vs_baseline_speedup": 0.75},
            "qpruner_release_packed_after_cache": True,
            "qpruner": {
                "targeted_layers": 196,
                "latency_speedup": 0.75,
                "code_cache_storage_reduction_pct": 50.0,
                "release_packed_after_cache": True,
            },
        },
    )
    (
        demo_root
        / "reports"
        / "qwen-qpruner-native-profile-qwen3_06b_native_profile_fulltarget_bits8_shapeaware_npu.md"
    ).write_text("speed\n")
    (
        demo_root
        / "reports"
        / "qwen-qpruner-native-profile-qwen3_06b_native_profile_fulltarget_bits8_code_releasepacked_npu.md"
    ).write_text("memory\n")

    text = playback.demo_storyline(demo_root)

    assert "reports/qwen-qpruner-native-profile-qwen3_06b_native_profile_fulltarget_bits8_shapeaware_npu.md" in text
    assert (
        "reports/qwen-qpruner-native-profile-qwen3_06b_native_profile_fulltarget_bits8_code_releasepacked_npu.md"
        in text
    )
    assert "artifacts/qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_shapeaware_npu.json" in text
    assert (
        "artifacts/qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_code_releasepacked_npu.json"
        in text
    )


def test_demo_storyline_lists_speed_first_qwen_native_profile_tradeoff(tmp_path):
    playback = load_playback_module()
    demo_root = tmp_path / "demo"
    (demo_root / "reports").mkdir(parents=True)
    (demo_root / "artifacts").mkdir(parents=True)
    speed_name = (
        "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_"
        "scaled_prebuilt_long16_npu.json"
    )
    write_json(
        demo_root / "artifacts" / "inference_bottleneck_report.json",
        {
            "status": "ACTIONABLE",
            "qwen_qpruner_native_speed_first_profile": {
                "artifact": speed_name,
                "speedups": {"qpruner_vs_baseline": 1.392},
                "qpruner_code_cache_storage_reduction_pct": -50.0,
                "qpruner_cached_scaled_code_bytes": 880803840.0,
                "qpruner_prebuild_scaled_code_dtype_cache": True,
            },
        },
    )
    (demo_root / "reports" / playback.qwen_native_profile_report_name(speed_name)).write_text("speed\n")

    text = playback.demo_storyline(demo_root)

    assert f"reports/{playback.qwen_native_profile_report_name(speed_name)}" in text
    assert f"artifacts/{speed_name}" in text
    assert (
        "Qwen3 speed-first native profile: 1.392x baseline, measurement missing, "
        "paired_rounds missing, code-cache storage -50.000%, scaled-code dtype-cache bytes "
        "880803840.000, dense-cache bytes missing, dense-cache budget bytes missing, "
        "prebuild dtype cache True"
    ) in text


def test_demo_storyline_memory_first_qwen_native_profile_breaks_memory_ties_by_speed(tmp_path):
    playback = load_playback_module()
    demo_root = tmp_path / "demo"
    (demo_root / "reports").mkdir(parents=True)
    (demo_root / "artifacts").mkdir(parents=True)
    faster_name = (
        "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_"
        "shapeaware_releasepacked_long16_npu.json"
    )
    slower_name = (
        "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_"
        "shapepolicy_releasepacked_long16_npu.json"
    )
    for name, speedup in ((faster_name, 0.913), (slower_name, 0.792)):
        write_json(
            demo_root / "artifacts" / name,
            {
                "status": "PASS",
                "target_layer_limit": 196,
                "targeted_layers_total": 196,
                "max_new_tokens": 16,
                "qpruner_release_packed_after_cache": True,
                "summary": {"qpruner_vs_baseline_speedup": speedup},
                "qpruner": {
                    "targeted_layers": 196,
                    "latency_speedup": speedup,
                    "code_cache_storage_reduction_pct": 50.0,
                    "release_packed_after_cache": True,
                },
            },
        )
        report_name = playback.qwen_native_profile_report_name(name)
        (demo_root / "reports" / report_name).write_text(f"{speedup}\n")

    text = playback.demo_storyline(demo_root)

    assert f"artifacts/{faster_name}" in text
    assert f"reports/{playback.qwen_native_profile_report_name(faster_name)}" in text
    assert f"artifacts/{slower_name}" not in text
    assert f"reports/{playback.qwen_native_profile_report_name(slower_name)}" not in text


def test_demo_storyline_prefers_stable_qwen_native_profile_over_single_shot_speedup(tmp_path):
    playback = load_playback_module()
    demo_root = tmp_path / "demo"
    (demo_root / "reports").mkdir(parents=True)
    (demo_root / "artifacts").mkdir(parents=True)
    stable_name = (
        "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_"
        "shapeaware_releasepacked_long16_npu.json"
    )
    exploratory_name = (
        "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_"
        "actualshapepolicy_releasepacked_long16_npu.json"
    )
    for name, speedup, iters, warmup in (
        (stable_name, 0.913, 3, 1),
        (exploratory_name, 1.259, 1, 0),
    ):
        write_json(
            demo_root / "artifacts" / name,
            {
                "status": "PASS",
                "target_layer_limit": 196,
                "targeted_layers_total": 196,
                "max_new_tokens": 16,
                "iters": iters,
                "warmup": warmup,
                "qpruner_release_packed_after_cache": True,
                "summary": {"qpruner_vs_baseline_speedup": speedup},
                "qpruner": {
                    "targeted_layers": 196,
                    "latency_speedup": speedup,
                    "code_cache_storage_reduction_pct": 50.0,
                    "release_packed_after_cache": True,
                },
            },
        )
        report_name = playback.qwen_native_profile_report_name(name)
        (demo_root / "reports" / report_name).write_text(f"{speedup}\n")

    text = playback.demo_storyline(demo_root)

    assert f"artifacts/{stable_name}" in text
    assert f"reports/{playback.qwen_native_profile_report_name(stable_name)}" in text
    assert f"artifacts/{exploratory_name}" not in text
    assert f"reports/{playback.qwen_native_profile_report_name(exploratory_name)}" not in text


def test_demo_storyline_prefers_paired_qwen_native_profile_and_prints_measurement_basis(tmp_path):
    playback = load_playback_module()
    demo_root = tmp_path / "demo"
    (demo_root / "reports").mkdir(parents=True)
    (demo_root / "artifacts").mkdir(parents=True)
    sequential_name = (
        "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_"
        "shapeaware_releasepacked_stable_long16_npu.json"
    )
    paired_name = (
        "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_"
        "shapeaware_releasepacked_paired3_long16_npu.json"
    )
    for name, speedup, iters, warmup, paired_rounds, measurement_basis in (
        (sequential_name, 0.913, 3, 1, 0, "single_sequential_run"),
        (paired_name, 0.797, 1, 0, 3, "paired_interleaved_median"),
    ):
        payload = {
            "status": "PASS",
            "target_layer_limit": 196,
            "targeted_layers_total": 196,
            "max_new_tokens": 16,
            "iters": iters,
            "warmup": warmup,
            "paired_rounds": paired_rounds,
            "qpruner_release_packed_after_cache": True,
            "summary": {
                "qpruner_vs_baseline_speedup": speedup,
                "measurement_basis": measurement_basis,
            },
            "qpruner": {
                "targeted_layers": 196,
                "latency_speedup": speedup,
                "code_cache_storage_reduction_pct": 50.0,
                "release_packed_after_cache": True,
            },
        }
        if paired_rounds:
            payload["paired_summary"] = {"rounds": paired_rounds, "samples": paired_rounds * 2}
        write_json(demo_root / "artifacts" / name, payload)
        report_name = playback.qwen_native_profile_report_name(name)
        (demo_root / "reports" / report_name).write_text(f"{speedup}\n")
    write_json(
        demo_root / "artifacts" / "inference_bottleneck_report.json",
        {
            "status": "ACTIONABLE",
            "qwen_qpruner_native_profile": {
                "artifact": paired_name,
                "paired_rounds": 3,
                "measurement_basis": "paired_interleaved_median",
                "speedups": {"qpruner_vs_baseline": 0.797},
                "qpruner_code_cache_storage_reduction_pct": 50.0,
            },
        },
    )

    text = playback.demo_storyline(demo_root)

    assert f"artifacts/{paired_name}" in text
    assert f"reports/{playback.qwen_native_profile_report_name(paired_name)}" in text
    assert f"artifacts/{sequential_name}" not in text
    assert (
        "Qwen3 paired native profile: 0.797x baseline, measurement paired_interleaved_median, "
        "paired_rounds 3, code-cache storage 50.000%"
    ) in text


def test_demo_storyline_lists_qwen_native_bits_tradeoff_profiles(tmp_path):
    playback = load_playback_module()
    demo_root = tmp_path / "demo"
    (demo_root / "reports").mkdir(parents=True)
    (demo_root / "artifacts").mkdir(parents=True)
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
                {"bits_label": "bits4", "artifact": bits4},
                {"bits_label": "bits6", "artifact": bits6},
                {"bits_label": "bits8", "artifact": bits8},
            ],
        },
    )
    for name in (bits4, bits6, bits8):
        write_json(
            demo_root / "artifacts" / name,
            {
                "status": "PASS",
                "target_layer_limit": 196,
                "targeted_layers_total": 196,
                "max_new_tokens": 16,
                "iters": 3,
                "warmup": 1,
                "summary": {"qpruner_vs_baseline_speedup": 0.8},
                "qpruner": {
                    "targeted_layers": 196,
                    "latency_speedup": 0.8,
                    "code_cache_storage_reduction_pct": 50.0,
                    "release_packed_after_cache": True,
                },
            },
        )
        (demo_root / "reports" / playback.qwen_native_profile_report_name(name)).write_text(name)

    text = playback.demo_storyline(demo_root)

    for name in (bits4, bits6, bits8):
        assert f"artifacts/{name}" in text
        assert f"reports/{playback.qwen_native_profile_report_name(name)}" in text


def test_demo_storyline_prints_qwen_native_bits_tradeoff_readout(tmp_path):
    playback = load_playback_module()
    demo_root = tmp_path / "demo"
    (demo_root / "reports").mkdir(parents=True)
    (demo_root / "artifacts").mkdir(parents=True)
    write_json(
        demo_root / "artifacts" / "inference_bottleneck_report.json",
        {
            "status": "ACTIONABLE",
            "qwen_qpruner_native_bits_tradeoff": [
                {
                    "bits_label": "bits4",
                    "speedups": {"qpruner_vs_baseline": 0.784},
                    "qpruner_storage_reduction_pct": 75.059,
                    "qpruner_code_cache_storage_reduction_pct": 50.0,
                },
                {
                    "bits_label": "bits6",
                    "speedups": {"qpruner_vs_baseline": 0.803},
                    "qpruner_storage_reduction_pct": 62.5,
                    "qpruner_code_cache_storage_reduction_pct": 50.0,
                },
                {
                    "bits_label": "bits8",
                    "speedups": {"qpruner_vs_baseline": 0.913},
                    "qpruner_storage_reduction_pct": 50.0,
                    "qpruner_code_cache_storage_reduction_pct": 50.0,
                },
            ],
        },
    )

    text = playback.demo_storyline(demo_root)

    assert (
        "Qwen3 native bits tradeoff: bits4 0.784x/75.059% packed storage/50.000% code-cache storage; "
        "bits6 0.803x/62.500% packed storage/50.000% code-cache storage; "
        "bits8 0.913x/50.000% packed storage/50.000% code-cache storage"
    ) in text


def test_demo_storyline_prefers_multitask_choice_accuracy_when_present(tmp_path):
    playback = load_playback_module()
    demo_root = tmp_path / "demo"
    (demo_root / "artifacts").mkdir(parents=True)
    (demo_root / "reports").mkdir(parents=True)
    write_choice_accuracy_fixture(demo_root)
    write_multitask_choice_accuracy_fixture(demo_root)

    text = playback.demo_storyline(demo_root)

    assert "Qwen3 compression choice accuracy: PASS" in text
    assert "multi-task choice accuracy baseline 75.000%, CAP 62.500%, QPruner 75.000%" in text
    assert "choice accuracy target layers 196/196" in text
    assert "tasks 4" in text
    assert "samples 8" in text


def test_demo_storyline_prefers_fulltarget_qpruner_choice_accuracy_when_present(tmp_path):
    playback = load_playback_module()
    demo_root = tmp_path / "demo"
    (demo_root / "artifacts").mkdir(parents=True)
    (demo_root / "reports").mkdir(parents=True)
    write_choice_accuracy_fixture(demo_root)
    write_multitask_choice_accuracy_fixture(demo_root)
    write_fulltarget_qpruner_choice_accuracy_fixture(demo_root)

    text = playback.demo_storyline(demo_root)

    assert "Qwen3 compression choice accuracy: PASS" in text
    assert "full-target QPruner choice accuracy baseline 75.000%, QPruner 87.500%" in text
    assert "choice accuracy target layers 196/196" in text
    assert "QPruner compression time 13.111s" in text
    assert "CAP missing" not in text
    assert "multi-task choice accuracy baseline 75.000%, CAP 62.500%, QPruner 75.000%" not in text


def test_vllm_benchmark_playback_surfaces_failed_engine_root_cause():
    playback = load_playback_module()
    benchmark = {
        "status": "FAIL",
        "backend": "vllm_ascend_generate",
        "summary": {},
        "exports": {
            "baseline": {
                "status": "FAIL",
                "error_type": "EngineDeadError",
                "stdout_tail": "AttributeError: 'list' object has no attribute 'slot_mapping'",
            },
            "cap": {
                "status": "FAIL",
                "error_type": "EngineDeadError",
                "stdout_tail": "AttributeError: 'list' object has no attribute 'slot_mapping'",
            },
            "qpruner": {
                "status": "FAIL",
                "error_type": "EngineDeadError",
                "stdout_tail": "AttributeError: 'list' object has no attribute 'slot_mapping'",
            },
        },
    }

    lines = playback.vllm_benchmark_lines(benchmark)

    assert "vLLM benchmark root cause: baseline, CAP, QPruner: EngineDeadError" in lines
    assert "AttributeError: 'list' object has no attribute 'slot_mapping'" in lines[-1]


def test_vllm_parallel_playback_uses_best_worker_per_repeated_method():
    playback = load_playback_module()
    parallel_suite = {
        "status": "PASS",
        "world_size": 4,
        "cards": [0, 1, 2, 3],
        "launched_synchronously": True,
        "start_window_s": 0.002,
        "task_counts": {"vllm_serving_benchmark": 4},
        "aggregate": {"best_vllm_tokens_per_s": 13.0},
        "workers": [
            {
                "metrics": {
                    "method": "qpruner",
                    "tokens_per_s": 13.0,
                    "metadata_shim_status": "OLD",
                    "metadata_backend_forward_status": "INSTALLED",
                }
            },
            {
                "metrics": {
                    "method": "cap",
                    "tokens_per_s": 11.0,
                    "metadata_shim_status": "OLD",
                    "metadata_backend_forward_status": "INSTALLED",
                }
            },
            {
                "metrics": {
                    "method": "baseline",
                    "tokens_per_s": 10.0,
                    "metadata_shim_status": "OLD",
                    "metadata_backend_forward_status": "INSTALLED",
                }
            },
            {
                "metrics": {
                    "method": "qpruner",
                    "tokens_per_s": 9.0,
                    "metadata_shim_status": "NEW",
                    "metadata_backend_forward_status": "INSTALLED",
                }
            },
        ],
    }

    text = "\n".join(playback.vllm_parallel_suite_lines(parallel_suite))

    assert "Synchronized vLLM methods: QPruner 13.000 tokens/s, CAP 11.000 tokens/s, baseline 10.000 tokens/s" in text
    assert "Synchronized vLLM metadata shim: QPruner OLD/INSTALLED" in text


def test_demo_storyline_marks_missing_artifacts(tmp_path):
    playback = load_playback_module()

    text = playback.demo_storyline(tmp_path / "demo")

    assert "Manifest: missing" in text
    assert "Model inventory: missing" in text
    assert "8-card HCCL sync: missing" in text
    assert "8-card BSLoRA sync: missing" in text
    assert "TinyQwen compressed generate: missing" in text
    assert "vLLM serving probe: missing" in text


def test_inference_bottleneck_playback_formats_missing_resource_blocker_cleanly():
    playback = load_playback_module()
    lines = playback.inference_bottleneck_lines(
        {
            "status": "ACTIONABLE",
            "diagnoses": [],
            "resource_blocker": {"status": "MISSING"},
            "next_optimization_target": {"title": "Optimize compressed serving export and kernels"},
        }
    )

    assert "Resource blocker: MISSING" not in "\n".join(lines)
    assert "Resource blocker: none" in lines


def test_inference_bottleneck_playback_lists_current_vllm_hbm_rerun():
    playback = load_playback_module()
    artifact = "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu_hbm_parallel_aclpath.json"
    lines = playback.inference_bottleneck_lines(
        {
            "status": "ACTIONABLE",
            "diagnoses": [],
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
            "resource_blocker": {"status": "MISSING"},
            "next_optimization_target": {"title": "Optimize compressed serving export and kernels"},
        }
    )
    text = "\n".join(lines)

    assert "Current vLLM HBM rerun: PASS" in text
    assert f"artifacts/{artifact}" in text
    assert "cards 0,1,2" in text
    assert "best qpruner 395.891 tokens/s" in text
    assert "qpruner_vs_baseline 1.050x" in text


def test_inference_bottleneck_playback_lists_secondary_diagnoses():
    playback = load_playback_module()
    lines = playback.inference_bottleneck_lines(
        {
            "status": "ACTIONABLE",
            "diagnoses": [
                {
                    "title": "vLLM compressed path is not faster than baseline",
                    "evidence": "baseline 200.000 tokens/s, CAP 210.000 tokens/s, QPruner 222.000 tokens/s.",
                },
                {
                    "title": "compressed-native QPruner preserves memory but is not end-to-end faster",
                    "evidence": "compressed-native QPruner qpruner_vs_baseline=0.876x.",
                },
                {
                    "title": "QPruner packed decode kernel microbenchmark is latency-positive",
                    "evidence": "QPruner packed decode scaled-code speedup 16.988x.",
                },
            ],
            "resource_blocker": {"status": "MISSING"},
            "next_optimization_target": {"title": "Optimize compressed serving export and kernels"},
        }
    )
    text = "\n".join(lines)

    assert "Bottleneck: vLLM compressed path is not faster than baseline" in text
    assert "Bottleneck detail: compressed-native QPruner preserves memory but is not end-to-end faster" in text
    assert "compressed-native QPruner qpruner_vs_baseline=0.876x" in text
    assert "Bottleneck detail: QPruner packed decode kernel microbenchmark is latency-positive" in text
    assert "QPruner packed decode scaled-code speedup 16.988x" in text


def test_record_command_uses_script_and_writes_recordings_path(tmp_path):
    playback = load_playback_module()
    demo_root = tmp_path / "demo"

    command, transcript = playback.record_command(demo_root)

    assert command[:2] == ["script", "-q"]
    assert transcript.parent == demo_root / "recordings"
    assert transcript.name.startswith("ascend-910b-demo-playback-")
    assert "play_ascend_910b_demo.py" in command[-1]


def test_playback_prefers_monitor_with_more_all_card_aicore_samples(tmp_path):
    playback = load_playback_module()
    artifacts = tmp_path / "demo" / "artifacts"
    artifacts.mkdir(parents=True)
    write_json(
        artifacts / "npu_monitor_old_more_process_samples.json",
        {
            "status": "PASS",
            "run_label": "old_more_process_samples",
            "samples": 57,
            "max_active_process_cards": 8,
            "samples_with_8_active_process_cards": 21,
            "samples_with_8_nonzero_aicore": 4,
            "max_aicore_by_card": {str(card): 9 for card in range(8)},
        },
    )
    write_json(
        artifacts / "npu_monitor_stronger_aicore_samples.json",
        {
            "status": "PASS",
            "run_label": "stronger_aicore_samples",
            "samples": 32,
            "max_active_process_cards": 8,
            "samples_with_8_active_process_cards": 15,
            "samples_with_8_nonzero_aicore": 32,
            "max_aicore_by_card": {str(card): 37 for card in range(8)},
        },
    )

    selected = playback.best_npu_utilization_monitor(artifacts)

    assert selected["_artifact_name"] == "npu_monitor_stronger_aicore_samples.json"


def test_playback_prefers_grouped_replay_monitor_over_unrelated_longer_monitor(tmp_path):
    playback = load_playback_module()
    artifacts = tmp_path / "demo" / "artifacts"
    artifacts.mkdir(parents=True)
    write_json(
        artifacts / "npu_monitor_qwen3_06b_compression_generate_8card_target8_utilmon.json",
        {
            "status": "PASS",
            "run_label": "qwen3_06b_compression_generate_8card_target8_utilmon",
            "command_text": "python scripts/multicard_qwen_compression_generate_benchmark.py",
            "samples": 180,
            "max_active_process_cards": 8,
            "samples_with_8_active_process_cards": 171,
            "samples_with_8_nonzero_aicore": 180,
            "max_aicore_by_card": {str(card): 37 for card in range(8)},
        },
    )
    write_json(
        artifacts / "npu_monitor_qwen3_06b_real_grouped_replay_8card_b32_i1000_utilmon.json",
        {
            "status": "PASS",
            "run_label": "qwen3_06b_real_grouped_replay_8card_b32_i1000_utilmon",
            "command_text": "python scripts/multicard_qwen_qpruner_grouped_replay.py --cards 0,1,2,3,4,5,6,7",
            "samples": 55,
            "max_active_process_cards": 8,
            "samples_with_8_active_process_cards": 27,
            "samples_with_8_nonzero_aicore": 55,
            "max_aicore_by_card": {str(card): 36 for card in range(8)},
        },
    )

    selected = playback.best_npu_utilization_monitor(artifacts)

    assert selected["_artifact_name"] == "npu_monitor_qwen3_06b_real_grouped_replay_8card_b32_i1000_utilmon.json"


def test_playback_prefers_grouped_replay_with_matching_monitor_over_higher_unmonitored_speedup(tmp_path):
    playback = load_playback_module()
    artifacts = tmp_path / "demo" / "artifacts"
    artifacts.mkdir(parents=True)
    common_aggregate = {
        "pass_count": 8,
        "all_workers_memory_native": True,
        "memory_native_worker_count": 8,
        "min_code_cache_storage_reduction_pct": 50.0,
    }
    write_json(
        artifacts / "multicard_qwen_qpruner_grouped_replay_qwen_old_higher_speedup.json",
        {
            "status": "PASS",
            "run_label": "qwen_old_higher_speedup",
            "world_size": 8,
            "aggregate": {**common_aggregate, "best_grouped_speedup": 1.218},
            "workers": [{"command": ["python", "worker.py", "--batch-size", "32", "--iters", "1000"]}],
        },
    )
    write_json(
        artifacts / "multicard_qwen_qpruner_grouped_replay_qwen_fast_monitored.json",
        {
            "status": "PASS",
            "run_label": "qwen_fast_monitored",
            "world_size": 8,
            "aggregate": {**common_aggregate, "best_grouped_speedup": 1.189},
            "workers": [{"command": ["python", "worker.py", "--batch-size", "32", "--iters", "1000"]}],
        },
    )
    write_json(
        artifacts / "npu_monitor_qwen_fast_monitored_utilmon.json",
        {
            "status": "PASS",
            "run_label": "qwen_fast_monitored_utilmon",
            "command_text": "python scripts/multicard_qwen_qpruner_grouped_replay.py --run-label qwen_fast_monitored",
            "max_active_process_cards": 8,
            "samples_with_8_nonzero_aicore": 55,
        },
    )

    selected = playback.best_multicard_qwen_qpruner_grouped_replay(artifacts)

    assert selected["_artifact_name"] == "multicard_qwen_qpruner_grouped_replay_qwen_fast_monitored.json"


def test_demo_storyline_prefers_selected_multicard_qwen_generate_artifact(tmp_path):
    playback = load_playback_module()
    demo_root = tmp_path / "demo"
    artifacts = demo_root / "artifacts"
    old_artifact = "multicard_qwen_compression_generate_qwen3_06b_compression_generate_npu.json"
    new_artifact = "multicard_qwen_compression_generate_qwen3_06b_compression_generate_8card_target8_long8_npu.json"
    write_json(
        artifacts / old_artifact,
        {
            "status": "PASS",
            "world_size": 8,
            "backend": "hccl",
            "target_layer_limit": 2,
            "targeted_layers_total": 196,
            "aggregate": {
                "distributed_reduce_consistent": True,
                "baseline_tokens_per_s_total": 70.0,
                "cap_tokens_per_s_total": 210.0,
                "qpruner_tokens_per_s_total": 224.0,
                "pass_count": 8,
            },
        },
    )
    write_json(
        artifacts / new_artifact,
        {
            "status": "PASS",
            "world_size": 8,
            "backend": "hccl",
            "target_layer_limit": 8,
            "targeted_layers_total": 196,
            "aggregate": {
                "distributed_reduce_consistent": True,
                "baseline_tokens_per_s_total": 104.491,
                "cap_tokens_per_s_total": 208.332,
                "qpruner_tokens_per_s_total": 211.526,
                "pass_count": 8,
            },
        },
    )

    text = playback.demo_storyline(demo_root)

    assert "target layers 8/196" in text
    assert "Qwen3 compressed all-reduce: True, CAP 208.332 tokens/s, QPruner 211.526 tokens/s" in text
    assert f"artifacts/{new_artifact}" in text
    assert f"artifacts/{old_artifact}" not in text


def test_record_playback_updates_latest_transcript(tmp_path, monkeypatch):
    playback = load_playback_module()
    demo_root = tmp_path / "demo"
    latest = demo_root / "recordings" / "ascend-910b-demo-playback-latest.typescript"
    latest.parent.mkdir(parents=True)
    latest.write_text("old transcript")

    def fake_run(command, check=False):
        transcript = Path(command[2])
        transcript.write_text("fresh transcript with reports/qwen-compression-generate-sweep.md")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(playback.shutil, "which", lambda name: "/usr/bin/script" if name == "script" else "/usr/bin/python")
    monkeypatch.setattr(playback.subprocess, "run", fake_run)

    code = playback.main(["--demo-root", str(demo_root), "--record"])

    assert code == 0
    assert latest.read_text() == "fresh transcript with reports/qwen-compression-generate-sweep.md"
