import importlib.util
import json
from pathlib import Path


def load_audit_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "write_objective_coverage_audit.py"
    spec = importlib.util.spec_from_file_location("write_objective_coverage_audit", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def populate_demo(demo_root: Path) -> None:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    reports.mkdir(parents=True)
    (reports / "ascend-910b-demo-progress.md").write_text("progress\n")
    (reports / "demo-storyboard.md").write_text("storyboard\n")
    (reports / "demo-readiness-report.md").write_text("readiness\n")
    (reports / "qwen-compression-choice-accuracy-qwen3_06b_choice_accuracy_npu.md").write_text("choice accuracy\n")
    (reports / "qwen-compression-choice-accuracy-qwen3_06b_choice_accuracy_npu.csv").write_text(
        "method,status,accuracy,correct,total\n"
    )
    (demo_root / "recordings").mkdir()
    (demo_root / "recordings" / "ascend-910b-demo-playback-latest.typescript").write_text("recording\n")
    write_json(
        artifacts / "multicard_sync_summary.json",
        {"status": "PASS", "world_size": 8, "backend": "hccl"},
    )
    write_json(
        artifacts / "compatibility_issues.json",
        {
            "status": "ISSUES_FOUND",
            "ready": [{"id": "cap_qpruner_rankadaptor_npu_smoke"}],
            "issues": [{"id": "vllm_attention_selector_api_mismatch"}],
        },
    )
    write_json(
        artifacts / "inference_acceleration_summary.json",
        {
            "status": "ACTIONABLE",
            "summary": {
                "best_throughput": {"track": "torch fallback", "method_label": "QPruner", "tokens_per_s": 456.0},
                "best_native_memory": {
                    "method_label": "QPruner",
                    "memory_reduction_pct": 75.0,
                    "storage_reduction_pct": 75.0,
                },
            },
            "vllm_bottleneck": {"title": "vLLM compressed path is not faster than baseline"},
        },
    )
    write_json(
        artifacts / "inference_bottleneck_report.json",
        {
            "status": "ACTIONABLE",
            "diagnoses": [{"title": "vLLM compressed path is not faster than baseline"}],
            "next_optimization_target": {"title": "Optimize compressed serving export and kernels"},
        },
    )
    write_json(
        artifacts / "qpruner_packed_decode_benchmark_tiny_qwen3_qpruner_decode_npu.json",
        {
            "status": "PASS",
            "device": "npu",
            "runtime_storage_format": "packed_nbit_weight_codes",
            "storage_reduction_pct": 74.561,
            "code_cache_storage_reduction_pct": 49.561,
            "uncached": {"runtime_strategy": "dequantize_per_forward", "latency_ms": 2.0},
            "code_cached": {
                "runtime_strategy": "int8_code_cache_dequantize_on_device",
                "latency_ms": 1.2,
                "speedup_vs_uncached": 1.667,
            },
            "scaled_code_matmul": {
                "runtime_strategy": "scaled_int8_code_matmul",
                "latency_ms": 1.1,
                "peak_mem_mb": 2.5,
                "speedup_vs_uncached": 1.818,
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
            "next_action": "Fuse packed/quantized decode kernels on NPU.",
        },
    )
    (reports / "qpruner-packed-decode-benchmark-tiny_qwen3_qpruner_decode_npu.md").write_text(
        "# QPruner Packed Decode Benchmark\n"
    )
    write_json(
        artifacts / "qwen_qpruner_native_profile_qwen3_06b_native_profile_npu.json",
        {
            "status": "PASS",
            "target_layer_limit": 8,
            "targeted_layers_total": 196,
            "serving_dense_export": False,
            "summary": {"qpruner_vs_baseline_speedup": 0.987},
            "qpruner": {
                "status": "PASS",
                "latency_speedup": 0.987,
                "runtime_strategy": "int8_code_cache_dequantize_on_device",
                "targeted_storage_reduction_pct": 76.47,
                "code_cache_storage_reduction_pct": 26.47,
                "runtime_profile": {
                    "status": "PASS",
                    "total_forward_calls": 32,
                },
            },
        },
    )
    (reports / "qwen-qpruner-native-profile-qwen3_06b_native_profile_npu.md").write_text(
        "# Qwen QPruner Native Runtime Profile\n"
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
        artifacts / "tiny_qwen_bslora_finetune_qwen3_06b_bslora_npu.json",
        {"status": "PASS", "loss_delta": -0.289, "trainable_adapter_params": 32768},
    )
    write_json(
        artifacts / "multicard_compression_sync_summary.json",
        {
            "status": "PASS",
            "readout": "8-card synchronized compression path is ready: QPruner 240.000 tokens/s total.",
            "qwen_compression_generate": {
                "qpruner_tokens_per_s_total": 240.0,
                "qpruner_speedup_vs_baseline": 1.25,
                "distributed_reduce_consistent": True,
            },
            "compressed_native_memory": {
                "qpruner_storage_reduction_pct": 76.316,
                "qpruner_cache_peak_mem_delta_mb": 0.3,
            },
            "baseline_alignment": {
                "QPruner": "pruning baseline: LLM-Pruner -> compressed-vs-uncompressed Ascend runtime reference"
            },
        },
    )
    write_json(
        artifacts / "qwen_compression_quality_qwen3_06b_quality_npu.json",
        {
            "status": "PASS",
            "baseline": {"loss": 6.797},
            "cap": {"loss_delta": -0.134},
            "qpruner": {"loss_delta": 0.276, "average_bits": 4.0},
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
            "max_target_layer_limit": 16,
            "targeted_layers_total": 196,
            "max_coverage_pct": 8.163,
            "best_under_loss_delta": {
                "run_label": "qwen3_06b_qpruner_sweep_layers16_bits6",
                "target_layer_limit": 16,
                "targeted_layers_total": 196,
                "coverage_pct": 8.163,
                "memory_reduction_pct": 62.5,
                "loss_delta": 0.25,
                "baseline_ppl": 403.429,
                "qpruner_ppl": 518.013,
                "ppl_delta_pct": 28.403,
            },
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


def test_build_audit_maps_objective_to_evidence_and_gaps(tmp_path):
    audit = load_audit_module()
    demo_root = tmp_path / "demo"
    populate_demo(demo_root)

    report = audit.build_audit(demo_root)

    assert report["status"] == "PARTIAL_READY"
    ids = [section["id"] for section in report["sections"]]
    assert ids == [
        "ascend_910b_baseline",
        "inference_acceleration",
        "finetuning_effect",
        "model_compression",
        "demo_materials",
    ]
    baseline = report["sections"][0]
    assert baseline["status"] == "READY"
    assert "8-card HCCL sync PASS" in baseline["evidence_summary"]
    inference = report["sections"][1]
    assert inference["status"] == "ACTIONABLE"
    assert "QPruner 456.000 tokens/s" in inference["evidence_summary"]
    assert "QPruner packed decode" in inference["evidence_summary"]
    assert "packed_nbit_weight_codes" in inference["evidence_summary"]
    assert "packed decode storage 74.561%" in inference["evidence_summary"]
    assert "code-cache storage 49.561%" in inference["evidence_summary"]
    assert "code-cache 1.200 ms" in inference["evidence_summary"]
    assert "code-cache speedup 1.667x" in inference["evidence_summary"]
    assert "scaled-code matmul 1.100 ms" in inference["evidence_summary"]
    assert "scaled-code speedup 1.818x" in inference["evidence_summary"]
    assert "scaled-code peak 2.500 MB" in inference["evidence_summary"]
    assert "cache speedup 2.000x" in inference["evidence_summary"]
    assert "Qwen3 QPruner native profile 0.987x baseline" in inference["evidence_summary"]
    assert "target layers 8/196" in inference["evidence_summary"]
    assert "storage 76.470% reduction" in inference["evidence_summary"]
    assert "code-cache storage 26.470% reduction" in inference["evidence_summary"]
    assert "32 QuantizedLinear forwards" in inference["evidence_summary"]
    assert "int8_code_cache_dequantize_on_device" in inference["evidence_summary"]
    assert "artifacts/qpruner_packed_decode_benchmark_tiny_qwen3_qpruner_decode_npu.json" in inference["evidence"]
    assert "reports/qpruner-packed-decode-benchmark-tiny_qwen3_qpruner_decode_npu.md" in inference["evidence"]
    assert "artifacts/qwen_qpruner_native_profile_qwen3_06b_native_profile_npu.json" in inference["evidence"]
    assert "reports/qwen-qpruner-native-profile-qwen3_06b_native_profile_npu.md" in inference["evidence"]
    assert "vLLM compressed path is not faster than baseline" in inference["remaining_gap"]
    assert "Fuse packed/quantized decode kernels on NPU" in inference["next_action"]
    finetuning = report["sections"][2]
    assert finetuning["status"] == "READY"
    assert "validation delta -0.133" in finetuning["evidence_summary"]
    assert (
        "artifacts/multicard_qwen_lora_finetune_qwen3_06b_lora_instruction_sync_8card_steps6_20260625.json"
        in finetuning["evidence"]
    )
    compression = report["sections"][3]
    assert compression["status"] == "PARTIAL_READY"
    assert "QPruner 240.000 tokens/s total" in compression["evidence_summary"]
    assert "QPruner storage reduction 76.316%" in compression["evidence_summary"]
    assert "scale sweep 16/196 target layers" in compression["evidence_summary"]
    assert "loss-derived approximate PPL 403.429 -> 518.013" in compression["evidence_summary"]
    assert "external choice accuracy baseline 75.000%, CAP 50.000%, QPruner 75.000%" in compression[
        "evidence_summary"
    ]
    assert "choice accuracy target layers 64/196" in compression["evidence_summary"]
    assert "LLM-Pruner" in compression["evidence_summary"]
    assert "Scale beyond 16/196 target layers" in compression["remaining_gap"]
    assert "broader external benchmark tasks" in compression["remaining_gap"]
    assert "artifacts/qwen_qpruner_scale_quality_summary.json" in compression["evidence"]
    assert "artifacts/qwen_compression_choice_accuracy_qwen3_06b_choice_accuracy_npu.json" in compression["evidence"]
    assert "reports/qwen-compression-choice-accuracy-qwen3_06b_choice_accuracy_npu.md" in compression["evidence"]
    materials = report["sections"][4]
    assert materials["status"] == "READY"
    assert "recordings/ascend-910b-demo-playback-latest.typescript" in materials["evidence"]


def test_build_audit_surfaces_inference_grouped_replay_evidence(tmp_path):
    audit = load_audit_module()
    demo_root = tmp_path / "demo"
    populate_demo(demo_root)
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
                    "method_label": "QPruner",
                    "tokens_per_s": 456.0,
                },
                "best_native_memory": {
                    "method_label": "QPruner",
                    "memory_reduction_pct": 75.0,
                    "storage_reduction_pct": 75.0,
                },
                "grouped_replay": {
                    "status": "PASS",
                    "artifact": replay_artifact,
                    "run_label": "qwen3_06b_real_grouped_replay_8card_bits8_releasepacked_npu_b64_i2000_20260625",
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
                        "samples": 55,
                        "samples_with_8_active_process_cards": 27,
                        "samples_with_8_nonzero_aicore": 55,
                    },
                },
            },
            "vllm_bottleneck": {"title": "vLLM compressed path is not faster than baseline"},
        },
    )

    inference = audit.build_audit(demo_root)["sections"][1]

    assert replay_artifact in inference["evidence"]
    assert monitor_artifact in inference["evidence"]
    assert "8-card grouped replay PASS" in inference["evidence_summary"]
    assert "world 8" in inference["evidence_summary"]
    assert "memory-native workers 8/8" in inference["evidence_summary"]
    assert "code-cache storage 50.000% reduction" in inference["evidence_summary"]
    assert "best grouped speedup 1.202x" in inference["evidence_summary"]
    assert "55 all-8 AICore samples" in inference["evidence_summary"]
    assert "released packed-code bytes 3523215360.000" in inference["evidence_summary"]
    assert "live compressed payload bytes 6272.000" in inference["evidence_summary"]


def test_build_audit_prefers_full_target_qwen_native_profile(tmp_path):
    audit = load_audit_module()
    demo_root = tmp_path / "demo"
    populate_demo(demo_root)
    write_json(
        demo_root / "artifacts" / "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_npu.json",
        {
            "status": "PASS",
            "target_layer_limit": 196,
            "targeted_layers_total": 196,
            "serving_dense_export": False,
            "summary": {"qpruner_vs_baseline_speedup": 0.787},
            "max_new_tokens": 4,
            "qpruner": {
                "status": "PASS",
                "latency_speedup": 0.787,
                "runtime_strategy": "int8_code_cache_dequantize_on_device",
                "targeted_layers": 196,
                "targeted_storage_reduction_pct": 50.0,
                "code_cache_storage_reduction_pct": -0.0,
                "runtime_profile": {
                    "status": "PASS",
                    "total_forward_calls": 784,
                },
            },
        },
    )
    (demo_root / "reports" / "qwen-qpruner-native-profile-qwen3_06b_native_profile_fulltarget_bits8_npu.md").write_text(
        "# Qwen QPruner Native Runtime Profile\n"
    )

    inference = audit.build_audit(demo_root)["sections"][1]

    assert "Qwen3 QPruner native profile 0.787x baseline" in inference["evidence_summary"]
    assert "target layers 196/196" in inference["evidence_summary"]
    assert "storage 50.000% reduction" in inference["evidence_summary"]
    assert "784 QuantizedLinear forwards" in inference["evidence_summary"]
    assert "artifacts/qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_npu.json" in inference[
        "evidence"
    ]
    assert "reports/qwen-qpruner-native-profile-qwen3_06b_native_profile_fulltarget_bits8_npu.md" in inference[
        "evidence"
    ]


def test_build_audit_tracks_memory_first_qwen_native_profile_separately(tmp_path):
    audit = load_audit_module()
    demo_root = tmp_path / "demo"
    populate_demo(demo_root)
    write_json(
        demo_root
        / "artifacts"
        / "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_shapeaware_npu.json",
        {
            "status": "PASS",
            "target_layer_limit": 196,
            "targeted_layers_total": 196,
            "summary": {"qpruner_vs_baseline_speedup": 0.8},
            "max_new_tokens": 4,
            "qpruner": {
                "status": "PASS",
                "latency_speedup": 0.8,
                "runtime_strategy": "shape_aware_mixed_int8_code_cache",
                "targeted_layers": 196,
                "targeted_storage_reduction_pct": 50.0,
                "code_cache_storage_reduction_pct": -0.0,
                "runtime_profile": {"status": "PASS", "total_forward_calls": 784},
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
            "summary": {"qpruner_vs_baseline_speedup": 0.75},
            "max_new_tokens": 4,
            "qpruner_release_packed_after_cache": True,
            "qpruner": {
                "status": "PASS",
                "latency_speedup": 0.75,
                "runtime_strategy": "int8_code_cache_dequantize_on_device",
                "targeted_layers": 196,
                "targeted_storage_reduction_pct": 50.0,
                "code_cache_storage_reduction_pct": 50.0,
                "release_packed_after_cache": True,
                "released_packed_code_bytes": 440401920,
                "packed_code_bytes": 0,
                "live_compressed_payload_storage_bytes": 784,
                "runtime_profile": {"status": "PASS", "total_forward_calls": 784},
            },
        },
    )
    (
        demo_root
        / "reports"
        / "qwen-qpruner-native-profile-qwen3_06b_native_profile_fulltarget_bits8_shapeaware_npu.md"
    ).write_text("# speed native\n")
    (
        demo_root
        / "reports"
        / "qwen-qpruner-native-profile-qwen3_06b_native_profile_fulltarget_bits8_code_releasepacked_npu.md"
    ).write_text("# memory native\n")

    inference = audit.build_audit(demo_root)["sections"][1]

    assert "Qwen3 QPruner native profile 0.800x baseline" in inference["evidence_summary"]
    assert "shape_aware_mixed_int8_code_cache" in inference["evidence_summary"]
    assert "Qwen3 QPruner memory-first native profile 0.750x baseline" in inference["evidence_summary"]
    assert "code-cache storage 50.000% reduction" in inference["evidence_summary"]
    assert "released packed codes True" in inference["evidence_summary"]
    assert "live payload bytes 784" in inference["evidence_summary"]
    assert (
        "artifacts/qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_shapeaware_npu.json"
        in inference["evidence"]
    )
    assert (
        "artifacts/qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_code_releasepacked_npu.json"
        in inference["evidence"]
    )
    assert (
        "reports/qwen-qpruner-native-profile-qwen3_06b_native_profile_fulltarget_bits8_shapeaware_npu.md"
        in inference["evidence"]
    )
    assert (
        "reports/qwen-qpruner-native-profile-qwen3_06b_native_profile_fulltarget_bits8_code_releasepacked_npu.md"
        in inference["evidence"]
    )


def test_build_audit_surfaces_qwen_native_long_decode_context(tmp_path):
    audit = load_audit_module()
    demo_root = tmp_path / "demo"
    populate_demo(demo_root)
    write_json(
        demo_root
        / "artifacts"
        / "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_shapeaware_releasepacked_long16_npu.json",
        {
            "status": "PASS",
            "target_layer_limit": 196,
            "targeted_layers_total": 196,
            "summary": {"qpruner_vs_baseline_speedup": 0.913},
            "max_new_tokens": 16,
            "qpruner_release_packed_after_cache": True,
            "qpruner": {
                "status": "PASS",
                "latency_speedup": 0.913,
                "runtime_strategy": "shape_aware_mixed_int8_code_cache",
                "targeted_layers": 196,
                "targeted_storage_reduction_pct": 50.0,
                "code_cache_storage_reduction_pct": 50.0,
                "release_packed_after_cache": True,
                "released_packed_code_bytes": 440401920,
                "packed_code_bytes": 0,
                "live_compressed_payload_storage_bytes": 784,
                "runtime_profile": {"status": "PASS", "total_forward_calls": 3136},
            },
        },
    )
    (
        demo_root
        / "reports"
        / "qwen-qpruner-native-profile-qwen3_06b_native_profile_fulltarget_bits8_shapeaware_releasepacked_long16_npu.md"
    ).write_text("# long decode native\n")

    inference = audit.build_audit(demo_root)["sections"][1]

    assert "Qwen3 QPruner native profile 0.913x baseline" in inference["evidence_summary"]
    assert "max_new_tokens 16" in inference["evidence_summary"]
    assert "3136 QuantizedLinear forwards" in inference["evidence_summary"]
    assert "Qwen3 QPruner memory-first native profile 0.913x baseline" in inference["evidence_summary"]
    assert "code-cache storage 50.000% reduction" in inference["evidence_summary"]
    assert (
        "artifacts/qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_shapeaware_releasepacked_long16_npu.json"
        in inference["evidence"]
    )
    assert (
        "reports/qwen-qpruner-native-profile-qwen3_06b_native_profile_fulltarget_bits8_shapeaware_releasepacked_long16_npu.md"
        in inference["evidence"]
    )


def test_build_audit_surfaces_8card_qwen_native_memory_profile_summary(tmp_path):
    audit = load_audit_module()
    demo_root = tmp_path / "demo"
    populate_demo(demo_root)
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

    inference = audit.build_audit(demo_root)["sections"][1]

    assert f"artifacts/{summary_name}" in inference["evidence"]
    assert f"artifacts/{monitor_name}" in inference["evidence"]
    assert f"reports/{report_name}" in inference["evidence"]
    assert "Qwen3 8-card memory-native native profile PASS" in inference["evidence_summary"]
    assert "workers 8/8" in inference["evidence_summary"]
    assert "target layers 196/196" in inference["evidence_summary"]
    assert "max_new_tokens 16" in inference["evidence_summary"]
    assert "code-cache storage 50.000% reduction" in inference["evidence_summary"]
    assert "released packed-code bytes 3523215360.000" in inference["evidence_summary"]
    assert "live compressed payload bytes 6272.000" in inference["evidence_summary"]
    assert "22 all-8 process samples" in inference["evidence_summary"]
    assert "47 all-8 AICore samples" in inference["evidence_summary"]


def test_build_audit_surfaces_actual_8card_qwen_native_summary_aggregate_fields(tmp_path):
    audit = load_audit_module()
    demo_root = tmp_path / "demo"
    populate_demo(demo_root)
    summary_name = "qwen_qpruner_native_profile_8card_bits8_code_releasepacked_long16_summary_20260625.json"
    report_name = "qwen-qpruner-native-profile-8card-bits8-code-releasepacked-long16-summary-20260625.md"
    monitor_name = "npu_monitor_qwen3_06b_native_profile_8card_bits8_code_releasepacked_long16_utilmon_20260625.json"
    write_json(
        demo_root / "artifacts" / summary_name,
        {
            "status": "PASS",
            "world_size": 8,
            "pass_count": 8,
            "monitor_artifact": monitor_name,
            "aggregate": {
                "memory_native_worker_count": 8,
                "target_layer_limit_min": 196,
                "targeted_layers_total_max": 196,
                "min_code_cache_storage_reduction_pct": 50.0,
                "released_packed_code_bytes_total": 3523215360,
                "live_compressed_payload_storage_bytes_total": 6272,
            },
            "monitor": {
                "status": "PASS",
                "samples_with_8_active_process_cards": 22,
                "samples_with_8_nonzero_aicore": 47,
                "max_active_process_cards": 8,
            },
        },
    )
    (demo_root / "reports" / report_name).write_text("# 8-card memory-native profile\n")

    inference = audit.build_audit(demo_root)["sections"][1]

    assert f"artifacts/{summary_name}" in inference["evidence"]
    assert f"artifacts/{monitor_name}" in inference["evidence"]
    assert f"reports/{report_name}" in inference["evidence"]
    assert "Qwen3 8-card memory-native native profile PASS" in inference["evidence_summary"]
    assert "workers 8/8" in inference["evidence_summary"]
    assert "target layers 196/196" in inference["evidence_summary"]
    assert "code-cache storage 50.000% reduction" in inference["evidence_summary"]
    assert "released packed-code bytes 3523215360.000" in inference["evidence_summary"]
    assert "live compressed payload bytes 6272.000" in inference["evidence_summary"]
    assert "22 all-8 process samples" in inference["evidence_summary"]
    assert "47 all-8 AICore samples" in inference["evidence_summary"]


def test_build_audit_memory_first_qwen_native_profile_breaks_memory_ties_by_speed(tmp_path):
    audit = load_audit_module()
    demo_root = tmp_path / "demo"
    populate_demo(demo_root)
    faster_name = (
        "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_"
        "shapeaware_releasepacked_long16_npu.json"
    )
    slower_name = (
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
    for name, speedup in ((faster_name, 0.913), (slower_name, 0.792)):
        write_json(
            demo_root / "artifacts" / name,
            {
                **base_payload,
                "summary": {"qpruner_vs_baseline_speedup": speedup},
                "qpruner": {
                    "status": "PASS",
                    "latency_speedup": speedup,
                    "runtime_strategy": "shape_aware_mixed_int8_code_cache",
                    "targeted_layers": 196,
                    "targeted_storage_reduction_pct": 50.0,
                    "code_cache_storage_reduction_pct": 50.0,
                    "release_packed_after_cache": True,
                    "released_packed_code_bytes": 440401920,
                    "live_compressed_payload_storage_bytes": 784,
                    "runtime_profile": {"status": "PASS", "total_forward_calls": 3136},
                },
            },
        )
        report_name = audit.qwen_native_profile_report_name(name)
        (demo_root / "reports" / report_name).write_text(f"# {speedup}\n")

    inference = audit.build_audit(demo_root)["sections"][1]

    assert "Qwen3 QPruner memory-first native profile 0.913x baseline" in inference["evidence_summary"]
    assert "Qwen3 QPruner memory-first native profile 0.792x baseline" not in inference["evidence_summary"]
    assert f"artifacts/{faster_name}" in inference["evidence"]


def test_build_audit_prefers_stable_qwen_native_profile_over_single_shot_speedup(tmp_path):
    audit = load_audit_module()
    demo_root = tmp_path / "demo"
    populate_demo(demo_root)
    stable_name = (
        "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_"
        "shapeaware_releasepacked_long16_npu.json"
    )
    exploratory_name = (
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
    for name, speedup, iters, warmup in (
        (stable_name, 0.913, 3, 1),
        (exploratory_name, 1.259, 1, 0),
    ):
        write_json(
            demo_root / "artifacts" / name,
            {
                **base_payload,
                "iters": iters,
                "warmup": warmup,
                "summary": {"qpruner_vs_baseline_speedup": speedup},
                "qpruner": {
                    "status": "PASS",
                    "latency_speedup": speedup,
                    "runtime_strategy": "shape_aware_int8_code_cache",
                    "targeted_layers": 196,
                    "targeted_storage_reduction_pct": 50.0,
                    "code_cache_storage_reduction_pct": 50.0,
                    "release_packed_after_cache": True,
                    "released_packed_code_bytes": 440401920,
                    "live_compressed_payload_storage_bytes": 784,
                    "runtime_profile": {"status": "PASS", "total_forward_calls": 3136},
                },
            },
        )
        report_name = audit.qwen_native_profile_report_name(name)
        (demo_root / "reports" / report_name).write_text(f"# {speedup}\n")

    inference = audit.build_audit(demo_root)["sections"][1]

    assert "Qwen3 QPruner native profile 0.913x baseline" in inference["evidence_summary"]
    assert "Qwen3 QPruner native profile 1.259x baseline" not in inference["evidence_summary"]
    assert f"artifacts/{stable_name}" in inference["evidence"]
    assert f"artifacts/{exploratory_name}" not in inference["evidence"]


def test_build_audit_prefers_paired_qwen_native_profile_over_stable_single_order(tmp_path):
    audit = load_audit_module()
    demo_root = tmp_path / "demo"
    populate_demo(demo_root)
    paired_name = (
        "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_"
        "shapeaware_releasepacked_paired3_long16_npu.json"
    )
    stable_name = (
        "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_"
        "shapeaware_releasepacked_long16_npu.json"
    )
    base_payload = {
        "status": "PASS",
        "target_layer_limit": 196,
        "targeted_layers_total": 196,
        "max_new_tokens": 16,
        "qpruner_release_packed_after_cache": True,
    }
    for name, speedup, paired_rounds, paired_samples, basis in (
        (paired_name, 0.797, 3, 6, "paired_interleaved_median"),
        (stable_name, 0.913, None, None, None),
    ):
        summary = {"qpruner_vs_baseline_speedup": speedup}
        if basis is not None:
            summary["measurement_basis"] = basis
        payload = {
            **base_payload,
            "iters": 3,
            "warmup": 1,
            "summary": summary,
            "qpruner": {
                "status": "PASS",
                "latency_speedup": speedup,
                "runtime_strategy": "shape_aware_int8_code_cache",
                "targeted_layers": 196,
                "targeted_storage_reduction_pct": 50.0,
                "code_cache_storage_reduction_pct": 50.0,
                "release_packed_after_cache": True,
                "released_packed_code_bytes": 440401920,
                "live_compressed_payload_storage_bytes": 784,
                "runtime_profile": {"status": "PASS", "total_forward_calls": 3136},
            },
        }
        if paired_rounds is not None:
            payload["paired_rounds"] = paired_rounds
            payload["paired_summary"] = {"samples": paired_samples}
        write_json(demo_root / "artifacts" / name, payload)
        report_name = audit.qwen_native_profile_report_name(name)
        (demo_root / "reports" / report_name).write_text(f"# {speedup}\n")

    inference = audit.build_audit(demo_root)["sections"][1]

    assert "Qwen3 QPruner native profile 0.797x baseline" in inference["evidence_summary"]
    assert "Qwen3 QPruner native profile 0.913x baseline" not in inference["evidence_summary"]
    assert f"artifacts/{paired_name}" in inference["evidence"]
    assert f"artifacts/{stable_name}" not in inference["evidence"]


def test_build_audit_lists_selected_qwen_shape_sweep_from_bottleneck_report(tmp_path):
    audit = load_audit_module()
    demo_root = tmp_path / "demo"
    populate_demo(demo_root)
    actual_artifact = "qpruner_packed_decode_benchmark_qwen3_06b_actual_projection_bits8_shape_sweep_npu.json"
    actual_report = "qpruner-packed-decode-benchmark-qwen3_06b_actual_projection_bits8_shape_sweep_npu.md"
    write_json(
        demo_root / "artifacts" / "inference_bottleneck_report.json",
        {
            "status": "ACTIONABLE",
            "diagnoses": [{"title": "QPruner full-model overhead remains"}],
            "qpruner_packed_decode_shape_sweep": {
                "artifact": actual_artifact,
                "bits": 8,
                "shape_count": 14,
                "best_memory_preserving": {
                    "label": "gate_proj_b2",
                    "path": "code_cached",
                    "strategy": "int8_code_cache_dequantize_on_device",
                    "latency_ms": 0.146129,
                    "speedup_vs_uncached": 249.151,
                    "code_cache_storage_reduction_pct": 49.854,
                },
            },
            "next_optimization_target": {"title": "Reduce full-model QuantizedLinear overhead"},
        },
    )
    (demo_root / "reports" / actual_report).write_text("# actual shape sweep\n")

    inference = audit.build_audit(demo_root)["sections"][1]

    assert f"artifacts/{actual_artifact}" in inference["evidence"]
    assert f"reports/{actual_report}" in inference["evidence"]
    assert "Qwen3 projection shape sweep bits 8, shapes 14" in inference["evidence_summary"]
    assert "gate_proj_b2" in inference["evidence_summary"]
    assert "code-cache storage 49.854%" in inference["evidence_summary"]


def test_build_audit_summarizes_qwen_native_bits_tradeoff_from_bottleneck_report(tmp_path):
    audit = load_audit_module()
    demo_root = tmp_path / "demo"
    populate_demo(demo_root)
    bits4 = "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits4_code_releasepacked_long16_npu.json"
    bits6 = "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits6_code_releasepacked_long16_npu.json"
    bits8 = "qwen_qpruner_native_profile_qwen3_06b_native_profile_fulltarget_bits8_code_releasepacked_long16_npu.json"
    write_json(
        demo_root / "artifacts" / "inference_bottleneck_report.json",
        {
            "status": "ACTIONABLE",
            "diagnoses": [{"title": "QPruner full-model overhead remains"}],
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

    inference = audit.build_audit(demo_root)["sections"][1]

    assert (
        "Qwen3 native bits tradeoff bits4 0.784x/75.059% packed storage/50.000% code-cache storage"
        in inference["evidence_summary"]
    )
    assert "bits8 0.913x/50.000% packed storage/50.000% code-cache storage" in inference["evidence_summary"]
    assert f"artifacts/{bits4}" in inference["evidence"]
    assert f"artifacts/{bits8}" in inference["evidence"]


def test_build_audit_does_not_request_more_scale_after_full_qpruner_coverage(tmp_path):
    audit = load_audit_module()
    demo_root = tmp_path / "demo"
    populate_demo(demo_root)
    write_json(
        demo_root / "artifacts" / "qwen_qpruner_scale_quality_summary.json",
        {
            "status": "PASS",
            "readout": (
                "QPruner scale-quality sweep reaches 196/196 target layers (100.000% coverage); "
                "best under loss delta <= 0.500 is qwen3_06b_qpruner_sweep_proj_layers160_196_npu_layers196_bits8 "
                "with 50.000% targeted memory reduction and loss-derived approximate PPL "
                "894.892 -> 891.377 (-0.393%)."
            ),
            "max_target_layer_limit": 196,
            "targeted_layers_total": 196,
            "max_coverage_pct": 100.0,
            "best_under_loss_delta": {
                "run_label": "qwen3_06b_qpruner_sweep_proj_layers160_196_npu_layers196_bits8",
                "target_layer_limit": 196,
                "targeted_layers_total": 196,
                "coverage_pct": 100.0,
                "memory_reduction_pct": 50.0,
                "loss_delta": -0.003935813903808594,
                "baseline_ppl": 894.892,
                "qpruner_ppl": 891.377,
                "ppl_delta_pct": -0.393,
            },
        },
    )

    compression = audit.build_audit(demo_root)["sections"][3]

    assert "scale sweep 196/196 target layers" in compression["evidence_summary"]
    assert "100.000% coverage" in compression["evidence_summary"]
    assert "Scale beyond 196/196 target layers" not in compression["remaining_gap"]
    assert "broader external benchmark tasks" in compression["remaining_gap"]
    assert "Increase layer coverage" not in compression["next_action"]
    assert "quality-memory frontier" in compression["next_action"]


def test_build_audit_uses_multitask_choice_accuracy_as_broader_benchmark_evidence(tmp_path):
    audit = load_audit_module()
    demo_root = tmp_path / "demo"
    populate_demo(demo_root)
    write_json(
        demo_root / "artifacts" / "qwen_qpruner_scale_quality_summary.json",
        {
            "status": "PASS",
            "max_target_layer_limit": 196,
            "targeted_layers_total": 196,
            "max_coverage_pct": 100.0,
            "best_under_loss_delta": {
                "target_layer_limit": 196,
                "targeted_layers_total": 196,
                "coverage_pct": 100.0,
                "memory_reduction_pct": 50.0,
                "baseline_ppl": 894.892,
                "qpruner_ppl": 891.377,
            },
        },
    )
    write_json(
        demo_root / "artifacts" / "qwen_compression_choice_accuracy_qwen3_06b_multitask_choice_npu.json",
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

    compression = audit.build_audit(demo_root)["sections"][3]

    assert "multi-task external choice accuracy" in compression["evidence_summary"]
    assert "tasks 4" in compression["evidence_summary"]
    assert "choice accuracy target layers 196/196" in compression["evidence_summary"]
    assert "fixed prompt-choice table" not in compression["remaining_gap"]
    assert "larger external benchmark suite" in compression["remaining_gap"]
    assert "artifacts/qwen_compression_choice_accuracy_qwen3_06b_multitask_choice_npu.json" in compression["evidence"]


def test_build_audit_marks_compression_ready_with_full_target_qpruner_choice_probe(tmp_path):
    audit = load_audit_module()
    demo_root = tmp_path / "demo"
    populate_demo(demo_root)
    write_json(
        demo_root / "artifacts" / "qwen_qpruner_scale_quality_summary.json",
        {
            "status": "PASS",
            "max_target_layer_limit": 196,
            "targeted_layers_total": 196,
            "max_coverage_pct": 100.0,
            "best_under_loss_delta": {
                "target_layer_limit": 196,
                "targeted_layers_total": 196,
                "coverage_pct": 100.0,
                "memory_reduction_pct": 50.0,
                "baseline_ppl": 894.892,
                "qpruner_ppl": 891.377,
            },
        },
    )
    write_json(
        demo_root / "artifacts" / "qwen_compression_choice_accuracy_qwen3_06b_multitask_choice_fulltarget_qpruner_npu.json",
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
    (demo_root / "reports" / "qwen-compression-choice-accuracy-qwen3_06b_multitask_choice_fulltarget_qpruner_npu.md").write_text(
        "full target choice\n"
    )
    (demo_root / "reports" / "qwen-compression-choice-accuracy-qwen3_06b_multitask_choice_fulltarget_qpruner_npu.csv").write_text(
        "method,task,status,accuracy,correct,total\n"
    )

    compression = audit.build_audit(demo_root)["sections"][3]

    assert compression["status"] == "READY"
    assert "full-target QPruner choice accuracy baseline 75.000%, QPruner 87.500%" in compression["evidence_summary"]
    assert "choice accuracy target layers 196/196" in compression["evidence_summary"]
    assert "QPruner compression time 13.111s" in compression["evidence_summary"]
    assert "larger external benchmark suite" in compression["remaining_gap"]
    assert "full-target QPruner probe is demo-ready" in compression["remaining_gap"]
    assert "artifacts/qwen_compression_choice_accuracy_qwen3_06b_multitask_choice_fulltarget_qpruner_npu.json" in compression[
        "evidence"
    ]


def test_markdown_surfaces_a_video_friendly_completion_audit(tmp_path):
    audit = load_audit_module()
    demo_root = tmp_path / "demo"
    populate_demo(demo_root)

    text = audit.markdown(audit.build_audit(demo_root))

    assert "# TIDAL-AI Ascend Objective Coverage Audit" in text
    assert "Overall status: `PARTIAL_READY`" in text
    assert "| Ascend 910B baseline | READY |" in text
    assert "| Inference acceleration | ACTIONABLE |" in text
    assert "| Model compression | PARTIAL_READY |" in text
    assert "QPruner storage reduction 76.316%" in text
    assert "scale sweep 16/196 target layers" in text
    assert "loss-derived approximate PPL 403.429 -> 518.013" in text
    assert "external choice accuracy baseline 75.000%, CAP 50.000%, QPruner 75.000%" in text
    assert "broader external benchmark tasks" in text
    assert "vLLM compressed path is not faster than baseline" in text
    assert "QPruner packed decode" in text
    assert "packed_nbit_weight_codes" in text
    assert "packed decode storage 74.561%" in text
    assert "code-cache storage 49.561%" in text
    assert "code-cache 1.200 ms" in text
    assert "code-cache speedup 1.667x" in text
    assert "scaled-code matmul 1.100 ms" in text
    assert "scaled-code speedup 1.818x" in text
    assert "scaled-code peak 2.500 MB" in text
    assert "cache speedup 2.000x" in text
    assert "Qwen3 QPruner native profile 0.987x baseline" in text
    assert "code-cache storage 26.470% reduction" in text
    assert "32 QuantizedLinear forwards" in text
    assert "reports/objective-coverage-audit.md" in text


def test_cli_writes_json_and_markdown(tmp_path):
    audit = load_audit_module()
    demo_root = tmp_path / "demo"
    populate_demo(demo_root)

    rc = audit.main(["--demo-root", str(demo_root)])

    assert rc == 0
    assert (demo_root / "artifacts" / "objective_coverage_audit.json").exists()
    assert (demo_root / "reports" / "objective-coverage-audit.md").exists()
