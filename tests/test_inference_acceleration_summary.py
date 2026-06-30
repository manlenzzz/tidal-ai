import importlib.util
import json
from pathlib import Path


def load_summary_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "summarize_inference_acceleration.py"
    assert script.exists(), f"missing summary script: {script}"
    spec = importlib.util.spec_from_file_location("summarize_inference_acceleration", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def populate_inference_artifacts(demo_root: Path) -> None:
    artifacts = demo_root / "artifacts"
    write_json(
        artifacts / "torch_serving_fallback_tiny_qwen3_serving_fallback_npu.json",
        {
            "status": "PASS",
            "backend": "torch_generate",
            "run_label": "tiny_qwen3_serving_fallback_npu",
            "summary": {
                "best_method": "qpruner",
                "best_tokens_per_s": 456.0,
                "cap_vs_baseline_speedup": 1.08,
                "qpruner_vs_baseline_speedup": 1.14,
            },
            "exports": {
                "baseline": {"status": "PASS", "tokens_per_s": 400.0, "latency_ms": 20.0, "peak_mem_mb": 44.0},
                "cap": {"status": "PASS", "tokens_per_s": 432.0, "latency_ms": 18.5, "peak_mem_mb": 43.0},
                "qpruner": {"status": "PASS", "tokens_per_s": 456.0, "latency_ms": 17.5, "peak_mem_mb": 42.0},
            },
        },
    )
    write_json(
        artifacts / "compressed_native_torch_serving_tiny_qwen3_native_serving_npu.json",
        {
            "status": "PASS",
            "backend": "torch_generate_compressed_native",
            "max_new_tokens": 8,
            "serving_dense_export": False,
            "qpruner_cache_mode": "dense",
            "summary": {
                "best_method": "qpruner",
                "best_tokens_per_s": 210.0,
                "qpruner_vs_baseline_speedup": 1.312,
                "dense_export_erases_storage_savings": False,
            },
            "memory_reference": {
                "baseline_targeted_storage_bytes": 8192,
                "cap_targeted_storage_bytes": 2731,
                "qpruner_targeted_storage_bytes": 2048,
                "cap_targeted_storage_reduction_pct": 66.663,
                "qpruner_targeted_storage_reduction_pct": 75.0,
                "cap_targeted_param_reduction_pct": 66.667,
                "qpruner_targeted_param_reduction_pct": 75.0,
            },
            "baseline": {"status": "PASS", "tokens_per_s": 160.0, "latency_ms": 25.0, "peak_mem_mb": 40.0},
            "cap": {
                "status": "PASS",
                "tokens_per_s": 170.0,
                "latency_ms": 23.4,
                "peak_mem_mb": 39.0,
                "runtime_storage_format": "coordinate_sparse_residual",
                "runtime_strategy": "dense_weight_cache",
                "packed_layers": 7,
                "exported_dense_linears": 0,
                "targeted_param_reduction_pct": 66.667,
            },
            "qpruner": {
                "status": "PASS",
                "tokens_per_s": 210.0,
                "latency_ms": 19.0,
                "peak_mem_mb": 38.0,
                "runtime_storage_format": "packed_nbit_weight_codes",
                "runtime_strategy": "dense_weight_cache",
                "quantized_layers": 7,
                "exported_dense_linears": 0,
                "targeted_param_reduction_pct": 75.0,
                "average_bits": 4.0,
            },
        },
    )
    write_json(
        artifacts / "compressed_native_torch_serving_tiny_qwen3_native_serving_code_cache_npu.json",
        {
            "status": "PASS",
            "backend": "torch_generate_compressed_native",
            "run_label": "tiny_qwen3_native_serving_code_cache_npu",
            "max_new_tokens": 8,
            "serving_dense_export": False,
            "qpruner_cache_mode": "code",
            "summary": {
                "best_method": "qpruner",
                "best_tokens_per_s": 205.0,
                "qpruner_vs_baseline_speedup": 1.281,
                "dense_export_erases_storage_savings": False,
            },
            "memory_reference": {
                "baseline_targeted_storage_bytes": 8192,
                "cap_targeted_storage_bytes": 2731,
                "qpruner_targeted_storage_bytes": 2048,
                "qpruner_code_cache_storage_bytes": 4096,
                "cap_targeted_storage_reduction_pct": 66.663,
                "qpruner_targeted_storage_reduction_pct": 75.0,
                "qpruner_code_cache_storage_reduction_pct": 50.0,
                "cap_targeted_param_reduction_pct": 66.667,
                "qpruner_targeted_param_reduction_pct": 75.0,
            },
            "baseline": {"status": "PASS", "tokens_per_s": 160.0, "latency_ms": 25.0, "peak_mem_mb": 40.0},
            "cap": {
                "status": "PASS",
                "tokens_per_s": 170.0,
                "latency_ms": 23.4,
                "peak_mem_mb": 39.0,
                "runtime_storage_format": "coordinate_sparse_residual",
                "runtime_strategy": "dense_weight_cache",
                "packed_layers": 7,
                "exported_dense_linears": 0,
                "targeted_param_reduction_pct": 66.667,
            },
            "qpruner": {
                "status": "PASS",
                "tokens_per_s": 205.0,
                "latency_ms": 19.5,
                "peak_mem_mb": 38.0,
                "runtime_storage_format": "packed_nbit_weight_codes",
                "runtime_strategy": "int8_code_cache_dequantize_on_device",
                "quantized_layers": 7,
                "exported_dense_linears": 0,
                "targeted_param_reduction_pct": 75.0,
                "targeted_storage_reduction_pct": 75.0,
                "code_cache_storage_reduction_pct": 50.0,
                "cached_code_modules": 7,
                "cached_dense_weight_modules": 0,
                "average_bits": 4.0,
            },
        },
    )
    write_json(
        artifacts / "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu.json",
        {
            "status": "PASS",
            "backend": "vllm_ascend_generate",
            "run_label": "tiny_qwen3_serving_vllm_metadata_shim_npu",
            "summary": {
                "best_method": "qpruner",
                "best_tokens_per_s": 222.0,
                "cap_vs_baseline_speedup": 1.05,
                "qpruner_vs_baseline_speedup": 1.11,
            },
            "exports": {
                "baseline": {
                    "status": "PASS",
                    "tokens_per_s": 200.0,
                    "latency_ms": 40.0,
                    "peak_mem_mb": 80.0,
                    "memory_measurement_source": "npu-smi process table",
                    "npu_smi_process_mem_mb": 80.0,
                    "torch_peak_mem_mb": 0.0,
                },
                "cap": {
                    "status": "PASS",
                    "tokens_per_s": 210.0,
                    "latency_ms": 38.1,
                    "peak_mem_mb": 80.0,
                    "memory_measurement_source": "npu-smi process table",
                    "npu_smi_process_mem_mb": 80.0,
                    "torch_peak_mem_mb": 0.0,
                },
                "qpruner": {
                    "status": "PASS",
                    "tokens_per_s": 222.0,
                    "latency_ms": 36.0,
                    "peak_mem_mb": 80.0,
                    "memory_measurement_source": "npu-smi process table",
                    "npu_smi_process_mem_mb": 80.0,
                    "torch_peak_mem_mb": 0.0,
                },
            },
        },
    )
    write_json(
        artifacts / "inference_bottleneck_report.json",
        {
            "status": "ACTIONABLE",
            "diagnoses": [
                {
                    "id": "vllm_compressed_not_faster_than_baseline",
                    "title": "vLLM compressed path is not faster than baseline",
                    "evidence": "baseline 200.000 tokens/s, CAP 210.000 tokens/s, QPruner 222.000 tokens/s.",
                }
            ],
            "next_optimization_target": {
                "title": "Optimize compressed serving export and kernels",
                "steps": ["Keep compressed modules native in the serving runtime."],
            },
        },
    )


def populate_grouped_replay_artifacts(demo_root: Path) -> None:
    artifacts = demo_root / "artifacts"
    write_json(
        artifacts / "multicard_qwen_qpruner_grouped_replay_qwen3_06b_real_grouped_replay_8card_fast.json",
        {
            "status": "PASS",
            "run_label": "qwen3_06b_real_grouped_replay_8card_fast",
            "world_size": 8,
            "aggregate": {
                "pass_count": 8,
                "best_role": "o_proj",
                "best_grouped_speedup": 1.189,
                "mean_grouped_speedup": 1.149,
                "all_workers_memory_native": True,
                "memory_native_worker_count": 8,
                "min_code_cache_storage_reduction_pct": 50.0,
                "released_packed_code_bytes_total": 3523215360.0,
                "live_compressed_payload_storage_bytes_total": 6272.0,
            },
            "workers": [
                {
                    "command": [
                        "python",
                        "worker.py",
                        "--batch-size",
                        "32",
                        "--iters",
                        "1000",
                    ]
                }
            ],
        },
    )
    write_json(
        artifacts / "npu_monitor_qwen3_06b_real_grouped_replay_8card_fast_utilmon.json",
        {
            "status": "PASS",
            "run_label": "qwen3_06b_real_grouped_replay_8card_fast_utilmon",
            "command": [
                "python",
                "scripts/multicard_qwen_qpruner_grouped_replay.py",
                "--run-label",
                "qwen3_06b_real_grouped_replay_8card_fast",
            ],
            "samples": 55,
            "expected_cards": 8,
            "max_active_process_cards": 8,
            "samples_with_8_active_process_cards": 27,
            "samples_with_8_nonzero_aicore": 55,
            "max_aicore_by_card": {str(card): 36 for card in range(8)},
        },
    )


def row_by(summary: dict, track_id: str, method: str) -> dict:
    for row in summary["rows"]:
        if row["track_id"] == track_id and row["method"] == method:
            return row
    raise AssertionError(f"missing row {track_id}/{method}")


def test_build_summary_combines_serving_tracks_and_bottleneck(tmp_path):
    module = load_summary_module()
    demo_root = tmp_path / "demo"
    populate_inference_artifacts(demo_root)

    summary = module.build_summary(demo_root)

    assert summary["status"] == "ACTIONABLE"
    assert summary["engineering_baseline"] == {
        "artifact_method_key": "baseline",
        "display_label": "engineering reference",
        "role": "uncompressed serving/runtime reference",
        "not_paper_baseline": True,
    }
    assert summary["paper_baseline_alignment"]["qpruner"]["primary"] == ["LLM-Pruner"]
    assert {"SparseGPT", "Wanda"} <= set(summary["paper_baseline_alignment"]["cap"]["pruning_baselines"])
    assert {"LoRA", "AdaLoRA", "without recovery"} <= set(
        summary["paper_baseline_alignment"]["rankadaptor"]["recovery_baselines"]
    )
    assert len(summary["rows"]) == 9
    assert row_by(summary, "torch_fallback", "qpruner")["speedup_vs_baseline"] == 1.14
    assert row_by(summary, "compressed_native", "qpruner")["speedup_vs_baseline"] == 1.281
    assert row_by(summary, "vllm_metadata_shim", "qpruner")["speedup_vs_baseline"] == 1.11
    vllm_qpruner = row_by(summary, "vllm_metadata_shim", "qpruner")
    assert vllm_qpruner["memory_measurement_source"] == "npu-smi process table"
    assert vllm_qpruner["npu_smi_process_mem_mb"] == 80.0
    assert vllm_qpruner["torch_peak_mem_mb"] == 0.0
    native_qpruner = row_by(summary, "compressed_native", "qpruner")
    assert native_qpruner["memory_reduction_pct"] == 75.0
    assert native_qpruner["storage_reduction_pct"] == 75.0
    assert native_qpruner["code_cache_storage_reduction_pct"] == 50.0
    assert native_qpruner["runtime_storage_format"] == "packed_nbit_weight_codes"
    assert native_qpruner["runtime_strategy"] == "int8_code_cache_dequantize_on_device"
    assert native_qpruner["qpruner_cache_mode"] == "code"
    assert native_qpruner["exported_dense_linears"] == 0
    assert native_qpruner["artifact"] == (
        "artifacts/compressed_native_torch_serving_tiny_qwen3_native_serving_code_cache_npu.json"
    )
    assert summary["summary"]["best_throughput"]["track"] == "torch fallback"
    assert summary["summary"]["best_throughput"]["method"] == "qpruner"
    assert summary["vllm_bottleneck"]["title"] == "vLLM compressed path is not faster than baseline"


def test_summary_surfaces_8card_grouped_replay_evidence(tmp_path):
    module = load_summary_module()
    demo_root = tmp_path / "demo"
    populate_inference_artifacts(demo_root)
    populate_grouped_replay_artifacts(demo_root)

    summary = module.build_summary(demo_root)

    grouped = summary["summary"]["grouped_replay"]
    assert grouped["status"] == "PASS"
    assert grouped["world_size"] == 8
    assert grouped["memory_native_worker_count"] == 8
    assert grouped["pass_count"] == 8
    assert grouped["code_cache_storage_reduction_pct"] == 50.0
    assert grouped["best_grouped_speedup"] == 1.189
    assert grouped["mean_grouped_speedup"] == 1.149
    assert grouped["released_packed_code_bytes_total"] == 3523215360.0
    assert grouped["live_compressed_payload_storage_bytes_total"] == 6272.0
    assert grouped["monitor"]["samples_with_8_nonzero_aicore"] == 55
    assert grouped["monitor"]["max_active_process_cards"] == 8
    assert grouped["artifact"] == (
        "artifacts/multicard_qwen_qpruner_grouped_replay_qwen3_06b_real_grouped_replay_8card_fast.json"
    )
    assert grouped["monitor"]["artifact"] == (
        "artifacts/npu_monitor_qwen3_06b_real_grouped_replay_8card_fast_utilmon.json"
    )
    assert grouped["artifact"] in summary["evidence"]
    assert grouped["monitor"]["artifact"] in summary["evidence"]

    markdown = module.markdown(summary)
    assert "8-card grouped replay" in markdown
    assert "memory-native workers 8/8" in markdown
    assert "55 samples with all 8 AICore nonzero" in markdown
    assert "released packed-code bytes 3523215360.000" in markdown


def test_summary_prefers_longer_decode_vllm_benchmark(tmp_path):
    module = load_summary_module()
    demo_root = tmp_path / "demo"
    populate_inference_artifacts(demo_root)
    write_json(
        demo_root
        / "artifacts"
        / "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu_long_decode.json",
        {
            "status": "PASS",
            "backend": "vllm_ascend_generate",
            "run_label": "tiny_qwen3_serving_vllm_metadata_shim_npu_long_decode",
            "max_new_tokens": 16,
            "summary": {
                "best_method": "baseline",
                "best_tokens_per_s": 15.0,
                "cap_vs_baseline_speedup": 0.95,
                "qpruner_vs_baseline_speedup": 0.90,
            },
            "exports": {
                "baseline": {
                    "status": "PASS",
                    "tokens_per_s": 15.0,
                    "latency_ms": 100.0,
                    "peak_mem_mb": 82.0,
                    "memory_measurement_source": "npu-smi process table",
                    "npu_smi_process_mem_mb": 82.0,
                    "torch_peak_mem_mb": 0.0,
                },
                "cap": {
                    "status": "PASS",
                    "tokens_per_s": 14.25,
                    "latency_ms": 105.0,
                    "peak_mem_mb": 82.0,
                    "memory_measurement_source": "npu-smi process table",
                    "npu_smi_process_mem_mb": 82.0,
                    "torch_peak_mem_mb": 0.0,
                },
                "qpruner": {
                    "status": "PASS",
                    "tokens_per_s": 13.5,
                    "latency_ms": 111.0,
                    "peak_mem_mb": 82.0,
                    "memory_measurement_source": "npu-smi process table",
                    "npu_smi_process_mem_mb": 82.0,
                    "torch_peak_mem_mb": 0.0,
                },
            },
        },
    )

    summary = module.build_summary(demo_root)
    vllm_qpruner = row_by(summary, "vllm_metadata_shim", "qpruner")

    assert vllm_qpruner["artifact"] == (
        "artifacts/vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu_long_decode.json"
    )
    assert vllm_qpruner["tokens_per_s"] == 13.5
    assert vllm_qpruner["speedup_vs_baseline"] == 0.9
    assert vllm_qpruner["max_new_tokens"] == 16
    assert vllm_qpruner["memory_measurement_source"] == "npu-smi process table"
    assert vllm_qpruner["npu_smi_process_mem_mb"] == 82.0
    assert vllm_qpruner["torch_peak_mem_mb"] == 0.0
    assert (
        "artifacts/vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu_long_decode.json"
        in summary["evidence"]
    )
    markdown = module.markdown(summary)
    csv_text = module.csv_text(summary)
    assert "Max new tokens" in markdown
    assert "Peak source" in markdown
    assert "npu_smi_process_mem_mb" in csv_text
    assert "torch_peak_mem_mb" in csv_text
    assert "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu_long_decode.json" in markdown
    assert "vLLM metadata-shim | QPruner | PASS | 16 | 13.500" in markdown
    assert "npu-smi process table" in markdown
    assert "max_new_tokens" in csv_text
    assert "runtime_storage_format" in csv_text


def test_summary_marks_legacy_zero_vllm_peak_memory_unavailable(tmp_path):
    module = load_summary_module()
    demo_root = tmp_path / "demo"
    populate_inference_artifacts(demo_root)
    write_json(
        demo_root
        / "artifacts"
        / "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu_legacy_zero_hbm.json",
        {
            "status": "PASS",
            "backend": "vllm_ascend_generate",
            "run_label": "tiny_qwen3_serving_vllm_metadata_shim_npu_legacy_zero_hbm",
            "max_new_tokens": 16,
            "exports": {
                "baseline": {"status": "PASS", "tokens_per_s": 15.0, "latency_ms": 100.0, "peak_mem_mb": 0.0},
                "cap": {"status": "PASS", "tokens_per_s": 14.25, "latency_ms": 105.0, "peak_mem_mb": 0.0},
                "qpruner": {"status": "PASS", "tokens_per_s": 13.5, "latency_ms": 111.0, "peak_mem_mb": 0.0},
            },
        },
    )

    summary = module.build_summary(demo_root)
    vllm_qpruner = row_by(summary, "vllm_metadata_shim", "qpruner")

    assert vllm_qpruner["peak_mem_mb"] is None
    assert vllm_qpruner["memory_measurement_source"] == "unavailable"
    markdown = module.markdown(summary)
    assert "vLLM metadata-shim | QPruner | PASS | 16 | 13.500 | 111.000 | missing | unavailable" in markdown


def test_summary_prefers_compressed_native_code_cache_over_longer_dense_cache(tmp_path):
    module = load_summary_module()
    demo_root = tmp_path / "demo"
    populate_inference_artifacts(demo_root)
    write_json(
        demo_root
        / "artifacts"
        / "compressed_native_torch_serving_tiny_qwen3_native_serving_long_dense_cache_npu.json",
        {
            "status": "PASS",
            "backend": "torch_generate_compressed_native",
            "run_label": "tiny_qwen3_native_serving_long_dense_cache_npu",
            "max_new_tokens": 16,
            "serving_dense_export": False,
            "qpruner_cache_mode": "dense",
            "summary": {
                "best_method": "qpruner",
                "best_tokens_per_s": 196.0,
                "qpruner_vs_baseline_speedup": 1.225,
            },
            "memory_reference": {
                "baseline_targeted_storage_bytes": 8192,
                "cap_targeted_storage_bytes": 2731,
                "qpruner_targeted_storage_bytes": 2048,
                "cap_targeted_storage_reduction_pct": 66.663,
                "qpruner_targeted_storage_reduction_pct": 75.0,
                "cap_targeted_param_reduction_pct": 66.667,
                "qpruner_targeted_param_reduction_pct": 75.0,
            },
            "baseline": {"status": "PASS", "tokens_per_s": 160.0, "latency_ms": 25.0, "peak_mem_mb": 40.0},
            "cap": {
                "status": "PASS",
                "tokens_per_s": 172.0,
                "latency_ms": 23.2,
                "peak_mem_mb": 39.0,
                "runtime_storage_format": "coordinate_sparse_residual",
                "runtime_strategy": "dense_weight_cache",
                "packed_layers": 7,
                "exported_dense_linears": 0,
                "targeted_param_reduction_pct": 66.667,
            },
            "qpruner": {
                "status": "PASS",
                "tokens_per_s": 196.0,
                "latency_ms": 20.4,
                "peak_mem_mb": 38.0,
                "runtime_storage_format": "packed_nbit_weight_codes",
                "runtime_strategy": "dense_weight_cache",
                "quantized_layers": 7,
                "exported_dense_linears": 0,
                "targeted_param_reduction_pct": 75.0,
                "average_bits": 4.0,
            },
        },
    )

    summary = module.build_summary(demo_root)
    native_qpruner = row_by(summary, "compressed_native", "qpruner")

    assert native_qpruner["artifact"] == (
        "artifacts/compressed_native_torch_serving_tiny_qwen3_native_serving_code_cache_npu.json"
    )
    assert native_qpruner["runtime_strategy"] == "int8_code_cache_dequantize_on_device"
    assert native_qpruner["qpruner_cache_mode"] == "code"


def test_summary_prefers_scaled_code_matmul_over_code_cache(tmp_path):
    module = load_summary_module()
    demo_root = tmp_path / "demo"
    populate_inference_artifacts(demo_root)
    write_json(
        demo_root
        / "artifacts"
        / "compressed_native_torch_serving_tiny_qwen3_native_serving_scaled_code_matmul_npu.json",
        {
            "status": "PASS",
            "backend": "torch_generate_compressed_native",
            "run_label": "tiny_qwen3_native_serving_scaled_code_matmul_npu",
            "max_new_tokens": 8,
            "serving_dense_export": False,
            "qpruner_cache_mode": "scaled-code-matmul",
            "summary": {
                "best_method": "qpruner",
                "best_tokens_per_s": 207.0,
                "qpruner_vs_baseline_speedup": 1.294,
            },
            "memory_reference": {
                "baseline_targeted_storage_bytes": 8192,
                "cap_targeted_storage_bytes": 2731,
                "qpruner_targeted_storage_bytes": 2048,
                "qpruner_code_cache_storage_bytes": 4096,
                "cap_targeted_storage_reduction_pct": 66.663,
                "qpruner_targeted_storage_reduction_pct": 75.0,
                "qpruner_code_cache_storage_reduction_pct": 50.0,
                "cap_targeted_param_reduction_pct": 66.667,
                "qpruner_targeted_param_reduction_pct": 75.0,
            },
            "baseline": {"status": "PASS", "tokens_per_s": 160.0, "latency_ms": 25.0, "peak_mem_mb": 40.0},
            "cap": {
                "status": "PASS",
                "tokens_per_s": 170.0,
                "latency_ms": 23.4,
                "peak_mem_mb": 39.0,
                "runtime_storage_format": "coordinate_sparse_residual",
                "runtime_strategy": "dense_weight_cache",
                "packed_layers": 7,
                "exported_dense_linears": 0,
                "targeted_param_reduction_pct": 66.667,
            },
            "qpruner": {
                "status": "PASS",
                "tokens_per_s": 207.0,
                "latency_ms": 19.3,
                "peak_mem_mb": 38.0,
                "runtime_storage_format": "packed_nbit_weight_codes",
                "runtime_strategy": "scaled_int8_code_matmul",
                "quantized_layers": 7,
                "exported_dense_linears": 0,
                "targeted_param_reduction_pct": 75.0,
                "targeted_storage_reduction_pct": 75.0,
                "code_cache_storage_reduction_pct": 50.0,
                "cached_code_modules": 7,
                "scaled_code_matmul_modules": 7,
                "cached_dense_weight_modules": 0,
                "average_bits": 4.0,
            },
        },
    )

    summary = module.build_summary(demo_root)
    native_qpruner = row_by(summary, "compressed_native", "qpruner")

    assert native_qpruner["artifact"] == (
        "artifacts/compressed_native_torch_serving_tiny_qwen3_native_serving_scaled_code_matmul_npu.json"
    )
    assert native_qpruner["runtime_strategy"] == "scaled_int8_code_matmul"
    assert native_qpruner["qpruner_cache_mode"] == "scaled-code-matmul"


def test_summary_prefers_shape_aware_code_over_scaled_code_matmul(tmp_path):
    module = load_summary_module()
    demo_root = tmp_path / "demo"
    populate_inference_artifacts(demo_root)
    write_json(
        demo_root
        / "artifacts"
        / "compressed_native_torch_serving_tiny_qwen3_native_serving_scaled_code_matmul_npu.json",
        {
            "status": "PASS",
            "backend": "torch_generate_compressed_native",
            "max_new_tokens": 8,
            "serving_dense_export": False,
            "qpruner_cache_mode": "scaled-code-matmul",
            "baseline": {"status": "PASS", "tokens_per_s": 160.0, "latency_ms": 25.0},
            "cap": {"status": "PASS", "tokens_per_s": 170.0, "latency_ms": 23.4},
            "qpruner": {
                "status": "PASS",
                "tokens_per_s": 207.0,
                "latency_ms": 19.3,
                "runtime_storage_format": "packed_nbit_weight_codes",
                "runtime_strategy": "scaled_int8_code_matmul",
                "targeted_param_reduction_pct": 75.0,
                "targeted_storage_reduction_pct": 75.0,
                "code_cache_storage_reduction_pct": 50.0,
            },
        },
    )
    write_json(
        demo_root
        / "artifacts"
        / "compressed_native_torch_serving_tiny_qwen3_native_serving_shape_aware_code_npu.json",
        {
            "status": "PASS",
            "backend": "torch_generate_compressed_native",
            "max_new_tokens": 8,
            "serving_dense_export": False,
            "qpruner_cache_mode": "shape-aware-code",
            "baseline": {"status": "PASS", "tokens_per_s": 160.0, "latency_ms": 25.0},
            "cap": {"status": "PASS", "tokens_per_s": 170.0, "latency_ms": 23.4},
            "qpruner": {
                "status": "PASS",
                "tokens_per_s": 206.0,
                "latency_ms": 19.4,
                "runtime_storage_format": "packed_nbit_weight_codes",
                "runtime_strategy": "shape_aware_int8_code_cache",
                "targeted_param_reduction_pct": 75.0,
                "targeted_storage_reduction_pct": 75.0,
                "code_cache_storage_reduction_pct": 50.0,
                "shape_aware_plan": {
                    "policy": "memory_preserving_code_cache_by_projection_shape",
                    "module_count": 7,
                },
            },
        },
    )

    summary = module.build_summary(demo_root)
    native_qpruner = row_by(summary, "compressed_native", "qpruner")

    assert native_qpruner["artifact"] == (
        "artifacts/compressed_native_torch_serving_tiny_qwen3_native_serving_shape_aware_code_npu.json"
    )
    assert native_qpruner["runtime_strategy"] == "shape_aware_int8_code_cache"
    assert native_qpruner["qpruner_cache_mode"] == "shape-aware-code"


def test_markdown_and_svg_surface_demo_messages(tmp_path):
    module = load_summary_module()
    demo_root = tmp_path / "demo"
    populate_inference_artifacts(demo_root)
    summary = module.build_summary(demo_root)

    markdown = module.markdown(summary)
    svg = module.svg(summary)

    assert "# Inference Acceleration Summary" in markdown
    assert "Engineering reference: `baseline` means uncompressed serving/runtime reference, not a paper baseline." in markdown
    assert "Paper baselines: QPruner -> LLM-Pruner; CAP -> SparseGPT, Wanda; RankAdaptor -> LoRA, AdaLoRA, without recovery." in markdown
    assert "compressed-native preserves quantized/pruned modules" in markdown
    assert "packed_nbit_weight_codes" in markdown
    assert "int8_code_cache_dequantize_on_device" in markdown
    assert "code-cache storage" in markdown
    assert "Runtime storage" in markdown
    assert "dense export `False`" in markdown
    assert "vLLM compressed path is not faster than baseline" in markdown
    assert "reports/inference-acceleration-summary.svg" in markdown
    assert "Inference acceleration: throughput, latency, memory" in svg
    assert "torch fallback" in svg
    assert "compressed-native" in svg
    assert "vLLM metadata-shim" in svg
    assert "QPruner" in svg
    assert "Peak MB" in svg
    assert "Best: torch fallback / QPruner" in svg
    assert "baseline = engineering reference, not paper baseline" in svg


def test_cli_writes_json_markdown_csv_and_svg(tmp_path):
    module = load_summary_module()
    demo_root = tmp_path / "demo"
    populate_inference_artifacts(demo_root)

    rc = module.main(["--demo-root", str(demo_root)])

    assert rc == 0
    payload = json.loads((demo_root / "artifacts" / "inference_acceleration_summary.json").read_text())
    assert payload["status"] == "ACTIONABLE"
    assert len(payload["rows"]) == 9
    assert (demo_root / "reports" / "inference-acceleration-summary.md").exists()
    assert (demo_root / "reports" / "inference-acceleration-summary.csv").exists()
    assert (demo_root / "reports" / "inference-acceleration-summary.svg").exists()
