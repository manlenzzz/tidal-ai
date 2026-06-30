import importlib.util
import json
from pathlib import Path


def load_summary_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "summarize_qwen_grouped_mlp_sweep.py"
    assert script.exists(), f"missing summary script: {script}"
    spec = importlib.util.spec_from_file_location("summarize_qwen_grouped_mlp_sweep", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def populate_profiles(demo_root: Path) -> None:
    artifacts = demo_root / "artifacts"
    base = {
        "status": "PASS",
        "device": "npu",
        "target_layer_limit": 196,
        "targeted_layers_total": 196,
        "target_layer_coverage_pct": 100.0,
        "qpruner_grouped_mlp": True,
        "grouped_mlp_pairs": 28,
        "qpruner_grouped_attention_kv": True,
        "grouped_attention_kv_pairs": 28,
        "qpruner_cache_mode": "scaled-code-matmul",
        "qpruner_runtime_strategy": "scaled_int8_code_matmul",
            "qpruner_grouped_mlp_strategy": "bmm",
            "qpruner_cached_aux_bytes": 392,
            "qpruner_cached_code_bytes": 440401920,
            "qpruner_cached_dense_weight_bytes": 0,
            "qpruner_cached_scaled_code_bytes": 0,
            "qpruner_scaled_code_dtype_cache_budget_bytes": 0,
            "qpruner_scaled_code_dtype_cache_selection_policy": "smallest_scaled_code_dtype_cache_first",
        "runtime_profile": {
            "total_forward_calls": 2240,
            "total_forward_time_ms": 470.0,
            "grouped_mlp_pairs": 28,
            "grouped_mlp_strategy": "bmm",
            "grouped_attention_kv_pairs": 28,
        },
        "baseline": {"latency_ms_median": 1200.0, "tokens_per_s_median": 26.0},
        "qpruner": {"latency_ms_median": 1500.0, "tokens_per_s_median": 20.8},
        "speedups": {"qpruner_vs_baseline": 0.8},
        "paired": {"latency_speedup": 0.8, "tokens_ratio": 0.8},
        "memory": {
            "qpruner_code_cache_storage_bytes": 440403096,
            "qpruner_code_cache_storage_reduction_pct": 50.0,
            "qpruner_compressed_payload_storage_bytes": 220000000,
            "qpruner_targeted_storage_reduction_pct": 75.0,
            "qpruner_live_compressed_payload_storage_bytes": 784,
        },
    }
    fast = dict(base)
    fast.update(
        {
            "run_label": "qwen3_06b_native_profile_8card_grouped_mlp_sweep_test_bits8_npu0",
            "qpruner_average_bits": 8.0,
            "qpruner_dense_cache_budget_bytes": 134217728,
            "qpruner_runtime_strategy": "mixed_dense_int8_code_cache",
            "paired": {"latency_speedup": 0.91, "tokens_ratio": 0.91},
            "speedups": {"qpruner_vs_baseline": 0.91},
        }
    )
    memory = dict(base)
    memory.update(
        {
            "run_label": "qwen3_06b_native_profile_8card_grouped_mlp_sweep_test_bits4_npu1",
            "qpruner_average_bits": 4.0,
            "paired": {"latency_speedup": 0.78, "tokens_ratio": 0.78},
            "speedups": {"qpruner_vs_baseline": 0.78},
        }
    )
    write_json(artifacts / f"qwen_qpruner_native_profile_{fast['run_label']}.json", fast)
    write_json(artifacts / f"qwen_qpruner_native_profile_{memory['run_label']}.json", memory)
    write_json(
        artifacts / "npu_monitor_qwen3_06b_native_profile_8card_grouped_mlp_sweep_test_utilmon.json",
        {
            "status": "PASS",
            "run_label": "qwen3_06b_native_profile_8card_grouped_mlp_sweep_test_utilmon",
            "samples": 12,
            "max_active_process_cards": 8,
            "samples_with_8_active_process_cards": 6,
            "samples_with_8_nonzero_aicore": 10,
            "max_aicore_by_card": {str(index): 30 + index for index in range(8)},
        },
    )


def test_build_summary_collects_profiles_and_monitor(tmp_path):
    module = load_summary_module()
    demo_root = tmp_path / "demo"
    populate_profiles(demo_root)

    summary = module.build_summary(
        demo_root,
        run_group="qwen3_06b_native_profile_8card_grouped_mlp_sweep_test",
        expected_profiles=2,
        monitor_label="qwen3_06b_native_profile_8card_grouped_mlp_sweep_test_utilmon",
    )

    assert summary["status"] == "PASS"
    assert summary["profile_count"] == 2
    assert summary["pass_count"] == 2
    assert summary["monitor_status"] == "PASS"
    assert summary["max_active_process_cards"] == 8
    assert summary["samples_with_8_nonzero_aicore"] == 10
    assert summary["best_speed"]["label"].endswith("bits8_npu0")
    assert summary["best_speed"]["paired_latency_speedup"] == 0.91
    assert summary["best_code_cache_reduction"]["label"].endswith("bits4_npu1")
    assert summary["best_code_cache_reduction"]["code_cache_storage_reduction_pct"] == 50.0
    assert summary["rows"][0]["grouped_mlp_pairs"] == 28
    assert summary["rows"][0]["grouped_mlp_strategy"] == "bmm"
    assert summary["rows"][0]["grouped_attention_kv_pairs"] == 28
    assert summary["rows"][0]["target_layer_limit"] == 196
    assert summary["rows"][0]["scaled_code_dtype_cache_budget_bytes"] == 0
    assert summary["rows"][0]["scaled_code_dtype_cache_selection_policy"] == "smallest_scaled_code_dtype_cache_first"


def test_write_outputs_preserves_video_ready_report(tmp_path):
    module = load_summary_module()
    demo_root = tmp_path / "demo"
    populate_profiles(demo_root)
    summary = module.build_summary(
        demo_root,
        run_group="qwen3_06b_native_profile_8card_grouped_mlp_sweep_test",
        expected_profiles=2,
        monitor_label="qwen3_06b_native_profile_8card_grouped_mlp_sweep_test_utilmon",
    )

    module.write_outputs(summary, demo_root, "qwen3_06b_native_profile_8card_grouped_mlp_sweep_test")

    artifact = demo_root / "artifacts" / "qwen3_06b_native_profile_8card_grouped_mlp_sweep_test.json"
    report = demo_root / "reports" / "qwen3_06b_native_profile_8card_grouped_mlp_sweep_test.md"
    assert json.loads(artifact.read_text())["status"] == "PASS"
    text = report.read_text()
    assert "Qwen3 grouped-MLP full-model sweep" in text
    assert "profiles 2/2" in text
    assert "max active cards 8" in text
    assert "Grouped K/V pairs" in text
    assert "Grouped MLP strategy" in text


def test_build_summary_reads_native_profile_schema(tmp_path):
    module = load_summary_module()
    demo_root = tmp_path / "demo"
    artifacts = demo_root / "artifacts"
    run_label = "qwen3_06b_native_profile_8card_grouped_mlp_nocopy_sweep_test_bits8_npu0"
    write_json(
        artifacts / f"qwen_qpruner_native_profile_{run_label}.json",
        {
            "status": "PASS",
            "run_label": run_label,
            "device": "npu",
            "target_layer_limit": 196,
            "targeted_layers_total": 196,
            "target_layer_coverage_pct": 100.0,
            "qpruner_cache_mode": "scaled-code-matmul",
            "qpruner_dense_cache_budget_bytes": 0,
            "qpruner_dense_cache_selection_policy": "qwen_projection_hotspot_first",
            "qpruner_prebuild_scaled_code_dtype_cache": False,
            "qpruner_scaled_code_dtype_cache_budget_bytes": 8388608,
            "qpruner_scaled_code_dtype_cache_selection_policy": "qwen_projection_hotspot_first",
            "paired_summary": {
                "baseline": {"latency_ms_median": 1066.87, "tokens_per_s_median": 29.994},
                "qpruner": {"latency_ms_median": 1380.86, "tokens_per_s_median": 23.174},
                "qpruner_vs_baseline_latency_median_speedup": 0.773,
                "qpruner_vs_baseline_tokens_median_ratio": 0.773,
            },
            "memory_reference": {
                "qpruner_code_cache_storage_bytes": 440403096,
                "qpruner_code_cache_storage_reduction_pct": 50.0,
                "qpruner_targeted_storage_bytes": 440402704,
                "qpruner_targeted_storage_reduction_pct": 50.0,
            },
            "qpruner": {
                "average_bits": 8.0,
                "runtime_strategy": "scaled_int8_code_matmul",
                "grouped_mlp_strategy": "fused-2d",
                "grouped_mlp_pairs": 28,
                "grouped_attention_kv_pairs": 28,
                "quantized_layers": 196,
                "dense_cache_selection_policy": "qwen_projection_hotspot_first",
                "cached_code_bytes": 440401920,
                "cached_scaled_code_bytes": 0,
                "cached_dense_weight_bytes": 0,
                "cached_aux_bytes": 392,
                "live_compressed_payload_storage_bytes": 784,
                "compressed_payload_storage_bytes": 440402704,
                "runtime_profile": {
                    "total_forward_calls": 2240,
                    "total_forward_time_ms": 405.184992,
                    "estimated_forward_share_pct": 27.275,
                    "grouped_mlp_pairs": 28,
                    "grouped_mlp_strategy": "fused-2d",
                    "grouped_attention_kv_pairs": 28,
                },
            },
        },
    )
    write_json(
        artifacts / "npu_monitor_qwen3_06b_native_profile_8card_grouped_mlp_nocopy_sweep_test_utilmon.json",
        {
            "status": "PASS",
            "samples": 3,
            "max_active_process_cards": 8,
            "samples_with_8_active_process_cards": 2,
            "samples_with_8_nonzero_aicore": 3,
        },
    )

    summary = module.build_summary(
        demo_root,
        run_group="qwen3_06b_native_profile_8card_grouped_mlp_nocopy_sweep_test",
        expected_profiles=1,
        monitor_label="qwen3_06b_native_profile_8card_grouped_mlp_nocopy_sweep_test_utilmon",
    )

    row = summary["rows"][0]
    assert summary["status"] == "PASS"
    assert summary["best_speed"]["paired_latency_speedup"] == 0.773
    assert summary["best_code_cache_reduction"]["code_cache_storage_reduction_pct"] == 50.0
    assert row["baseline_latency_ms_median"] == 1066.87
    assert row["qpruner_latency_ms_median"] == 1380.86
    assert row["runtime_forward_time_ms"] == 405.185
    assert row["grouped_mlp_pairs"] == 28
    assert row["grouped_mlp_strategy"] == "fused-2d"
    assert row["dense_cache_selection_policy"] == "qwen_projection_hotspot_first"
    assert row["scaled_code_dtype_cache_budget_bytes"] == 8388608
    assert row["scaled_code_dtype_cache_selection_policy"] == "qwen_projection_hotspot_first"
    assert row["grouped_attention_kv_pairs"] == 28
    assert row["cached_code_bytes"] == 440401920
    text = module.markdown(summary)
    assert "Dense-cache policy" in text
    assert "Scaled dtype-cache budget" in text
    assert "qwen_projection_hotspot_first" in text
