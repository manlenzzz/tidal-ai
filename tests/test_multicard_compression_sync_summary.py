import importlib.util
import json
from pathlib import Path


def load_summary_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "summarize_multicard_compression_sync.py"
    assert script.exists(), f"missing summary script: {script}"
    spec = importlib.util.spec_from_file_location("summarize_multicard_compression_sync", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def populate_artifacts(demo_root: Path) -> None:
    artifacts = demo_root / "artifacts"
    write_json(
        artifacts / "multicard_sync_summary.json",
        {"status": "PASS", "world_size": 8, "backend": "hccl"},
    )
    write_json(
        artifacts / "multicard_qwen_compression_generate_qwen3_06b_compression_generate_npu.json",
        {
            "status": "PASS",
            "model_id": "Qwen/Qwen3-0.6B",
            "world_size": 8,
            "backend": "hccl",
            "target_layer_limit": 8,
            "targeted_layers_total": 196,
            "aggregate": {
                "distributed_reduce_consistent": True,
                "baseline_tokens_per_s_total": 192.0,
                "cap_tokens_per_s_total": 224.0,
                "qpruner_tokens_per_s_total": 240.0,
                "peak_mem_mb_total": 9288.0,
                "pass_count": 8,
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
            "target_layer_limit": 8,
            "targeted_layers_total": 196,
            "aggregate": {
                "distributed_reduce_consistent": True,
                "baseline_loss_avg": 6.797,
                "cap_loss_delta_avg": -0.134,
                "qpruner_loss_delta_avg": 0.276,
                "baseline_tokens_per_s_total": 95.0,
                "cap_tokens_per_s_total": 210.0,
                "qpruner_tokens_per_s_total": 224.0,
                "peak_mem_mb_total": 9216.0,
                "pass_count": 8,
            },
        },
    )
    write_json(
        artifacts / "multicard_parallel_suite_demo_parallel_suite_npu.json",
        {
            "status": "PASS",
            "world_size": 8,
            "launched_synchronously": True,
            "start_window_s": 0.005,
            "release_lag_window_s": 0.001,
            "task_counts": {"workflow_cap": 2, "workflow_qpruner": 2, "rankadaptor": 2, "torch_serving_fallback": 2},
        },
    )
    write_json(
        artifacts / "compressed_native_sweep_summary.json",
        {
            "status": "PASS",
            "best": {
                "method": "qpruner",
                "method_label": "QPruner",
                "speedup": 1.148,
                "max_new_tokens": 1,
                "inference_cache_enabled": True,
                "storage_reduction_pct": 76.316,
            },
            "cache_effect": {"max_new_tokens": 1, "qpruner_speedup_delta": 0.086},
            "long_decode_effect": {
                "from_max_new_tokens": 1,
                "to_max_new_tokens": 16,
                "qpruner_speedup_delta": -0.154,
            },
            "runtime_diagnosis": {
                "qpruner_scaling_gap_vs_baseline": -0.236,
                "qpruner_cache_peak_mem_delta_mb": 0.3,
                "next_action": "Keep compressed weights live, then fuse packed/quantized decode kernels on NPU for long generation.",
            },
            "memory": {"best_qpruner_storage_reduction_pct": 76.316},
        },
    )
    write_json(
        artifacts / "compression_memory_report.json",
        {
            "status": "ACTIONABLE",
            "baseline_alignment_matrix": [
                {
                    "method": "QPruner",
                    "paper_baseline_role": "pruning baseline",
                    "paper_baselines": ["LLM-Pruner"],
                    "demo_reference_role": "compressed-vs-uncompressed Ascend runtime reference",
                },
                {
                    "method": "Engineering runtime baseline",
                    "paper_baseline_role": "not a paper baseline",
                    "paper_baselines": ["uncompressed serving export"],
                    "demo_reference_role": "runtime sanity, latency, throughput, and memory reference",
                },
            ],
        },
    )


def test_build_summary_combines_multicard_sync_and_memory_evidence(tmp_path):
    module = load_summary_module()
    demo_root = tmp_path / "demo"
    populate_artifacts(demo_root)

    summary = module.build_summary(demo_root)

    assert summary["status"] == "PASS"
    assert summary["readout"].startswith("8-card synchronized compression path is ready")
    assert summary["sync"]["world_size"] == 8
    assert summary["qwen_compression_generate"]["qpruner_speedup_vs_baseline"] == 1.25
    assert summary["qwen_compression_generate"]["cap_speedup_vs_baseline"] == 1.167
    assert summary["qwen_compression_quality"]["qpruner_loss_delta_avg"] == 0.276
    assert summary["compressed_native_memory"]["qpruner_storage_reduction_pct"] == 76.316
    assert summary["compressed_native_memory"]["qpruner_cache_peak_mem_delta_mb"] == 0.3
    assert summary["baseline_alignment"]["QPruner"] == "pruning baseline: LLM-Pruner -> compressed-vs-uncompressed Ascend runtime reference"
    assert summary["parallel_suite"]["start_window_s"] == 0.005
    assert "artifacts/multicard_compression_sync_summary.json" in summary["evidence"]


def test_build_summary_prefers_stronger_multicard_generate_artifact(tmp_path):
    module = load_summary_module()
    demo_root = tmp_path / "demo"
    populate_artifacts(demo_root)
    artifacts = demo_root / "artifacts"
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
                "cap_tokens_per_s_total": 140.0,
                "qpruner_tokens_per_s_total": 150.0,
                "pass_count": 8,
            },
        },
    )
    write_json(
        artifacts / "multicard_qwen_compression_generate_qwen3_06b_compression_generate_8card_target8_long8_npu.json",
        {
            "status": "PASS",
            "model_id": "Qwen/Qwen3-0.6B",
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
        },
    )

    summary = module.build_summary(demo_root)

    assert summary["qwen_compression_generate"]["artifact"] == (
        "multicard_qwen_compression_generate_qwen3_06b_compression_generate_8card_target8_long8_npu.json"
    )
    assert summary["qwen_compression_generate"]["target_layer_limit"] == 8
    assert summary["qwen_compression_generate"]["qpruner_tokens_per_s_total"] == 211.526
    assert (
        "artifacts/multicard_qwen_compression_generate_qwen3_06b_compression_generate_8card_target8_long8_npu.json"
        in summary["evidence"]
    )


def test_markdown_csv_and_svg_surface_video_readout(tmp_path):
    module = load_summary_module()
    demo_root = tmp_path / "demo"
    populate_artifacts(demo_root)
    summary = module.build_summary(demo_root)

    markdown = module.markdown(summary)
    csv_text = module.csv_text(summary)
    svg = module.svg(summary)

    assert "# Multi-Card Compression Sync Summary" in markdown
    assert "not extrapolated from one card" in markdown
    assert "QPruner 240.000 tokens/s total" in markdown
    assert "QPruner storage reduction 76.316%" in markdown
    assert "Baseline alignment: QPruner pruning baseline: LLM-Pruner" in markdown
    assert "fuse packed/quantized decode kernels on NPU" in markdown
    assert "metric,value" in csv_text
    assert "qwen_compression_generate_qpruner_speedup_vs_baseline,1.25" in csv_text
    assert "Multi-card compression sync" in svg
    assert "8-card" in svg
    assert "76.316%" in svg


def test_cli_writes_multicard_compression_sync_outputs(tmp_path):
    module = load_summary_module()
    demo_root = tmp_path / "demo"
    populate_artifacts(demo_root)

    rc = module.main(["--demo-root", str(demo_root)])

    assert rc == 0
    assert (demo_root / "artifacts" / "multicard_compression_sync_summary.json").exists()
    assert (demo_root / "reports" / "multicard-compression-sync-summary.md").exists()
    assert (demo_root / "reports" / "multicard-compression-sync-summary.csv").exists()
    assert (demo_root / "reports" / "multicard-compression-sync-summary.svg").exists()
