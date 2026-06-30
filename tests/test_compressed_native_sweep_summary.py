import importlib.util
import json
from pathlib import Path


def load_summary_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "summarize_compressed_native_sweep.py"
    assert script.exists(), f"missing sweep summary script: {script}"
    spec = importlib.util.spec_from_file_location("summarize_compressed_native_sweep", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def populate_sweep_artifacts(demo_root: Path) -> list[Path]:
    artifacts = demo_root / "artifacts"
    cases = [
        (
            "compressed_native_torch_serving_tiny_short_cache_off.json",
            {
                "status": "PASS",
                "run_label": "tiny_short_cache_off",
                "backend": "torch_generate_compressed_native",
                "max_new_tokens": 1,
                "inference_cache_enabled": False,
                "serving_dense_export": False,
                "summary": {
                    "best_method": "baseline",
                    "cap_vs_baseline_speedup": 0.92,
                    "qpruner_vs_baseline_speedup": 0.90,
                },
                "memory_reference": {
                    "cap_targeted_storage_reduction_pct": 66.667,
                    "qpruner_targeted_storage_reduction_pct": 75.0,
                },
                "baseline": {"status": "PASS", "tokens_per_s": 100.0, "latency_ms": 20.0, "peak_mem_mb": 40.0},
                "cap": {
                    "status": "PASS",
                    "tokens_per_s": 92.0,
                    "latency_ms": 21.7,
                    "peak_mem_mb": 41.0,
                    "runtime_storage_format": "coordinate_sparse_residual",
                    "runtime_strategy": "coordinate_sparse_residual",
                    "dense_sparse_buffers": 0,
                    "sparse_entries": 128,
                    "cache_modules": 0,
                },
                "qpruner": {
                    "status": "PASS",
                    "tokens_per_s": 90.0,
                    "latency_ms": 22.2,
                    "peak_mem_mb": 41.0,
                    "runtime_storage_format": "packed_nbit_weight_codes",
                    "runtime_strategy": "dequantize_per_forward",
                    "cache_modules": 0,
                },
            },
        ),
        (
            "compressed_native_torch_serving_tiny_short_cache_on.json",
            {
                "status": "PASS",
                "run_label": "tiny_short_cache_on",
                "backend": "torch_generate_compressed_native",
                "max_new_tokens": 1,
                "inference_cache_enabled": True,
                "serving_dense_export": False,
                "summary": {
                    "best_method": "baseline",
                    "cap_vs_baseline_speedup": 0.99,
                    "qpruner_vs_baseline_speedup": 0.98,
                },
                "memory_reference": {
                    "cap_targeted_storage_reduction_pct": 66.667,
                    "qpruner_targeted_storage_reduction_pct": 75.0,
                },
                "baseline": {"status": "PASS", "tokens_per_s": 100.0, "latency_ms": 20.0, "peak_mem_mb": 40.0},
                "cap": {
                    "status": "PASS",
                    "tokens_per_s": 99.0,
                    "latency_ms": 20.2,
                    "peak_mem_mb": 42.0,
                    "runtime_storage_format": "coordinate_sparse_residual",
                    "runtime_strategy": "dense_weight_cache",
                    "dense_sparse_buffers": 0,
                    "sparse_entries": 128,
                    "cache_modules": 7,
                },
                "qpruner": {
                    "status": "PASS",
                    "tokens_per_s": 98.0,
                    "latency_ms": 20.4,
                    "peak_mem_mb": 42.0,
                    "runtime_storage_format": "packed_nbit_weight_codes",
                    "runtime_strategy": "dense_weight_cache",
                    "cache_modules": 7,
                },
            },
        ),
        (
            "compressed_native_torch_serving_tiny_long_cache_on.json",
            {
                "status": "PASS",
                "run_label": "tiny_long_cache_on",
                "backend": "torch_generate_compressed_native",
                "max_new_tokens": 16,
                "inference_cache_enabled": True,
                "serving_dense_export": False,
                "summary": {
                    "best_method": "qpruner",
                    "cap_vs_baseline_speedup": 1.10,
                    "qpruner_vs_baseline_speedup": 1.20,
                },
                "memory_reference": {
                    "cap_targeted_storage_reduction_pct": 66.667,
                    "qpruner_targeted_storage_reduction_pct": 75.0,
                },
                "baseline": {"status": "PASS", "tokens_per_s": 200.0, "latency_ms": 160.0, "peak_mem_mb": 48.0},
                "cap": {
                    "status": "PASS",
                    "tokens_per_s": 220.0,
                    "latency_ms": 145.5,
                    "peak_mem_mb": 50.0,
                    "runtime_storage_format": "coordinate_sparse_residual",
                    "runtime_strategy": "dense_weight_cache",
                    "dense_sparse_buffers": 0,
                    "sparse_entries": 128,
                    "cache_modules": 7,
                },
                "qpruner": {
                    "status": "PASS",
                    "tokens_per_s": 240.0,
                    "latency_ms": 133.3,
                    "peak_mem_mb": 50.0,
                    "runtime_storage_format": "packed_nbit_weight_codes",
                    "runtime_strategy": "dense_weight_cache",
                    "cache_modules": 7,
                },
            },
        ),
    ]
    paths = []
    for name, payload in cases:
        path = artifacts / name
        write_json(path, payload)
        paths.append(path)
    return paths


def test_build_summary_identifies_cache_and_long_decode_effects(tmp_path):
    module = load_summary_module()
    demo_root = tmp_path / "demo"
    paths = populate_sweep_artifacts(demo_root)

    summary = module.build_summary(paths, demo_root=demo_root)

    assert summary["status"] == "PASS"
    assert len(summary["rows"]) == 3
    assert summary["rows"][0]["cap_runtime_storage_format"] == "coordinate_sparse_residual"
    assert summary["rows"][0]["cap_runtime_strategy"] == "coordinate_sparse_residual"
    assert summary["rows"][0]["cap_dense_sparse_buffers"] == 0
    assert summary["rows"][1]["cap_runtime_strategy"] == "dense_weight_cache"
    assert summary["best"]["label"] == "tiny_long_cache_on"
    assert summary["best"]["method"] == "qpruner"
    assert summary["best"]["qpruner_vs_baseline_speedup"] == 1.2
    assert summary["cache_effect"]["qpruner_speedup_delta"] == 0.08
    assert summary["long_decode_effect"]["qpruner_speedup_delta"] == 0.22
    assert summary["memory"]["best_qpruner_storage_reduction_pct"] == 75.0
    assert summary["runtime_diagnosis"]["baseline_decode_scaling"] == 2.0
    assert summary["runtime_diagnosis"]["qpruner_decode_scaling"] == 2.449
    assert summary["runtime_diagnosis"]["qpruner_scaling_gap_vs_baseline"] == 0.449
    assert summary["runtime_diagnosis"]["qpruner_cache_peak_mem_delta_mb"] == 1.0
    assert summary["runtime_diagnosis"]["qpruner_cache_vs_baseline_peak_mem_delta_mb"] == 2.0
    assert summary["runtime_diagnosis"]["cap_coordinate_uncached_speedup"] == 0.92
    assert summary["runtime_diagnosis"]["cap_coordinate_uncached_peak_mem_mb"] == 41.0
    assert summary["runtime_diagnosis"]["cap_cached_runtime_strategy"] == "dense_weight_cache"
    assert summary["runtime_diagnosis"]["cap_coordinate_runtime_strategy"] == "coordinate_sparse_residual"
    assert summary["runtime_diagnosis"]["cap_dense_sparse_buffers"] == 0
    assert summary["runtime_diagnosis"]["cap_runtime_tradeoff"] == (
        "CAP stores sparse residuals as coordinates with 0 dense sparse buffers; "
        "uncached coordinate path reaches 0.920x baseline at 41.000 MB peak, "
        "while cache switches runtime strategy to dense_weight_cache for +0.070x speedup delta."
    )
    assert summary["runtime_diagnosis"]["cache_tradeoff"] == (
        "QPruner cache adds 1.000 MB over uncached QPruner peak at max_new_tokens=1 "
        "for 0.080x speedup delta; +2.000 MB vs uncached baseline peak."
    )
    assert summary["runtime_diagnosis"]["next_action"] == (
        "Keep compressed weights live, then fuse packed/quantized decode kernels on NPU for long generation."
    )
    assert summary["readout"].startswith("Cached long-decode compressed-native path is latency-positive")


def test_build_summary_identifies_best_memory_preserving_qpruner_runtime(tmp_path):
    module = load_summary_module()
    demo_root = tmp_path / "demo"
    paths = populate_sweep_artifacts(demo_root)
    artifacts = demo_root / "artifacts"
    runtime_cases = [
        (
            "compressed_native_torch_serving_tiny_short_code_cache.json",
            "tiny_short_code_cache",
            "code",
            "int8_code_cache_dequantize_on_device",
            1.05,
            105.0,
            49.5,
            {"cached_code_modules": 7, "cached_code_bytes": 4096},
        ),
        (
            "compressed_native_torch_serving_tiny_short_scaled_code.json",
            "tiny_short_scaled_code",
            "scaled-code-matmul",
            "scaled_int8_code_matmul",
            1.18,
            118.0,
            49.5,
            {"scaled_code_matmul_modules": 7, "cached_code_modules": 7, "cached_code_bytes": 4096},
        ),
        (
            "compressed_native_torch_serving_tiny_short_shape_aware_mixed.json",
            "tiny_short_shape_aware_mixed",
            "shape-aware-code",
            "shape_aware_mixed_int8_code_cache",
            1.21,
            121.0,
            49.5,
            {
                "shape_aware_cache_modules": 7,
                "shape_aware_scaled_code_modules": 4,
                "cached_code_modules": 7,
                "cached_code_bytes": 4096,
            },
        ),
        (
            "compressed_native_torch_serving_tiny_short_shape_policy.json",
            "tiny_short_shape_policy",
            "shape-aware-code",
            "shape_aware_int8_code_cache",
            1.19,
            119.0,
            49.5,
            {
                "shape_aware_cache_modules": 7,
                "shape_aware_scaled_code_modules": 0,
                "cached_code_modules": 7,
                "cached_code_bytes": 4096,
                "shape_aware_plan": {
                    "shape_policy_source": "artifacts/qpruner_packed_decode_benchmark_tiny_shape_policy.json",
                    "shape_policy_match_count": 7,
                    "strategy_source_counts": {"shape_sweep_artifact": 7},
                },
            },
        ),
    ]
    for filename, label, cache_mode, runtime_strategy, speedup, tokens_per_s, code_storage, extra in runtime_cases:
        path = artifacts / filename
        payload = {
            "status": "PASS",
            "run_label": label,
            "backend": "torch_generate_compressed_native",
            "max_new_tokens": 1,
            "inference_cache_enabled": False,
            "qpruner_cache_mode": cache_mode,
            "serving_dense_export": False,
            "summary": {
                "best_method": "qpruner",
                "cap_vs_baseline_speedup": 0.92,
                "qpruner_vs_baseline_speedup": speedup,
            },
            "memory_reference": {
                "cap_targeted_storage_reduction_pct": 66.667,
                "qpruner_targeted_storage_reduction_pct": 75.0,
                "qpruner_code_cache_storage_reduction_pct": code_storage,
            },
            "baseline": {"status": "PASS", "tokens_per_s": 100.0, "latency_ms": 20.0, "peak_mem_mb": 40.0},
            "cap": {
                "status": "PASS",
                "tokens_per_s": 92.0,
                "latency_ms": 21.7,
                "peak_mem_mb": 41.0,
                "runtime_storage_format": "coordinate_sparse_residual",
                "runtime_strategy": "coordinate_sparse_residual",
                "dense_sparse_buffers": 0,
                "sparse_entries": 128,
                "cache_modules": 0,
            },
            "qpruner": {
                "status": "PASS",
                "tokens_per_s": tokens_per_s,
                "latency_ms": round(20.0 / speedup, 3),
                "peak_mem_mb": 41.5,
                "runtime_storage_format": "packed_nbit_weight_codes",
                "runtime_strategy": runtime_strategy,
                "cache_modules": 7,
                "cache_mode": cache_mode,
                "cached_dense_weight_modules": 0,
                "code_cache_storage_reduction_pct": code_storage,
                **extra,
            },
        }
        write_json(path, payload)
        paths.append(path)

    summary = module.build_summary(paths, demo_root=demo_root)

    best = summary["memory_preserving_qpruner_best"]
    assert best["label"] == "tiny_short_shape_policy"
    assert best["qpruner_cache_mode"] == "shape-aware-code"
    assert best["qpruner_runtime_strategy"] == "shape_aware_int8_code_cache"
    assert best["qpruner_vs_baseline_speedup"] == 1.19
    assert best["qpruner_code_cache_storage_reduction_pct"] == 49.5
    assert best["qpruner_cached_dense_weight_modules"] == 0
    assert best["qpruner_cached_code_modules"] == 7
    assert best["qpruner_shape_policy_match_count"] == 7
    assert best["qpruner_shape_policy_source"] == "artifacts/qpruner_packed_decode_benchmark_tiny_shape_policy.json"
    assert best["qpruner_shape_strategy_source_counts"] == {"shape_sweep_artifact": 7}
    assert summary["runtime_diagnosis"]["memory_preserving_qpruner_tradeoff"] == (
        "Best memory-preserving QPruner runtime tiny_short_shape_policy uses shape_aware_int8_code_cache "
        "with cache mode shape-aware-code: 1.190x baseline while retaining 49.500% code-cache storage reduction "
        "and 0 dense cached weight modules; packed-decode shape policy matched 7 modules."
    )


def test_build_summary_describes_cached_short_decode_when_long_decode_is_not_positive(tmp_path):
    module = load_summary_module()
    demo_root = tmp_path / "demo"
    paths = populate_sweep_artifacts(demo_root)
    long_path = demo_root / "artifacts" / "compressed_native_torch_serving_tiny_long_cache_on.json"
    long_payload = json.loads(long_path.read_text())
    long_payload["summary"]["qpruner_vs_baseline_speedup"] = 0.99
    long_payload["cap"]["tokens_per_s"] = 202.0
    long_payload["qpruner"]["tokens_per_s"] = 198.0
    long_path.write_text(json.dumps(long_payload))
    short_cache_path = demo_root / "artifacts" / "compressed_native_torch_serving_tiny_short_cache_on.json"
    short_cache_payload = json.loads(short_cache_path.read_text())
    short_cache_payload["summary"]["qpruner_vs_baseline_speedup"] = 1.14
    short_cache_payload["qpruner"]["tokens_per_s"] = 114.0
    short_cache_path.write_text(json.dumps(short_cache_payload))

    summary = module.build_summary(paths, demo_root=demo_root)

    assert summary["best"]["label"] == "tiny_short_cache_on"
    assert summary["best"]["qpruner_vs_baseline_speedup"] == 1.14
    assert summary["readout"].startswith("Cached compressed-native path is latency-positive")
    assert "max_new_tokens=1" in summary["readout"]
    assert "75.000% targeted storage reduction" in summary["readout"]
    assert "long-decode remains the next optimization target" in summary["readout"]


def test_markdown_csv_and_svg_surface_demo_readout(tmp_path):
    module = load_summary_module()
    demo_root = tmp_path / "demo"
    paths = populate_sweep_artifacts(demo_root)
    summary = module.build_summary(paths, demo_root=demo_root)

    markdown = module.markdown(summary)
    csv_text = module.csv_text(summary)
    svg = module.svg(summary)

    assert "# Compressed-Native Cache/Long-Decode Sweep" in markdown
    assert "Cached long-decode compressed-native path is latency-positive" in markdown
    assert "tiny_long_cache_on" in markdown
    assert "QPruner storage reduction 75.000%" in markdown
    assert "Runtime Diagnosis" in markdown
    assert "Memory-preserving QPruner runtime" in markdown
    assert "dequantize_per_forward" in markdown
    assert "CAP stores sparse residuals as coordinates with 0 dense sparse buffers" in markdown
    assert "CAP strategy" in markdown
    assert "coordinate_sparse_residual" in markdown
    assert "dense_weight_cache" in markdown
    assert "QPruner cache adds 1.000 MB over uncached QPruner peak at max_new_tokens=1" in markdown
    assert "fuse packed/quantized decode kernels on NPU" in markdown
    assert (
        "run_label,max_new_tokens,inference_cache_enabled,qpruner_cache_mode,cap_runtime_strategy"
        in csv_text
    )
    assert "qpruner_code_cache_storage_reduction_pct" in csv_text
    assert "tiny_long_cache_on,16,True" in csv_text
    assert "Compressed-native cache/long-decode sweep" in svg
    assert "cache off" in svg
    assert "cache on" in svg
    assert "long decode" in svg
    assert "QPruner" in svg


def test_cli_writes_sweep_outputs(tmp_path):
    module = load_summary_module()
    demo_root = tmp_path / "demo"
    paths = populate_sweep_artifacts(demo_root)

    rc = module.main(["--demo-root", str(demo_root), *[str(path) for path in paths]])

    assert rc == 0
    assert (demo_root / "artifacts" / "compressed_native_sweep_summary.json").exists()
    assert (demo_root / "reports" / "compressed-native-cache-long-decode-sweep.md").exists()
    assert (demo_root / "reports" / "compressed-native-cache-long-decode-sweep.csv").exists()
    assert (demo_root / "reports" / "compressed-native-cache-long-decode-sweep.svg").exists()
