import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn


def load_fixture_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "create_tiny_qwen_fixture.py"
    spec = importlib.util.spec_from_file_location("create_tiny_qwen_fixture", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_profile_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "qwen_qpruner_native_profile.py"
    spec = importlib.util.spec_from_file_location("qwen_qpruner_native_profile", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def quantized_linear(in_features, out_features, *, bias=True):
    linear = nn.Linear(in_features, out_features, bias=bias)
    with torch.no_grad():
        linear.weight.copy_(
            torch.arange(out_features * in_features, dtype=torch.float32).reshape(out_features, in_features) / 16
        )
        if bias:
            linear.bias.copy_(torch.arange(out_features, dtype=torch.float32) / 32)
    module = load_profile_module().QuantizedLinear.from_linear(linear, bits=8)
    module.enable_scaled_code_matmul(dtype=torch.float32, device=torch.device("cpu"))
    module.enable_aux_cache(dtype=torch.float32, device=torch.device("cpu"))
    return module


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj = quantized_linear(4, 3)
        self.v_proj = quantized_linear(4, 3)

    def forward(self, hidden_states):
        return self.k_proj(hidden_states), self.v_proj(hidden_states)


class TinyAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = TinyAttention()


def test_paired_summary_uses_median_metrics_and_alternating_order():
    profile = load_profile_module()
    samples = [
        {"round": 1, "method": "baseline", "order_index": 0, "latency_ms": 100.0, "tokens_per_s": 10.0},
        {"round": 1, "method": "qpruner", "order_index": 1, "latency_ms": 50.0, "tokens_per_s": 20.0},
        {"round": 2, "method": "qpruner", "order_index": 0, "latency_ms": 60.0, "tokens_per_s": 18.0},
        {"round": 2, "method": "baseline", "order_index": 1, "latency_ms": 120.0, "tokens_per_s": 9.0},
        {"round": 3, "method": "baseline", "order_index": 0, "latency_ms": 110.0, "tokens_per_s": 11.0},
        {"round": 3, "method": "qpruner", "order_index": 1, "latency_ms": 55.0, "tokens_per_s": 21.0},
    ]

    summary = profile.summarize_paired_samples(samples, paired_rounds=3)

    assert summary["rounds"] == 3
    assert summary["samples"] == 6
    assert summary["order"] == "alternating"
    assert summary["baseline"]["latency_ms_median"] == 110.0
    assert summary["baseline"]["tokens_per_s_median"] == 10.0
    assert summary["qpruner"]["latency_ms_median"] == 55.0
    assert summary["qpruner"]["tokens_per_s_median"] == 20.0
    assert summary["qpruner_vs_baseline_latency_median_speedup"] == 2.0
    assert summary["qpruner_vs_baseline_tokens_median_ratio"] == 2.0


def test_grouped_attention_kv_pairs_return_same_outputs_and_cache_v_projection():
    profile = load_profile_module()
    model = TinyAttentionModel()
    hidden_states = torch.randn(2, 5, 4)
    expected_k, expected_v = model.self_attn(hidden_states)

    grouped_pairs = profile.enable_grouped_qpruner_attention_kv_pairs(model)

    assert grouped_pairs == 1
    actual_k, actual_v = model.self_attn(hidden_states)
    torch.testing.assert_close(actual_k, expected_k)
    torch.testing.assert_close(actual_v, expected_v)
    assert model._tidal_qpruner_grouped_attention_kv_pairs == 1


def test_qwen_qpruner_native_profile_runs_offline_cpu_with_code_cache(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)
    script = Path(__file__).resolve().parents[1] / "scripts" / "qwen_qpruner_native_profile.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--demo-root",
            str(demo_root),
            "--model-id",
            "Qwen/Qwen3-0.6B",
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
            "--target-layer-limit",
            "2",
            "--target-layer-pattern",
            "self_attn\\.(q_proj|k_proj)$",
            "--qpruner-average-bits",
            "4.0",
            "--qpruner-cache-mode",
            "code",
            "--run-label",
            "cpu_profile",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads((demo_root / "artifacts" / "qwen_qpruner_native_profile_cpu_profile.json").read_text())
    assert payload["status"] == "PASS"
    assert payload["backend"] == "torch_generate_qwen_qpruner_native_profile"
    assert payload["model_id"] == "Qwen/Qwen3-0.6B"
    assert payload["serving_dense_export"] is False
    assert payload["qpruner_cache_mode"] == "code"
    assert payload["target_layer_limit"] == 2
    assert payload["target_layer_pattern"] == "self_attn\\.(q_proj|k_proj)$"
    assert payload["baseline"]["status"] == "PASS"
    assert payload["baseline"]["targeted_layers"] == 2
    qpruner = payload["qpruner"]
    assert qpruner["status"] == "PASS"
    assert qpruner["targeted_layers"] == 2
    assert qpruner["runtime_storage_format"] == "packed_nbit_weight_codes"
    assert qpruner["runtime_strategy"] == "int8_code_cache_dequantize_on_device"
    assert qpruner["targeted_storage_reduction_pct"] is not None
    assert qpruner["targeted_storage_reduction_pct"] > 0
    assert qpruner["code_cache_storage_reduction_pct"] is not None
    assert qpruner["runtime_profile"]["status"] == "PASS"
    assert qpruner["runtime_profile"]["quantized_layers"] == 2
    assert qpruner["runtime_profile"]["total_forward_calls"] > 0
    assert qpruner["runtime_profile"]["top_modules"]
    assert payload["summary"]["qpruner_vs_baseline_speedup"] is not None

    report = (demo_root / "reports" / "qwen-qpruner-native-profile-cpu_profile.md").read_text()
    assert "Qwen QPruner Native Runtime Profile" in report
    assert "QPruner runtime profile" in report
    assert "int8_code_cache_dequantize_on_device" in report


def test_qwen_qpruner_native_profile_records_paired_interleaved_measurement(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)
    script = Path(__file__).resolve().parents[1] / "scripts" / "qwen_qpruner_native_profile.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--demo-root",
            str(demo_root),
            "--model-id",
            "Qwen/Qwen3-0.6B",
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
            "--paired-rounds",
            "2",
            "--target-layer-limit",
            "2",
            "--target-layer-pattern",
            "self_attn\\.(q_proj|k_proj)$",
            "--qpruner-average-bits",
            "4.0",
            "--qpruner-cache-mode",
            "code",
            "--run-label",
            "cpu_paired_profile",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads((demo_root / "artifacts" / "qwen_qpruner_native_profile_cpu_paired_profile.json").read_text())
    assert payload["status"] == "PASS"
    assert payload["paired_rounds"] == 2
    assert payload["paired_summary"]["rounds"] == 2
    assert payload["paired_summary"]["samples"] == 4
    assert payload["paired_summary"]["baseline"]["count"] == 2
    assert payload["paired_summary"]["qpruner"]["count"] == 2
    assert payload["paired_samples"][0]["method"] == "baseline"
    assert payload["paired_samples"][1]["method"] == "qpruner"
    assert payload["paired_samples"][2]["method"] == "qpruner"
    assert payload["paired_samples"][3]["method"] == "baseline"
    assert payload["summary"]["qpruner_vs_baseline_speedup"] == payload["paired_summary"][
        "qpruner_vs_baseline_latency_median_speedup"
    ]
    assert payload["summary"]["qpruner_tokens_per_s_ratio"] == payload["paired_summary"][
        "qpruner_vs_baseline_tokens_median_ratio"
    ]

    report = (demo_root / "reports" / "qwen-qpruner-native-profile-cpu_paired_profile.md").read_text()
    assert "Paired measurement" in report
    assert "Paired rounds: `2`" in report


def test_qwen_qpruner_native_profile_records_effective_layer_limit_for_all_layers(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)
    script = Path(__file__).resolve().parents[1] / "scripts" / "qwen_qpruner_native_profile.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--demo-root",
            str(demo_root),
            "--model-id",
            "Qwen/Qwen3-0.6B",
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
            "--target-layer-limit",
            "0",
            "--qpruner-average-bits",
            "8.0",
            "--qpruner-cache-mode",
            "code",
            "--run-label",
            "cpu_all_layers",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads((demo_root / "artifacts" / "qwen_qpruner_native_profile_cpu_all_layers.json").read_text())
    assert payload["status"] == "PASS"
    assert payload["target_layer_requested_limit"] == 0
    assert payload["target_layer_limit"] == payload["targeted_layers_total"]
    assert payload["baseline"]["targeted_layers"] == payload["targeted_layers_total"]
    assert payload["qpruner"]["targeted_layers"] == payload["targeted_layers_total"]

    report = (demo_root / "reports" / "qwen-qpruner-native-profile-cpu_all_layers.md").read_text()
    assert f"Target layer limit: `{payload['targeted_layers_total']} / {payload['targeted_layers_total']}`" in report
    assert "Requested target layer limit: `0`" in report


def test_qwen_qpruner_native_profile_runs_shape_aware_code_cache_with_policy(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)
    policy_path = tmp_path / "qpruner_packed_decode_shape_sweep.json"
    policy_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "backend": "torch_qpruner_packed_decode_shape_sweep",
                "shape_sweep": [
                    {
                        "label": "tiny_qwen_attn",
                        "shape": {"batch_size": 1, "in_features": 32, "out_features": 512},
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
                    }
                ],
            }
        )
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "qwen_qpruner_native_profile.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--demo-root",
            str(demo_root),
            "--model-id",
            "Qwen/Qwen3-0.6B",
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
            "--target-layer-limit",
            "2",
            "--target-layer-pattern",
            "self_attn\\.(q_proj|k_proj)$",
            "--qpruner-average-bits",
            "8.0",
            "--qpruner-cache-mode",
            "shape-aware-code",
            "--qpruner-shape-policy-artifact",
            str(policy_path),
            "--run-label",
            "cpu_shape_aware",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads((demo_root / "artifacts" / "qwen_qpruner_native_profile_cpu_shape_aware.json").read_text())
    assert payload["status"] == "PASS"
    assert payload["qpruner_cache_mode"] == "shape-aware-code"
    assert payload["qpruner_shape_policy_artifact"] == str(policy_path)
    qpruner = payload["qpruner"]
    assert qpruner["runtime_strategy"] == "shape_aware_mixed_int8_code_cache"
    assert qpruner["shape_aware_cache_modules"] == 2
    assert qpruner["shape_aware_scaled_code_modules"] == 2
    assert qpruner["cached_dense_weight_modules"] == 0
    assert qpruner["shape_aware_plan"]["shape_policy_source"] == str(policy_path)
    assert qpruner["shape_aware_plan"]["shape_policy_match_count"] == 2
    assert qpruner["shape_aware_plan"]["strategy_source_counts"]["shape_sweep_artifact"] == 2
    assert {entry["shape_policy_label"] for entry in qpruner["shape_aware_plan"]["module_shapes"]} == {
        "tiny_qwen_attn"
    }

    report = (demo_root / "reports" / "qwen-qpruner-native-profile-cpu_shape_aware.md").read_text()
    assert "shape-aware-code" in report
    assert "shape_aware_mixed_int8_code_cache" in report


def test_qwen_qpruner_native_profile_can_release_packed_codes_after_code_cache(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)
    script = Path(__file__).resolve().parents[1] / "scripts" / "qwen_qpruner_native_profile.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--demo-root",
            str(demo_root),
            "--model-id",
            "Qwen/Qwen3-0.6B",
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
            "--target-layer-limit",
            "2",
            "--target-layer-pattern",
            "self_attn\\.(q_proj|k_proj)$",
            "--qpruner-average-bits",
            "8.0",
            "--qpruner-cache-mode",
            "code",
            "--qpruner-release-packed-after-cache",
            "--run-label",
            "cpu_release_packed",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads((demo_root / "artifacts" / "qwen_qpruner_native_profile_cpu_release_packed.json").read_text())
    assert payload["status"] == "PASS"
    assert payload["qpruner_release_packed_after_cache"] is True
    qpruner = payload["qpruner"]
    assert qpruner["runtime_strategy"] == "int8_code_cache_dequantize_on_device"
    assert qpruner["packed_code_bytes"] == 0
    assert qpruner["released_packed_code_bytes"] > 0
    assert qpruner["cached_code_bytes"] > 0
    assert qpruner["compressed_payload_storage_bytes"] > qpruner["live_compressed_payload_storage_bytes"]
    assert 49.0 < qpruner["targeted_storage_reduction_pct"] <= 50.0
    assert qpruner["code_cache_storage_bytes"] == (
        qpruner["live_compressed_payload_storage_bytes"] + qpruner["cached_code_bytes"]
    )
    assert 49.0 < qpruner["code_cache_storage_reduction_pct"] <= qpruner["targeted_storage_reduction_pct"]

    report = (demo_root / "reports" / "qwen-qpruner-native-profile-cpu_release_packed.md").read_text()
    assert "Release packed codes after cache: `True`" in report


def test_qwen_qpruner_native_profile_can_prebuild_scaled_code_dtype_cache(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)
    script = Path(__file__).resolve().parents[1] / "scripts" / "qwen_qpruner_native_profile.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--demo-root",
            str(demo_root),
            "--model-id",
            "Qwen/Qwen3-0.6B",
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
            "--target-layer-limit",
            "2",
            "--target-layer-pattern",
            "self_attn\\.(q_proj|k_proj)$",
            "--qpruner-average-bits",
            "8.0",
            "--qpruner-cache-mode",
            "scaled-code-matmul",
            "--qpruner-release-packed-after-cache",
            "--qpruner-prebuild-scaled-code-dtype-cache",
            "--run-label",
            "cpu_scaled_prebuilt",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads((demo_root / "artifacts" / "qwen_qpruner_native_profile_cpu_scaled_prebuilt.json").read_text())
    assert payload["status"] == "PASS"
    assert payload["qpruner_cache_mode"] == "scaled-code-matmul"
    assert payload["qpruner_prebuild_scaled_code_dtype_cache"] is True
    qpruner = payload["qpruner"]
    assert qpruner["runtime_strategy"] == "scaled_int8_code_matmul"
    assert qpruner["cached_code_bytes"] > 0
    assert qpruner["cached_scaled_code_bytes"] > 0
    assert qpruner["code_cache_storage_bytes"] == (
        qpruner["live_compressed_payload_storage_bytes"]
        + qpruner["cached_code_bytes"]
        + qpruner["cached_scaled_code_bytes"]
    )
    assert payload["memory_reference"]["qpruner_scaled_code_dtype_cache_bytes"] == qpruner["cached_scaled_code_bytes"]

    report = (demo_root / "reports" / "qwen-qpruner-native-profile-cpu_scaled_prebuilt.md").read_text()
    assert "Prebuild scaled-code dtype cache: `True`" in report
    assert "QPruner scaled-code dtype-cache bytes" in report


def test_qwen_qpruner_native_profile_can_apply_scaled_code_dtype_cache_budget(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)
    script = Path(__file__).resolve().parents[1] / "scripts" / "qwen_qpruner_native_profile.py"
    scaled_budget_bytes = 32 * 512 * 4

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--demo-root",
            str(demo_root),
            "--model-id",
            "Qwen/Qwen3-0.6B",
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
            "--target-layer-limit",
            "2",
            "--target-layer-pattern",
            "self_attn\\.(q_proj|k_proj)$",
            "--qpruner-average-bits",
            "8.0",
            "--qpruner-cache-mode",
            "scaled-code-matmul",
            "--qpruner-release-packed-after-cache",
            "--qpruner-scaled-code-dtype-cache-budget-bytes",
            str(scaled_budget_bytes),
            "--run-label",
            "cpu_scaled_dtype_budget",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(
        (demo_root / "artifacts" / "qwen_qpruner_native_profile_cpu_scaled_dtype_budget.json").read_text()
    )
    assert payload["status"] == "PASS"
    assert payload["qpruner_cache_mode"] == "scaled-code-matmul"
    assert payload["qpruner_prebuild_scaled_code_dtype_cache"] is False
    assert payload["qpruner_scaled_code_dtype_cache_budget_bytes"] == scaled_budget_bytes
    assert payload["qpruner_scaled_code_dtype_cache_selection_policy"] == "smallest_scaled_code_dtype_cache_first"
    qpruner = payload["qpruner"]
    assert qpruner["runtime_strategy"] == "scaled_int8_code_matmul"
    assert qpruner["scaled_code_dtype_cache_budget_bytes"] == scaled_budget_bytes
    assert qpruner["scaled_code_dtype_cache_budget_plan"]["selected_modules"] == 1
    assert qpruner["cached_code_bytes"] > 0
    assert qpruner["cached_scaled_code_bytes"] == scaled_budget_bytes
    assert qpruner["code_cache_storage_bytes"] == (
        qpruner["live_compressed_payload_storage_bytes"]
        + qpruner["cached_code_bytes"]
        + qpruner["cached_scaled_code_bytes"]
    )
    assert payload["memory_reference"]["qpruner_scaled_code_dtype_cache_bytes"] == scaled_budget_bytes

    report = (demo_root / "reports" / "qwen-qpruner-native-profile-cpu_scaled_dtype_budget.md").read_text()
    assert f"QPruner scaled-code dtype-cache budget bytes: `{scaled_budget_bytes}`" in report
    assert "QPruner scaled-code dtype-cache budget" in report


def test_qwen_qpruner_native_profile_can_cache_aux_tensors(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)
    script = Path(__file__).resolve().parents[1] / "scripts" / "qwen_qpruner_native_profile.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--demo-root",
            str(demo_root),
            "--model-id",
            "Qwen/Qwen3-0.6B",
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
            "--target-layer-limit",
            "2",
            "--target-layer-pattern",
            "self_attn\\.(q_proj|k_proj)$",
            "--qpruner-average-bits",
            "8.0",
            "--qpruner-cache-mode",
            "code",
            "--qpruner-release-packed-after-cache",
            "--qpruner-cache-aux-tensors",
            "--run-label",
            "cpu_aux_cache",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads((demo_root / "artifacts" / "qwen_qpruner_native_profile_cpu_aux_cache.json").read_text())
    assert payload["status"] == "PASS"
    assert payload["qpruner_cache_aux_tensors"] is True
    qpruner = payload["qpruner"]
    assert qpruner["runtime_strategy"] == "int8_code_cache_dequantize_on_device"
    assert qpruner["cached_aux_modules"] == 2
    assert qpruner["cached_scale_bytes"] > 0
    assert qpruner["cached_bias_bytes"] == 0
    assert qpruner["cached_aux_bytes"] == qpruner["cached_scale_bytes"]
    assert qpruner["code_cache_storage_bytes"] == (
        qpruner["live_compressed_payload_storage_bytes"]
        + qpruner["cached_code_bytes"]
        + qpruner["cached_aux_bytes"]
    )
    assert payload["memory_reference"]["qpruner_aux_cache_bytes"] == qpruner["cached_aux_bytes"]

    report = (demo_root / "reports" / "qwen-qpruner-native-profile-cpu_aux_cache.md").read_text()
    assert "Cache QPruner aux tensors: `True`" in report
    assert "QPruner aux-cache bytes" in report


def test_qwen_qpruner_native_profile_can_apply_dense_cache_budget(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)
    script = Path(__file__).resolve().parents[1] / "scripts" / "qwen_qpruner_native_profile.py"
    dense_budget_bytes = 32 * 512 * 4

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--demo-root",
            str(demo_root),
            "--model-id",
            "Qwen/Qwen3-0.6B",
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
            "--target-layer-limit",
            "2",
            "--target-layer-pattern",
            "self_attn\\.(q_proj|k_proj)$",
            "--qpruner-average-bits",
            "8.0",
            "--qpruner-cache-mode",
            "code",
            "--qpruner-release-packed-after-cache",
            "--qpruner-dense-cache-budget-bytes",
            str(dense_budget_bytes),
            "--run-label",
            "cpu_dense_budget",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads((demo_root / "artifacts" / "qwen_qpruner_native_profile_cpu_dense_budget.json").read_text())
    assert payload["status"] == "PASS"
    assert payload["qpruner_dense_cache_budget_bytes"] == dense_budget_bytes
    qpruner = payload["qpruner"]
    assert qpruner["runtime_strategy"] == "mixed_dense_int8_code_cache"
    assert qpruner["dense_cache_budget_plan"]["selected_modules"] == 1
    assert qpruner["cached_dense_weight_modules"] == 1
    assert qpruner["cached_dense_weight_bytes"] == dense_budget_bytes
    assert qpruner["code_cache_storage_bytes"] == (
        qpruner["live_compressed_payload_storage_bytes"]
        + qpruner["cached_code_bytes"]
        + qpruner["cached_scaled_code_bytes"]
        + qpruner["cached_dense_weight_bytes"]
    )
    assert payload["memory_reference"]["qpruner_dense_cache_bytes"] == qpruner["cached_dense_weight_bytes"]

    report = (demo_root / "reports" / "qwen-qpruner-native-profile-cpu_dense_budget.md").read_text()
    assert "QPruner dense-cache bytes" in report
    assert f"QPruner dense-cache budget bytes: `{dense_budget_bytes}`" in report


def test_qwen_qpruner_native_profile_records_dense_cache_budget_policy(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)
    script = Path(__file__).resolve().parents[1] / "scripts" / "qwen_qpruner_native_profile.py"
    dense_budget_bytes = 32 * 512 * 4

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--demo-root",
            str(demo_root),
            "--model-id",
            "Qwen/Qwen3-0.6B",
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
            "--target-layer-limit",
            "2",
            "--target-layer-pattern",
            "self_attn\\.(q_proj|k_proj)$",
            "--qpruner-average-bits",
            "8.0",
            "--qpruner-cache-mode",
            "code",
            "--qpruner-release-packed-after-cache",
            "--qpruner-dense-cache-budget-bytes",
            str(dense_budget_bytes),
            "--qpruner-dense-cache-selection-policy",
            "qwen_projection_hotspot_first",
            "--run-label",
            "cpu_dense_budget_hotspot",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(
        (demo_root / "artifacts" / "qwen_qpruner_native_profile_cpu_dense_budget_hotspot.json").read_text()
    )
    assert payload["status"] == "PASS"
    assert payload["qpruner_dense_cache_selection_policy"] == "qwen_projection_hotspot_first"
    qpruner = payload["qpruner"]
    assert qpruner["dense_cache_budget_plan"]["selection_policy"] == "qwen_projection_hotspot_first"

    report = (demo_root / "reports" / "qwen-qpruner-native-profile-cpu_dense_budget_hotspot.md").read_text()
    assert "QPruner dense-cache selection policy: `qwen_projection_hotspot_first`" in report


def test_qwen_qpruner_native_profile_can_enable_grouped_mlp_pairs(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)
    script = Path(__file__).resolve().parents[1] / "scripts" / "qwen_qpruner_native_profile.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--demo-root",
            str(demo_root),
            "--model-id",
            "Qwen/Qwen3-0.6B",
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
            "--target-layer-limit",
            "0",
            "--target-layer-pattern",
            "mlp\\.(gate_proj|up_proj|down_proj)$",
            "--qpruner-average-bits",
            "8.0",
            "--qpruner-cache-mode",
            "scaled-code-matmul",
            "--qpruner-release-packed-after-cache",
            "--qpruner-cache-aux-tensors",
            "--qpruner-grouped-mlp",
            "--run-label",
            "cpu_grouped_mlp",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads((demo_root / "artifacts" / "qwen_qpruner_native_profile_cpu_grouped_mlp.json").read_text())
    assert payload["status"] == "PASS"
    assert payload["qpruner_grouped_mlp"] is True
    qpruner = payload["qpruner"]
    assert qpruner["grouped_mlp_pairs"] == 1
    assert qpruner["runtime_profile"]["grouped_mlp_pairs"] == 1
    assert qpruner["runtime_profile"]["total_forward_calls"] > 0
    assert qpruner["runtime_profile"]["by_strategy"]["scaled_int8_code_matmul"]["calls"] > 0

    report = (demo_root / "reports" / "qwen-qpruner-native-profile-cpu_grouped_mlp.md").read_text()
    assert "QPruner grouped MLP pairs: `1`" in report
    assert "Grouped MLP pairs: `1`" in report


def test_qwen_qpruner_native_profile_records_grouped_mlp_strategy(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)
    script = Path(__file__).resolve().parents[1] / "scripts" / "qwen_qpruner_native_profile.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--demo-root",
            str(demo_root),
            "--model-id",
            "Qwen/Qwen3-0.6B",
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
            "--target-layer-limit",
            "0",
            "--target-layer-pattern",
            "mlp\\.(gate_proj|up_proj|down_proj)$",
            "--qpruner-average-bits",
            "8.0",
            "--qpruner-cache-mode",
            "scaled-code-matmul",
            "--qpruner-release-packed-after-cache",
            "--qpruner-cache-aux-tensors",
            "--qpruner-grouped-mlp",
            "--qpruner-grouped-mlp-strategy",
            "fused-2d",
            "--run-label",
            "cpu_grouped_mlp_fused2d",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(
        (demo_root / "artifacts" / "qwen_qpruner_native_profile_cpu_grouped_mlp_fused2d.json").read_text()
    )
    assert payload["qpruner_grouped_mlp_strategy"] == "fused-2d"
    assert payload["qpruner"]["grouped_mlp_strategy"] == "fused-2d"
    assert payload["qpruner"]["runtime_profile"]["grouped_mlp_strategy"] == "fused-2d"
    report = (demo_root / "reports" / "qwen-qpruner-native-profile-cpu_grouped_mlp_fused2d.md").read_text()
    assert "QPruner grouped MLP strategy: `fused-2d`" in report


def test_qwen_qpruner_native_profile_can_enable_grouped_attention_kv_pairs(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)
    script = Path(__file__).resolve().parents[1] / "scripts" / "qwen_qpruner_native_profile.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--demo-root",
            str(demo_root),
            "--model-id",
            "Qwen/Qwen3-0.6B",
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
            "--target-layer-limit",
            "0",
            "--target-layer-pattern",
            "self_attn\\.(k_proj|v_proj)$",
            "--qpruner-average-bits",
            "8.0",
            "--qpruner-cache-mode",
            "scaled-code-matmul",
            "--qpruner-release-packed-after-cache",
            "--qpruner-cache-aux-tensors",
            "--qpruner-grouped-attention-kv",
            "--run-label",
            "cpu_grouped_attention_kv",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(
        (demo_root / "artifacts" / "qwen_qpruner_native_profile_cpu_grouped_attention_kv.json").read_text()
    )
    assert payload["status"] == "PASS"
    assert payload["qpruner_grouped_attention_kv"] is True
    qpruner = payload["qpruner"]
    assert qpruner["grouped_attention_kv_pairs"] == 1
    assert qpruner["runtime_profile"]["grouped_attention_kv_pairs"] == 1
    assert qpruner["runtime_profile"]["by_strategy"]["scaled_int8_code_matmul"]["calls"] > 0

    report = (demo_root / "reports" / "qwen-qpruner-native-profile-cpu_grouped_attention_kv.md").read_text()
    assert "QPruner grouped attention K/V pairs: `1`" in report
    assert "Grouped attention K/V pairs: `1`" in report
