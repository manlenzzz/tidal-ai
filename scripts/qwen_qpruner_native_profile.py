#!/usr/bin/env python
"""Profile Qwen-family QPruner native runtime without dense serving export."""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Any, Sequence

import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from ascend_inference_benchmark import peak_memory_mb, reset_peak_memory_stats, synchronize_device  # noqa: E402
from compressed_native_torch_serving_benchmark import (  # noqa: E402
    QPRUNER_SCALED_CODE_DTYPE_CACHE_SELECTION_POLICIES,
    bits_to_storage_bytes,
    compressed_payload_storage_bytes,
    enable_qpruner_aux_cache,
    enable_qpruner_code_cache,
    enable_qpruner_dense_cache_budget,
    enable_qpruner_scaled_code_dtype_cache_budget,
    enable_qpruner_shape_aware_code_cache,
    enable_qpruner_scaled_code_matmul,
    fmt,
    load_qpruner_shape_policy,
    profile_qpruner_runtime,
    qpruner_profile_workload,
    qpruner_reduction_from_bits,
    qpruner_runtime_storage_bytes,
    qpruner_runtime_metadata,
    reduction_pct,
    speedup,
)
from qwen_compression_generate_benchmark import DEFAULT_PROMPTS, limited_target_filter, select_target_names  # noqa: E402
from tiny_qwen_compression_benchmark import (  # noqa: E402
    enable_model_inference_cache,
    load_model,
    load_tokenizer,
    qpruner_average_bits,
    targeted_parameter_count,
)
from tiny_qwen_compression_generate_benchmark import run_generate  # noqa: E402
from tidal.device import resolve_device, resolve_dtype  # noqa: E402
from tidal.methods.qpruner.grouped import (  # noqa: E402
    grouped_same_input_scaled_code,
    grouped_same_input_scaled_code_fused_2d,
)
from tidal.methods.qpruner.torch import QuantizedLinear  # noqa: E402
from tidal.workflows.compression import qpruner_compress  # noqa: E402


def benchmark_model(
    *,
    model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    max_new_tokens: int,
    iters: int,
    warmup: int,
) -> dict[str, Any]:
    reset_peak_memory_stats(device)
    metrics = run_generate(
        model=model,
        tokenizer=tokenizer,
        prompts=list(DEFAULT_PROMPTS),
        device=device,
        max_new_tokens=max_new_tokens,
        iters=iters,
        warmup=warmup,
    )
    metrics["peak_mem_mb"] = peak_memory_mb(device)
    return metrics


def apply_qpruner_cache_mode(
    model: nn.Module,
    *,
    mode: str,
    dtype: torch.dtype,
    device: torch.device,
    shape_policy_artifact: str | None = None,
    release_packed_codes: bool = False,
    prebuild_scaled_code_dtype_cache: bool = False,
) -> int:
    if mode == "dense":
        return enable_model_inference_cache(model, dtype=dtype, device=device)
    if mode == "code":
        return enable_qpruner_code_cache(model, device=device, release_packed_codes=release_packed_codes)
    if mode == "scaled-code-matmul":
        return enable_qpruner_scaled_code_matmul(
            model,
            device=device,
            dtype=dtype if prebuild_scaled_code_dtype_cache else None,
            release_packed_codes=release_packed_codes,
        )
    if mode == "shape-aware-code":
        shape_policy = load_qpruner_shape_policy(shape_policy_artifact)
        return enable_qpruner_shape_aware_code_cache(
            model,
            device=device,
            dtype=dtype if prebuild_scaled_code_dtype_cache else None,
            shape_policy=shape_policy,
            release_packed_codes=release_packed_codes,
        )
    return 0


def enable_grouped_qpruner_mlp_pairs(model: nn.Module, *, strategy: str = "bmm") -> int:
    grouped_pairs = 0
    if strategy not in {"bmm", "fused-2d"}:
        raise ValueError("grouped MLP strategy must be one of: bmm, fused-2d")
    grouped_project = grouped_same_input_scaled_code_fused_2d if strategy == "fused-2d" else grouped_same_input_scaled_code
    for module in model.modules():
        if bool(getattr(module, "_tidal_qpruner_grouped_mlp_pair", False)):
            continue
        gate_proj = getattr(module, "gate_proj", None)
        up_proj = getattr(module, "up_proj", None)
        down_proj = getattr(module, "down_proj", None)
        act_fn = getattr(module, "act_fn", None)
        if not (
            isinstance(gate_proj, QuantizedLinear)
            and isinstance(up_proj, QuantizedLinear)
            and isinstance(down_proj, nn.Module)
            and callable(act_fn)
        ):
            continue
        if gate_proj.in_features != up_proj.in_features or gate_proj.out_features != up_proj.out_features:
            continue
        if getattr(gate_proj, "_cached_weight_codes", None) is None:
            continue
        if getattr(up_proj, "_cached_weight_codes", None) is None:
            continue

        def grouped_forward(
            self: nn.Module,
            hidden_states: torch.Tensor,
            *,
            _gate_proj: QuantizedLinear = gate_proj,
            _up_proj: QuantizedLinear = up_proj,
            _down_proj: nn.Module = down_proj,
            _act_fn: Any = act_fn,
            _grouped_project: Any = grouped_project,
        ) -> torch.Tensor:
            gate_output, up_output = _grouped_project((_gate_proj, _up_proj), hidden_states)
            return _down_proj(_act_fn(gate_output) * up_output)

        module._tidal_original_forward = module.forward
        module.forward = MethodType(grouped_forward, module)
        module._tidal_qpruner_grouped_mlp_pair = True
        module._tidal_qpruner_grouped_mlp_strategy = strategy
        grouped_pairs += 1
    model._tidal_qpruner_grouped_mlp_pairs = grouped_pairs
    model._tidal_qpruner_grouped_mlp_strategy = strategy
    return grouped_pairs


def enable_grouped_qpruner_attention_kv_pairs(model: nn.Module) -> int:
    grouped_pairs = 0
    for module in model.modules():
        if bool(getattr(module, "_tidal_qpruner_grouped_attention_kv_pair", False)):
            continue
        k_proj = getattr(module, "k_proj", None)
        v_proj = getattr(module, "v_proj", None)
        if not (isinstance(k_proj, QuantizedLinear) and isinstance(v_proj, QuantizedLinear)):
            continue
        if k_proj.in_features != v_proj.in_features or k_proj.out_features != v_proj.out_features:
            continue
        if getattr(k_proj, "_cached_weight_codes", None) is None:
            continue
        if getattr(v_proj, "_cached_weight_codes", None) is None:
            continue

        original_k_forward = k_proj.forward
        original_v_forward = v_proj.forward
        cached_v: dict[str, Any] = {"signature": None, "output": None}

        def input_signature(inputs: torch.Tensor) -> tuple[Any, ...]:
            data_ptr = inputs.data_ptr() if inputs.numel() else 0
            return (id(inputs), data_ptr, tuple(inputs.shape), inputs.dtype, inputs.device, inputs.storage_offset())

        def grouped_k_forward(
            self: QuantizedLinear,
            hidden_states: torch.Tensor,
            *,
            _k_proj: QuantizedLinear = k_proj,
            _v_proj: QuantizedLinear = v_proj,
            _original_k_forward: Any = original_k_forward,
        ) -> torch.Tensor:
            if hidden_states.shape[-1] != _k_proj.in_features:
                cached_v["signature"] = None
                cached_v["output"] = None
                return _original_k_forward(hidden_states)
            key_output, value_output = grouped_same_input_scaled_code((_k_proj, _v_proj), hidden_states)
            cached_v["signature"] = input_signature(hidden_states)
            cached_v["output"] = value_output
            return key_output

        def cached_v_forward(
            self: QuantizedLinear,
            hidden_states: torch.Tensor,
            *,
            _original_v_forward: Any = original_v_forward,
        ) -> torch.Tensor:
            if cached_v.get("signature") == input_signature(hidden_states):
                value_output = cached_v["output"]
                cached_v["signature"] = None
                cached_v["output"] = None
                return value_output
            cached_v["signature"] = None
            cached_v["output"] = None
            return _original_v_forward(hidden_states)

        k_proj._tidal_original_forward_before_grouped_attention_kv = original_k_forward
        v_proj._tidal_original_forward_before_grouped_attention_kv = original_v_forward
        k_proj.forward = MethodType(grouped_k_forward, k_proj)
        v_proj.forward = MethodType(cached_v_forward, v_proj)
        module._tidal_qpruner_grouped_attention_kv_pair = True
        grouped_pairs += 1
    model._tidal_qpruner_grouped_attention_kv_pairs = grouped_pairs
    return grouped_pairs


def summarize(exports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline_latency = exports.get("baseline", {}).get("latency_ms")
    qpruner_latency = exports.get("qpruner", {}).get("latency_ms")
    baseline_tokens = exports.get("baseline", {}).get("tokens_per_s")
    qpruner_tokens = exports.get("qpruner", {}).get("tokens_per_s")
    best_method = None
    best_tokens = None
    passing = {
        name: payload
        for name, payload in exports.items()
        if payload.get("status") == "PASS" and payload.get("tokens_per_s") is not None
    }
    if passing:
        best_method, best_payload = max(passing.items(), key=lambda item: float(item[1]["tokens_per_s"]))
        best_tokens = best_payload.get("tokens_per_s")
    return {
        "best_method": best_method,
        "best_tokens_per_s": best_tokens,
        "qpruner_vs_baseline_speedup": speedup(baseline_latency, qpruner_latency),
        "qpruner_tokens_per_s_ratio": speedup(qpruner_tokens, baseline_tokens),
    }


def _numeric_values(samples: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for sample in samples:
        value = sample.get(key)
        if value is None:
            continue
        values.append(float(value))
    return values


def summarize_metric_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = _numeric_values(samples, "latency_ms")
    tokens = _numeric_values(samples, "tokens_per_s")
    return {
        "count": len(samples),
        "latency_ms_median": statistics.median(latencies) if latencies else None,
        "latency_ms_mean": statistics.fmean(latencies) if latencies else None,
        "latency_ms_min": min(latencies) if latencies else None,
        "latency_ms_max": max(latencies) if latencies else None,
        "latency_ms_stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0.0 if latencies else None,
        "tokens_per_s_median": statistics.median(tokens) if tokens else None,
        "tokens_per_s_mean": statistics.fmean(tokens) if tokens else None,
        "tokens_per_s_min": min(tokens) if tokens else None,
        "tokens_per_s_max": max(tokens) if tokens else None,
        "tokens_per_s_stdev": statistics.stdev(tokens) if len(tokens) > 1 else 0.0 if tokens else None,
    }


def summarize_paired_samples(samples: list[dict[str, Any]], *, paired_rounds: int) -> dict[str, Any]:
    baseline_samples = [sample for sample in samples if sample.get("method") == "baseline"]
    qpruner_samples = [sample for sample in samples if sample.get("method") == "qpruner"]
    baseline = summarize_metric_samples(baseline_samples)
    qpruner = summarize_metric_samples(qpruner_samples)
    return {
        "rounds": paired_rounds,
        "samples": len(samples),
        "order": "alternating",
        "baseline": baseline,
        "qpruner": qpruner,
        "qpruner_vs_baseline_latency_median_speedup": speedup(
            baseline.get("latency_ms_median"),
            qpruner.get("latency_ms_median"),
        ),
        "qpruner_vs_baseline_tokens_median_ratio": speedup(
            qpruner.get("tokens_per_s_median"),
            baseline.get("tokens_per_s_median"),
        ),
    }


def collect_paired_samples(
    *,
    baseline_model: nn.Module,
    qpruner_model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    max_new_tokens: int,
    iters: int,
    warmup: int,
    paired_rounds: int,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    models = {"baseline": baseline_model, "qpruner": qpruner_model}
    for round_index in range(paired_rounds):
        order = ("baseline", "qpruner") if round_index % 2 == 0 else ("qpruner", "baseline")
        for order_index, method in enumerate(order):
            metrics = benchmark_model(
                model=models[method],
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=max_new_tokens,
                iters=iters,
                warmup=warmup,
            )
            samples.append(
                {
                    "round": round_index + 1,
                    "method": method,
                    "order_index": order_index,
                    "status": metrics.get("status"),
                    "latency_ms": metrics.get("latency_ms"),
                    "tokens_per_s": metrics.get("tokens_per_s"),
                    "peak_mem_mb": metrics.get("peak_mem_mb"),
                }
            )
    return samples


def markdown_report(report: dict[str, Any]) -> str:
    baseline = report.get("baseline", {}) if isinstance(report.get("baseline"), dict) else {}
    qpruner = report.get("qpruner", {}) if isinstance(report.get("qpruner"), dict) else {}
    summary = report.get("summary", {}) if isinstance(report.get("summary"), dict) else {}
    paired = report.get("paired_summary") if isinstance(report.get("paired_summary"), dict) else None
    memory = report.get("memory_reference", {}) if isinstance(report.get("memory_reference"), dict) else {}
    lines = [
        "# Qwen QPruner Native Runtime Profile",
        "",
        "QPruner-only torch.generate profile for the target Qwen family model. It keeps QuantizedLinear modules live and avoids dense serving export so the memory numbers remain meaningful.",
        "",
        "| Metric | Baseline | QPruner |",
        "|---|---:|---:|",
        f"| Status | {baseline.get('status')} | {qpruner.get('status')} |",
        f"| Latency ms | {fmt(baseline.get('latency_ms'))} | {fmt(qpruner.get('latency_ms'))} |",
        f"| Tokens/s | {fmt(baseline.get('tokens_per_s'))} | {fmt(qpruner.get('tokens_per_s'))} |",
        f"| Speedup | - | {fmt(summary.get('qpruner_vs_baseline_speedup'))} |",
        f"| Peak MB | {fmt(baseline.get('peak_mem_mb'))} | {fmt(qpruner.get('peak_mem_mb'))} |",
        f"| Targeted layers | {fmt(baseline.get('targeted_layers'))} | {fmt(qpruner.get('targeted_layers'))} |",
        f"| Targeted storage bytes | {fmt(baseline.get('targeted_storage_bytes'))} | {fmt(qpruner.get('targeted_storage_bytes'))} |",
        f"| Targeted storage reduction | - | {fmt(qpruner.get('targeted_storage_reduction_pct'))}% |",
        f"| Code-cache storage bytes | - | {fmt(qpruner.get('code_cache_storage_bytes'))} |",
        f"| Dense-cache bytes | - | {fmt(qpruner.get('cached_dense_weight_bytes'))} |",
        f"| Scaled-code dtype-cache bytes | - | {fmt(qpruner.get('cached_scaled_code_bytes'))} |",
        f"| QPruner aux-cache bytes | - | {fmt(qpruner.get('cached_aux_bytes'))} |",
        f"| QPruner grouped MLP pairs | - | {fmt(qpruner.get('grouped_mlp_pairs'))} |",
        f"| QPruner grouped attention K/V pairs | - | {fmt(qpruner.get('grouped_attention_kv_pairs'))} |",
        f"| Code-cache storage reduction | - | {fmt(qpruner.get('code_cache_storage_reduction_pct'))}% |",
        f"| Runtime storage | {baseline.get('runtime_storage_format', 'dense fp16/fp32 weights')} | {qpruner.get('runtime_storage_format')} |",
        f"| Runtime strategy | {baseline.get('runtime_strategy', 'dense_linear')} | {qpruner.get('runtime_strategy')} |",
        "",
        "## Memory Reference",
        "",
        f"- Baseline targeted params: `{memory.get('baseline_targeted_params')}`",
        f"- Target layer coverage: `{memory.get('target_layer_coverage_pct')}`%",
        f"- QPruner average bits: `{qpruner.get('average_bits')}`",
        f"- QPruner targeted memory reduction: `{memory.get('qpruner_targeted_param_reduction_pct')}`%",
        f"- Native compressed-module storage measurement: `{memory.get('storage_measurement')}`",
        f"- QPruner targeted storage reduction: `{memory.get('qpruner_targeted_storage_reduction_pct')}`%",
        f"- QPruner code-cache storage reduction: `{memory.get('qpruner_code_cache_storage_reduction_pct')}`%",
        f"- QPruner dense-cache bytes: `{memory.get('qpruner_dense_cache_bytes')}`",
        f"- QPruner scaled-code dtype-cache bytes: `{memory.get('qpruner_scaled_code_dtype_cache_bytes')}`",
        f"- QPruner aux-cache bytes: `{memory.get('qpruner_aux_cache_bytes')}`",
        "",
        "## Metadata",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Backend: `{report.get('backend')}`",
        f"- Model: `{report.get('model_id')}`",
        f"- Model path: `{report.get('model_path')}`",
        f"- Device: `{report.get('device')}`",
        f"- Dtype: `{report.get('dtype')}`",
        f"- QPruner cache mode: `{report.get('qpruner_cache_mode')}`",
        f"- QPruner shape policy artifact: `{report.get('qpruner_shape_policy_artifact')}`",
        f"- Release packed codes after cache: `{report.get('qpruner_release_packed_after_cache')}`",
        f"- Cache QPruner aux tensors: `{report.get('qpruner_cache_aux_tensors')}`",
        f"- Prebuild scaled-code dtype cache: `{report.get('qpruner_prebuild_scaled_code_dtype_cache')}`",
        f"- QPruner scaled-code dtype-cache budget bytes: `{report.get('qpruner_scaled_code_dtype_cache_budget_bytes')}`",
        f"- QPruner scaled-code dtype-cache selection policy: `{report.get('qpruner_scaled_code_dtype_cache_selection_policy')}`",
        f"- QPruner dense-cache budget bytes: `{report.get('qpruner_dense_cache_budget_bytes')}`",
        f"- QPruner dense-cache selection policy: `{report.get('qpruner_dense_cache_selection_policy')}`",
        f"- QPruner grouped MLP: `{report.get('qpruner_grouped_mlp')}`",
        f"- QPruner grouped MLP strategy: `{report.get('qpruner_grouped_mlp_strategy')}`",
        f"- QPruner grouped MLP pairs: `{qpruner.get('grouped_mlp_pairs')}`",
        f"- QPruner grouped attention K/V: `{report.get('qpruner_grouped_attention_kv')}`",
        f"- QPruner grouped attention K/V pairs: `{qpruner.get('grouped_attention_kv_pairs')}`",
        f"- Serving dense export: `{report.get('serving_dense_export')}`",
        f"- Target layer pattern: `{report.get('target_layer_pattern')}`",
        f"- Target layer limit: `{report.get('target_layer_limit')} / {report.get('targeted_layers_total')}`",
        f"- Requested target layer limit: `{report.get('target_layer_requested_limit')}`",
        f"- Target layer sample: `{', '.join(report.get('targeted_layer_names_sample', []))}`",
        f"- Peak MB overall: `{fmt(report.get('peak_mem_mb'))}`",
        f"- Download attempted: `{report.get('download_attempted')}`",
    ]
    if paired:
        paired_baseline = paired.get("baseline", {}) if isinstance(paired.get("baseline"), dict) else {}
        paired_qpruner = paired.get("qpruner", {}) if isinstance(paired.get("qpruner"), dict) else {}
        lines.extend(
            [
                "",
                "## Paired measurement",
                "",
                f"- Paired rounds: `{paired.get('rounds')}`",
                f"- Paired samples: `{paired.get('samples')}`",
                f"- Paired order: `{paired.get('order')}`",
                f"- Baseline median latency ms: `{fmt(paired_baseline.get('latency_ms_median'))}`",
                f"- QPruner median latency ms: `{fmt(paired_qpruner.get('latency_ms_median'))}`",
                f"- Latency median speedup: `{fmt(paired.get('qpruner_vs_baseline_latency_median_speedup'))}`",
                f"- Baseline median tokens/s: `{fmt(paired_baseline.get('tokens_per_s_median'))}`",
                f"- QPruner median tokens/s: `{fmt(paired_qpruner.get('tokens_per_s_median'))}`",
                f"- Tokens/s median ratio: `{fmt(paired.get('qpruner_vs_baseline_tokens_median_ratio'))}`",
            ]
        )
    profile = qpruner.get("runtime_profile") if isinstance(qpruner.get("runtime_profile"), dict) else None
    shape_plan = qpruner.get("shape_aware_plan") if isinstance(qpruner.get("shape_aware_plan"), dict) else None
    if shape_plan:
        lines.extend(
            [
                "",
                "## Shape-aware QPruner policy",
                "",
                f"- Policy: `{shape_plan.get('policy')}`",
                f"- Source: `{shape_plan.get('shape_policy_source')}`",
                f"- Shape rows examined: `{shape_plan.get('shape_policy_rows_examined')}`",
                f"- Shape matches: `{shape_plan.get('shape_policy_match_count')} / {shape_plan.get('module_count')}`",
                f"- Strategy counts: `{shape_plan.get('strategy_counts')}`",
                f"- Strategy source counts: `{shape_plan.get('strategy_source_counts')}`",
            ]
        )
    dense_plan = qpruner.get("dense_cache_budget_plan") if isinstance(qpruner.get("dense_cache_budget_plan"), dict) else None
    scaled_plan = (
        qpruner.get("scaled_code_dtype_cache_budget_plan")
        if isinstance(qpruner.get("scaled_code_dtype_cache_budget_plan"), dict)
        else None
    )
    if scaled_plan:
        lines.extend(
            [
                "",
                "## QPruner scaled-code dtype-cache budget",
                "",
                f"- Budget bytes: `{scaled_plan.get('budget_bytes')}`",
                f"- Selection policy: `{scaled_plan.get('selection_policy')}`",
                f"- Selected modules: `{scaled_plan.get('selected_modules')}`",
                f"- Selected scaled-code dtype-cache bytes: `{scaled_plan.get('selected_scaled_code_dtype_cache_bytes')}`",
                f"- Remaining budget bytes: `{scaled_plan.get('remaining_budget_bytes')}`",
            ]
        )
    if dense_plan:
        lines.extend(
            [
                "",
                "## QPruner dense-cache budget",
                "",
                f"- Budget bytes: `{dense_plan.get('budget_bytes')}`",
                f"- Selected modules: `{dense_plan.get('selected_modules')}`",
                f"- Selected dense-cache bytes: `{dense_plan.get('selected_dense_cache_bytes')}`",
                f"- Remaining budget bytes: `{dense_plan.get('remaining_budget_bytes')}`",
            ]
        )
    if profile:
        lines.extend(
            [
                "",
                "## QPruner runtime profile",
                "",
                f"- Status: `{profile.get('status')}`",
                f"- Quantized layers: `{profile.get('quantized_layers')}`",
                f"- QuantizedLinear forward calls: `{profile.get('total_forward_calls')}`",
                f"- QuantizedLinear forward time ms: `{fmt(profile.get('total_forward_time_ms'))}`",
                f"- Estimated forward share: `{fmt(profile.get('estimated_forward_share_pct'))}`%",
                f"- Grouped MLP pairs: `{profile.get('grouped_mlp_pairs')}`",
                f"- Grouped attention K/V pairs: `{profile.get('grouped_attention_kv_pairs')}`",
                f"- Runtime strategy: `{profile.get('runtime_strategy')}`",
                f"- Strategy timing: `{profile.get('by_strategy')}`",
            ]
        )
        top_modules = profile.get("top_modules", [])
        if isinstance(top_modules, list) and top_modules:
            first = top_modules[0]
            if isinstance(first, dict):
                lines.append(
                    "- Slowest QuantizedLinear: `{name}` strategy `{strategy}`, calls `{calls}`, time `{time}` ms".format(
                        name=first.get("name"),
                        strategy=first.get("strategy"),
                        calls=first.get("calls"),
                        time=fmt(first.get("time_ms")),
                    )
                )
    if report.get("error"):
        lines.extend(["", f"Error: `{report.get('error_type')}: {report.get('error')}`"])
    return "\n".join(lines) + "\n"


def write_outputs(report: dict[str, Any], demo_root: Path, run_label: str) -> None:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    (artifacts / f"qwen_qpruner_native_profile_{run_label}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    (reports / f"qwen-qpruner-native-profile-{run_label}.md").write_text(markdown_report(report))


def qpruner_profile_result(
    *,
    model: nn.Module,
    model_id: str,
    tokenizer: Any,
    target_names: list[str],
    baseline_latency_ms: float,
    baseline_targeted_params: int,
    baseline_targeted_storage_bytes: int,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> dict[str, Any]:
    start = time.perf_counter()
    result = qpruner_compress(
        model=model,
        model_id=model_id,
        importances={name: 1.0 for name in target_names},
        candidate_bits=(2, 4, 8),
        max_average_bits=args.qpruner_average_bits,
        target_roles=None,
        name_filter=limited_target_filter(set(target_names)),
        device=device,
        dtype=dtype,
        seed=args.seed,
        inplace=bool(args.inplace_compression),
    )
    compression_time_s = time.perf_counter() - start
    cache_modules = apply_qpruner_cache_mode(
        result.model,
        mode=args.qpruner_cache_mode,
        dtype=dtype,
        device=device,
        shape_policy_artifact=args.qpruner_shape_policy_artifact,
        release_packed_codes=bool(args.qpruner_release_packed_after_cache),
        prebuild_scaled_code_dtype_cache=bool(args.qpruner_prebuild_scaled_code_dtype_cache),
    )
    aux_cache_modules = 0
    if bool(args.qpruner_cache_aux_tensors):
        aux_cache_modules = enable_qpruner_aux_cache(result.model, device=device, dtype=dtype)
    dense_cache_budget_plan = None
    if args.qpruner_dense_cache_budget_bytes:
        dense_cache_budget_plan = enable_qpruner_dense_cache_budget(
            result.model,
            device=device,
            dtype=dtype,
            budget_bytes=int(args.qpruner_dense_cache_budget_bytes),
            selection_policy=args.qpruner_dense_cache_selection_policy,
        )
    scaled_code_dtype_cache_budget_plan = None
    if args.qpruner_scaled_code_dtype_cache_budget_bytes:
        scaled_code_dtype_cache_budget_plan = enable_qpruner_scaled_code_dtype_cache_budget(
            result.model,
            device=device,
            dtype=dtype,
            budget_bytes=int(args.qpruner_scaled_code_dtype_cache_budget_bytes),
            selection_policy=args.qpruner_scaled_code_dtype_cache_selection_policy,
        )
    grouped_mlp_pairs = 0
    if bool(args.qpruner_grouped_mlp):
        grouped_mlp_pairs = enable_grouped_qpruner_mlp_pairs(
            result.model,
            strategy=args.qpruner_grouped_mlp_strategy,
        )
    grouped_attention_kv_pairs = 0
    if bool(args.qpruner_grouped_attention_kv):
        grouped_attention_kv_pairs = enable_grouped_qpruner_attention_kv_pairs(result.model)
    metrics = benchmark_model(
        model=result.model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=args.max_new_tokens,
        iters=args.iters,
        warmup=args.warmup,
    )
    runtime_profile = profile_qpruner_runtime(
        result.model,
        qpruner_profile_workload(
            model=result.model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=args.max_new_tokens,
        ),
        device=device,
        full_latency_ms=metrics.get("latency_ms"),
    )
    runtime_profile["grouped_mlp_pairs"] = grouped_mlp_pairs
    runtime_profile["grouped_mlp_strategy"] = args.qpruner_grouped_mlp_strategy if grouped_mlp_pairs else None
    runtime_profile["grouped_attention_kv_pairs"] = grouped_attention_kv_pairs
    average_bits = qpruner_average_bits(result.model, target_names)
    targeted_live_storage_bytes = compressed_payload_storage_bytes(result.model, target_names)
    targeted_storage_bytes = compressed_payload_storage_bytes(
        result.model,
        target_names,
        include_released_packed_codes=True,
    )
    runtime_metadata = qpruner_runtime_metadata(result.model)
    code_cache_storage_bytes = qpruner_runtime_storage_bytes(targeted_live_storage_bytes, runtime_metadata)
    metrics.update(
        {
            "compression_time_s": compression_time_s,
            "cache_modules": cache_modules,
            "cache_mode": args.qpruner_cache_mode,
            "release_packed_after_cache": bool(args.qpruner_release_packed_after_cache),
            "cache_aux_tensors": bool(args.qpruner_cache_aux_tensors),
            "aux_cache_modules": aux_cache_modules,
            "prebuild_scaled_code_dtype_cache": bool(args.qpruner_prebuild_scaled_code_dtype_cache),
            "scaled_code_dtype_cache_budget_bytes": int(
                args.qpruner_scaled_code_dtype_cache_budget_bytes or 0
            ),
            "scaled_code_dtype_cache_selection_policy": args.qpruner_scaled_code_dtype_cache_selection_policy,
            "scaled_code_dtype_cache_budget_plan": scaled_code_dtype_cache_budget_plan,
            "dense_cache_budget_bytes": int(args.qpruner_dense_cache_budget_bytes or 0),
            "dense_cache_selection_policy": args.qpruner_dense_cache_selection_policy,
            "dense_cache_budget_plan": dense_cache_budget_plan,
            "grouped_mlp_pairs": grouped_mlp_pairs,
            "grouped_mlp_strategy": args.qpruner_grouped_mlp_strategy if grouped_mlp_pairs else None,
            "grouped_attention_kv_pairs": grouped_attention_kv_pairs,
            "exported_dense_linears": 0,
            "targeted_layers": len(target_names),
            "targeted_params_original": baseline_targeted_params,
            "targeted_params_quantized": targeted_parameter_count(result.model, target_names),
            "compressed_payload_storage_bytes": targeted_storage_bytes,
            "live_compressed_payload_storage_bytes": targeted_live_storage_bytes,
            "targeted_storage_measurement": "compressed_payload_model_storage_bytes",
            "average_bits": average_bits,
            "targeted_param_reduction_pct": qpruner_reduction_from_bits(average_bits, args.dense_weight_bits),
            "targeted_storage_bytes": targeted_storage_bytes,
            "targeted_storage_reduction_pct": reduction_pct(
                baseline_targeted_storage_bytes,
                targeted_storage_bytes,
            ),
            "code_cache_storage_bytes": code_cache_storage_bytes,
            "code_cache_storage_reduction_pct": reduction_pct(
                baseline_targeted_storage_bytes,
                code_cache_storage_bytes,
            ),
            "latency_speedup": speedup(baseline_latency_ms, metrics.get("latency_ms")),
            "runtime_profile": runtime_profile,
            "_profile_model": result.model,
            **runtime_metadata,
        }
    )
    return metrics


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if args.paired_rounds < 0:
        raise ValueError("--paired-rounds must be >= 0")
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    torch_dtype = dtype if isinstance(dtype, torch.dtype) else torch.float32
    model_path = Path(args.model_path)
    tokenizer = load_tokenizer(model_path)

    baseline_model = load_model(model_path, device, torch_dtype)
    selected_target_names, total_targeted_layers = select_target_names(
        baseline_model,
        args.target_layer_limit,
        args.target_layer_pattern,
    )
    if not selected_target_names:
        raise ValueError("model does not contain matching Qwen projection layers")
    baseline_targeted_params = targeted_parameter_count(baseline_model, selected_target_names)
    baseline_targeted_storage_bytes = bits_to_storage_bytes(baseline_targeted_params, args.dense_weight_bits)
    baseline = benchmark_model(
        model=baseline_model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=args.max_new_tokens,
        iters=args.iters,
        warmup=args.warmup,
    )
    baseline.update(
        {
            "targeted_layers": len(selected_target_names),
            "targeted_params": baseline_targeted_params,
            "targeted_storage_bytes": baseline_targeted_storage_bytes,
            "runtime_storage_format": "dense fp16/fp32 weights",
            "runtime_strategy": "dense_linear",
        }
    )
    qpruner = qpruner_profile_result(
        model=load_model(model_path, device, torch_dtype),
        model_id=args.model_id,
        tokenizer=tokenizer,
        target_names=selected_target_names,
        baseline_latency_ms=float(baseline["latency_ms"]),
        baseline_targeted_params=baseline_targeted_params,
        baseline_targeted_storage_bytes=int(baseline_targeted_storage_bytes or 0),
        device=device,
        dtype=torch_dtype,
        args=args,
    )
    synchronize_device(device)
    exports = {"baseline": baseline, "qpruner": qpruner}
    summary = summarize(exports)
    paired_samples: list[dict[str, Any]] = []
    paired_summary: dict[str, Any] | None = None
    if args.paired_rounds > 0:
        paired_samples = collect_paired_samples(
            baseline_model=baseline_model,
            qpruner_model=qpruner.pop("_profile_model"),
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=args.max_new_tokens,
            iters=args.iters,
            warmup=args.warmup,
            paired_rounds=args.paired_rounds,
        )
        paired_summary = summarize_paired_samples(paired_samples, paired_rounds=args.paired_rounds)
        summary.update(
            {
                "qpruner_vs_baseline_speedup": paired_summary.get(
                    "qpruner_vs_baseline_latency_median_speedup"
                ),
                "qpruner_tokens_per_s_ratio": paired_summary.get(
                    "qpruner_vs_baseline_tokens_median_ratio"
                ),
                "measurement_basis": "paired_interleaved_median",
            }
        )
    else:
        qpruner.pop("_profile_model", None)
        summary["measurement_basis"] = "single_sequential_run"
    effective_target_layer_limit = len(selected_target_names)
    coverage_pct = round(100.0 * len(selected_target_names) / total_targeted_layers, 3) if total_targeted_layers else None
    report = {
        "status": "PASS",
        "backend": "torch_generate_qwen_qpruner_native_profile",
        "model_id": args.model_id,
        "model_path": str(model_path),
        "device": str(device),
        "dtype": str(torch_dtype),
        "download_attempted": False,
        "serving_dense_export": False,
        "inplace_compression": bool(args.inplace_compression),
        "qpruner_cache_mode": args.qpruner_cache_mode,
        "qpruner_shape_policy_artifact": args.qpruner_shape_policy_artifact,
        "qpruner_release_packed_after_cache": bool(args.qpruner_release_packed_after_cache),
        "qpruner_cache_aux_tensors": bool(args.qpruner_cache_aux_tensors),
        "qpruner_prebuild_scaled_code_dtype_cache": bool(args.qpruner_prebuild_scaled_code_dtype_cache),
        "qpruner_scaled_code_dtype_cache_budget_bytes": int(
            args.qpruner_scaled_code_dtype_cache_budget_bytes or 0
        ),
        "qpruner_scaled_code_dtype_cache_selection_policy": args.qpruner_scaled_code_dtype_cache_selection_policy,
        "qpruner_dense_cache_budget_bytes": int(args.qpruner_dense_cache_budget_bytes or 0),
        "qpruner_dense_cache_selection_policy": args.qpruner_dense_cache_selection_policy,
        "qpruner_grouped_mlp": bool(args.qpruner_grouped_mlp),
        "qpruner_grouped_mlp_strategy": args.qpruner_grouped_mlp_strategy,
        "qpruner_grouped_attention_kv": bool(args.qpruner_grouped_attention_kv),
        "target_layer_pattern": args.target_layer_pattern,
        "target_layer_limit": effective_target_layer_limit,
        "target_layer_requested_limit": args.target_layer_limit,
        "targeted_layers_total": total_targeted_layers,
        "targeted_layer_names_sample": selected_target_names[:8],
        "target_layer_coverage_pct": coverage_pct,
        "max_new_tokens": args.max_new_tokens,
        "iters": args.iters,
        "warmup": args.warmup,
        "paired_rounds": args.paired_rounds,
        "baseline": baseline,
        "qpruner": qpruner,
        "summary": summary,
        "memory_reference": {
            "baseline_targeted_params": baseline_targeted_params,
            "targeted_layers": len(selected_target_names),
            "target_layer_coverage_pct": coverage_pct,
            "qpruner_targeted_param_reduction_pct": qpruner.get("targeted_param_reduction_pct"),
            "storage_measurement": "compressed_payload_model_storage_bytes",
            "baseline_targeted_storage_bytes": baseline_targeted_storage_bytes,
            "qpruner_targeted_storage_bytes": qpruner.get("targeted_storage_bytes"),
            "qpruner_code_cache_storage_bytes": qpruner.get("code_cache_storage_bytes"),
            "qpruner_aux_cache_bytes": qpruner.get("cached_aux_bytes"),
            "qpruner_scaled_code_dtype_cache_bytes": qpruner.get("cached_scaled_code_bytes"),
            "qpruner_dense_cache_bytes": qpruner.get("cached_dense_weight_bytes"),
            "qpruner_targeted_storage_reduction_pct": qpruner.get("targeted_storage_reduction_pct"),
            "qpruner_code_cache_storage_reduction_pct": qpruner.get("code_cache_storage_reduction_pct"),
        },
        "peak_mem_mb": peak_memory_mb(device),
        "platform": platform.platform(),
    }
    if paired_summary is not None:
        report["paired_summary"] = paired_summary
        report["paired_samples"] = paired_samples
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Qwen-family QPruner native torch.generate runtime")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-path", default="/mnt/nvme/622/models/Qwen3-0.6B")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument(
        "--paired-rounds",
        type=int,
        default=0,
        help="Run additional alternating baseline/QPruner measurements and summarize medians.",
    )
    parser.add_argument("--target-layer-limit", type=int, default=8)
    parser.add_argument("--target-layer-pattern")
    parser.add_argument("--qpruner-average-bits", type=float, default=4.0)
    parser.add_argument("--dense-weight-bits", type=float, default=16.0)
    parser.add_argument(
        "--qpruner-cache-mode",
        choices=("none", "dense", "code", "scaled-code-matmul", "shape-aware-code"),
        default="code",
        help="Runtime strategy for live QuantizedLinear modules.",
    )
    parser.add_argument(
        "--qpruner-shape-policy-artifact",
        help="Optional shape sweep artifact used by --qpruner-cache-mode shape-aware-code.",
    )
    parser.add_argument(
        "--qpruner-release-packed-after-cache",
        action="store_true",
        help="After building a QPruner code cache, release packed payload buffers to keep live memory low.",
    )
    parser.add_argument(
        "--qpruner-prebuild-scaled-code-dtype-cache",
        action="store_true",
        help="Prebuild scaled-code dtype tensors for scaled-code QPruner paths to avoid per-forward int8->dtype casts.",
    )
    parser.add_argument(
        "--qpruner-cache-aux-tensors",
        action="store_true",
        help="Cache QPruner scale and bias tensors in serving dtype/device to avoid per-forward aux casts.",
    )
    parser.add_argument(
        "--qpruner-scaled-code-dtype-cache-budget-bytes",
        type=int,
        default=0,
        help="Optional byte budget for prebuilding dtype copies of scaled-code QPruner caches.",
    )
    parser.add_argument(
        "--qpruner-scaled-code-dtype-cache-selection-policy",
        choices=QPRUNER_SCALED_CODE_DTYPE_CACHE_SELECTION_POLICIES,
        default="smallest_scaled_code_dtype_cache_first",
        help="How to spend --qpruner-scaled-code-dtype-cache-budget-bytes across scaled-code QPruner layers.",
    )
    parser.add_argument(
        "--qpruner-dense-cache-budget-bytes",
        type=int,
        default=0,
        help="Optional byte budget for promoting largest QPruner code-cache layers to dense dtype cache.",
    )
    parser.add_argument(
        "--qpruner-dense-cache-selection-policy",
        choices=("smallest_dense_cache_first", "qwen_projection_hotspot_first"),
        default="smallest_dense_cache_first",
        help="How to spend --qpruner-dense-cache-budget-bytes across QPruner layers.",
    )
    parser.add_argument(
        "--qpruner-grouped-mlp",
        action="store_true",
        help="Patch Qwen MLP gate/up QuantizedLinear pairs to use one grouped same-input scaled-code matmul.",
    )
    parser.add_argument(
        "--qpruner-grouped-mlp-strategy",
        choices=("bmm", "fused-2d"),
        default="bmm",
        help="Implementation strategy for --qpruner-grouped-mlp.",
    )
    parser.add_argument(
        "--qpruner-grouped-attention-kv",
        action="store_true",
        help="Patch Qwen attention k/v QuantizedLinear pairs to share one grouped same-input scaled-code matmul.",
    )
    parser.add_argument("--inplace-compression", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-label", default="qwen3_06b_native_profile_npu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_benchmark(args)
    except Exception as exc:
        report = {
            "status": "FAIL",
            "backend": "torch_generate_qwen_qpruner_native_profile",
            "model_id": args.model_id,
            "model_path": args.model_path,
            "device": args.device,
            "download_attempted": False,
            "serving_dense_export": False,
            "qpruner_cache_mode": args.qpruner_cache_mode,
            "qpruner_shape_policy_artifact": args.qpruner_shape_policy_artifact,
            "qpruner_release_packed_after_cache": bool(args.qpruner_release_packed_after_cache),
            "qpruner_cache_aux_tensors": bool(args.qpruner_cache_aux_tensors),
            "qpruner_prebuild_scaled_code_dtype_cache": bool(args.qpruner_prebuild_scaled_code_dtype_cache),
            "qpruner_scaled_code_dtype_cache_budget_bytes": int(
                args.qpruner_scaled_code_dtype_cache_budget_bytes or 0
            ),
            "qpruner_scaled_code_dtype_cache_selection_policy": args.qpruner_scaled_code_dtype_cache_selection_policy,
            "qpruner_dense_cache_selection_policy": args.qpruner_dense_cache_selection_policy,
            "qpruner_grouped_mlp": bool(args.qpruner_grouped_mlp),
            "qpruner_grouped_mlp_strategy": args.qpruner_grouped_mlp_strategy,
            "qpruner_grouped_attention_kv": bool(args.qpruner_grouped_attention_kv),
            "target_layer_limit": args.target_layer_limit,
            "target_layer_requested_limit": args.target_layer_limit,
            "target_layer_pattern": args.target_layer_pattern,
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "platform": platform.platform(),
        }
    write_outputs(report, Path(args.demo_root), args.run_label)
    print("QWEN_QPRUNER_NATIVE_PROFILE " + json.dumps(report, sort_keys=True))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
