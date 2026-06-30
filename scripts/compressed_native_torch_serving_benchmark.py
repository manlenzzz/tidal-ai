#!/usr/bin/env python
"""Benchmark torch.generate while keeping TIDAL compressed modules live."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from ascend_inference_benchmark import peak_memory_mb, reset_peak_memory_stats, synchronize_device  # noqa: E402
from tiny_qwen_compression_benchmark import (  # noqa: E402
    enable_model_inference_cache,
    load_model,
    load_tokenizer,
    qpruner_average_bits,
    qwen_projection_filter,
    target_linear_names,
    targeted_parameter_count,
)
from tiny_qwen_compression_generate_benchmark import DEFAULT_PROMPTS, run_generate  # noqa: E402
from tidal.device import resolve_device, resolve_dtype  # noqa: E402
from tidal.methods.global_rank_sparsity.torch import CAPPackedLinear  # noqa: E402
from tidal.methods.qpruner.torch import QuantizedLinear  # noqa: E402
from tidal.workflows.compression import cap_compress, qpruner_compress  # noqa: E402


SHAPE_AWARE_SCORE_THRESHOLD = 256
MEMORY_PRESERVING_QPRUNER_STRATEGIES = {
    "int8_code_cache_dequantize_on_device",
    "scaled_int8_code_matmul",
}
QWEN_PROJECTION_FAMILIES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
QPRUNER_DENSE_CACHE_SELECTION_POLICIES = (
    "smallest_dense_cache_first",
    "qwen_projection_hotspot_first",
)
QPRUNER_SCALED_CODE_DTYPE_CACHE_SELECTION_POLICIES = (
    "smallest_scaled_code_dtype_cache_first",
    "qwen_projection_hotspot_first",
)
QWEN_HOTSPOT_DENSE_CACHE_PRIORITY = {
    "q_proj": 0,
    "o_proj": 1,
    "down_proj": 2,
    "k_proj": 3,
    "v_proj": 4,
    "gate_proj": 5,
    "up_proj": 6,
}


def fmt(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def speedup(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return round(float(numerator) / float(denominator), 3)


def reduction_pct(original: int | float | None, compressed: int | float | None) -> float | None:
    if original in (None, 0) or compressed is None:
        return None
    return round(100.0 * (1.0 - float(compressed) / float(original)), 3)


def qpruner_reduction_from_bits(bits: float | None, dense_bits: float) -> float | None:
    if bits is None or dense_bits <= 0:
        return None
    return round(100.0 * (1.0 - float(bits) / dense_bits), 3)


def bits_to_storage_bytes(parameter_count: int | float | None, bits_per_parameter: int | float | None) -> int | None:
    if parameter_count is None or bits_per_parameter is None:
        return None
    return int((float(parameter_count) * float(bits_per_parameter) + 7) // 8)


def tensor_storage_bytes(tensor: torch.Tensor | None) -> int:
    if not torch.is_tensor(tensor):
        return 0
    return int(tensor.numel() * tensor.element_size())


def module_payload_storage_bytes(
    module: nn.Module,
    *,
    include_released_packed_codes: bool = False,
) -> int:
    if isinstance(module, CAPPackedLinear):
        return (
            tensor_storage_bytes(module.low_rank_left)
            + tensor_storage_bytes(module.low_rank_right)
            + tensor_storage_bytes(module.sparse_indices)
            + tensor_storage_bytes(module.sparse_values)
            + tensor_storage_bytes(module.bias)
        )
    if isinstance(module, QuantizedLinear):
        packed_code_bytes = tensor_storage_bytes(module.packed_weight_codes)
        if include_released_packed_codes:
            packed_code_bytes += int(getattr(module, "released_packed_code_bytes", 0) or 0)
        return (
            packed_code_bytes
            + tensor_storage_bytes(module.scale)
            + tensor_storage_bytes(module.bias)
        )
    weight = getattr(module, "weight", None)
    bias = getattr(module, "bias", None)
    return tensor_storage_bytes(weight) + tensor_storage_bytes(bias)


def compressed_payload_storage_bytes(
    model: nn.Module,
    names: list[str],
    *,
    include_released_packed_codes: bool = False,
) -> int:
    modules = dict(model.named_modules())
    return sum(
        module_payload_storage_bytes(
            modules[name],
            include_released_packed_codes=include_released_packed_codes,
        )
        for name in names
        if name in modules
    )


def _cached_weight_is_live(module: nn.Module) -> bool:
    return getattr(module, "_cached_weight", None) is not None


def _cached_weight_codes_are_live(module: nn.Module) -> bool:
    return getattr(module, "_cached_weight_codes", None) is not None


def cap_runtime_metadata(model: nn.Module) -> dict[str, Any]:
    packed = [module for module in model.modules() if isinstance(module, CAPPackedLinear)]
    cached_modules = sum(1 for module in packed if _cached_weight_is_live(module))
    dense_sparse_buffers = 0
    sparse_entries = 0
    coordinate_value_entries = 0
    for module in packed:
        buffers = dict(module.named_buffers(recurse=False))
        if "sparse" in buffers:
            dense_sparse_buffers += 1
        sparse_entries += int(getattr(module, "sparse_entries", 0) or 0)
        sparse_values = getattr(module, "sparse_values", None)
        if torch.is_tensor(sparse_values):
            coordinate_value_entries += int(sparse_values.numel())
    return {
        "runtime_storage_format": "coordinate_sparse_residual",
        "runtime_strategy": "dense_weight_cache" if cached_modules else "coordinate_sparse_residual",
        "packed_layers": len(packed),
        "sparse_entries": sparse_entries,
        "coordinate_value_entries": coordinate_value_entries,
        "dense_sparse_buffers": dense_sparse_buffers,
        "cached_dense_weight_modules": cached_modules,
    }


def qpruner_runtime_metadata(model: nn.Module) -> dict[str, Any]:
    quantized = [module for module in model.modules() if isinstance(module, QuantizedLinear)]
    cached_modules = sum(1 for module in quantized if _cached_weight_is_live(module))
    cached_code_modules = sum(1 for module in quantized if _cached_weight_codes_are_live(module))
    scaled_code_matmul_modules = sum(1 for module in quantized if getattr(module, "scaled_code_matmul_enabled", False))
    shape_aware_plan = getattr(model, "_tidal_qpruner_shape_aware_plan", None)
    shape_aware_scaled_code_modules = 0
    shape_aware_dense_cache_modules = 0
    if isinstance(shape_aware_plan, dict):
        shape_entries = shape_aware_plan.get("module_shapes", [])
        if isinstance(shape_entries, list):
            shape_aware_scaled_code_modules = sum(
                1
                for entry in shape_entries
                if isinstance(entry, dict) and entry.get("strategy") == "scaled_int8_code_matmul"
            )
            shape_aware_dense_cache_modules = sum(
                1
                for entry in shape_entries
                if isinstance(entry, dict) and entry.get("strategy") == "dense_weight_cache"
            )
    weight_code_entries = 0
    packed_code_bytes = 0
    cached_code_bytes = 0
    cached_scaled_code_bytes = 0
    cached_dense_weight_bytes = 0
    cached_scale_bytes = 0
    cached_bias_bytes = 0
    cached_aux_bytes = 0
    released_packed_code_bytes = 0
    for module in quantized:
        weight_code_entries += int(getattr(module, "code_count", 0) or 0)
        packed_code_bytes += tensor_storage_bytes(getattr(module, "packed_weight_codes", None))
        cached_code_bytes += int(getattr(module, "cached_code_bytes", 0) or 0)
        cached_scaled_code_bytes += int(getattr(module, "cached_scaled_code_bytes", 0) or 0)
        cached_dense_weight_bytes += int(getattr(module, "cached_weight_bytes", 0) or 0)
        cached_scale_bytes += int(getattr(module, "cached_scale_bytes", 0) or 0)
        cached_bias_bytes += int(getattr(module, "cached_bias_bytes", 0) or 0)
        cached_aux_bytes += int(getattr(module, "cached_aux_bytes", 0) or 0)
        released_packed_code_bytes += int(getattr(module, "released_packed_code_bytes", 0) or 0)
    if isinstance(shape_aware_plan, dict):
        if cached_modules and (cached_code_modules or scaled_code_matmul_modules):
            runtime_strategy = "shape_aware_mixed_dense_int8_code_cache"
        elif cached_modules:
            runtime_strategy = "shape_aware_dense_weight_cache"
        elif shape_aware_scaled_code_modules:
            runtime_strategy = "shape_aware_mixed_int8_code_cache"
        else:
            runtime_strategy = "shape_aware_int8_code_cache"
    elif cached_modules and (cached_code_modules or scaled_code_matmul_modules):
        runtime_strategy = "mixed_dense_int8_code_cache"
    elif cached_modules:
        runtime_strategy = "dense_weight_cache"
    elif scaled_code_matmul_modules:
        runtime_strategy = "scaled_int8_code_matmul"
    elif cached_code_modules:
        runtime_strategy = "int8_code_cache_dequantize_on_device"
    else:
        runtime_strategy = "dequantize_per_forward"
    return {
        "runtime_storage_format": "packed_nbit_weight_codes",
        "runtime_strategy": runtime_strategy,
        "quantized_layers": len(quantized),
        "weight_code_entries": weight_code_entries,
        "packed_code_bytes": packed_code_bytes,
        "released_packed_code_bytes": released_packed_code_bytes,
        "cached_dense_weight_modules": cached_modules,
        "cached_dense_weight_bytes": cached_dense_weight_bytes,
        "cached_code_modules": cached_code_modules,
        "cached_code_bytes": cached_code_bytes,
        "cached_scaled_code_bytes": cached_scaled_code_bytes,
        "cached_aux_modules": sum(1 for module in quantized if int(getattr(module, "cached_aux_bytes", 0) or 0)),
        "cached_scale_bytes": cached_scale_bytes,
        "cached_bias_bytes": cached_bias_bytes,
        "cached_aux_bytes": cached_aux_bytes,
        "scaled_code_matmul_modules": scaled_code_matmul_modules,
        "shape_aware_cache_modules": cached_code_modules if isinstance(shape_aware_plan, dict) else 0,
        "shape_aware_scaled_code_modules": shape_aware_scaled_code_modules,
        "shape_aware_dense_cache_modules": shape_aware_dense_cache_modules,
        "shape_aware_plan": shape_aware_plan,
    }


def qpruner_module_runtime_strategy(module: QuantizedLinear) -> str:
    cached_weight = getattr(module, "_cached_weight", None)
    cached_codes = getattr(module, "_cached_weight_codes", None)
    if cached_weight is not None:
        return "dense_weight_cache"
    if getattr(module, "scaled_code_matmul_enabled", False):
        return "scaled_int8_code_matmul"
    if cached_codes is not None:
        return "int8_code_cache_dequantize_on_device"
    return "dequantize_per_forward"


def profile_qpruner_runtime(
    model: nn.Module,
    workload: Any,
    *,
    device: torch.device,
    full_latency_ms: float | None = None,
    top_k: int = 8,
) -> dict[str, Any]:
    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, QuantizedLinear)]
    if not modules:
        return {
            "status": "SKIPPED",
            "reason": "no_quantized_layers",
            "quantized_layers": 0,
            "total_forward_calls": 0,
            "total_forward_time_ms": 0.0,
        }

    stats: dict[str, dict[str, Any]] = {
        name: {
            "name": name,
            "strategy": qpruner_module_runtime_strategy(module),
            "calls": 0,
            "time_ms": 0.0,
            "in_features": int(module.in_features),
            "out_features": int(module.out_features),
        }
        for name, module in modules
    }
    handles = []

    def make_pre_hook(name: str):
        def pre_hook(_module: nn.Module, _inputs: tuple[Any, ...]) -> None:
            synchronize_device(device)
            stats[name]["_start_s"] = time.perf_counter()

        return pre_hook

    def make_post_hook(name: str):
        def post_hook(_module: nn.Module, _inputs: tuple[Any, ...], _output: Any) -> None:
            synchronize_device(device)
            start_s = stats[name].pop("_start_s", None)
            if start_s is None:
                return
            stats[name]["calls"] += 1
            stats[name]["time_ms"] += (time.perf_counter() - float(start_s)) * 1000.0

        return post_hook

    try:
        for name, module in modules:
            handles.append(module.register_forward_pre_hook(make_pre_hook(name)))
            handles.append(module.register_forward_hook(make_post_hook(name)))
        workload()
        synchronize_device(device)
    finally:
        for handle in handles:
            handle.remove()

    module_rows: list[dict[str, Any]] = []
    by_strategy: dict[str, dict[str, Any]] = {}
    for row in stats.values():
        row.pop("_start_s", None)
        calls = int(row["calls"])
        time_ms = round(float(row["time_ms"]), 6)
        avg_ms = round(time_ms / calls, 6) if calls else None
        module_row = {
            "name": row["name"],
            "strategy": row["strategy"],
            "calls": calls,
            "time_ms": time_ms,
            "avg_ms_per_call": avg_ms,
            "in_features": row["in_features"],
            "out_features": row["out_features"],
        }
        module_rows.append(module_row)
        strategy = str(row["strategy"])
        bucket = by_strategy.setdefault(
            strategy,
            {
                "modules": 0,
                "calls": 0,
                "time_ms": 0.0,
                "avg_ms_per_call": None,
            },
        )
        bucket["modules"] += 1
        bucket["calls"] += calls
        bucket["time_ms"] += time_ms

    total_calls = sum(int(row["calls"]) for row in module_rows)
    total_time_ms = round(sum(float(row["time_ms"]) for row in module_rows), 6)
    for bucket in by_strategy.values():
        bucket["time_ms"] = round(float(bucket["time_ms"]), 6)
        bucket["avg_ms_per_call"] = (
            round(float(bucket["time_ms"]) / int(bucket["calls"]), 6) if int(bucket["calls"]) else None
        )
    estimated_share = None
    if full_latency_ms not in (None, 0):
        estimated_share = round(100.0 * total_time_ms / float(full_latency_ms), 3)
    runtime_metadata = qpruner_runtime_metadata(model)
    top_modules = sorted(module_rows, key=lambda row: float(row["time_ms"]), reverse=True)[:top_k]
    return {
        "status": "PASS",
        "measurement": "synchronized_forward_hooks",
        "synchronizes_per_forward": device.type in {"cuda", "npu"},
        "quantized_layers": len(modules),
        "total_forward_calls": total_calls,
        "total_forward_time_ms": total_time_ms,
        "estimated_forward_share_pct": estimated_share,
        "runtime_strategy": runtime_metadata.get("runtime_strategy"),
        "by_strategy": by_strategy,
        "top_modules": top_modules,
    }


def resolved_qpruner_cache_mode(args: argparse.Namespace) -> str:
    mode = str(getattr(args, "qpruner_cache_mode", "auto"))
    if mode == "auto":
        return "dense" if getattr(args, "enable_inference_cache", False) else "none"
    return mode


def enable_qpruner_code_cache(
    model: nn.Module,
    *,
    device: torch.device,
    release_packed_codes: bool = False,
) -> int:
    modules = [module for module in model.modules() if isinstance(module, QuantizedLinear)]
    for module in modules:
        module.enable_code_cache(device=device, release_packed_codes=release_packed_codes)
    return len(modules)


def enable_qpruner_scaled_code_matmul(
    model: nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
    release_packed_codes: bool = False,
) -> int:
    modules = [module for module in model.modules() if isinstance(module, QuantizedLinear)]
    for module in modules:
        module.enable_scaled_code_matmul(device=device, dtype=dtype, release_packed_codes=release_packed_codes)
    return len(modules)


def enable_qpruner_aux_cache(
    model: nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> int:
    modules = [module for module in model.modules() if isinstance(module, QuantizedLinear)]
    for module in modules:
        module.enable_aux_cache(dtype=dtype, device=device)
    return len(modules)


def qpruner_runtime_storage_bytes(
    compressed_payload_live_storage_bytes: int | float | None,
    runtime_metadata: dict[str, Any],
) -> int | None:
    if compressed_payload_live_storage_bytes is None:
        return None
    return int(compressed_payload_live_storage_bytes) + sum(
        int(runtime_metadata.get(key) or 0)
        for key in ("cached_code_bytes", "cached_scaled_code_bytes", "cached_dense_weight_bytes", "cached_aux_bytes")
    )


def _dtype_element_size(dtype: torch.dtype | None) -> int:
    return int(torch.empty((), dtype=dtype or torch.float32).element_size())


def _enable_dense_cache_from_qpruner_codes(
    module: QuantizedLinear,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> int:
    cached_codes = getattr(module, "_cached_weight_codes", None)
    if cached_codes is not None:
        weight = module.dequantized_weight_from_codes(cached_codes, dtype=dtype, device=device)
    else:
        weight = module.dequantized_weight().to(dtype=dtype, device=device)
    module._cached_weight = weight.detach()
    module._cached_weight_codes = None
    module._cached_scaled_weight_codes = None
    module._scaled_code_matmul_enabled = False
    return int(module.cached_weight_bytes)


def _refresh_shape_aware_strategy_counts(plan: dict[str, Any]) -> None:
    entries = plan.get("module_shapes", [])
    if not isinstance(entries, list):
        entries = []
    strategy_counts: dict[str, int] = {}
    strategy_source_counts: dict[str, int] = {}
    families: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        strategy = str(entry.get("strategy", "unknown"))
        source = str(entry.get("strategy_source", "unknown"))
        family = str(entry.get("family", "unknown"))
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        strategy_source_counts[source] = strategy_source_counts.get(source, 0) + 1
        families[family] = families.get(family, 0) + 1
    plan["strategy_counts"] = strategy_counts
    plan["strategy_source_counts"] = strategy_source_counts
    plan["families"] = families


def _module_name_leaf(name: str) -> str:
    leaf = str(name).rsplit(".", 1)[-1]
    return leaf if leaf else str(name)


def _dense_cache_candidate_sort_key(row: dict[str, Any], selection_policy: str) -> tuple[Any, ...]:
    if selection_policy == "smallest_dense_cache_first":
        return (int(row["dense_cache_bytes"]), -int(row["shape_score"]), str(row["name"]))
    if selection_policy == "qwen_projection_hotspot_first":
        family = _module_name_leaf(str(row["name"]))
        return (
            int(QWEN_HOTSPOT_DENSE_CACHE_PRIORITY.get(family, len(QWEN_HOTSPOT_DENSE_CACHE_PRIORITY))),
            int(row["dense_cache_bytes"]),
            str(row["name"]),
        )
    raise ValueError(
        "dense-cache selection policy must be one of: "
        + ", ".join(QPRUNER_DENSE_CACHE_SELECTION_POLICIES)
    )


def _scaled_code_dtype_cache_candidate_sort_key(row: dict[str, Any], selection_policy: str) -> tuple[Any, ...]:
    if selection_policy == "smallest_scaled_code_dtype_cache_first":
        return (int(row["scaled_code_dtype_cache_bytes"]), -int(row["shape_score"]), str(row["name"]))
    if selection_policy == "qwen_projection_hotspot_first":
        family = _module_name_leaf(str(row["name"]))
        return (
            int(QWEN_HOTSPOT_DENSE_CACHE_PRIORITY.get(family, len(QWEN_HOTSPOT_DENSE_CACHE_PRIORITY))),
            int(row["scaled_code_dtype_cache_bytes"]),
            str(row["name"]),
        )
    raise ValueError(
        "scaled-code dtype-cache selection policy must be one of: "
        + ", ".join(QPRUNER_SCALED_CODE_DTYPE_CACHE_SELECTION_POLICIES)
    )


def _enable_scaled_code_dtype_cache_from_qpruner_codes(
    module: QuantizedLinear,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> int:
    cached_codes = getattr(module, "_cached_weight_codes", None)
    if cached_codes is None:
        return 0
    module._cached_scaled_weight_codes = cached_codes.to(device=device, dtype=dtype).detach()
    return int(module.cached_scaled_code_bytes)


def enable_qpruner_scaled_code_dtype_cache_budget(
    model: nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
    budget_bytes: int | float | None,
    selection_policy: str = "smallest_scaled_code_dtype_cache_first",
) -> dict[str, Any]:
    if selection_policy not in QPRUNER_SCALED_CODE_DTYPE_CACHE_SELECTION_POLICIES:
        raise ValueError(
            "scaled-code dtype-cache selection policy must be one of: "
            + ", ".join(QPRUNER_SCALED_CODE_DTYPE_CACHE_SELECTION_POLICIES)
        )
    budget = int(budget_bytes or 0)
    if budget < 0:
        raise ValueError("scaled-code dtype-cache budget must be >= 0")
    element_size = _dtype_element_size(dtype)
    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, QuantizedLinear)]
    for _, module in modules:
        module._cached_scaled_weight_codes = None
    candidates = [
        {
            "name": name,
            "module": module,
            "scaled_code_dtype_cache_bytes": int(module.code_count) * element_size,
            "shape_score": int(module.in_features) * int(module.out_features),
            "current_strategy": qpruner_module_runtime_strategy(module),
        }
        for name, module in modules
        if getattr(module, "scaled_code_matmul_enabled", False)
    ]
    candidates.sort(key=lambda row: _scaled_code_dtype_cache_candidate_sort_key(row, selection_policy))

    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    remaining = budget
    candidate_names = {str(row["name"]) for row in candidates}
    for name, module in modules:
        if name in candidate_names:
            continue
        skipped.append(
            {
                "name": name,
                "scaled_code_dtype_cache_bytes": int(module.code_count) * element_size,
                "reason": "not_scaled_code_matmul",
            }
        )
    for row in candidates:
        cache_bytes = int(row["scaled_code_dtype_cache_bytes"])
        if cache_bytes <= 0:
            skipped.append({"name": row["name"], "scaled_code_dtype_cache_bytes": cache_bytes, "reason": "empty_weight"})
            continue
        if cache_bytes > remaining:
            skipped.append({"name": row["name"], "scaled_code_dtype_cache_bytes": cache_bytes, "reason": "over_budget"})
            continue
        module = row["module"]
        cached_codes = getattr(module, "_cached_weight_codes", None)
        if cached_codes is None:
            skipped.append({"name": row["name"], "scaled_code_dtype_cache_bytes": cache_bytes, "reason": "no_code_source"})
            continue
        actual_bytes = _enable_scaled_code_dtype_cache_from_qpruner_codes(module, device=device, dtype=dtype)
        selected.append(
            {
                "name": row["name"],
                "scaled_code_dtype_cache_bytes": actual_bytes,
                "previous_strategy": row["current_strategy"],
            }
        )
        remaining -= actual_bytes

    selected_by_name = {str(row["name"]): row for row in selected}
    shape_aware_plan = getattr(model, "_tidal_qpruner_shape_aware_plan", None)
    if isinstance(shape_aware_plan, dict):
        entries = shape_aware_plan.get("module_shapes", [])
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                selected_row = selected_by_name.get(str(entry.get("name")))
                if selected_row is None:
                    continue
                entry["scaled_code_dtype_cache_bytes"] = selected_row["scaled_code_dtype_cache_bytes"]
        shape_aware_plan["scaled_code_dtype_cache_budget_bytes"] = budget
        shape_aware_plan["scaled_code_dtype_cache_selection_policy"] = selection_policy
        shape_aware_plan["scaled_code_dtype_cache_selected_modules"] = len(selected)
        shape_aware_plan["scaled_code_dtype_cache_selected_bytes"] = sum(
            int(row["scaled_code_dtype_cache_bytes"]) for row in selected
        )
        shape_aware_plan["scaled_code_dtype_cache_remaining_budget_bytes"] = max(0, remaining)
        _refresh_shape_aware_strategy_counts(shape_aware_plan)

    return {
        "budget_bytes": budget,
        "selection_policy": selection_policy,
        "selected_modules": len(selected),
        "selected_scaled_code_dtype_cache_bytes": sum(
            int(row["scaled_code_dtype_cache_bytes"]) for row in selected
        ),
        "remaining_budget_bytes": max(0, remaining),
        "selected": selected,
        "skipped": skipped,
    }


def enable_qpruner_dense_cache_budget(
    model: nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
    budget_bytes: int | float | None,
    selection_policy: str = "smallest_dense_cache_first",
) -> dict[str, Any]:
    if selection_policy not in QPRUNER_DENSE_CACHE_SELECTION_POLICIES:
        raise ValueError(
            "dense-cache selection policy must be one of: "
            + ", ".join(QPRUNER_DENSE_CACHE_SELECTION_POLICIES)
        )
    budget = int(budget_bytes or 0)
    element_size = _dtype_element_size(dtype)
    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, QuantizedLinear)]
    candidates = [
        {
            "name": name,
            "module": module,
            "dense_cache_bytes": int(module.code_count) * element_size,
            "shape_score": int(module.in_features) * int(module.out_features),
            "current_strategy": qpruner_module_runtime_strategy(module),
        }
        for name, module in modules
    ]
    candidates.sort(key=lambda row: _dense_cache_candidate_sort_key(row, selection_policy))

    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    remaining = budget
    for row in candidates:
        dense_bytes = int(row["dense_cache_bytes"])
        if dense_bytes <= 0:
            skipped.append({"name": row["name"], "dense_cache_bytes": dense_bytes, "reason": "empty_weight"})
            continue
        if dense_bytes > remaining:
            skipped.append({"name": row["name"], "dense_cache_bytes": dense_bytes, "reason": "over_budget"})
            continue
        module = row["module"]
        cached_codes = getattr(module, "_cached_weight_codes", None)
        packed_codes = getattr(module, "packed_weight_codes", None)
        if cached_codes is None and tensor_storage_bytes(packed_codes) == 0:
            skipped.append({"name": row["name"], "dense_cache_bytes": dense_bytes, "reason": "no_code_source"})
            continue
        actual_bytes = _enable_dense_cache_from_qpruner_codes(module, device=device, dtype=dtype)
        selected.append(
            {
                "name": row["name"],
                "dense_cache_bytes": actual_bytes,
                "previous_strategy": row["current_strategy"],
            }
        )
        remaining -= actual_bytes

    selected_by_name = {str(row["name"]): row for row in selected}
    shape_aware_plan = getattr(model, "_tidal_qpruner_shape_aware_plan", None)
    if isinstance(shape_aware_plan, dict):
        entries = shape_aware_plan.get("module_shapes", [])
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                selected_row = selected_by_name.get(str(entry.get("name")))
                if selected_row is None:
                    continue
                entry["previous_strategy"] = entry.get("strategy")
                entry["previous_strategy_source"] = entry.get("strategy_source")
                entry["strategy"] = "dense_weight_cache"
                entry["strategy_source"] = "dense_cache_budget"
                entry["dense_cache_bytes"] = selected_row["dense_cache_bytes"]
        shape_aware_plan["dense_cache_budget_bytes"] = budget
        shape_aware_plan["dense_cache_selection_policy"] = selection_policy
        shape_aware_plan["dense_cache_selected_modules"] = len(selected)
        shape_aware_plan["dense_cache_selected_bytes"] = sum(int(row["dense_cache_bytes"]) for row in selected)
        shape_aware_plan["dense_cache_remaining_budget_bytes"] = max(0, remaining)
        _refresh_shape_aware_strategy_counts(shape_aware_plan)

    return {
        "budget_bytes": budget,
        "selection_policy": selection_policy,
        "selected_modules": len(selected),
        "selected_dense_cache_bytes": sum(int(row["dense_cache_bytes"]) for row in selected),
        "remaining_budget_bytes": max(0, remaining),
        "selected": selected,
        "skipped": skipped,
    }


def projection_family(name: str) -> str:
    leaf = name.rsplit(".", 1)[-1]
    return leaf if leaf else name


def sweep_row_projection_family(row: dict[str, Any]) -> str | None:
    family = row.get("family")
    if isinstance(family, str) and family in QWEN_PROJECTION_FAMILIES:
        return family
    label = row.get("label")
    if isinstance(label, str):
        for candidate in QWEN_PROJECTION_FAMILIES:
            if candidate in label:
                return candidate
    return None


def qpruner_shape_key(*, in_features: int, out_features: int) -> str:
    return f"{int(in_features)}x{int(out_features)}"


def _shape_dims_from_sweep_row(row: dict[str, Any]) -> tuple[int, int] | None:
    shape = row.get("shape", {}) if isinstance(row.get("shape"), dict) else {}
    try:
        in_features = int(shape.get("in_features"))
        out_features = int(shape.get("out_features"))
    except (TypeError, ValueError):
        return None
    if in_features <= 0 or out_features <= 0:
        return None
    return in_features, out_features


def _shape_policy_candidate(row: dict[str, Any], path_key: str) -> dict[str, Any] | None:
    payload = row.get(path_key, {}) if isinstance(row.get(path_key), dict) else {}
    latency_ms = payload.get("latency_ms")
    if latency_ms is None:
        return None
    strategy = payload.get("runtime_strategy")
    if strategy not in MEMORY_PRESERVING_QPRUNER_STRATEGIES:
        return None
    try:
        latency_value = float(latency_ms)
    except (TypeError, ValueError):
        return None
    return {
        "label": row.get("label"),
        "path": path_key,
        "strategy": strategy,
        "latency_ms": latency_value,
        "speedup_vs_uncached": payload.get("speedup_vs_uncached"),
        "storage_reduction_pct": row.get("storage_reduction_pct"),
        "code_cache_storage_reduction_pct": row.get("code_cache_storage_reduction_pct"),
    }


def _best_shape_policy_candidate(row: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [
        candidate
        for candidate in (
            _shape_policy_candidate(row, "code_cached"),
            _shape_policy_candidate(row, "scaled_code_matmul"),
        )
        if candidate is not None
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: float(item["latency_ms"]))


def load_qpruner_shape_policy(path: str | Path | None) -> dict[str, Any] | None:
    if path in (None, ""):
        return None
    policy_path = Path(path)
    payload = json.loads(policy_path.read_text())
    rows = payload.get("shape_sweep")
    if not isinstance(rows, list):
        raise ValueError(f"shape policy artifact lacks shape_sweep rows: {policy_path}")

    strategies: dict[str, dict[str, Any]] = {}
    family_strategies: dict[str, dict[str, Any]] = {}
    rows_examined = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        rows_examined += 1
        dims = _shape_dims_from_sweep_row(row)
        candidate = _best_shape_policy_candidate(row)
        if candidate is None:
            continue
        if dims is not None:
            key = qpruner_shape_key(in_features=dims[0], out_features=dims[1])
            existing = strategies.get(key)
            if existing is None or float(candidate["latency_ms"]) < float(existing["latency_ms"]):
                strategies[key] = {**candidate, "in_features": dims[0], "out_features": dims[1], "shape_key": key}
        family = sweep_row_projection_family(row)
        if family is not None:
            existing_family = family_strategies.get(family)
            if existing_family is None or float(candidate["latency_ms"]) < float(existing_family["latency_ms"]):
                family_strategies[family] = {
                    **candidate,
                    "family": family,
                    "strategy_source": "shape_sweep_family_fallback",
                    "shape_policy_label": candidate.get("label"),
                }

    if not strategies and not family_strategies:
        raise ValueError(f"shape policy artifact has no memory-preserving QPruner rows: {policy_path}")
    return {
        "source": str(policy_path),
        "shape_count": len(strategies),
        "family_count": len(family_strategies),
        "rows_examined": rows_examined,
        "strategies": strategies,
        "family_strategies": family_strategies,
    }


def shape_aware_qpruner_decision(
    module: QuantizedLinear,
    *,
    shape_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    shape_score = int(module.in_features) * int(module.out_features)
    shape_key = qpruner_shape_key(in_features=int(module.in_features), out_features=int(module.out_features))
    if isinstance(shape_policy, dict):
        strategies = shape_policy.get("strategies", {})
        policy_entry = strategies.get(shape_key) if isinstance(strategies, dict) else None
        if isinstance(policy_entry, dict) and policy_entry.get("strategy") in MEMORY_PRESERVING_QPRUNER_STRATEGIES:
            return {
                "strategy": policy_entry["strategy"],
                "strategy_source": "shape_sweep_artifact",
                "shape_key": shape_key,
                "shape_policy_source": shape_policy.get("source"),
                "shape_policy_label": policy_entry.get("label"),
                "shape_policy_path": policy_entry.get("path"),
                "shape_policy_latency_ms": policy_entry.get("latency_ms"),
                "shape_policy_speedup_vs_uncached": policy_entry.get("speedup_vs_uncached"),
            }
        family_strategies = shape_policy.get("family_strategies", {})
        family = projection_family(getattr(module, "_tidal_module_name", ""))
        if family not in QWEN_PROJECTION_FAMILIES:
            family = None
        family_entry = family_strategies.get(family) if isinstance(family_strategies, dict) else None
        if isinstance(family_entry, dict) and family_entry.get("strategy") in MEMORY_PRESERVING_QPRUNER_STRATEGIES:
            return {
                "strategy": family_entry["strategy"],
                "strategy_source": "shape_sweep_family_fallback",
                "shape_key": shape_key,
                "shape_policy_source": shape_policy.get("source"),
                "shape_policy_family": family,
                "shape_policy_label": family_entry.get("label") or family_entry.get("shape_policy_label"),
                "shape_policy_path": family_entry.get("path"),
                "shape_policy_latency_ms": family_entry.get("latency_ms"),
                "shape_policy_speedup_vs_uncached": family_entry.get("speedup_vs_uncached"),
            }

    strategy = (
        "scaled_int8_code_matmul"
        if shape_score >= SHAPE_AWARE_SCORE_THRESHOLD
        else "int8_code_cache_dequantize_on_device"
    )
    return {
        "strategy": strategy,
        "strategy_source": "shape_score_threshold",
        "shape_key": shape_key,
        "scaled_code_shape_score_threshold": SHAPE_AWARE_SCORE_THRESHOLD,
    }


def shape_aware_qpruner_strategy(
    module: QuantizedLinear,
    *,
    shape_policy: dict[str, Any] | None = None,
) -> str:
    return str(shape_aware_qpruner_decision(module, shape_policy=shape_policy)["strategy"])


def enable_qpruner_shape_aware_code_cache(
    model: nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
    shape_policy: dict[str, Any] | None = None,
    release_packed_codes: bool = False,
) -> int:
    entries: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, QuantizedLinear):
            continue
        module._tidal_module_name = name
        decision = shape_aware_qpruner_decision(module, shape_policy=shape_policy)
        strategy = str(decision["strategy"])
        if strategy == "scaled_int8_code_matmul":
            module.enable_scaled_code_matmul(device=device, dtype=dtype, release_packed_codes=release_packed_codes)
        else:
            module.enable_code_cache(device=device, release_packed_codes=release_packed_codes)
        entry = {
            "name": name,
            "family": projection_family(name),
            "in_features": int(module.in_features),
            "out_features": int(module.out_features),
            "shape_score": int(module.in_features) * int(module.out_features),
        }
        entry.update(decision)
        entries.append(entry)
    families: dict[str, int] = {}
    strategy_counts: dict[str, int] = {}
    strategy_source_counts: dict[str, int] = {}
    for entry in entries:
        family = str(entry["family"])
        families[family] = families.get(family, 0) + 1
        strategy = str(entry["strategy"])
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        source = str(entry.get("strategy_source", "unknown"))
        strategy_source_counts[source] = strategy_source_counts.get(source, 0) + 1
    model._tidal_qpruner_shape_aware_plan = {
        "policy": "memory_preserving_code_cache_by_projection_shape",
        "module_count": len(entries),
        "families": families,
        "strategy_counts": strategy_counts,
        "strategy_source_counts": strategy_source_counts,
        "scaled_code_shape_score_threshold": SHAPE_AWARE_SCORE_THRESHOLD,
        "shape_policy_source": shape_policy.get("source") if isinstance(shape_policy, dict) else None,
        "shape_policy_shape_count": shape_policy.get("shape_count") if isinstance(shape_policy, dict) else 0,
        "shape_policy_family_count": shape_policy.get("family_count") if isinstance(shape_policy, dict) else 0,
        "shape_policy_rows_examined": shape_policy.get("rows_examined") if isinstance(shape_policy, dict) else 0,
        "shape_policy_match_count": strategy_source_counts.get("shape_sweep_artifact", 0),
        "shape_policy_family_match_count": strategy_source_counts.get("shape_sweep_family_fallback", 0),
        "release_packed_codes": bool(release_packed_codes),
        "module_shapes": entries,
    }
    return len(entries)


def summarize(exports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    passing = {
        method: payload
        for method, payload in exports.items()
        if payload.get("status") == "PASS" and payload.get("tokens_per_s") is not None
    }
    best_method = None
    best_tokens = None
    if passing:
        best_method, best_payload = max(passing.items(), key=lambda item: float(item[1]["tokens_per_s"]))
        best_tokens = best_payload.get("tokens_per_s")
    baseline_latency = exports.get("baseline", {}).get("latency_ms")
    cap_latency = exports.get("cap", {}).get("latency_ms")
    qpruner_latency = exports.get("qpruner", {}).get("latency_ms")
    return {
        "best_method": best_method,
        "best_tokens_per_s": best_tokens,
        "cap_vs_baseline_speedup": speedup(baseline_latency, cap_latency),
        "qpruner_vs_baseline_speedup": speedup(baseline_latency, qpruner_latency),
        "dense_export_erases_storage_savings": None,
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report.get("summary", {}) if isinstance(report.get("summary"), dict) else {}
    memory = report.get("memory_reference", {}) if isinstance(report.get("memory_reference"), dict) else {}
    baseline = report.get("baseline", {}) if isinstance(report.get("baseline"), dict) else {}
    cap = report.get("cap", {}) if isinstance(report.get("cap"), dict) else {}
    qpruner = report.get("qpruner", {}) if isinstance(report.get("qpruner"), dict) else {}
    lines = [
        "# Compressed-Native Torch Serving Benchmark",
        "",
        "torch.generate reference path that keeps CAPPackedLinear/QuantizedLinear modules live. This is the memory-first serving target before wiring the same representation into vLLM-Ascend.",
        "",
        "| Metric | Baseline | CAP | QPruner |",
        "|---|---:|---:|---:|",
        f"| Status | {baseline.get('status')} | {cap.get('status')} | {qpruner.get('status')} |",
        f"| Latency ms | {baseline.get('latency_ms')} | {cap.get('latency_ms')} | {qpruner.get('latency_ms')} |",
        f"| Tokens/s | {baseline.get('tokens_per_s')} | {cap.get('tokens_per_s')} | {qpruner.get('tokens_per_s')} |",
        f"| Peak MB | {baseline.get('peak_mem_mb')} | {cap.get('peak_mem_mb')} | {qpruner.get('peak_mem_mb')} |",
        f"| Speedup | - | {summary.get('cap_vs_baseline_speedup')} | {summary.get('qpruner_vs_baseline_speedup')} |",
        f"| Targeted layers | {baseline.get('targeted_layers')} | {cap.get('targeted_layers')} | {qpruner.get('targeted_layers')} |",
        f"| Compressed modules | - | {cap.get('packed_layers')} packed | {qpruner.get('quantized_layers')} quantized |",
        f"| Compression | - | {cap.get('targeted_compression_ratio')}x | {qpruner.get('average_bits')} avg bits |",
        f"| Targeted memory reduction | - | {cap.get('targeted_param_reduction_pct')}% | {qpruner.get('targeted_param_reduction_pct')}% |",
        f"| Targeted storage bytes | {baseline.get('targeted_storage_bytes')} | {cap.get('targeted_storage_bytes')} | {qpruner.get('targeted_storage_bytes')} |",
        f"| Targeted storage reduction | - | {cap.get('targeted_storage_reduction_pct')}% | {qpruner.get('targeted_storage_reduction_pct')}% |",
        f"| Runtime storage | {baseline.get('runtime_storage_format', 'dense fp16/fp32 weights')} | {cap.get('runtime_storage_format')} | {qpruner.get('runtime_storage_format')} |",
        f"| Runtime strategy | {baseline.get('runtime_strategy', 'dense_linear')} | {cap.get('runtime_strategy')} | {qpruner.get('runtime_strategy')} |",
        f"| Dense sparse buffers | - | {cap.get('dense_sparse_buffers')} | - |",
        f"| Sparse entries | - | {cap.get('sparse_entries')} | - |",
        f"| Cache modules | - | {cap.get('cache_modules')} | {qpruner.get('cache_modules')} |",
        f"| QPruner code-cache bytes | - | - | {qpruner.get('cached_code_bytes')} |",
        f"| QPruner scaled-code dtype-cache bytes | - | - | {qpruner.get('cached_scaled_code_bytes')} |",
        f"| QPruner dense-cache bytes | - | - | {qpruner.get('cached_dense_weight_bytes')} |",
        f"| QPruner aux-cache bytes | - | - | {qpruner.get('cached_aux_bytes')} |",
        f"| Serving dense export | {report.get('serving_dense_export')} | {report.get('serving_dense_export')} | {report.get('serving_dense_export')} |",
        f"| Exported dense linears | - | {cap.get('exported_dense_linears')} | {qpruner.get('exported_dense_linears')} |",
        "",
        "## Memory Reference",
        "",
        f"- Baseline targeted params: `{memory.get('baseline_targeted_params')}`",
        f"- Target layer coverage: `{memory.get('target_layer_coverage_pct')}`%",
        f"- CAP targeted memory reduction: `{memory.get('cap_targeted_param_reduction_pct')}`%",
        f"- QPruner targeted memory reduction: `{memory.get('qpruner_targeted_param_reduction_pct')}`%",
        f"- Native compressed-module storage measurement: `{memory.get('storage_measurement')}`",
        f"- Baseline targeted storage bytes: `{memory.get('baseline_targeted_storage_bytes')}`",
        f"- CAP targeted storage bytes: `{memory.get('cap_targeted_storage_bytes')}`",
        f"- QPruner targeted storage bytes: `{memory.get('qpruner_targeted_storage_bytes')}`",
        f"- QPruner code-cache storage bytes: `{memory.get('qpruner_code_cache_storage_bytes')}`",
        f"- QPruner aux-cache bytes: `{memory.get('qpruner_aux_cache_bytes')}`",
        f"- CAP targeted storage reduction: `{memory.get('cap_targeted_storage_reduction_pct')}`%",
        f"- QPruner targeted storage reduction: `{memory.get('qpruner_targeted_storage_reduction_pct')}`%",
        f"- QPruner code-cache storage reduction: `{memory.get('qpruner_code_cache_storage_reduction_pct')}`%",
        f"- dense_export_erases_storage_savings: `{summary.get('dense_export_erases_storage_savings')}`",
        "",
        "## Metadata",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Backend: `{report.get('backend')}`",
        f"- Model: `{report.get('model_id')}`",
        f"- Model path: `{report.get('model_path')}`",
        f"- Device: `{report.get('device')}`",
        f"- Dtype: `{report.get('dtype')}`",
        f"- Inference cache enabled: `{report.get('inference_cache_enabled')}`",
        f"- QPruner cache mode: `{report.get('qpruner_cache_mode')}`",
        f"- Cache QPruner aux tensors: `{report.get('qpruner_cache_aux_tensors')}`",
        f"- Prebuild scaled-code dtype cache: `{report.get('qpruner_prebuild_scaled_code_dtype_cache')}`",
        f"- QPruner scaled-code dtype-cache budget bytes: `{report.get('qpruner_scaled_code_dtype_cache_budget_bytes')}`",
        f"- QPruner scaled-code dtype-cache selection policy: `{report.get('qpruner_scaled_code_dtype_cache_selection_policy')}`",
        f"- QPruner dense-cache budget bytes: `{report.get('qpruner_dense_cache_budget_bytes')}`",
        f"- QPruner dense-cache selection policy: `{report.get('qpruner_dense_cache_selection_policy')}`",
        f"- Peak MB overall: `{fmt(report.get('peak_mem_mb'))}`",
    ]
    if isinstance(qpruner.get("shape_aware_plan"), dict):
        plan = qpruner["shape_aware_plan"]
        lines.extend(
            [
                "",
                "## Shape-aware QPruner policy",
                "",
                f"- Policy: `{plan.get('policy')}`",
                f"- Module count: `{plan.get('module_count')}`",
                f"- Families: `{plan.get('families')}`",
                f"- Strategy counts: `{plan.get('strategy_counts')}`",
                f"- Strategy source counts: `{plan.get('strategy_source_counts')}`",
                f"- Shape policy source: `{plan.get('shape_policy_source')}`",
                f"- Shape policy matches: `{plan.get('shape_policy_match_count')}`",
            ]
        )
    if isinstance(qpruner.get("scaled_code_dtype_cache_budget_plan"), dict):
        plan = qpruner["scaled_code_dtype_cache_budget_plan"]
        lines.extend(
            [
                "",
                "## QPruner scaled-code dtype-cache budget",
                "",
                f"- Budget bytes: `{plan.get('budget_bytes')}`",
                f"- Selection policy: `{plan.get('selection_policy')}`",
                f"- Selected modules: `{plan.get('selected_modules')}`",
                f"- Selected scaled-code dtype-cache bytes: `{plan.get('selected_scaled_code_dtype_cache_bytes')}`",
                f"- Remaining budget bytes: `{plan.get('remaining_budget_bytes')}`",
            ]
        )
    if isinstance(qpruner.get("runtime_profile"), dict):
        profile = qpruner["runtime_profile"]
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
                f"- Strategy timing: `{profile.get('by_strategy')}`",
            ]
        )
        top_modules = profile.get("top_modules", [])
        if isinstance(top_modules, list) and top_modules:
            first = top_modules[0]
            if isinstance(first, dict):
                lines.append(
                    "- Slowest QuantizedLinear: `{name}` strategy `{strategy}`, calls `{calls}`, "
                    "time `{time}` ms".format(
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
    (artifacts / f"compressed_native_torch_serving_{run_label}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    (reports / f"compressed-native-torch-serving-{run_label}.md").write_text(markdown_report(report))


@torch.no_grad()
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


def qpruner_profile_workload(
    *,
    model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    max_new_tokens: int,
) -> Any:
    def workload() -> None:
        run_generate(
            model=model,
            tokenizer=tokenizer,
            prompts=list(DEFAULT_PROMPTS),
            device=device,
            max_new_tokens=max_new_tokens,
            iters=1,
            warmup=0,
        )

    return workload


def cap_result(
    *,
    model: nn.Module,
    model_id: str,
    tokenizer: Any,
    target_names: list[str],
    baseline_targeted_params: int,
    baseline_targeted_storage_bytes: int,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> dict[str, Any]:
    start = time.perf_counter()
    result = cap_compress(
        model=model,
        model_id=model_id,
        budget=args.cap_budget,
        target_roles=None,
        name_filter=qwen_projection_filter,
        device=device,
        dtype=dtype,
        max_iter=args.cap_max_iter,
        policy_steps=args.cap_policy_steps,
        samples_per_step=args.cap_samples_per_step,
        seed=args.seed,
        rpca_backend=args.cap_rpca_backend,
    )
    compression_time_s = time.perf_counter() - start
    cache_modules = 0
    if args.enable_inference_cache:
        cache_modules = enable_model_inference_cache(result.model, dtype=dtype, device=device)
    metrics = benchmark_model(
        model=result.model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=args.max_new_tokens,
        iters=args.iters,
        warmup=args.warmup,
    )
    target_names = [target.name for target in result.method_result.targets]
    compressed_params = targeted_parameter_count(result.model, target_names)
    targeted_storage_bytes = compressed_payload_storage_bytes(result.model, target_names)
    metrics.update(
        {
            "compression_time_s": compression_time_s,
            "cache_modules": cache_modules,
            "exported_dense_linears": 0,
            "targeted_layers": len(target_names),
            "targeted_params_original": baseline_targeted_params,
            "targeted_params_compressed": compressed_params,
            "compressed_payload_storage_bytes": targeted_storage_bytes,
            "targeted_storage_measurement": "compressed_payload_model_storage_bytes",
            "targeted_compression_ratio": round(baseline_targeted_params / compressed_params, 3)
            if compressed_params
            else None,
            "targeted_param_reduction_pct": reduction_pct(baseline_targeted_params, compressed_params),
            "targeted_storage_bytes": targeted_storage_bytes,
            "targeted_storage_reduction_pct": reduction_pct(
                baseline_targeted_storage_bytes,
                targeted_storage_bytes,
            ),
            **cap_runtime_metadata(result.model),
        }
    )
    return metrics


def qpruner_result(
    *,
    model: nn.Module,
    model_id: str,
    tokenizer: Any,
    target_names: list[str],
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
        name_filter=qwen_projection_filter,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )
    compression_time_s = time.perf_counter() - start
    cache_modules = 0
    cache_mode = resolved_qpruner_cache_mode(args)
    shape_policy = None
    release_packed_codes = bool(getattr(args, "qpruner_release_packed_after_cache", False))
    if cache_mode == "dense":
        cache_modules = enable_model_inference_cache(result.model, dtype=dtype, device=device)
    elif cache_mode == "code":
        cache_modules = enable_qpruner_code_cache(
            result.model,
            device=device,
            release_packed_codes=release_packed_codes,
        )
    elif cache_mode == "scaled-code-matmul":
        cache_modules = enable_qpruner_scaled_code_matmul(
            result.model,
            device=device,
            dtype=dtype if bool(getattr(args, "qpruner_prebuild_scaled_code_dtype_cache", False)) else None,
            release_packed_codes=release_packed_codes,
        )
    elif cache_mode == "shape-aware-code":
        shape_policy = load_qpruner_shape_policy(getattr(args, "qpruner_shape_policy_artifact", None))
        cache_modules = enable_qpruner_shape_aware_code_cache(
            result.model,
            device=device,
            dtype=dtype if bool(getattr(args, "qpruner_prebuild_scaled_code_dtype_cache", False)) else None,
            shape_policy=shape_policy,
            release_packed_codes=release_packed_codes,
        )
    aux_cache_modules = 0
    if bool(getattr(args, "qpruner_cache_aux_tensors", False)):
        aux_cache_modules = enable_qpruner_aux_cache(result.model, device=device, dtype=dtype)
    dense_cache_budget_plan = None
    dense_cache_budget_bytes = getattr(args, "qpruner_dense_cache_budget_bytes", 0)
    if dense_cache_budget_bytes:
        dense_cache_budget_plan = enable_qpruner_dense_cache_budget(
            result.model,
            device=device,
            dtype=dtype,
            budget_bytes=int(dense_cache_budget_bytes),
            selection_policy=getattr(
                args,
                "qpruner_dense_cache_selection_policy",
                "smallest_dense_cache_first",
            ),
        )
    scaled_code_dtype_cache_budget_plan = None
    scaled_code_dtype_cache_budget_bytes = getattr(args, "qpruner_scaled_code_dtype_cache_budget_bytes", 0)
    if scaled_code_dtype_cache_budget_bytes:
        scaled_code_dtype_cache_budget_plan = enable_qpruner_scaled_code_dtype_cache_budget(
            result.model,
            device=device,
            dtype=dtype,
            budget_bytes=int(scaled_code_dtype_cache_budget_bytes),
            selection_policy=getattr(
                args,
                "qpruner_scaled_code_dtype_cache_selection_policy",
                "smallest_scaled_code_dtype_cache_first",
            ),
        )
    metrics = benchmark_model(
        model=result.model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=args.max_new_tokens,
        iters=args.iters,
        warmup=args.warmup,
    )
    runtime_profile = None
    if getattr(args, "profile_qpruner_runtime", False):
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
            "cache_mode": cache_mode,
            "release_packed_after_cache": release_packed_codes,
            "cache_aux_tensors": bool(getattr(args, "qpruner_cache_aux_tensors", False)),
            "aux_cache_modules": aux_cache_modules,
            "prebuild_scaled_code_dtype_cache": bool(
                getattr(args, "qpruner_prebuild_scaled_code_dtype_cache", False)
            ),
            "scaled_code_dtype_cache_budget_bytes": int(scaled_code_dtype_cache_budget_bytes or 0),
            "scaled_code_dtype_cache_selection_policy": getattr(
                args,
                "qpruner_scaled_code_dtype_cache_selection_policy",
                "smallest_scaled_code_dtype_cache_first",
            ),
            "scaled_code_dtype_cache_budget_plan": scaled_code_dtype_cache_budget_plan,
            "dense_cache_budget_bytes": int(dense_cache_budget_bytes or 0),
            "dense_cache_selection_policy": getattr(
                args,
                "qpruner_dense_cache_selection_policy",
                "smallest_dense_cache_first",
            ),
            "dense_cache_budget_plan": dense_cache_budget_plan,
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
            **runtime_metadata,
        }
    )
    if runtime_profile is not None:
        metrics["runtime_profile"] = runtime_profile
    return metrics


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    torch_dtype = dtype if isinstance(dtype, torch.dtype) else torch.float32
    model_path = Path(args.model_path)
    tokenizer = load_tokenizer(model_path)

    baseline_model = load_model(model_path, device, torch_dtype)
    target_names = target_linear_names(baseline_model)
    if not target_names:
        raise ValueError("model does not contain matching Qwen projection layers")
    baseline_targeted_params = targeted_parameter_count(baseline_model, target_names)
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
            "targeted_layers": len(target_names),
            "targeted_params": baseline_targeted_params,
            "targeted_storage_bytes": baseline_targeted_storage_bytes,
            "runtime_storage_format": "dense fp16/fp32 weights",
            "runtime_strategy": "dense_linear",
        }
    )
    cap = cap_result(
        model=load_model(model_path, device, torch_dtype),
        model_id=args.model_id,
        tokenizer=tokenizer,
        target_names=target_names,
        baseline_targeted_params=baseline_targeted_params,
        baseline_targeted_storage_bytes=baseline_targeted_storage_bytes,
        device=device,
        dtype=torch_dtype,
        args=args,
    )
    qpruner = qpruner_result(
        model=load_model(model_path, device, torch_dtype),
        model_id=args.model_id,
        tokenizer=tokenizer,
        target_names=target_names,
        baseline_targeted_params=baseline_targeted_params,
        baseline_targeted_storage_bytes=baseline_targeted_storage_bytes,
        device=device,
        dtype=torch_dtype,
        args=args,
    )
    cap["latency_speedup"] = speedup(baseline.get("latency_ms"), cap.get("latency_ms"))
    qpruner["latency_speedup"] = speedup(baseline.get("latency_ms"), qpruner.get("latency_ms"))
    synchronize_device(device)
    exports = {"baseline": baseline, "cap": cap, "qpruner": qpruner}
    summary = summarize(exports)
    summary["dense_export_erases_storage_savings"] = bool(
        cap.get("exported_dense_linears") == 0
        and qpruner.get("exported_dense_linears") == 0
        and not args.export_dense_for_serving
    )
    return {
        "status": "PASS",
        "backend": "torch_generate_compressed_native",
        "model_id": args.model_id,
        "model_path": str(model_path),
        "device": str(device),
        "dtype": str(torch_dtype),
        "download_attempted": False,
        "serving_dense_export": bool(args.export_dense_for_serving),
        "inference_cache_enabled": bool(args.enable_inference_cache),
        "qpruner_cache_mode": resolved_qpruner_cache_mode(args),
        "qpruner_shape_policy_artifact": getattr(args, "qpruner_shape_policy_artifact", None),
        "qpruner_cache_aux_tensors": bool(getattr(args, "qpruner_cache_aux_tensors", False)),
        "qpruner_prebuild_scaled_code_dtype_cache": bool(
            getattr(args, "qpruner_prebuild_scaled_code_dtype_cache", False)
        ),
        "qpruner_scaled_code_dtype_cache_budget_bytes": int(
            getattr(args, "qpruner_scaled_code_dtype_cache_budget_bytes", 0) or 0
        ),
        "qpruner_scaled_code_dtype_cache_selection_policy": getattr(
            args,
            "qpruner_scaled_code_dtype_cache_selection_policy",
            "smallest_scaled_code_dtype_cache_first",
        ),
        "qpruner_dense_cache_budget_bytes": int(getattr(args, "qpruner_dense_cache_budget_bytes", 0) or 0),
        "qpruner_dense_cache_selection_policy": getattr(
            args,
            "qpruner_dense_cache_selection_policy",
            "smallest_dense_cache_first",
        ),
        "max_new_tokens": args.max_new_tokens,
        "baseline": baseline,
        "cap": cap,
        "qpruner": qpruner,
        "summary": summary,
        "memory_reference": {
            "baseline_targeted_params": baseline_targeted_params,
            "targeted_layers": len(target_names),
            "target_layer_coverage_pct": 100.0,
            "cap_targeted_param_reduction_pct": cap.get("targeted_param_reduction_pct"),
            "qpruner_targeted_param_reduction_pct": qpruner.get("targeted_param_reduction_pct"),
            "storage_measurement": "compressed_payload_model_storage_bytes",
            "baseline_targeted_storage_bytes": baseline_targeted_storage_bytes,
            "cap_targeted_storage_bytes": cap.get("targeted_storage_bytes"),
            "qpruner_targeted_storage_bytes": qpruner.get("targeted_storage_bytes"),
            "qpruner_code_cache_storage_bytes": qpruner.get("code_cache_storage_bytes"),
            "qpruner_aux_cache_bytes": qpruner.get("cached_aux_bytes"),
            "qpruner_scaled_code_dtype_cache_bytes": qpruner.get("cached_scaled_code_bytes"),
            "qpruner_dense_cache_bytes": qpruner.get("cached_dense_weight_bytes"),
            "cap_targeted_storage_reduction_pct": cap.get("targeted_storage_reduction_pct"),
            "qpruner_targeted_storage_reduction_pct": qpruner.get("targeted_storage_reduction_pct"),
            "qpruner_code_cache_storage_reduction_pct": qpruner.get("code_cache_storage_reduction_pct"),
        },
        "peak_mem_mb": peak_memory_mb(device),
        "platform": platform.platform(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark compressed-native torch.generate serving reference")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--model-id", default="TinyQwen3-Offline")
    parser.add_argument("--model-path", default="/mnt/nvme/622/models/TinyQwen3-Offline")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--cap-budget", type=int, default=8192)
    parser.add_argument("--cap-max-iter", type=int, default=4)
    parser.add_argument("--cap-policy-steps", type=int, default=1)
    parser.add_argument("--cap-samples-per-step", type=int, default=1)
    parser.add_argument("--cap-rpca-backend", default="numpy")
    parser.add_argument("--qpruner-average-bits", type=float, default=4.0)
    parser.add_argument("--dense-weight-bits", type=float, default=16.0)
    parser.add_argument("--enable-inference-cache", action="store_true")
    parser.add_argument(
        "--qpruner-cache-mode",
        choices=("auto", "none", "dense", "code", "scaled-code-matmul", "shape-aware-code"),
        default="auto",
        help="QPruner runtime cache strategy. auto keeps legacy --enable-inference-cache behavior.",
    )
    parser.add_argument(
        "--qpruner-shape-policy-artifact",
        default=None,
        help="Packed-decode shape-sweep JSON used to choose memory-preserving QPruner runtime per projection shape.",
    )
    parser.add_argument(
        "--profile-qpruner-runtime",
        action="store_true",
        help="Run one extra measured QPruner generate pass and record QuantizedLinear runtime-path profile.",
    )
    parser.add_argument(
        "--qpruner-release-packed-after-cache",
        action="store_true",
        help="After building a QPruner code cache, release packed payload buffers to avoid double-counting live memory.",
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
        choices=QPRUNER_DENSE_CACHE_SELECTION_POLICIES,
        default="smallest_dense_cache_first",
        help="How to spend --qpruner-dense-cache-budget-bytes across QPruner layers.",
    )
    parser.add_argument("--export-dense-for-serving", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-label", default="tiny_qwen3_native_npu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_benchmark(args)
    except Exception as exc:
        report = {
            "status": "FAIL",
            "backend": "torch_generate_compressed_native",
            "model_id": args.model_id,
            "model_path": args.model_path,
            "device": args.device,
            "download_attempted": False,
            "serving_dense_export": bool(args.export_dense_for_serving),
            "inference_cache_enabled": bool(args.enable_inference_cache),
            "qpruner_cache_mode": resolved_qpruner_cache_mode(args),
            "qpruner_shape_policy_artifact": getattr(args, "qpruner_shape_policy_artifact", None),
            "qpruner_release_packed_after_cache": getattr(args, "qpruner_release_packed_after_cache", False),
            "qpruner_cache_aux_tensors": getattr(args, "qpruner_cache_aux_tensors", False),
            "qpruner_prebuild_scaled_code_dtype_cache": getattr(
                args,
                "qpruner_prebuild_scaled_code_dtype_cache",
                False,
            ),
            "qpruner_scaled_code_dtype_cache_budget_bytes": int(
                getattr(args, "qpruner_scaled_code_dtype_cache_budget_bytes", 0) or 0
            ),
            "qpruner_scaled_code_dtype_cache_selection_policy": getattr(
                args,
                "qpruner_scaled_code_dtype_cache_selection_policy",
                "smallest_scaled_code_dtype_cache_first",
            ),
            "qpruner_dense_cache_selection_policy": getattr(
                args,
                "qpruner_dense_cache_selection_policy",
                "smallest_dense_cache_first",
            ),
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "platform": platform.platform(),
        }
    write_outputs(report, Path(args.demo_root), args.run_label)
    print("COMPRESSED_NATIVE_TORCH_SERVING_BENCH " + json.dumps(report, sort_keys=True))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
