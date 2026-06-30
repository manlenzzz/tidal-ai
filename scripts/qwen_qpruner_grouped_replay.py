#!/usr/bin/env python
"""Replay grouped QPruner projection matmul on real Qwen QuantizedLinear modules."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from ascend_inference_benchmark import peak_memory_mb, reset_peak_memory_stats, synchronize_device  # noqa: E402
from compressed_native_torch_serving_benchmark import (  # noqa: E402
    bits_to_storage_bytes,
    compressed_payload_storage_bytes,
    enable_qpruner_aux_cache,
    enable_qpruner_scaled_code_matmul,
    qpruner_runtime_metadata,
    qpruner_runtime_storage_bytes,
    reduction_pct,
)
from qwen_compression_generate_benchmark import limited_target_filter, select_target_names  # noqa: E402
from tiny_qwen_compression_benchmark import load_model, targeted_parameter_count  # noqa: E402
from tidal.device import resolve_device, resolve_dtype  # noqa: E402
from tidal.methods.qpruner.grouped import (  # noqa: E402
    grouped_qpruner_cache_summary,
    grouped_scaled_code_bmm,
    sequential_scaled_code_group,
)
from tidal.methods.qpruner.torch import QuantizedLinear  # noqa: E402
from tidal.workflows.compression import qpruner_compress  # noqa: E402


ROLE_ORDER = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
DEFAULT_NEXT_ACTION = (
    "Move the real-module grouped replay into a fused NPU QuantizedLinear path, then rerun "
    "the full-model Qwen QPruner native profile before claiming end-to-end acceleration."
)


class QuantizedModuleRef:
    def __init__(self, *, name: str, module: QuantizedLinear) -> None:
        self.name = name
        self.module = module


def fmt(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def speedup(*, baseline_ms: float | None, optimized_ms: float | None) -> float | None:
    if baseline_ms is None or optimized_ms is None or baseline_ms <= 0 or optimized_ms <= 0:
        return None
    return round(baseline_ms / optimized_ms, 3)


def projection_role_from_name(name: Any) -> str | None:
    text = str(name or "")
    for role in ROLE_ORDER:
        if text.endswith(role) or f".{role}" in text or f"_{role}" in text:
            return role
    return None


def _device_type_index(value: Any) -> tuple[str, int | None]:
    if isinstance(value, torch.device):
        return value.type, value.index
    text = str(value)
    if ":" not in text:
        return text, None
    device_type, raw_index = text.split(":", 1)
    try:
        return device_type, int(raw_index)
    except ValueError:
        return device_type, None


def device_matches_request(actual: Any, requested: Any) -> bool:
    actual_type, actual_index = _device_type_index(actual)
    requested_type, requested_index = _device_type_index(requested)
    if (actual_type, actual_index) == (requested_type, requested_index):
        return True
    if actual_type != requested_type:
        return False
    return requested_index is None and actual_index in (None, 0)


def collect_groupable_quantized_linears(
    model: nn.Module,
    *,
    require_scaled_code_cache: bool = True,
) -> dict[tuple[str, int, int], list[QuantizedModuleRef]]:
    groups: dict[tuple[str, int, int], list[QuantizedModuleRef]] = {}
    for name, module in model.named_modules():
        if not name or not isinstance(module, QuantizedLinear):
            continue
        role = projection_role_from_name(name)
        if role is None:
            continue
        if require_scaled_code_cache and not module.scaled_code_matmul_enabled:
            continue
        key = (role, int(module.in_features), int(module.out_features))
        groups.setdefault(key, []).append(QuantizedModuleRef(name=name, module=module))
    return groups


def _sorted_group_items(
    groups: dict[tuple[str, int, int], list[QuantizedModuleRef]],
) -> list[tuple[tuple[str, int, int], list[QuantizedModuleRef]]]:
    role_index = {role: index for index, role in enumerate(ROLE_ORDER)}
    return sorted(
        groups.items(),
        key=lambda item: (role_index.get(item[0][0], 999), item[0][1], item[0][2], item[1][0].name),
    )


def make_grouped_inputs(
    *,
    module_count: int,
    batch_size: int,
    in_features: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    inputs = torch.randn(module_count, batch_size, in_features, generator=generator)
    return inputs.to(device=device, dtype=dtype)


@torch.no_grad()
def _measure_sequential(
    modules: Sequence[QuantizedLinear],
    grouped_inputs: torch.Tensor,
    *,
    device: torch.device,
    warmup: int,
    iters: int,
) -> tuple[float, torch.Tensor, float | None]:
    for _ in range(warmup):
        sequential_scaled_code_group(modules, grouped_inputs)
    synchronize_device(device)
    reset_peak_memory_stats(device)
    start = time.perf_counter()
    output = None
    for _ in range(iters):
        output = sequential_scaled_code_group(modules, grouped_inputs)
    synchronize_device(device)
    latency_ms = (time.perf_counter() - start) / iters * 1000.0
    if output is None:
        output = sequential_scaled_code_group(modules, grouped_inputs)
    return round(latency_ms, 6), output.detach(), peak_memory_mb(device)


@torch.no_grad()
def _measure_grouped(
    modules: Sequence[QuantizedLinear],
    grouped_inputs: torch.Tensor,
    *,
    device: torch.device,
    warmup: int,
    iters: int,
) -> tuple[float, torch.Tensor, float | None]:
    for _ in range(warmup):
        grouped_scaled_code_bmm(modules, grouped_inputs)
    synchronize_device(device)
    reset_peak_memory_stats(device)
    start = time.perf_counter()
    output = None
    for _ in range(iters):
        output = grouped_scaled_code_bmm(modules, grouped_inputs)
    synchronize_device(device)
    latency_ms = (time.perf_counter() - start) / iters * 1000.0
    if output is None:
        output = grouped_scaled_code_bmm(modules, grouped_inputs)
    return round(latency_ms, 6), output.detach(), peak_memory_mb(device)


def _enable_replay_caches(
    modules: Sequence[QuantizedLinear],
    *,
    dtype: torch.dtype,
    device: torch.device,
    prebuild_scaled_code_dtype_cache: bool,
) -> None:
    for module in modules:
        module.eval()
        cached_codes = getattr(module, "_cached_weight_codes", None)
        if (
            cached_codes is None
            or not device_matches_request(cached_codes.device, device)
            or not module.scaled_code_matmul_enabled
        ):
            module.enable_scaled_code_matmul(
                device=device,
                dtype=dtype if prebuild_scaled_code_dtype_cache else None,
            )
        elif prebuild_scaled_code_dtype_cache:
            scaled_codes = getattr(module, "_cached_scaled_weight_codes", None)
            if scaled_codes is None or scaled_codes.dtype != dtype or scaled_codes.device != device:
                module._cached_scaled_weight_codes = cached_codes.to(dtype=dtype, device=device).detach()
        module.enable_aux_cache(dtype=dtype, device=device)


def replay_quantized_group(
    *,
    group: Sequence[QuantizedModuleRef],
    role: str,
    in_features: int,
    out_features: int,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    iters: int,
    seed: int,
    max_modules: int,
    prebuild_scaled_code_dtype_cache: bool = False,
) -> dict[str, Any]:
    if iters <= 0:
        raise ValueError("--iters must be positive")
    if warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if max_modules <= 0:
        raise ValueError("--max-modules must be positive")
    selected = list(group[:max_modules])
    if not selected:
        raise ValueError("at least one QuantizedLinear module is required")
    modules = [item.module for item in selected]
    _enable_replay_caches(
        modules,
        dtype=dtype,
        device=device,
        prebuild_scaled_code_dtype_cache=prebuild_scaled_code_dtype_cache,
    )
    grouped_inputs = make_grouped_inputs(
        module_count=len(modules),
        batch_size=batch_size,
        in_features=in_features,
        dtype=dtype,
        device=device,
        seed=seed,
    )
    sequential_ms, sequential_output, sequential_peak = _measure_sequential(
        modules,
        grouped_inputs,
        device=device,
        warmup=warmup,
        iters=iters,
    )
    grouped_ms, grouped_output, grouped_peak = _measure_grouped(
        modules,
        grouped_inputs,
        device=device,
        warmup=warmup,
        iters=iters,
    )
    cache_summary = grouped_qpruner_cache_summary(modules)
    return {
        "role": role,
        "shape": {
            "batch_size": batch_size,
            "in_features": int(in_features),
            "out_features": int(out_features),
        },
        "module_count": len(modules),
        "available_module_count": len(group),
        "sample_module_names": [item.name for item in selected[:8]],
        "bits": sorted({int(module.bits) for module in modules}),
        "sequential_scaled_code_matmul_group": {
            "runtime_strategy": "sequential_scaled_int8_code_matmul_group",
            "latency_ms": sequential_ms,
            "peak_mem_mb": sequential_peak,
            "module_count": len(modules),
            "projection_calls_per_iter": len(modules),
        },
        "grouped_scaled_code_matmul": {
            "runtime_strategy": "grouped_scaled_int8_code_bmm",
            "latency_ms": grouped_ms,
            "peak_mem_mb": grouped_peak,
            "module_count": len(modules),
            "projection_calls_per_iter": len(modules),
            "speedup_vs_sequential_scaled_code": speedup(
                baseline_ms=sequential_ms,
                optimized_ms=grouped_ms,
            ),
            "memory_strategy": "int8_code_cache_plus_aux_tensors",
            **cache_summary,
        },
        "grouped_code_cache_bytes": cache_summary["grouped_code_cache_bytes"],
        "grouped_aux_cache_bytes": cache_summary["grouped_aux_cache_bytes"],
        "grouped_scaled_code_cache_bytes": cache_summary["grouped_scaled_code_cache_bytes"],
        "max_abs_diff_vs_sequential_scaled_code": round(
            float((sequential_output - grouped_output).abs().max().item()),
            8,
        ),
    }


def run_replay_on_model(
    model: nn.Module,
    *,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    iters: int,
    seed: int,
    max_modules: int,
    min_group_modules: int,
    max_groups: int,
    prebuild_scaled_code_dtype_cache: bool = False,
) -> dict[str, Any]:
    groups = collect_groupable_quantized_linears(model)
    candidate_items = [
        (key, group)
        for key, group in _sorted_group_items(groups)
        if len(group) >= min_group_modules
    ]
    if max_groups > 0:
        candidate_items = candidate_items[:max_groups]
    rows = [
        replay_quantized_group(
            group=group,
            role=key[0],
            in_features=key[1],
            out_features=key[2],
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            warmup=warmup,
            iters=iters,
            seed=seed + index,
            max_modules=max_modules,
            prebuild_scaled_code_dtype_cache=prebuild_scaled_code_dtype_cache,
        )
        for index, (key, group) in enumerate(candidate_items)
    ]
    best = max(
        rows,
        key=lambda row: row.get("grouped_scaled_code_matmul", {}).get(
            "speedup_vs_sequential_scaled_code",
            0.0,
        )
        or 0.0,
        default=None,
    )
    return {
        "status": "PASS" if rows else "NO_GROUPS",
        "group_count": len(rows),
        "available_group_count": len(candidate_items),
        "groups": rows,
        "best_group": best,
    }


def target_group_summary(groups: dict[tuple[str, int, int], list[QuantizedModuleRef]]) -> list[dict[str, Any]]:
    return [
        {
            "role": key[0],
            "in_features": key[1],
            "out_features": key[2],
            "module_count": len(group),
            "sample_module_names": [item.name for item in group[:4]],
        }
        for key, group in _sorted_group_items(groups)
    ]


def markdown_report(report: dict[str, Any]) -> str:
    best = report.get("best_group", {}) if isinstance(report.get("best_group"), dict) else {}
    best_grouped = (
        best.get("grouped_scaled_code_matmul", {})
        if isinstance(best.get("grouped_scaled_code_matmul"), dict)
        else {}
    )
    best_seq = (
        best.get("sequential_scaled_code_matmul_group", {})
        if isinstance(best.get("sequential_scaled_code_matmul_group"), dict)
        else {}
    )
    lines = [
        "# Qwen QPruner Grouped Replay",
        "",
        "Real QuantizedLinear-module replay for grouped QPruner projection matmul. This is module-level evidence, not an end-to-end serving speedup claim.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Status | {report.get('status')} |",
        f"| Model | {report.get('model_id')} |",
        f"| Targeted layers | {fmt(report.get('target_layer_limit'))} / {fmt(report.get('targeted_layers_total'))} |",
        f"| Quantized layers | {fmt(report.get('quantized_layers'))} |",
        f"| Group count | {fmt(report.get('group_count'))} |",
        f"| Best role | {fmt(best.get('role'))} |",
        f"| Best grouped speedup | {fmt(best_grouped.get('speedup_vs_sequential_scaled_code'))}x |",
        f"| Best sequential ms | {fmt(best_seq.get('latency_ms'))} |",
        f"| Best grouped ms | {fmt(best_grouped.get('latency_ms'))} |",
        f"| Best code-cache bytes | {fmt(best.get('grouped_code_cache_bytes'))} |",
        f"| Best aux-cache bytes | {fmt(best.get('grouped_aux_cache_bytes'))} |",
        f"| Best scaled-code dtype-cache bytes | {fmt(best.get('grouped_scaled_code_cache_bytes'))} |",
        f"| Best max abs diff | {fmt(best.get('max_abs_diff_vs_sequential_scaled_code'))} |",
        "",
        "## Groups",
        "",
        "| Role | Shape | Modules | Sequential ms | Grouped ms | Speedup | Code bytes | Aux bytes |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("groups", []):
        if not isinstance(row, dict):
            continue
        shape = row.get("shape", {}) if isinstance(row.get("shape"), dict) else {}
        grouped = row.get("grouped_scaled_code_matmul", {})
        seq = row.get("sequential_scaled_code_matmul_group", {})
        lines.append(
            "| {role} | {batch}x{in_f}x{out_f} | {modules} | {seq} | {grouped_ms} | {speedup}x | {codes} | {aux} |".format(
                role=row.get("role"),
                batch=fmt(shape.get("batch_size")),
                in_f=fmt(shape.get("in_features")),
                out_f=fmt(shape.get("out_features")),
                modules=fmt(row.get("module_count")),
                seq=fmt(seq.get("latency_ms") if isinstance(seq, dict) else None),
                grouped_ms=fmt(grouped.get("latency_ms") if isinstance(grouped, dict) else None),
                speedup=fmt(
                    grouped.get("speedup_vs_sequential_scaled_code")
                    if isinstance(grouped, dict)
                    else None
                ),
                codes=fmt(row.get("grouped_code_cache_bytes")),
                aux=fmt(row.get("grouped_aux_cache_bytes")),
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
    (artifacts / f"qwen_qpruner_grouped_replay_{run_label}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    (reports / f"qwen-qpruner-grouped-replay-{run_label}.md").write_text(markdown_report(report))


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.max_modules <= 0:
        raise ValueError("--max-modules must be positive")
    if args.min_group_modules <= 0:
        raise ValueError("--min-group-modules must be positive")
    device = resolve_device(args.device)
    resolved_dtype = resolve_dtype(args.dtype, device)
    dtype = resolved_dtype if isinstance(resolved_dtype, torch.dtype) else torch.float32
    model_path = Path(args.model_path)
    model = load_model(model_path, device, dtype)
    target_names, total_targeted_layers = select_target_names(
        model,
        args.target_layer_limit,
        args.target_layer_pattern,
    )
    if not target_names:
        raise ValueError("model does not contain matching Qwen projection layers")
    baseline_targeted_params = targeted_parameter_count(model, target_names)
    baseline_targeted_storage_bytes = bits_to_storage_bytes(baseline_targeted_params, args.dense_weight_bits)
    start = time.perf_counter()
    result = qpruner_compress(
        model=model,
        model_id=args.model_id,
        importances={name: 1.0 for name in target_names},
        candidate_bits=(2, 4, 8),
        max_average_bits=args.qpruner_average_bits,
        target_roles=None,
        name_filter=limited_target_filter(set(target_names)),
        device=device,
        dtype=dtype,
        seed=args.seed,
        inplace=True,
    )
    compression_time_s = time.perf_counter() - start
    cache_modules = enable_qpruner_scaled_code_matmul(
        result.model,
        device=device,
        dtype=dtype if args.qpruner_prebuild_scaled_code_dtype_cache else None,
        release_packed_codes=bool(args.qpruner_release_packed_after_cache),
    )
    aux_cache_modules = enable_qpruner_aux_cache(result.model, device=device, dtype=dtype)
    target_groups = collect_groupable_quantized_linears(result.model)
    replay = run_replay_on_model(
        result.model,
        batch_size=args.batch_size,
        dtype=dtype,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
        seed=args.seed + 10_000,
        max_modules=args.max_modules,
        min_group_modules=args.min_group_modules,
        max_groups=args.max_groups,
        prebuild_scaled_code_dtype_cache=bool(args.qpruner_prebuild_scaled_code_dtype_cache),
    )
    targeted_live_storage_bytes = compressed_payload_storage_bytes(result.model, target_names)
    targeted_storage_bytes = compressed_payload_storage_bytes(
        result.model,
        target_names,
        include_released_packed_codes=True,
    )
    runtime_metadata = qpruner_runtime_metadata(result.model)
    code_cache_storage_bytes = qpruner_runtime_storage_bytes(targeted_live_storage_bytes, runtime_metadata)
    report = {
        "status": replay["status"],
        "backend": "qwen_qpruner_real_quantizedlinear_grouped_replay",
        "model_id": args.model_id,
        "model_path": str(model_path),
        "device": str(device),
        "dtype": str(dtype),
        "run_label": args.run_label,
        "target_layer_limit": len(target_names),
        "target_layer_requested_limit": args.target_layer_limit,
        "targeted_layers_total": total_targeted_layers,
        "targeted_layer_names_sample": target_names[:8],
        "target_group_summary": target_group_summary(target_groups),
        "quantized_layers": sum(isinstance(module, QuantizedLinear) for module in result.model.modules()),
        "cache_modules": cache_modules,
        "aux_cache_modules": aux_cache_modules,
        "qpruner_average_bits": args.qpruner_average_bits,
        "qpruner_release_packed_after_cache": bool(args.qpruner_release_packed_after_cache),
        "qpruner_prebuild_scaled_code_dtype_cache": bool(args.qpruner_prebuild_scaled_code_dtype_cache),
        "compression_time_s": compression_time_s,
        "batch_size": args.batch_size,
        "iters": args.iters,
        "warmup": args.warmup,
        "max_modules": args.max_modules,
        "min_group_modules": args.min_group_modules,
        "max_groups": args.max_groups,
        "baseline_targeted_params": baseline_targeted_params,
        "baseline_targeted_storage_bytes": baseline_targeted_storage_bytes,
        "targeted_storage_bytes": targeted_storage_bytes,
        "live_compressed_payload_storage_bytes": targeted_live_storage_bytes,
        "code_cache_storage_bytes": code_cache_storage_bytes,
        "targeted_storage_reduction_pct": reduction_pct(
            baseline_targeted_storage_bytes,
            targeted_storage_bytes,
        ),
        "code_cache_storage_reduction_pct": reduction_pct(
            baseline_targeted_storage_bytes,
            code_cache_storage_bytes,
        ),
        **runtime_metadata,
        "group_count": replay["group_count"],
        "available_group_count": replay["available_group_count"],
        "groups": replay["groups"],
        "best_group": replay["best_group"],
        "next_action": DEFAULT_NEXT_ACTION,
        "platform": platform.platform(),
    }
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay real Qwen QPruner grouped QuantizedLinear projections")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-path", default="/mnt/nvme/622/models/Qwen3-0.6B")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-modules", type=int, default=8)
    parser.add_argument("--min-group-modules", type=int, default=2)
    parser.add_argument("--max-groups", type=int, default=0, help="0 means replay every eligible role/shape group.")
    parser.add_argument("--target-layer-limit", type=int, default=0)
    parser.add_argument("--target-layer-pattern")
    parser.add_argument("--qpruner-average-bits", type=float, default=8.0)
    parser.add_argument("--dense-weight-bits", type=float, default=16.0)
    parser.add_argument("--qpruner-release-packed-after-cache", action="store_true")
    parser.add_argument("--qpruner-prebuild-scaled-code-dtype-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-label", default="qwen3_06b_real_module_grouped_replay_npu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_benchmark(args)
    except Exception as exc:
        report = {
            "status": "FAIL",
            "backend": "qwen_qpruner_real_quantizedlinear_grouped_replay",
            "model_id": args.model_id,
            "model_path": args.model_path,
            "device": args.device,
            "dtype": args.dtype,
            "run_label": args.run_label,
            "target_layer_limit": args.target_layer_limit,
            "target_layer_pattern": args.target_layer_pattern,
            "qpruner_release_packed_after_cache": bool(args.qpruner_release_packed_after_cache),
            "qpruner_prebuild_scaled_code_dtype_cache": bool(args.qpruner_prebuild_scaled_code_dtype_cache),
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "platform": platform.platform(),
        }
    write_outputs(report, Path(args.demo_root), args.run_label)
    print("QWEN_QPRUNER_GROUPED_REPLAY " + json.dumps(report, sort_keys=True))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
