#!/usr/bin/env python
"""Microbenchmark QPruner packed-code decode overhead."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from ascend_inference_benchmark import peak_memory_mb, reset_peak_memory_stats, synchronize_device
from tidal.device import resolve_device, resolve_dtype
from tidal.methods.qpruner.grouped import (
    grouped_qpruner_cache_summary,
    grouped_scaled_code_bmm,
    sequential_scaled_code_group,
)
from tidal.methods.qpruner.torch import QuantizedLinear


DEFAULT_NEXT_ACTION = (
    "Use the int8 code-cache path as the next memory-native serving step, then fuse "
    "packed/quantized decode kernels on NPU so packed storage avoids host unpack and "
    "dense materialization cost per forward."
)


class ShapeSpec(NamedTuple):
    label: str
    batch_size: int
    in_features: int
    out_features: int


def tensor_storage_bytes(tensor: torch.Tensor | nn.Parameter | None) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def pct_reduction(*, baseline_bytes: int, compressed_bytes: int) -> float | None:
    if baseline_bytes <= 0:
        return None
    return round((1.0 - compressed_bytes / baseline_bytes) * 100.0, 3)


def speedup(*, baseline_ms: float, optimized_ms: float) -> float | None:
    if baseline_ms <= 0 or optimized_ms <= 0:
        return None
    return round(baseline_ms / optimized_ms, 3)


def overhead_pct(*, baseline_ms: float, slower_ms: float) -> float | None:
    if baseline_ms <= 0:
        return None
    return round((slower_ms / baseline_ms - 1.0) * 100.0, 3)


def parse_shape_spec(text: str) -> ShapeSpec:
    label = "shape"
    shape_text = text.strip()
    if ":" in shape_text:
        label, shape_text = shape_text.split(":", 1)
        label = label.strip()
    parts = shape_text.lower().replace(",", "x").split("x")
    if len(parts) != 3:
        raise ValueError(f"shape must be label:batchxin_featuresxout_features, got {text!r}")
    try:
        batch_size, in_features, out_features = (int(part.strip()) for part in parts)
    except ValueError as exc:
        raise ValueError(f"shape dimensions must be integers, got {text!r}") from exc
    if batch_size <= 0 or in_features <= 0 or out_features <= 0:
        raise ValueError(f"shape dimensions must be positive, got {text!r}")
    if not label:
        label = f"{batch_size}x{in_features}x{out_features}"
    return ShapeSpec(label=label, batch_size=batch_size, in_features=in_features, out_features=out_features)


def shape_sweep_specs(*, shape_sweep: str | None, shape_preset: str | None) -> list[ShapeSpec]:
    if shape_sweep:
        return [parse_shape_spec(item) for item in shape_sweep.split(",") if item.strip()]
    if shape_preset in (None, "", "none"):
        return []
    if shape_preset == "qwen3-0.6b":
        return [
            ShapeSpec("qwen3_06b_attn_qkv", 8, 1024, 1024),
            ShapeSpec("qwen3_06b_attn_o", 8, 1024, 1024),
            ShapeSpec("qwen3_06b_mlp_gate_up", 8, 1024, 2816),
            ShapeSpec("qwen3_06b_mlp_down", 8, 2816, 1024),
        ]
    if shape_preset == "qwen3-0.6b-actual":
        return [
            ShapeSpec("qwen3_06b_actual_q_proj", 8, 1024, 2048),
            ShapeSpec("qwen3_06b_actual_k_proj", 8, 1024, 1024),
            ShapeSpec("qwen3_06b_actual_v_proj", 8, 1024, 1024),
            ShapeSpec("qwen3_06b_actual_o_proj", 8, 2048, 1024),
            ShapeSpec("qwen3_06b_actual_gate_proj", 8, 1024, 3072),
            ShapeSpec("qwen3_06b_actual_up_proj", 8, 1024, 3072),
            ShapeSpec("qwen3_06b_actual_down_proj", 8, 3072, 1024),
        ]
    raise ValueError(f"unknown shape preset: {shape_preset}")


def storage_summary(quantized: QuantizedLinear, dense: nn.Linear) -> dict[str, Any]:
    packed_weight_code_bytes = tensor_storage_bytes(quantized.packed_weight_codes)
    scale_bytes = tensor_storage_bytes(quantized.scale)
    quantized_bias_bytes = tensor_storage_bytes(quantized.bias)
    dense_weight_bytes = tensor_storage_bytes(dense.weight)
    dense_bias_bytes = tensor_storage_bytes(dense.bias)
    quantized_payload_bytes = packed_weight_code_bytes + scale_bytes + quantized_bias_bytes
    code_cache_bytes = int(quantized.code_count)
    code_cache_payload_bytes = code_cache_bytes + scale_bytes + quantized_bias_bytes
    dense_payload_bytes = dense_weight_bytes + dense_bias_bytes
    return {
        "runtime_storage_format": "packed_nbit_weight_codes",
        "code_count": int(quantized.code_count),
        "packed_weight_code_bytes": packed_weight_code_bytes,
        "code_cache_bytes": code_cache_bytes,
        "scale_bytes": scale_bytes,
        "quantized_bias_bytes": quantized_bias_bytes,
        "quantized_payload_bytes": quantized_payload_bytes,
        "code_cache_payload_bytes": code_cache_payload_bytes,
        "dense_weight_bytes": dense_weight_bytes,
        "dense_bias_bytes": dense_bias_bytes,
        "dense_payload_bytes": dense_payload_bytes,
        "storage_reduction_pct": pct_reduction(
            baseline_bytes=dense_payload_bytes,
            compressed_bytes=quantized_payload_bytes,
        ),
        "code_cache_storage_reduction_pct": pct_reduction(
            baseline_bytes=dense_payload_bytes,
            compressed_bytes=code_cache_payload_bytes,
        ),
    }


def make_base_linear(*, in_features: int, out_features: int, seed: int) -> nn.Linear:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    linear = nn.Linear(in_features, out_features)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(out_features, in_features, generator=generator) * 0.02)
        if linear.bias is not None:
            linear.bias.copy_(torch.randn(out_features, generator=generator) * 0.02)
    return linear


def make_dense_dequantized(quantized: QuantizedLinear, *, dtype: torch.dtype, device: torch.device) -> nn.Linear:
    dense = nn.Linear(
        quantized.in_features,
        quantized.out_features,
        bias=quantized.bias is not None,
    )
    with torch.no_grad():
        dense.weight.copy_(quantized.dequantized_weight().to(dtype=torch.float32, device="cpu"))
        if dense.bias is not None and quantized.bias is not None:
            dense.bias.copy_(quantized.bias.detach().to(dtype=torch.float32, device="cpu"))
    return dense.to(device=device, dtype=dtype).eval()


def build_layers(
    *,
    in_features: int,
    out_features: int,
    bits: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> tuple[QuantizedLinear, QuantizedLinear, QuantizedLinear, QuantizedLinear, nn.Linear]:
    base = make_base_linear(in_features=in_features, out_features=out_features, seed=seed)
    uncached = QuantizedLinear.from_linear(base, bits=bits).to(device=device).eval()
    code_cached = QuantizedLinear.from_linear(base, bits=bits).to(device=device).eval()
    scaled_code_matmul = QuantizedLinear.from_linear(base, bits=bits).to(device=device).eval()
    cached = QuantizedLinear.from_linear(base, bits=bits).to(device=device).eval()
    dense = make_dense_dequantized(QuantizedLinear.from_linear(base, bits=bits), dtype=dtype, device=device)
    code_cached.enable_code_cache(device=device)
    scaled_code_matmul.enable_scaled_code_matmul(device=device)
    cached.enable_inference_cache(dtype=dtype, device=device)
    return uncached, code_cached, scaled_code_matmul, cached, dense


def make_inputs(*, batch_size: int, in_features: int, dtype: torch.dtype, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    inputs = torch.randn(batch_size, in_features, generator=generator)
    return inputs.to(device=device, dtype=dtype)


@torch.no_grad()
def measure_layer(
    *,
    module: nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
    warmup: int,
    iters: int,
) -> tuple[float, torch.Tensor, float | None]:
    module.eval()
    for _ in range(warmup):
        module(inputs)
    synchronize_device(device)
    reset_peak_memory_stats(device)
    start = time.perf_counter()
    output = None
    for _ in range(iters):
        output = module(inputs)
    synchronize_device(device)
    latency_ms = (time.perf_counter() - start) / iters * 1000.0
    if output is None:
        output = module(inputs)
    return round(latency_ms, 6), output.detach(), peak_memory_mb(device)


def build_grouped_scaled_code_layers(
    *,
    module_count: int,
    in_features: int,
    out_features: int,
    bits: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    batch_size: int,
) -> tuple[list[QuantizedLinear], torch.Tensor]:
    modules: list[QuantizedLinear] = []
    inputs: list[torch.Tensor] = []
    for index in range(module_count):
        base = make_base_linear(in_features=in_features, out_features=out_features, seed=seed + index)
        module = QuantizedLinear.from_linear(base, bits=bits).to(device=device).eval()
        module.enable_scaled_code_matmul(device=device)
        module.enable_aux_cache(dtype=dtype, device=device)
        modules.append(module)
        inputs.append(
            make_inputs(
                batch_size=batch_size,
                in_features=in_features,
                dtype=dtype,
                device=device,
                seed=seed + 1000 + index,
            )
        )
    return modules, torch.stack(inputs, dim=0)


@torch.no_grad()
def measure_sequential_scaled_code_group(
    *,
    modules: list[QuantizedLinear],
    grouped_inputs: torch.Tensor,
    device: torch.device,
    warmup: int,
    iters: int,
) -> tuple[float, torch.Tensor, float | None]:
    for module in modules:
        module.eval()
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
def measure_grouped_scaled_code_bmm(
    *,
    modules: list[QuantizedLinear],
    grouped_inputs: torch.Tensor,
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


def grouped_projection_diagnostic(
    *,
    module_count: int,
    in_features: int,
    out_features: int,
    batch_size: int,
    bits: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    modules, grouped_inputs = build_grouped_scaled_code_layers(
        module_count=module_count,
        in_features=in_features,
        out_features=out_features,
        bits=bits,
        dtype=dtype,
        device=device,
        seed=seed,
        batch_size=batch_size,
    )
    sequential_ms, sequential_out, sequential_peak = measure_sequential_scaled_code_group(
        modules=modules,
        grouped_inputs=grouped_inputs,
        device=device,
        warmup=warmup,
        iters=iters,
    )
    grouped_ms, grouped_out, grouped_peak = measure_grouped_scaled_code_bmm(
        modules=modules,
        grouped_inputs=grouped_inputs,
        device=device,
        warmup=warmup,
        iters=iters,
    )
    cache_summary = grouped_qpruner_cache_summary(modules)
    return {
        "sequential_scaled_code_matmul_group": {
            "runtime_strategy": "sequential_scaled_int8_code_matmul_group",
            "latency_ms": sequential_ms,
            "peak_mem_mb": sequential_peak,
            "module_count": module_count,
            "projection_calls_per_iter": module_count,
        },
        "grouped_scaled_code_matmul": {
            "runtime_strategy": "grouped_scaled_int8_code_bmm",
            "latency_ms": grouped_ms,
            "peak_mem_mb": grouped_peak,
            "module_count": module_count,
            "projection_calls_per_iter": module_count,
            "speedup_vs_sequential_scaled_code": speedup(
                baseline_ms=sequential_ms,
                optimized_ms=grouped_ms,
            ),
            "grouped_code_cache_bytes": cache_summary["grouped_code_cache_bytes"],
            "grouped_aux_cache_bytes": cache_summary["grouped_aux_cache_bytes"],
            "grouped_scaled_code_cache_bytes": cache_summary["grouped_scaled_code_cache_bytes"],
            "memory_strategy": "int8_code_cache_plus_aux_tensors",
        },
        "grouped_max_abs_diff_vs_sequential_scaled_code": round(
            float((sequential_out - grouped_out).abs().max().item()),
            8,
        ),
    }


def path_result(
    *,
    runtime_strategy: str,
    latency_ms: float,
    peak_mem_mb: float | None,
    baseline_latency_ms: float | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "runtime_strategy": runtime_strategy,
        "latency_ms": latency_ms,
        "peak_mem_mb": peak_mem_mb,
    }
    if baseline_latency_ms is not None:
        result["speedup_vs_uncached"] = speedup(
            baseline_ms=baseline_latency_ms,
            optimized_ms=latency_ms,
        )
    return result


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.in_features <= 0 or args.out_features <= 0 or args.batch_size <= 0:
        raise ValueError("feature sizes and batch size must be positive")

    device = resolve_device(args.device)
    resolved_dtype = resolve_dtype(args.dtype, device)
    dtype = resolved_dtype if isinstance(resolved_dtype, torch.dtype) else torch.float32

    uncached, code_cached, scaled_code_matmul, cached, dense = build_layers(
        in_features=args.in_features,
        out_features=args.out_features,
        bits=args.bits,
        dtype=dtype,
        device=device,
        seed=args.seed,
    )
    inputs = make_inputs(
        batch_size=args.batch_size,
        in_features=args.in_features,
        dtype=dtype,
        device=device,
        seed=args.seed + 1,
    )

    uncached_ms, uncached_out, uncached_peak = measure_layer(
        module=uncached,
        inputs=inputs,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    code_cached_ms, code_cached_out, code_cached_peak = measure_layer(
        module=code_cached,
        inputs=inputs,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    scaled_code_ms, scaled_code_out, scaled_code_peak = measure_layer(
        module=scaled_code_matmul,
        inputs=inputs,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    cached_ms, cached_out, cached_peak = measure_layer(
        module=cached,
        inputs=inputs,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    dense_ms, dense_out, dense_peak = measure_layer(
        module=dense,
        inputs=inputs,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )

    storage = storage_summary(uncached, dense)
    report: dict[str, Any] = {
        "status": "PASS",
        "backend": "torch_qpruner_packed_decode_microbenchmark",
        "device": str(device),
        "dtype": str(dtype),
        "bits": args.bits,
        "shape": {
            "batch_size": args.batch_size,
            "in_features": args.in_features,
            "out_features": args.out_features,
        },
        "iters": args.iters,
        "warmup": args.warmup,
        "uncached": path_result(
            runtime_strategy="dequantize_per_forward",
            latency_ms=uncached_ms,
            peak_mem_mb=uncached_peak,
        ),
        "cached": path_result(
            runtime_strategy="dense_weight_cache",
            latency_ms=cached_ms,
            peak_mem_mb=cached_peak,
            baseline_latency_ms=uncached_ms,
        ),
        "code_cached": path_result(
            runtime_strategy="int8_code_cache_dequantize_on_device",
            latency_ms=code_cached_ms,
            peak_mem_mb=code_cached_peak,
            baseline_latency_ms=uncached_ms,
        ),
        "scaled_code_matmul": path_result(
            runtime_strategy="scaled_int8_code_matmul",
            latency_ms=scaled_code_ms,
            peak_mem_mb=scaled_code_peak,
            baseline_latency_ms=uncached_ms,
        ),
        "dense": path_result(
            runtime_strategy="dense_dequantized_linear",
            latency_ms=dense_ms,
            peak_mem_mb=dense_peak,
            baseline_latency_ms=uncached_ms,
        ),
        "decode_overhead_vs_cached_pct": overhead_pct(baseline_ms=cached_ms, slower_ms=uncached_ms),
        "decode_overhead_vs_code_cached_pct": overhead_pct(baseline_ms=code_cached_ms, slower_ms=uncached_ms),
        "decode_overhead_vs_scaled_code_matmul_pct": overhead_pct(baseline_ms=scaled_code_ms, slower_ms=uncached_ms),
        "decode_overhead_vs_dense_pct": overhead_pct(baseline_ms=dense_ms, slower_ms=uncached_ms),
        "max_abs_diff_uncached_code_cached": round(float((uncached_out - code_cached_out).abs().max().item()), 8),
        "max_abs_diff_uncached_scaled_code_matmul": round(
            float((uncached_out - scaled_code_out).abs().max().item()),
            8,
        ),
        "max_abs_diff_uncached_cached": round(float((uncached_out - cached_out).abs().max().item()), 8),
        "max_abs_diff_uncached_dense": round(float((uncached_out - dense_out).abs().max().item()), 8),
        "next_action": DEFAULT_NEXT_ACTION,
    }
    report.update(storage)
    if args.grouped_modules:
        if args.grouped_modules < 2:
            raise ValueError("--grouped-modules must be at least 2 when provided")
        report.update(
            grouped_projection_diagnostic(
                module_count=args.grouped_modules,
                in_features=args.in_features,
                out_features=args.out_features,
                batch_size=args.batch_size,
                bits=args.bits,
                dtype=dtype,
                device=device,
                seed=args.seed + 10_000,
                warmup=args.warmup,
                iters=args.iters,
            )
        )
    return report


def shape_row(label: str, report: dict[str, Any]) -> dict[str, Any]:
    row = {
        "label": label,
        "shape": report.get("shape"),
        "storage_reduction_pct": report.get("storage_reduction_pct"),
        "code_cache_storage_reduction_pct": report.get("code_cache_storage_reduction_pct"),
        "uncached": report.get("uncached"),
        "code_cached": report.get("code_cached"),
        "scaled_code_matmul": report.get("scaled_code_matmul"),
        "cached": report.get("cached"),
        "dense": report.get("dense"),
        "max_abs_diff_uncached_code_cached": report.get("max_abs_diff_uncached_code_cached"),
        "max_abs_diff_uncached_scaled_code_matmul": report.get("max_abs_diff_uncached_scaled_code_matmul"),
    }
    for key in (
        "sequential_scaled_code_matmul_group",
        "grouped_scaled_code_matmul",
        "grouped_max_abs_diff_vs_sequential_scaled_code",
    ):
        if key in report:
            row[key] = report.get(key)
    return row


def _best_path_for_shape(row: dict[str, Any], path_keys: tuple[str, ...]) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for key in path_keys:
        payload = row.get(key, {}) if isinstance(row.get(key), dict) else {}
        latency = payload.get("latency_ms")
        if latency is None:
            continue
        candidates.append(
            {
                "label": row.get("label"),
                "path": key,
                "strategy": payload.get("runtime_strategy"),
                "latency_ms": latency,
                "speedup_vs_uncached": payload.get("speedup_vs_uncached"),
                "storage_reduction_pct": row.get("storage_reduction_pct"),
                "code_cache_storage_reduction_pct": row.get("code_cache_storage_reduction_pct"),
            }
        )
    if not candidates:
        return None
    return min(candidates, key=lambda item: float(item["latency_ms"]))


def sweep_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    memory_candidates = [
        candidate
        for row in rows
        for candidate in [_best_path_for_shape(row, ("code_cached", "scaled_code_matmul"))]
        if candidate is not None
    ]
    latency_candidates = [
        candidate
        for row in rows
        for candidate in [_best_path_for_shape(row, ("uncached", "code_cached", "scaled_code_matmul", "cached", "dense"))]
        if candidate is not None
    ]
    return {
        "shape_count": len(rows),
        "best_memory_preserving": min(memory_candidates, key=lambda item: float(item["latency_ms"]))
        if memory_candidates
        else None,
        "best_latency": min(latency_candidates, key=lambda item: float(item["latency_ms"]))
        if latency_candidates
        else None,
    }


def run_shape_sweep(args: argparse.Namespace, specs: list[ShapeSpec]) -> dict[str, Any]:
    if not specs:
        raise ValueError("shape sweep requested but no shapes were provided")
    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        shape_args = argparse.Namespace(**vars(args))
        shape_args.batch_size = spec.batch_size
        shape_args.in_features = spec.in_features
        shape_args.out_features = spec.out_features
        shape_args.seed = int(args.seed) + index
        rows.append(shape_row(spec.label, run_benchmark(shape_args)))
    return {
        "status": "PASS",
        "backend": "torch_qpruner_packed_decode_shape_sweep",
        "device": str(resolve_device(args.device)),
        "dtype": args.dtype,
        "bits": args.bits,
        "iters": args.iters,
        "warmup": args.warmup,
        "shape_preset": args.shape_preset,
        "shape_sweep": rows,
        "summary": sweep_summary(rows),
        "next_action": DEFAULT_NEXT_ACTION,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# QPruner Packed Decode Benchmark",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Status | {report.get('status', 'missing')} |",
        f"| Device | {report.get('device', 'missing')} |",
        f"| Dtype | {report.get('dtype', 'missing')} |",
        f"| Bits | {report.get('bits', 'missing')} |",
        f"| Runtime storage | {report.get('runtime_storage_format', 'missing')} |",
        f"| Packed code bytes | {report.get('packed_weight_code_bytes', 'missing')} |",
        f"| Int8 code-cache bytes | {report.get('code_cache_bytes', 'missing')} |",
        f"| Quantized payload bytes | {report.get('quantized_payload_bytes', 'missing')} |",
        f"| Code-cache payload bytes | {report.get('code_cache_payload_bytes', 'missing')} |",
        f"| Dense payload bytes | {report.get('dense_payload_bytes', 'missing')} |",
        f"| Storage reduction pct | {report.get('storage_reduction_pct', 'missing')} |",
        f"| Code-cache storage reduction pct | {report.get('code_cache_storage_reduction_pct', 'missing')} |",
        "",
        "| Path | Runtime strategy | Latency ms | Speedup vs uncached | Peak MB |",
        "|---|---|---:|---:|---:|",
    ]
    for key, label in (
        ("uncached", "QPruner uncached"),
        ("code_cached", "QPruner int8 code cache"),
        ("scaled_code_matmul", "QPruner scaled code matmul"),
        ("cached", "QPruner cache"),
        ("dense", "Dense dequantized"),
    ):
        item = report.get(key, {}) if isinstance(report.get(key), dict) else {}
        lines.append(
            "| {label} | {strategy} | {latency} | {speedup} | {peak} |".format(
                label=label,
                strategy=item.get("runtime_strategy", "missing"),
                latency=item.get("latency_ms", "missing"),
                speedup=item.get("speedup_vs_uncached", "baseline" if key == "uncached" else "missing"),
                peak=item.get("peak_mem_mb", "missing"),
            )
        )
    if report.get("decode_overhead_vs_cached_pct") is not None:
        lines.extend(
            [
                "",
                "Decode overhead vs cached dense weight: "
                f"{report['decode_overhead_vs_cached_pct']}%.",
            ]
        )
    if isinstance(report.get("shape_sweep"), list):
        summary = report.get("summary", {}) if isinstance(report.get("summary"), dict) else {}
        best_memory = summary.get("best_memory_preserving") if isinstance(summary.get("best_memory_preserving"), dict) else {}
        best_latency = summary.get("best_latency") if isinstance(summary.get("best_latency"), dict) else {}
        lines.extend(
            [
                "",
                "## Shape Sweep",
                "",
                "Best memory-preserving path: {label} / {strategy} at {latency} ms, speedup {speedup}x.".format(
                    label=best_memory.get("label", "missing"),
                    strategy=best_memory.get("strategy", "missing"),
                    latency=best_memory.get("latency_ms", "missing"),
                    speedup=best_memory.get("speedup_vs_uncached", "missing"),
                ),
                "Best latency path: {label} / {strategy} at {latency} ms.".format(
                    label=best_latency.get("label", "missing"),
                    strategy=best_latency.get("strategy", "missing"),
                    latency=best_latency.get("latency_ms", "missing"),
                ),
                "",
                "| Label | Shape | Code-cache ms | Scaled-code ms | Dense-cache ms | Dense ms | Storage reduction | Code-cache storage |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in report.get("shape_sweep", []):
            if not isinstance(row, dict):
                continue
            shape = row.get("shape", {}) if isinstance(row.get("shape"), dict) else {}
            code_cached = row.get("code_cached", {}) if isinstance(row.get("code_cached"), dict) else {}
            scaled = row.get("scaled_code_matmul", {}) if isinstance(row.get("scaled_code_matmul"), dict) else {}
            cached = row.get("cached", {}) if isinstance(row.get("cached"), dict) else {}
            dense = row.get("dense", {}) if isinstance(row.get("dense"), dict) else {}
            shape_text = "{batch}x{in_features}x{out_features}".format(
                batch=shape.get("batch_size", "missing"),
                in_features=shape.get("in_features", "missing"),
                out_features=shape.get("out_features", "missing"),
            )
            lines.append(
                "| {label} | {shape} | {code} | {scaled} | {cached} | {dense} | {storage}% | {code_storage}% |".format(
                    label=row.get("label", "missing"),
                    shape=shape_text,
                    code=code_cached.get("latency_ms", "missing"),
                    scaled=scaled.get("latency_ms", "missing"),
                    cached=cached.get("latency_ms", "missing"),
                    dense=dense.get("latency_ms", "missing"),
                    storage=row.get("storage_reduction_pct", "missing"),
                    code_storage=row.get("code_cache_storage_reduction_pct", "missing"),
                )
            )
    grouped = report.get("grouped_scaled_code_matmul") if isinstance(report.get("grouped_scaled_code_matmul"), dict) else {}
    sequential = (
        report.get("sequential_scaled_code_matmul_group")
        if isinstance(report.get("sequential_scaled_code_matmul_group"), dict)
        else {}
    )
    if grouped:
        lines.extend(
            [
                "",
                "## Grouped Projection Diagnostic",
                "",
                "| Path | Runtime strategy | Modules | Latency ms | Speedup | Peak MB | Code-cache bytes | Aux-cache bytes |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
                "| Sequential scaled-code group | {strategy} | {modules} | {latency} | baseline | {peak} | missing | missing |".format(
                    strategy=sequential.get("runtime_strategy", "missing"),
                    modules=sequential.get("module_count", "missing"),
                    latency=sequential.get("latency_ms", "missing"),
                    peak=sequential.get("peak_mem_mb", "missing"),
                ),
                "| Grouped scaled-code bmm | {strategy} | {modules} | {latency} | {speedup} | {peak} | {codes} | {aux} |".format(
                    strategy=grouped.get("runtime_strategy", "missing"),
                    modules=grouped.get("module_count", "missing"),
                    latency=grouped.get("latency_ms", "missing"),
                    speedup=grouped.get("speedup_vs_sequential_scaled_code", "missing"),
                    peak=grouped.get("peak_mem_mb", "missing"),
                    codes=grouped.get("grouped_code_cache_bytes", "missing"),
                    aux=grouped.get("grouped_aux_cache_bytes", "missing"),
                ),
                "",
                "Max abs diff vs sequential scaled-code group: "
                f"{report.get('grouped_max_abs_diff_vs_sequential_scaled_code', 'missing')}.",
            ]
        )
    if isinstance(report.get("shape_sweep"), list) and any(
        isinstance(row, dict) and isinstance(row.get("grouped_scaled_code_matmul"), dict)
        for row in report.get("shape_sweep", [])
    ):
        lines.extend(
            [
                "",
                "## Shape Sweep Grouped Projection Diagnostic",
                "",
                "| Label | Modules | Sequential ms | Grouped ms | Grouped speedup | Grouped code-cache bytes | Grouped aux-cache bytes | Max abs diff |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in report.get("shape_sweep", []):
            if not isinstance(row, dict):
                continue
            grouped_row = (
                row.get("grouped_scaled_code_matmul")
                if isinstance(row.get("grouped_scaled_code_matmul"), dict)
                else {}
            )
            sequential_row = (
                row.get("sequential_scaled_code_matmul_group")
                if isinstance(row.get("sequential_scaled_code_matmul_group"), dict)
                else {}
            )
            if not grouped_row:
                continue
            lines.append(
                "| {label} | {modules} | {seq_ms} | {group_ms} | {speedup} | {codes} | {aux} | {diff} |".format(
                    label=row.get("label", "missing"),
                    modules=grouped_row.get("module_count", "missing"),
                    seq_ms=sequential_row.get("latency_ms", "missing"),
                    group_ms=grouped_row.get("latency_ms", "missing"),
                    speedup=grouped_row.get("speedup_vs_sequential_scaled_code", "missing"),
                    codes=grouped_row.get("grouped_code_cache_bytes", "missing"),
                    aux=grouped_row.get("grouped_aux_cache_bytes", "missing"),
                    diff=row.get("grouped_max_abs_diff_vs_sequential_scaled_code", "missing"),
                )
            )
    if report.get("next_action"):
        lines.extend(["", f"Next action: {report['next_action']}"])
    if report.get("error"):
        lines.extend(["", f"Error: `{report.get('error_type')}: {report.get('error')}`"])
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], demo_root: Path, *, run_label: str) -> None:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    suffix = f"_{run_label}" if run_label else ""
    report_suffix = f"-{run_label}" if run_label else ""
    (artifacts / f"qpruner_packed_decode_benchmark{suffix}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    (reports / f"qpruner-packed-decode-benchmark{report_suffix}.md").write_text(markdown_report(report))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QPruner packed-code decode microbenchmark")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--in-features", type=int, default=512)
    parser.add_argument("--out-features", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--shape-sweep",
        default=None,
        help="Comma-separated label:batchxin_featuresxout_features specs, e.g. attn:8x1024x1024,mlp:8x1024x2816.",
    )
    parser.add_argument(
        "--shape-preset",
        choices=["qwen3-0.6b", "qwen3-0.6b-actual"],
        default=None,
        help="Known projection-shape preset for shape-sweep profiling.",
    )
    parser.add_argument(
        "--grouped-modules",
        type=int,
        default=0,
        help="Also benchmark a same-shape grouped scaled-code projection diagnostic with this many modules.",
    )
    parser.add_argument("--run-label", default="tiny_qwen3_qpruner_decode_npu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        specs = shape_sweep_specs(shape_sweep=args.shape_sweep, shape_preset=args.shape_preset)
        report = run_shape_sweep(args, specs) if specs else run_benchmark(args)
    except Exception as exc:
        report = {
            "status": "FAIL",
            "backend": "torch_qpruner_packed_decode_microbenchmark",
            "device": args.device,
            "dtype": args.dtype,
            "bits": args.bits,
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
            "next_action": DEFAULT_NEXT_ACTION,
        }
    report.update({"platform": platform.platform(), "download_attempted": False})
    write_report(report, Path(args.demo_root), run_label=args.run_label)
    print("QPRUNER_PACKED_DECODE_BENCH " + json.dumps(report, sort_keys=True))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
