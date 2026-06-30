#!/usr/bin/env python
"""Offline TinyQwen compressed torch.generate benchmark."""
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

from ascend_inference_benchmark import (
    generation_batch,
    peak_memory_mb,
    reset_peak_memory_stats,
    synchronize_device,
    tokens_per_second,
)
from generation_padding import ensure_left_padding_for_generate
from tiny_qwen_compression_benchmark import (
    enable_model_inference_cache,
    load_model,
    load_tokenizer,
    qpruner_average_bits,
    qwen_projection_filter,
    target_linear_names,
    targeted_parameter_count,
)
from tidal.device import resolve_device, resolve_dtype
from tidal.workflows.export import export_compressed_linears_to_dense
from tidal.workflows.compression import cap_compress, qpruner_compress


DEFAULT_PROMPTS = (
    "hello world",
    "Explain compression",
)


def dense_linear_count(model: nn.Module) -> int:
    return sum(isinstance(module, nn.Linear) for module in model.modules())


@torch.no_grad()
def run_generate(
    *,
    model: nn.Module,
    tokenizer: Any,
    prompts: list[str],
    device: torch.device,
    max_new_tokens: int,
    iters: int,
    warmup: int,
) -> dict[str, Any]:
    tokenizer_padding_side = ensure_left_padding_for_generate(tokenizer)
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    batch = generation_batch(encoded, device)
    kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
    for _ in range(warmup):
        model.generate(**batch, **kwargs)
    synchronize_device(device)
    start = time.perf_counter()
    output = None
    for _ in range(iters):
        output = model.generate(**batch, **kwargs)
    synchronize_device(device)
    latency_ms = (time.perf_counter() - start) / max(1, iters) * 1000.0
    prompt_tokens = int(batch["input_ids"].numel())
    generated_tokens = len(prompts) * max_new_tokens
    if isinstance(output, torch.Tensor):
        generated_tokens = max(0, int(output.numel() - prompt_tokens))
    return {
        "status": "PASS",
        "generated_tokens": generated_tokens,
        "prompt_tokens": prompt_tokens,
        "batch_size": len(prompts),
        "latency_ms": latency_ms,
        "tokens_per_s": tokens_per_second(generated_tokens=generated_tokens, latency_ms=latency_ms),
        "tokenizer_padding_side": tokenizer_padding_side,
    }


def cap_generate_result(
    *,
    model: nn.Module,
    model_id: str,
    tokenizer: Any,
    baseline_latency_ms: float,
    baseline_targeted_params: int,
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
    exported_dense_linears = None
    generate_model = result.model
    if args.export_dense_for_serving:
        before_dense = dense_linear_count(result.model)
        generate_model = export_compressed_linears_to_dense(result.model, inplace=False, dtype=dtype, device=device)
        exported_dense_linears = dense_linear_count(generate_model) - before_dense
    metrics = run_generate(
        model=generate_model,
        tokenizer=tokenizer,
        prompts=list(DEFAULT_PROMPTS),
        device=device,
        max_new_tokens=args.max_new_tokens,
        iters=args.iters,
        warmup=args.warmup,
    )
    target_names = [target.name for target in result.method_result.targets]
    compressed_params = targeted_parameter_count(result.model, target_names)
    metrics.update(
        {
            "compression_time_s": compression_time_s,
            "cache_modules": cache_modules,
            "exported_dense_linears": exported_dense_linears,
            "targeted_layers": len(target_names),
            "targeted_compression_ratio": (
                round(baseline_targeted_params / compressed_params, 3) if compressed_params else None
            ),
            "latency_speedup": (
                round(baseline_latency_ms / metrics["latency_ms"], 3) if metrics.get("latency_ms") else None
            ),
        }
    )
    return metrics


def qpruner_generate_result(
    *,
    model: nn.Module,
    model_id: str,
    tokenizer: Any,
    baseline_latency_ms: float,
    baseline_target_names: list[str],
    baseline_targeted_params: int,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> dict[str, Any]:
    importances = {name: 1.0 for name in baseline_target_names}
    start = time.perf_counter()
    result = qpruner_compress(
        model=model,
        model_id=model_id,
        importances=importances,
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
    if args.enable_inference_cache:
        cache_modules = enable_model_inference_cache(result.model, dtype=dtype, device=device)
    exported_dense_linears = None
    generate_model = result.model
    if args.export_dense_for_serving:
        before_dense = dense_linear_count(result.model)
        generate_model = export_compressed_linears_to_dense(result.model, inplace=False, dtype=dtype, device=device)
        exported_dense_linears = dense_linear_count(generate_model) - before_dense
    metrics = run_generate(
        model=generate_model,
        tokenizer=tokenizer,
        prompts=list(DEFAULT_PROMPTS),
        device=device,
        max_new_tokens=args.max_new_tokens,
        iters=args.iters,
        warmup=args.warmup,
    )
    metrics.update(
        {
            "compression_time_s": compression_time_s,
            "cache_modules": cache_modules,
            "exported_dense_linears": exported_dense_linears,
            "targeted_layers": len(baseline_target_names),
            "targeted_params_original": baseline_targeted_params,
            "average_bits": qpruner_average_bits(result.model, baseline_target_names),
            "latency_speedup": (
                round(baseline_latency_ms / metrics["latency_ms"], 3) if metrics.get("latency_ms") else None
            ),
        }
    )
    return metrics


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# TinyQwen Compressed Generate Benchmark",
        "",
        "| Metric | Baseline | CAP | QPruner |",
        "|---|---:|---:|---:|",
        f"| Status | {report.get('baseline', {}).get('status')} | {report.get('cap', {}).get('status')} | {report.get('qpruner', {}).get('status')} |",
        f"| Latency ms | {report.get('baseline', {}).get('latency_ms')} | {report.get('cap', {}).get('latency_ms')} | {report.get('qpruner', {}).get('latency_ms')} |",
        f"| Tokens/s | {report.get('baseline', {}).get('tokens_per_s')} | {report.get('cap', {}).get('tokens_per_s')} | {report.get('qpruner', {}).get('tokens_per_s')} |",
        f"| Speedup | - | {report.get('cap', {}).get('latency_speedup')} | {report.get('qpruner', {}).get('latency_speedup')} |",
        f"| Compression | - | {report.get('cap', {}).get('targeted_compression_ratio')}x | {report.get('qpruner', {}).get('average_bits')} avg bits |",
        f"| Cache modules | - | {report.get('cap', {}).get('cache_modules')} | {report.get('qpruner', {}).get('cache_modules')} |",
        f"| Serving dense export | {report.get('serving_dense_export')} | {report.get('serving_dense_export')} | {report.get('serving_dense_export')} |",
        f"| Exported dense linears | - | {report.get('cap', {}).get('exported_dense_linears')} | {report.get('qpruner', {}).get('exported_dense_linears')} |",
        "",
        "## Metadata",
        "",
        f"- Model: `{report.get('model_id')}`",
        f"- Model path: `{report.get('model_path')}`",
        f"- Device: `{report.get('device')}`",
        f"- Dtype: `{report.get('dtype')}`",
        f"- Peak MB: `{report.get('peak_mem_mb')}`",
        f"- Download attempted: `{report.get('download_attempted')}`",
    ]
    if report.get("error"):
        lines.extend(["", f"Error: `{report.get('error_type')}: {report.get('error')}`"])
    return "\n".join(lines) + "\n"


def write_outputs(report: dict[str, Any], demo_root: Path, run_label: str) -> None:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    (artifacts / f"tiny_qwen_compression_generate_{run_label}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    (reports / f"tiny-qwen-compression-generate-{run_label}.md").write_text(markdown_report(report))


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    torch_dtype = dtype if isinstance(dtype, torch.dtype) else torch.float32
    model_path = Path(args.model_path)
    tokenizer = load_tokenizer(model_path)
    reset_peak_memory_stats(device)

    baseline_model = load_model(model_path, device, torch_dtype)
    baseline_target_names = target_linear_names(baseline_model)
    baseline_targeted_params = targeted_parameter_count(baseline_model, baseline_target_names)
    baseline = run_generate(
        model=baseline_model,
        tokenizer=tokenizer,
        prompts=list(DEFAULT_PROMPTS),
        device=device,
        max_new_tokens=args.max_new_tokens,
        iters=args.iters,
        warmup=args.warmup,
    )
    baseline["targeted_layers"] = len(baseline_target_names)
    baseline["targeted_params"] = baseline_targeted_params

    cap = cap_generate_result(
        model=load_model(model_path, device, torch_dtype),
        model_id=args.model_id,
        tokenizer=tokenizer,
        baseline_latency_ms=float(baseline["latency_ms"]),
        baseline_targeted_params=baseline_targeted_params,
        device=device,
        dtype=torch_dtype,
        args=args,
    )
    qpruner = qpruner_generate_result(
        model=load_model(model_path, device, torch_dtype),
        model_id=args.model_id,
        tokenizer=tokenizer,
        baseline_latency_ms=float(baseline["latency_ms"]),
        baseline_target_names=baseline_target_names,
        baseline_targeted_params=baseline_targeted_params,
        device=device,
        dtype=torch_dtype,
        args=args,
    )
    synchronize_device(device)

    return {
        "status": "PASS",
        "backend": "torch_generate",
        "model_id": args.model_id,
        "model_path": str(model_path),
        "device": str(device),
        "dtype": str(torch_dtype),
        "download_attempted": False,
        "inference_cache_enabled": bool(args.enable_inference_cache),
        "serving_dense_export": bool(args.export_dense_for_serving),
        "max_new_tokens": args.max_new_tokens,
        "baseline": baseline,
        "cap": cap,
        "qpruner": qpruner,
        "peak_mem_mb": peak_memory_mb(device),
        "platform": platform.platform(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark TinyQwen compressed torch.generate")
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
    parser.add_argument("--enable-inference-cache", action="store_true")
    parser.add_argument("--export-dense-for-serving", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-label", default="tiny_qwen3_generate_npu")
    args = parser.parse_args()

    try:
        report = run_benchmark(args)
    except Exception as exc:
        report = {
            "status": "FAIL",
            "backend": "torch_generate",
            "model_id": args.model_id,
            "model_path": args.model_path,
            "device": args.device,
            "download_attempted": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
            "platform": platform.platform(),
        }
    write_outputs(report, Path(args.demo_root), args.run_label)
    print("TINY_QWEN_COMPRESSION_GENERATE_BENCH " + json.dumps(report, sort_keys=True))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
