#!/usr/bin/env python
"""Qwen-family CAP/QPruner compressed torch.generate benchmark."""
from __future__ import annotations

import argparse
import json
import platform
import re
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

from ascend_inference_benchmark import peak_memory_mb, reset_peak_memory_stats, synchronize_device
from tiny_qwen_compression_benchmark import (
    enable_model_inference_cache,
    load_model,
    load_tokenizer,
    qpruner_average_bits,
    qwen_projection_filter,
    target_linear_names,
    targeted_parameter_count,
)
from tiny_qwen_compression_generate_benchmark import dense_linear_count, run_generate
from tidal.device import resolve_device, resolve_dtype
from tidal.workflows.compression import cap_compress, qpruner_compress
from tidal.workflows.export import export_compressed_linears_to_dense


DEFAULT_PROMPTS = (
    "Explain why low-rank compression can reduce inference cost.",
    "Summarize Ascend NPU synchronized benchmarking in one sentence.",
)


def limited_target_filter(selected_names: set[str]):
    def keep(name: str) -> bool:
        return name in selected_names

    return keep


def select_target_names(model: nn.Module, limit: int | None, pattern: str | None = None) -> tuple[list[str], int]:
    names = target_linear_names(model)
    total_names = len(names)
    if pattern:
        regex = re.compile(pattern)
        names = [name for name in names if regex.search(name)]
    if limit is None or limit <= 0:
        return names, total_names
    return names[:limit], total_names


def build_prompts(*, prompt_repeat: int) -> list[str]:
    repeat = int(prompt_repeat)
    if repeat < 1:
        raise ValueError("prompt_repeat must be >= 1")
    prompts: list[str] = []
    for index in range(repeat):
        suffix = "" if index == 0 else f" [repeat {index + 1}]"
        prompts.extend(f"{prompt}{suffix}" for prompt in DEFAULT_PROMPTS)
    return prompts


def fmt(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Qwen CAP/QPruner Compressed Generate Benchmark",
        "",
        "| Metric | Baseline | CAP | QPruner |",
        "|---|---:|---:|---:|",
        f"| Status | {report.get('baseline', {}).get('status')} | {report.get('cap', {}).get('status')} | {report.get('qpruner', {}).get('status')} |",
        f"| Latency ms | {fmt(report.get('baseline', {}).get('latency_ms'))} | {fmt(report.get('cap', {}).get('latency_ms'))} | {fmt(report.get('qpruner', {}).get('latency_ms'))} |",
        f"| Tokens/s | {fmt(report.get('baseline', {}).get('tokens_per_s'))} | {fmt(report.get('cap', {}).get('tokens_per_s'))} | {fmt(report.get('qpruner', {}).get('tokens_per_s'))} |",
        f"| Speedup | - | {fmt(report.get('cap', {}).get('latency_speedup'))} | {fmt(report.get('qpruner', {}).get('latency_speedup'))} |",
        f"| Compression | - | {fmt(report.get('cap', {}).get('targeted_compression_ratio'))}x | {fmt(report.get('qpruner', {}).get('average_bits'))} avg bits |",
        f"| Targeted layers | {fmt(report.get('baseline', {}).get('targeted_layers'))} | {fmt(report.get('cap', {}).get('targeted_layers'))} | {fmt(report.get('qpruner', {}).get('targeted_layers'))} |",
        f"| Cache modules | - | {fmt(report.get('cap', {}).get('cache_modules'))} | {fmt(report.get('qpruner', {}).get('cache_modules'))} |",
        f"| Serving dense export | {report.get('serving_dense_export')} | {report.get('serving_dense_export')} | {report.get('serving_dense_export')} |",
        f"| Exported dense linears | - | {fmt(report.get('cap', {}).get('exported_dense_linears'))} | {fmt(report.get('qpruner', {}).get('exported_dense_linears'))} |",
        "",
        "## Metadata",
        "",
        f"- Model: `{report.get('model_id')}`",
        f"- Model path: `{report.get('model_path')}`",
        f"- Device: `{report.get('device')}`",
        f"- Dtype: `{report.get('dtype')}`",
        f"- Prompt repeat: `{report.get('prompt_repeat')}`",
        f"- Per-rank prompt count: `{report.get('per_rank_prompt_count')}`",
        f"- Per-rank generated-token target: `{report.get('per_rank_generated_tokens_target')}`",
        f"- Target layer pattern: `{report.get('target_layer_pattern')}`",
        f"- Target layer limit: `{report.get('target_layer_limit')} / {report.get('targeted_layers_total')}`",
        f"- Target layer sample: `{', '.join(report.get('targeted_layer_names_sample', []))}`",
        f"- Peak MB: `{fmt(report.get('peak_mem_mb'))}`",
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
    (artifacts / f"qwen_compression_generate_{run_label}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    (reports / f"qwen-compression-generate-{run_label}.md").write_text(markdown_report(report))


def cap_generate_result(
    *,
    model: nn.Module,
    model_id: str,
    tokenizer: Any,
    baseline_latency_ms: float,
    baseline_targeted_params: int,
    target_names: list[str],
    prompts: list[str],
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
        name_filter=limited_target_filter(set(target_names)),
        device=device,
        dtype=dtype,
        max_iter=args.cap_max_iter,
        policy_steps=args.cap_policy_steps,
        samples_per_step=args.cap_samples_per_step,
        seed=args.seed,
        inplace=bool(args.inplace_compression),
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
        prompts=prompts,
        device=device,
        max_new_tokens=args.max_new_tokens,
        iters=args.iters,
        warmup=args.warmup,
    )
    compressed_params = targeted_parameter_count(result.model, target_names)
    metrics.update(
        {
            "compression_time_s": compression_time_s,
            "cache_modules": cache_modules,
            "exported_dense_linears": exported_dense_linears,
            "targeted_layers": len(target_names),
            "targeted_params_original": baseline_targeted_params,
            "targeted_params_compressed": compressed_params,
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
    target_names: list[str],
    baseline_targeted_params: int,
    prompts: list[str],
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> dict[str, Any]:
    importances = {name: 1.0 for name in target_names}
    start = time.perf_counter()
    result = qpruner_compress(
        model=model,
        model_id=model_id,
        importances=importances,
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
        prompts=prompts,
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
            "targeted_layers": len(target_names),
            "targeted_params_original": baseline_targeted_params,
            "average_bits": qpruner_average_bits(result.model, target_names),
            "latency_speedup": (
                round(baseline_latency_ms / metrics["latency_ms"], 3) if metrics.get("latency_ms") else None
            ),
        }
    )
    return metrics


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    torch_dtype = dtype if isinstance(dtype, torch.dtype) else torch.float32
    model_path = Path(args.model_path)
    tokenizer = load_tokenizer(model_path)
    prompts = build_prompts(prompt_repeat=args.prompt_repeat)
    reset_peak_memory_stats(device)

    baseline_model = load_model(model_path, device, torch_dtype)
    selected_target_names, total_targeted_layers = select_target_names(
        baseline_model,
        args.target_layer_limit,
        args.target_layer_pattern,
    )
    if not selected_target_names:
        raise ValueError("model does not contain matching Qwen projection layers")
    baseline_targeted_params = targeted_parameter_count(baseline_model, selected_target_names)
    baseline = run_generate(
        model=baseline_model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
        max_new_tokens=args.max_new_tokens,
        iters=args.iters,
        warmup=args.warmup,
    )
    baseline["targeted_layers"] = len(selected_target_names)
    baseline["targeted_params"] = baseline_targeted_params

    cap = cap_generate_result(
        model=load_model(model_path, device, torch_dtype),
        model_id=args.model_id,
        tokenizer=tokenizer,
        baseline_latency_ms=float(baseline["latency_ms"]),
        baseline_targeted_params=baseline_targeted_params,
        target_names=selected_target_names,
        prompts=prompts,
        device=device,
        dtype=torch_dtype,
        args=args,
    )
    qpruner = qpruner_generate_result(
        model=load_model(model_path, device, torch_dtype),
        model_id=args.model_id,
        tokenizer=tokenizer,
        baseline_latency_ms=float(baseline["latency_ms"]),
        target_names=selected_target_names,
        baseline_targeted_params=baseline_targeted_params,
        prompts=prompts,
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
        "inplace_compression": bool(args.inplace_compression),
        "target_layer_pattern": args.target_layer_pattern,
        "target_layer_limit": args.target_layer_limit,
        "targeted_layers_total": total_targeted_layers,
        "targeted_layer_names_sample": selected_target_names[:8],
        "max_new_tokens": args.max_new_tokens,
        "prompt_repeat": int(args.prompt_repeat),
        "base_prompt_count": len(DEFAULT_PROMPTS),
        "per_rank_prompt_count": len(prompts),
        "per_rank_generated_tokens_target": len(prompts) * int(args.max_new_tokens),
        "baseline": baseline,
        "cap": cap,
        "qpruner": qpruner,
        "peak_mem_mb": peak_memory_mb(device),
        "platform": platform.platform(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Qwen-family CAP/QPruner compressed torch.generate")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-path", default="/mnt/nvme/622/models/Qwen3-0.6B")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--prompt-repeat", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--target-layer-limit", type=int, default=2)
    parser.add_argument("--target-layer-pattern")
    parser.add_argument("--cap-budget", type=int, default=1048576)
    parser.add_argument("--cap-max-iter", type=int, default=1)
    parser.add_argument("--cap-policy-steps", type=int, default=1)
    parser.add_argument("--cap-samples-per-step", type=int, default=1)
    parser.add_argument("--cap-rpca-backend", default="numpy")
    parser.add_argument("--qpruner-average-bits", type=float, default=4.0)
    parser.add_argument("--enable-inference-cache", action="store_true")
    parser.add_argument("--export-dense-for-serving", action="store_true")
    parser.add_argument("--inplace-compression", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-label", default="qwen3_06b_generate_npu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
            "target_layer_limit": args.target_layer_limit,
            "prompt_repeat": args.prompt_repeat,
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "platform": platform.platform(),
        }
    write_outputs(report, Path(args.demo_root), args.run_label)
    print("QWEN_COMPRESSION_GENERATE_BENCH " + json.dumps(report, sort_keys=True))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
