#!/usr/bin/env python
"""Qwen-family LLM-Pruner-style structural pruning baseline benchmark."""
from __future__ import annotations

import argparse
from copy import deepcopy
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

from ascend_inference_benchmark import peak_memory_mb, reset_peak_memory_stats, synchronize_device
from qwen_compression_generate_benchmark import select_target_names
from qwen_compression_quality_benchmark import (
    build_eval_batch,
    collect_linear_input_scales,
    evaluate_quality_model,
    fmt,
    get_submodule,
    set_submodule,
)
from tiny_qwen_compression_benchmark import enable_model_inference_cache, load_model, load_tokenizer, targeted_parameter_count
from tidal.device import resolve_device, resolve_dtype


BASELINE_VARIANT = "llm_pruner_style_structured_channel_pruning"
CLAIM_SCOPE = "lightweight Ascend structural-pruning baseline evidence, not official full LLM-Pruner reproduction"


def llm_pruner_structured_prune_linear(
    module: nn.Linear,
    input_scale: torch.Tensor | None,
    prune_ratio: float,
) -> tuple[nn.Linear, dict[str, int]]:
    if prune_ratio < 0.0 or prune_ratio >= 1.0:
        raise ValueError("prune ratio must be in [0, 1)")
    pruned = deepcopy(module)
    weight = module.weight.detach()
    out_features, in_features = weight.shape
    if weight.numel() == 0 or prune_ratio == 0.0:
        return pruned, {"pruned_rows": 0, "pruned_params": 0}
    scale = input_scale
    if scale is None or scale.numel() != in_features:
        scale = torch.ones(in_features, dtype=torch.float32)
    scale = scale.to(device=weight.device, dtype=torch.float32).reshape(1, -1)
    row_score = (weight.detach().to(torch.float32).abs() * scale).mean(dim=1)
    pruned_rows = min(out_features - 1, max(1, int(round(out_features * prune_ratio))))
    prune_indices = torch.topk(row_score, pruned_rows, largest=False).indices
    keep_rows = torch.ones(out_features, dtype=torch.bool, device=weight.device)
    keep_rows[prune_indices] = False
    with torch.no_grad():
        pruned.weight.copy_(weight * keep_rows.reshape(-1, 1).to(dtype=weight.dtype))
        if pruned.bias is not None:
            pruned.bias[~keep_rows] = 0
    bias_params = pruned_rows if module.bias is not None else 0
    return pruned, {"pruned_rows": pruned_rows, "pruned_params": pruned_rows * in_features + bias_params}


def apply_llm_pruner_structured_pruning(
    model: nn.Module,
    *,
    target_names: list[str],
    input_scales: dict[str, torch.Tensor],
    prune_ratio: float,
    inplace: bool,
) -> tuple[nn.Module, dict[str, int]]:
    target = model if inplace else deepcopy(model)
    totals = {"pruned_rows": 0, "pruned_params": 0}
    for name in target_names:
        module = get_submodule(target, name)
        if not isinstance(module, nn.Linear):
            raise ValueError(f"{name} is not a torch.nn.Linear module")
        pruned, stats = llm_pruner_structured_prune_linear(module, input_scales.get(name), prune_ratio)
        set_submodule(target, name, pruned)
        totals["pruned_rows"] += stats["pruned_rows"]
        totals["pruned_params"] += stats["pruned_params"]
    return target, totals


def llm_pruner_quality_result(
    *,
    model: nn.Module,
    baseline_loss: float,
    baseline_latency_ms: float,
    baseline_targeted_params: int,
    target_names: list[str],
    batch: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> dict[str, Any]:
    start = time.perf_counter()
    input_scales = collect_linear_input_scales(model, target_names, batch)
    pruned_model, pruning_stats = apply_llm_pruner_structured_pruning(
        model,
        target_names=target_names,
        input_scales=input_scales,
        prune_ratio=args.prune_ratio,
        inplace=bool(args.inplace_compression),
    )
    compression_time_s = time.perf_counter() - start
    cache_modules = 0
    if args.enable_inference_cache:
        cache_modules = enable_model_inference_cache(pruned_model, dtype=dtype, device=device)
    metrics = evaluate_quality_model(
        pruned_model,
        batch,
        iters=args.iters,
        warmup=args.warmup,
        device=device,
    )
    pruned_params = min(pruning_stats["pruned_params"], baseline_targeted_params)
    retained_params = max(0, baseline_targeted_params - pruned_params)
    metrics.update(
        {
            "compression_time_s": compression_time_s,
            "cache_modules": cache_modules,
            "targeted_layers": len(target_names),
            "structured_prune_ratio": args.prune_ratio,
            "pruned_rows": pruning_stats["pruned_rows"],
            "targeted_params_original": baseline_targeted_params,
            "targeted_params_retained": retained_params,
            "targeted_params_pruned": pruned_params,
            "targeted_param_reduction_pct": (
                round(100.0 * pruned_params / baseline_targeted_params, 3)
                if baseline_targeted_params
                else None
            ),
            "latency_speedup": (
                round(baseline_latency_ms / metrics["latency_ms"], 3) if metrics.get("latency_ms") else None
            ),
            "loss_delta": metrics["loss"] - baseline_loss,
        }
    )
    return metrics


def markdown_report(report: dict[str, Any]) -> str:
    llm_pruner = report.get("llm_pruner", {})
    baseline = report.get("baseline", {})
    lines = [
        "# Qwen LLM-Pruner Baseline",
        "",
        "LLM-Pruner-style structural channel pruning is used here as a memory-first paper-baseline compatibility point.",
        f"Claim scope: {report.get('claim_scope')}.",
        "",
        "| Metric | Baseline | LLM-Pruner-style |",
        "|---|---:|---:|",
        f"| Status | {baseline.get('status')} | {llm_pruner.get('status')} |",
        f"| Loss | {fmt(baseline.get('loss'))} | {fmt(llm_pruner.get('loss'))} |",
        f"| Loss delta | - | {fmt(llm_pruner.get('loss_delta'))} |",
        f"| Latency ms | {fmt(baseline.get('latency_ms'))} | {fmt(llm_pruner.get('latency_ms'))} |",
        f"| Tokens/s | {fmt(baseline.get('tokens_per_s'))} | {fmt(llm_pruner.get('tokens_per_s'))} |",
        f"| Speedup | - | {fmt(llm_pruner.get('latency_speedup'))} |",
        f"| Targeted memory reduction | - | {fmt(llm_pruner.get('targeted_param_reduction_pct'))}% |",
        f"| Structured prune ratio | - | {fmt(llm_pruner.get('structured_prune_ratio'))} |",
        f"| Targeted layers | {fmt(baseline.get('targeted_layers'))} | {fmt(llm_pruner.get('targeted_layers'))} |",
        f"| Compression seconds | - | {fmt(llm_pruner.get('compression_time_s'))} |",
        "",
        "## Metadata",
        "",
        f"- Paper baseline: `{report.get('paper_baseline')}`",
        f"- Baseline variant: `{report.get('baseline_variant')}`",
        f"- Model: `{report.get('model_id')}`",
        f"- Model path: `{report.get('model_path')}`",
        f"- Device: `{report.get('device')}`",
        f"- Dtype: `{report.get('dtype')}`",
        f"- Target layer pattern: `{report.get('target_layer_pattern')}`",
        f"- Target layer limit: `{report.get('target_layer_limit')} / {report.get('targeted_layers_total')}`",
        f"- Target layer sample: `{', '.join(report.get('targeted_layer_names_sample', []))}`",
        f"- Max length: `{report.get('max_length')}`",
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
    (artifacts / f"qwen_llm_pruner_baseline_{run_label}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    (reports / f"qwen-llm-pruner-baseline-{run_label}.md").write_text(markdown_report(report))


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    torch_dtype = dtype if isinstance(dtype, torch.dtype) else torch.float32
    model_path = Path(args.model_path)
    tokenizer = load_tokenizer(model_path)
    batch = build_eval_batch(tokenizer=tokenizer, max_length=args.max_length, device=device)

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
    baseline = evaluate_quality_model(
        baseline_model,
        batch,
        iters=args.iters,
        warmup=args.warmup,
        device=device,
    )
    baseline["targeted_layers"] = len(selected_target_names)
    baseline["targeted_params"] = baseline_targeted_params

    llm_pruner = llm_pruner_quality_result(
        model=load_model(model_path, device, torch_dtype),
        baseline_loss=float(baseline["loss"]),
        baseline_latency_ms=float(baseline["latency_ms"]),
        baseline_targeted_params=baseline_targeted_params,
        target_names=selected_target_names,
        batch=batch,
        device=device,
        dtype=torch_dtype,
        args=args,
    )
    synchronize_device(device)
    return {
        "status": "PASS",
        "backend": "torch_forward",
        "paper_baseline": "LLM-Pruner",
        "baseline_family": "LLM-Pruner",
        "baseline_variant": BASELINE_VARIANT,
        "claim_scope": CLAIM_SCOPE,
        "model_id": args.model_id,
        "model_path": str(model_path),
        "device": str(device),
        "dtype": str(torch_dtype),
        "download_attempted": False,
        "inference_cache_enabled": bool(args.enable_inference_cache),
        "inplace_compression": bool(args.inplace_compression),
        "target_layer_pattern": args.target_layer_pattern,
        "target_layer_limit": args.target_layer_limit,
        "targeted_layers_total": total_targeted_layers,
        "targeted_layer_names_sample": selected_target_names[:8],
        "max_length": args.max_length,
        "baseline": baseline,
        "llm_pruner": llm_pruner,
        "peak_mem_mb": peak_memory_mb(device),
        "platform": platform.platform(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Qwen-family LLM-Pruner-style structural pruning baseline")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-path", default="/mnt/nvme/622/models/Qwen3-0.6B")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--target-layer-limit", type=int, default=2)
    parser.add_argument("--target-layer-pattern")
    parser.add_argument("--prune-ratio", type=float, default=0.25)
    parser.add_argument("--enable-inference-cache", action="store_true")
    parser.add_argument("--inplace-compression", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-label", default="qwen3_06b_llm_pruner_npu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    try:
        report = run_benchmark(args)
    except Exception as exc:
        report = {
            "status": "FAIL",
            "backend": "torch_forward",
            "paper_baseline": "LLM-Pruner",
            "baseline_family": "LLM-Pruner",
            "baseline_variant": BASELINE_VARIANT,
            "claim_scope": CLAIM_SCOPE,
            "model_id": args.model_id,
            "model_path": args.model_path,
            "device": args.device,
            "download_attempted": False,
            "target_layer_limit": args.target_layer_limit,
            "target_layer_pattern": args.target_layer_pattern,
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "platform": platform.platform(),
        }
    write_outputs(report, Path(args.demo_root), args.run_label)
    print("QWEN_LLM_PRUNER_BASELINE " + json.dumps(report, sort_keys=True))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
