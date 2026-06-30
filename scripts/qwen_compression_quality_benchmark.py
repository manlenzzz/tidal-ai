#!/usr/bin/env python
"""Qwen-family CAP/QPruner compression quality and forward benchmark."""
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
from qwen_compression_generate_benchmark import limited_target_filter, select_target_names
from tiny_qwen_compression_benchmark import (
    enable_model_inference_cache,
    evaluate_loss,
    forward_latency_ms,
    load_model,
    load_tokenizer,
    qpruner_average_bits,
    targeted_parameter_count,
)
from tidal.device import resolve_device, resolve_dtype
from tidal.workflows.compression import cap_compress, qpruner_compress


DEFAULT_EVAL_SAMPLES = (
    {
        "prompt": "Explain CAP compression in one sentence.",
        "target": "CAP uses compact low-rank factors to reduce selected dense projection costs.",
    },
    {
        "prompt": "Why measure loss after compression?",
        "target": "Held-out loss checks whether compressed weights preserve model quality.",
    },
)


def set_submodule(root: nn.Module, name: str, module: nn.Module) -> None:
    parent = root
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def get_submodule(root: nn.Module, name: str) -> nn.Module:
    module: nn.Module = root
    for part in name.split("."):
        module = getattr(module, part)
    return module


def fmt(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def sample_text(sample: dict[str, str]) -> str:
    return f"{sample['prompt']} {sample['target']}"


def build_eval_batch(
    *,
    tokenizer: Any,
    max_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        [sample_text(sample) for sample in DEFAULT_EVAL_SAMPLES],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    if not torch.any(labels != -100):
        labels = input_ids.clone()
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
    }


def eval_token_count(batch: dict[str, torch.Tensor]) -> int:
    return int(batch["attention_mask"].detach().sum().cpu().item())


def tokens_per_second(*, tokens: int, latency_ms: float) -> float | None:
    if latency_ms <= 0:
        return None
    return round(tokens / (latency_ms / 1000.0), 3)


def evaluate_quality_model(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    iters: int,
    warmup: int,
    device: torch.device,
) -> dict[str, Any]:
    tokens = eval_token_count(batch)
    loss = evaluate_loss(model, batch)
    latency_ms = forward_latency_ms(model, batch, iters=iters, warmup=warmup, device=device)
    return {
        "status": "PASS",
        "loss": loss,
        "latency_ms": latency_ms,
        "tokens_per_s": tokens_per_second(tokens=tokens, latency_ms=latency_ms),
        "eval_tokens": tokens,
    }


def collect_linear_input_stats(
    model: nn.Module,
    target_names: list[str],
    batch: dict[str, torch.Tensor],
) -> dict[str, dict[str, torch.Tensor]]:
    modules = dict(model.named_modules())
    stats: dict[str, dict[str, torch.Tensor]] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
            if not inputs or not torch.is_tensor(inputs[0]):
                return
            tensor = inputs[0].detach().to(torch.float32)
            if tensor.ndim == 1:
                tensor = tensor.reshape(1, -1)
            tensor = tensor.reshape(-1, tensor.shape[-1])
            tensor = tensor.cpu()
            stats[name] = {
                "abs_mean": tensor.abs().mean(dim=0),
                "hessian_diag": tensor.pow(2).mean(dim=0).clamp_min(1e-8),
            }

        return hook

    for name in target_names:
        module = modules.get(name)
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(make_hook(name)))
    with torch.no_grad():
        model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
    for handle in handles:
        handle.remove()
    return stats


def collect_linear_input_scales(
    model: nn.Module,
    target_names: list[str],
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {name: payload["abs_mean"] for name, payload in collect_linear_input_stats(model, target_names, batch).items()}


def wanda_prune_linear(module: nn.Linear, input_scale: torch.Tensor | None, sparsity: float) -> nn.Linear:
    if sparsity < 0.0 or sparsity >= 1.0:
        raise ValueError("wanda sparsity must be in [0, 1)")
    pruned = deepcopy(module)
    weight = module.weight.detach()
    if weight.numel() == 0 or sparsity == 0.0:
        return pruned
    scale = input_scale
    if scale is None or scale.numel() != weight.shape[1]:
        scale = torch.ones(weight.shape[1], dtype=torch.float32)
    scale = scale.to(device=weight.device, dtype=torch.float32).reshape(1, -1)
    score = weight.detach().to(torch.float32).abs() * scale
    keep_count = max(1, int(round(score.numel() * (1.0 - sparsity))))
    keep_count = min(keep_count, score.numel())
    flat_score = score.reshape(-1)
    threshold = torch.topk(flat_score, keep_count, largest=True).values[-1]
    keep_mask = score >= threshold
    if int(keep_mask.sum().item()) > keep_count:
        keep_indices = torch.topk(flat_score, keep_count, largest=True).indices
        keep_mask = torch.zeros_like(flat_score, dtype=torch.bool)
        keep_mask[keep_indices] = True
        keep_mask = keep_mask.reshape_as(score)
    with torch.no_grad():
        pruned.weight.copy_(weight * keep_mask.to(dtype=weight.dtype))
    return pruned


def apply_wanda_pruning(
    model: nn.Module,
    *,
    target_names: list[str],
    input_scales: dict[str, torch.Tensor],
    sparsity: float,
    inplace: bool,
) -> nn.Module:
    target = model if inplace else deepcopy(model)
    for name in target_names:
        module = get_submodule(target, name)
        if not isinstance(module, nn.Linear):
            raise ValueError(f"{name} is not a torch.nn.Linear module")
        set_submodule(target, name, wanda_prune_linear(module, input_scales.get(name), sparsity))
    return target


def sparsegpt_prune_linear(module: nn.Linear, hessian_diag: torch.Tensor | None, sparsity: float) -> nn.Linear:
    if sparsity < 0.0 or sparsity >= 1.0:
        raise ValueError("SparseGPT sparsity must be in [0, 1)")
    pruned = deepcopy(module)
    weight = module.weight.detach()
    if weight.numel() == 0 or sparsity == 0.0:
        return pruned
    diag = hessian_diag
    if diag is None or diag.numel() != weight.shape[1]:
        diag = torch.ones(weight.shape[1], dtype=torch.float32)
    diag = diag.to(device=weight.device, dtype=torch.float32).reshape(1, -1).clamp_min(1e-8)
    score = weight.detach().to(torch.float32).pow(2) / diag
    keep_count = max(1, int(round(score.numel() * (1.0 - sparsity))))
    keep_count = min(keep_count, score.numel())
    flat_score = score.reshape(-1)
    keep_indices = torch.topk(flat_score, keep_count, largest=True).indices
    keep_mask = torch.zeros_like(flat_score, dtype=torch.bool)
    keep_mask[keep_indices] = True
    keep_mask = keep_mask.reshape_as(score)
    with torch.no_grad():
        pruned.weight.copy_(weight * keep_mask.to(dtype=weight.dtype))
    return pruned


def apply_sparsegpt_pruning(
    model: nn.Module,
    *,
    target_names: list[str],
    input_stats: dict[str, dict[str, torch.Tensor]],
    sparsity: float,
    inplace: bool,
) -> nn.Module:
    target = model if inplace else deepcopy(model)
    for name in target_names:
        module = get_submodule(target, name)
        if not isinstance(module, nn.Linear):
            raise ValueError(f"{name} is not a torch.nn.Linear module")
        stats = input_stats.get(name, {})
        hessian_diag = stats.get("hessian_diag") if isinstance(stats, dict) else None
        set_submodule(target, name, sparsegpt_prune_linear(module, hessian_diag, sparsity))
    return target


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Qwen CAP/QPruner Compression Quality Benchmark",
        "",
        "| Metric | Baseline | CAP | WANDA | SparseGPT | QPruner |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Status | {report.get('baseline', {}).get('status')} | {report.get('cap', {}).get('status')} | {report.get('wanda', {}).get('status')} | {report.get('sparsegpt', {}).get('status')} | {report.get('qpruner', {}).get('status')} |",
        f"| Loss | {fmt(report.get('baseline', {}).get('loss'))} | {fmt(report.get('cap', {}).get('loss'))} | {fmt(report.get('wanda', {}).get('loss'))} | {fmt(report.get('sparsegpt', {}).get('loss'))} | {fmt(report.get('qpruner', {}).get('loss'))} |",
        f"| Loss delta | - | {fmt(report.get('cap', {}).get('loss_delta'))} | {fmt(report.get('wanda', {}).get('loss_delta'))} | {fmt(report.get('sparsegpt', {}).get('loss_delta'))} | {fmt(report.get('qpruner', {}).get('loss_delta'))} |",
        f"| Latency ms | {fmt(report.get('baseline', {}).get('latency_ms'))} | {fmt(report.get('cap', {}).get('latency_ms'))} | {fmt(report.get('wanda', {}).get('latency_ms'))} | {fmt(report.get('sparsegpt', {}).get('latency_ms'))} | {fmt(report.get('qpruner', {}).get('latency_ms'))} |",
        f"| Tokens/s | {fmt(report.get('baseline', {}).get('tokens_per_s'))} | {fmt(report.get('cap', {}).get('tokens_per_s'))} | {fmt(report.get('wanda', {}).get('tokens_per_s'))} | {fmt(report.get('sparsegpt', {}).get('tokens_per_s'))} | {fmt(report.get('qpruner', {}).get('tokens_per_s'))} |",
        f"| Speedup | - | {fmt(report.get('cap', {}).get('latency_speedup'))} | {fmt(report.get('wanda', {}).get('latency_speedup'))} | {fmt(report.get('sparsegpt', {}).get('latency_speedup'))} | {fmt(report.get('qpruner', {}).get('latency_speedup'))} |",
        f"| Compression | - | {fmt(report.get('cap', {}).get('targeted_compression_ratio'))}x | {fmt(report.get('wanda', {}).get('targeted_param_reduction_pct'))}% sparse | {fmt(report.get('sparsegpt', {}).get('targeted_param_reduction_pct'))}% sparse | {fmt(report.get('qpruner', {}).get('average_bits'))} avg bits |",
        f"| Targeted layers | {fmt(report.get('baseline', {}).get('targeted_layers'))} | {fmt(report.get('cap', {}).get('targeted_layers'))} | {fmt(report.get('wanda', {}).get('targeted_layers'))} | {fmt(report.get('sparsegpt', {}).get('targeted_layers'))} | {fmt(report.get('qpruner', {}).get('targeted_layers'))} |",
        f"| Compression seconds | - | {fmt(report.get('cap', {}).get('compression_time_s'))} | {fmt(report.get('wanda', {}).get('compression_time_s'))} | {fmt(report.get('sparsegpt', {}).get('compression_time_s'))} | {fmt(report.get('qpruner', {}).get('compression_time_s'))} |",
        f"| Cache modules | - | {fmt(report.get('cap', {}).get('cache_modules'))} | {fmt(report.get('wanda', {}).get('cache_modules'))} | {fmt(report.get('sparsegpt', {}).get('cache_modules'))} | {fmt(report.get('qpruner', {}).get('cache_modules'))} |",
        "",
        "## Metadata",
        "",
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
    (artifacts / f"qwen_compression_quality_{run_label}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    (reports / f"qwen-compression-quality-{run_label}.md").write_text(markdown_report(report))


def cap_quality_result(
    *,
    model: nn.Module,
    model_id: str,
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
    metrics = evaluate_quality_model(
        result.model,
        batch,
        iters=args.iters,
        warmup=args.warmup,
        device=device,
    )
    compressed_params = targeted_parameter_count(result.model, target_names)
    metrics.update(
        {
            "compression_time_s": compression_time_s,
            "cache_modules": cache_modules,
            "targeted_layers": len(target_names),
            "targeted_params_original": baseline_targeted_params,
            "targeted_params_compressed": compressed_params,
            "targeted_compression_ratio": (
                round(baseline_targeted_params / compressed_params, 3) if compressed_params else None
            ),
            "latency_speedup": (
                round(baseline_latency_ms / metrics["latency_ms"], 3) if metrics.get("latency_ms") else None
            ),
            "loss_delta": metrics["loss"] - baseline_loss,
        }
    )
    return metrics


def wanda_quality_result(
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
    pruned_model = apply_wanda_pruning(
        model,
        target_names=target_names,
        input_scales=input_scales,
        sparsity=args.wanda_sparsity,
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
    metrics.update(
        {
            "compression_time_s": compression_time_s,
            "cache_modules": cache_modules,
            "targeted_layers": len(target_names),
            "targeted_params_original": baseline_targeted_params,
            "targeted_params_retained": int(round(baseline_targeted_params * (1.0 - args.wanda_sparsity))),
            "targeted_sparsity": args.wanda_sparsity,
            "targeted_param_reduction_pct": round(args.wanda_sparsity * 100.0, 3),
            "latency_speedup": (
                round(baseline_latency_ms / metrics["latency_ms"], 3) if metrics.get("latency_ms") else None
            ),
            "loss_delta": metrics["loss"] - baseline_loss,
        }
    )
    return metrics


def sparsegpt_quality_result(
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
    input_stats = collect_linear_input_stats(model, target_names, batch)
    pruned_model = apply_sparsegpt_pruning(
        model,
        target_names=target_names,
        input_stats=input_stats,
        sparsity=args.sparsegpt_sparsity,
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
    metrics.update(
        {
            "compression_time_s": compression_time_s,
            "cache_modules": cache_modules,
            "targeted_layers": len(target_names),
            "targeted_params_original": baseline_targeted_params,
            "targeted_params_retained": int(round(baseline_targeted_params * (1.0 - args.sparsegpt_sparsity))),
            "targeted_sparsity": args.sparsegpt_sparsity,
            "targeted_param_reduction_pct": round(args.sparsegpt_sparsity * 100.0, 3),
            "calibration": "activation_hessian_diag",
            "latency_speedup": (
                round(baseline_latency_ms / metrics["latency_ms"], 3) if metrics.get("latency_ms") else None
            ),
            "loss_delta": metrics["loss"] - baseline_loss,
        }
    )
    return metrics


def qpruner_quality_result(
    *,
    model: nn.Module,
    model_id: str,
    baseline_loss: float,
    baseline_latency_ms: float,
    baseline_targeted_params: int,
    target_names: list[str],
    batch: dict[str, torch.Tensor],
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
    metrics = evaluate_quality_model(
        result.model,
        batch,
        iters=args.iters,
        warmup=args.warmup,
        device=device,
    )
    metrics.update(
        {
            "compression_time_s": compression_time_s,
            "cache_modules": cache_modules,
            "targeted_layers": len(target_names),
            "targeted_params_original": baseline_targeted_params,
            "targeted_params_quantized": targeted_parameter_count(result.model, target_names),
            "average_bits": qpruner_average_bits(result.model, target_names),
            "memory_bits": result.summary.get("memory_bits"),
            "bitwidths": result.summary.get("bitwidths", {}),
            "latency_speedup": (
                round(baseline_latency_ms / metrics["latency_ms"], 3) if metrics.get("latency_ms") else None
            ),
            "loss_delta": metrics["loss"] - baseline_loss,
        }
    )
    return metrics


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

    cap = cap_quality_result(
        model=load_model(model_path, device, torch_dtype),
        model_id=args.model_id,
        baseline_loss=float(baseline["loss"]),
        baseline_latency_ms=float(baseline["latency_ms"]),
        baseline_targeted_params=baseline_targeted_params,
        target_names=selected_target_names,
        batch=batch,
        device=device,
        dtype=torch_dtype,
        args=args,
    )
    wanda = wanda_quality_result(
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
    sparsegpt = sparsegpt_quality_result(
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
    qpruner = qpruner_quality_result(
        model=load_model(model_path, device, torch_dtype),
        model_id=args.model_id,
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
        "cap": cap,
        "wanda": wanda,
        "sparsegpt": sparsegpt,
        "qpruner": qpruner,
        "peak_mem_mb": peak_memory_mb(device),
        "platform": platform.platform(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Qwen-family CAP/QPruner compression quality")
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
    parser.add_argument("--cap-budget", type=int, default=1048576)
    parser.add_argument("--cap-max-iter", type=int, default=1)
    parser.add_argument("--cap-policy-steps", type=int, default=1)
    parser.add_argument("--cap-samples-per-step", type=int, default=1)
    parser.add_argument("--cap-rpca-backend", default="numpy")
    parser.add_argument("--wanda-sparsity", type=float, default=0.5)
    parser.add_argument("--sparsegpt-sparsity", type=float, default=0.5)
    parser.add_argument("--qpruner-average-bits", type=float, default=4.0)
    parser.add_argument("--enable-inference-cache", action="store_true")
    parser.add_argument("--inplace-compression", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-label", default="qwen3_06b_quality_npu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_benchmark(args)
    except Exception as exc:
        report = {
            "status": "FAIL",
            "backend": "torch_forward",
            "model_id": args.model_id,
            "model_path": args.model_path,
            "device": args.device,
            "download_attempted": False,
            "target_layer_limit": args.target_layer_limit,
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "platform": platform.platform(),
        }
    write_outputs(report, Path(args.demo_root), args.run_label)
    print("QWEN_COMPRESSION_QUALITY_BENCH " + json.dumps(report, sort_keys=True))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
