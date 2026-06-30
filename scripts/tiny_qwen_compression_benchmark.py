#!/usr/bin/env python
"""Offline TinyQwen CAP/QPruner compression inference benchmark."""
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

from ascend_inference_benchmark import peak_memory_mb, reset_peak_memory_stats, synchronize_device
from tiny_qwen_lora_finetune import seeded_token_batch
from tidal.device import resolve_device, resolve_dtype
from tidal.methods.global_rank_sparsity.torch import CAPPackedLinear
from tidal.methods.qpruner.torch import QuantizedLinear
from tidal.workflows.compression import cap_compress, qpruner_compress


def qwen_projection_filter(name: str) -> bool:
    return name.endswith("_proj")


def load_model(model_path: Path, device: torch.device, dtype: torch.dtype) -> nn.Module:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=dtype,
    )
    model.eval()
    return model.to(device)


def load_tokenizer(model_path: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)


def build_eval_batch(
    *,
    tokenizer: Any,
    batch_size: int,
    seq_len: int,
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    input_ids = seeded_token_batch(
        tokenizer=tokenizer,
        batch_size=batch_size,
        seq_len=seq_len,
        seed=seed,
        device=device,
    )
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }


@torch.no_grad()
def evaluate_loss(model: nn.Module, batch: dict[str, torch.Tensor]) -> float:
    output = model(**batch)
    return float(output.loss.detach().cpu())


@torch.no_grad()
def forward_latency_ms(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    iters: int,
    warmup: int,
    device: torch.device,
) -> float:
    forward_batch = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
    }
    for _ in range(warmup):
        model(**forward_batch)
    synchronize_device(device)
    start = time.perf_counter()
    for _ in range(iters):
        model(**forward_batch)
    synchronize_device(device)
    return (time.perf_counter() - start) / max(1, iters) * 1000.0


def tokens_per_second(*, batch_size: int, seq_len: int, latency_ms: float) -> float | None:
    if latency_ms <= 0:
        return None
    return round((batch_size * seq_len) / (latency_ms / 1000.0), 3)


def target_linear_names(model: nn.Module) -> list[str]:
    return [
        name
        for name, module in model.named_modules()
        if name and isinstance(module, nn.Linear) and qwen_projection_filter(name)
    ]


def module_parameter_count(module: nn.Module) -> int:
    if isinstance(module, CAPPackedLinear):
        return int(module.parameter_count)
    if isinstance(module, QuantizedLinear):
        return int(module.code_count)
    if hasattr(module, "weight"):
        return int(module.weight.numel())
    return 0


def targeted_parameter_count(model: nn.Module, names: list[str]) -> int:
    modules = dict(model.named_modules())
    return sum(module_parameter_count(modules[name]) for name in names if name in modules)


def qpruner_average_bits(model: nn.Module, names: list[str]) -> float | None:
    modules = dict(model.named_modules())
    total_params = 0
    total_bits = 0
    for name in names:
        module = modules.get(name)
        if not isinstance(module, QuantizedLinear):
            continue
        params = int(module.code_count)
        total_params += params
        total_bits += params * int(module.bits)
    if total_params == 0:
        return None
    return total_bits / total_params


def evaluate_model(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    batch_size: int,
    seq_len: int,
    iters: int,
    warmup: int,
    device: torch.device,
) -> dict[str, Any]:
    loss = evaluate_loss(model, batch)
    latency_ms = forward_latency_ms(model, batch, iters=iters, warmup=warmup, device=device)
    return {
        "loss": loss,
        "latency_ms": latency_ms,
        "tokens_per_s": tokens_per_second(batch_size=batch_size, seq_len=seq_len, latency_ms=latency_ms),
    }


def enable_model_inference_cache(model: nn.Module, *, dtype: torch.dtype, device: torch.device) -> int:
    enabled = 0
    for module in model.modules():
        enable_cache = getattr(module, "enable_inference_cache", None)
        if callable(enable_cache):
            enable_cache(dtype=dtype, device=device)
            enabled += 1
    return enabled


def summarize_cap(
    *,
    result: Any,
    baseline_targeted_params: int,
    eval_metrics: dict[str, Any],
    compression_time_s: float,
) -> dict[str, Any]:
    target_names = [target.name for target in result.method_result.targets]
    compressed_targeted_params = targeted_parameter_count(result.model, target_names)
    ratio = baseline_targeted_params / compressed_targeted_params if compressed_targeted_params else None
    return {
        "status": "PASS",
        "targeted_layers": len(target_names),
        "packed_layers": sum(isinstance(module, CAPPackedLinear) for module in result.model.modules()),
        "targeted_params_original": baseline_targeted_params,
        "targeted_params_compressed": compressed_targeted_params,
        "targeted_compression_ratio": round(ratio, 3) if ratio is not None else None,
        "targeted_param_reduction_pct": (
            round(100.0 * (1.0 - compressed_targeted_params / baseline_targeted_params), 2)
            if baseline_targeted_params
            else None
        ),
        "compression_time_s": compression_time_s,
        "layers": [
            {
                "name": layer.name,
                "rank": layer.rank,
                "sparse_entries": layer.sparse_entries,
                "params": layer.parameter_count,
            }
            for layer in result.method_result.layer_summaries[:8]
        ],
        **eval_metrics,
    }


def summarize_qpruner(
    *,
    result: Any,
    target_names: list[str],
    baseline_targeted_params: int,
    eval_metrics: dict[str, Any],
    compression_time_s: float,
) -> dict[str, Any]:
    compressed_targeted_params = targeted_parameter_count(result.model, target_names)
    average_bits = qpruner_average_bits(result.model, target_names)
    return {
        "status": "PASS",
        "targeted_layers": len(target_names),
        "quantized_layers": sum(isinstance(module, QuantizedLinear) for module in result.model.modules()),
        "targeted_params_original": baseline_targeted_params,
        "targeted_params_quantized": compressed_targeted_params,
        "average_bits": average_bits,
        "memory_bits": result.summary.get("memory_bits"),
        "bitwidths": result.summary.get("bitwidths", {}),
        "compression_time_s": compression_time_s,
        **eval_metrics,
    }


def cached_section_rows(report: dict[str, Any]) -> list[str]:
    if not report.get("inference_cache_enabled"):
        return []
    cap_cached = report.get("cap_cached", {})
    qpruner_cached = report.get("qpruner_cached", {})
    return [
        f"| Cached latency ms | - | {cap_cached.get('latency_ms')} | {qpruner_cached.get('latency_ms')} |",
        f"| Cached speedup | - | {cap_cached.get('latency_speedup')} | {qpruner_cached.get('latency_speedup')} |",
        f"| Cache modules | - | {cap_cached.get('cache_modules')} | {qpruner_cached.get('cache_modules')} |",
    ]


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# TinyQwen Compression Inference Benchmark",
        "",
        "| Metric | Baseline | CAP | QPruner |",
        "|---|---:|---:|---:|",
        f"| Status | {report.get('status')} | {report.get('cap', {}).get('status')} | {report.get('qpruner', {}).get('status')} |",
        f"| Loss | {report.get('baseline', {}).get('loss')} | {report.get('cap', {}).get('loss')} | {report.get('qpruner', {}).get('loss')} |",
        f"| Latency ms | {report.get('baseline', {}).get('latency_ms')} | {report.get('cap', {}).get('latency_ms')} | {report.get('qpruner', {}).get('latency_ms')} |",
        f"| Tokens/s | {report.get('baseline', {}).get('tokens_per_s')} | {report.get('cap', {}).get('tokens_per_s')} | {report.get('qpruner', {}).get('tokens_per_s')} |",
        f"| Targeted compression | - | {report.get('cap', {}).get('targeted_compression_ratio')} | {report.get('qpruner', {}).get('average_bits')} bits avg |",
        f"| Compression seconds | - | {report.get('cap', {}).get('compression_time_s')} | {report.get('qpruner', {}).get('compression_time_s')} |",
        *cached_section_rows(report),
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
    (artifacts / f"tiny_qwen_compression_{run_label}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    (reports / f"tiny-qwen-compression-{run_label}.md").write_text(markdown_report(report))


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    torch_dtype = dtype if isinstance(dtype, torch.dtype) else torch.float32
    model_path = Path(args.model_path)
    tokenizer = load_tokenizer(model_path)
    batch = build_eval_batch(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        seed=args.seed,
        device=device,
    )

    reset_peak_memory_stats(device)
    baseline_model = load_model(model_path, device, torch_dtype)
    baseline_target_names = target_linear_names(baseline_model)
    baseline_targeted_params = targeted_parameter_count(baseline_model, baseline_target_names)
    baseline = evaluate_model(
        baseline_model,
        batch,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        iters=args.iters,
        warmup=args.warmup,
        device=device,
    )
    baseline["targeted_layers"] = len(baseline_target_names)
    baseline["targeted_params"] = baseline_targeted_params

    cap_model = load_model(model_path, device, torch_dtype)
    cap_start = time.perf_counter()
    cap_result = cap_compress(
        model=cap_model,
        model_id=args.model_id,
        budget=args.cap_budget,
        target_roles=None,
        name_filter=qwen_projection_filter,
        device=device,
        dtype=torch_dtype,
        max_iter=args.cap_max_iter,
        policy_steps=args.cap_policy_steps,
        samples_per_step=args.cap_samples_per_step,
        seed=args.seed,
        rpca_backend=args.cap_rpca_backend,
    )
    cap_time_s = time.perf_counter() - cap_start
    cap_metrics = evaluate_model(
        cap_result.model,
        batch,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        iters=args.iters,
        warmup=args.warmup,
        device=device,
    )
    cap = summarize_cap(
        result=cap_result,
        baseline_targeted_params=baseline_targeted_params,
        eval_metrics=cap_metrics,
        compression_time_s=cap_time_s,
    )
    cap["latency_speedup"] = (
        round(baseline["latency_ms"] / cap["latency_ms"], 3) if cap.get("latency_ms") else None
    )
    cap["loss_delta"] = cap["loss"] - baseline["loss"]
    cap_cached = None
    if args.enable_inference_cache:
        cache_modules = enable_model_inference_cache(cap_result.model, dtype=torch_dtype, device=device)
        cap_cached = evaluate_model(
            cap_result.model,
            batch,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            iters=args.iters,
            warmup=args.warmup,
            device=device,
        )
        cap_cached.update(
            {
                "status": "PASS",
                "cache_modules": cache_modules,
                "latency_speedup": (
                    round(baseline["latency_ms"] / cap_cached["latency_ms"], 3)
                    if cap_cached.get("latency_ms")
                    else None
                ),
                "loss_delta": cap_cached["loss"] - baseline["loss"],
            }
        )

    qpruner_model = load_model(model_path, device, torch_dtype)
    qpruner_importances = {name: 1.0 for name in baseline_target_names}
    qpruner_start = time.perf_counter()
    qpruner_result = qpruner_compress(
        model=qpruner_model,
        model_id=args.model_id,
        importances=qpruner_importances,
        candidate_bits=(2, 4, 8),
        max_average_bits=args.qpruner_average_bits,
        target_roles=None,
        name_filter=qwen_projection_filter,
        device=device,
        dtype=torch_dtype,
        seed=args.seed,
    )
    qpruner_time_s = time.perf_counter() - qpruner_start
    qpruner_metrics = evaluate_model(
        qpruner_result.model,
        batch,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        iters=args.iters,
        warmup=args.warmup,
        device=device,
    )
    qpruner = summarize_qpruner(
        result=qpruner_result,
        target_names=baseline_target_names,
        baseline_targeted_params=baseline_targeted_params,
        eval_metrics=qpruner_metrics,
        compression_time_s=qpruner_time_s,
    )
    qpruner["latency_speedup"] = (
        round(baseline["latency_ms"] / qpruner["latency_ms"], 3) if qpruner.get("latency_ms") else None
    )
    qpruner["loss_delta"] = qpruner["loss"] - baseline["loss"]
    qpruner_cached = None
    if args.enable_inference_cache:
        cache_modules = enable_model_inference_cache(qpruner_result.model, dtype=torch_dtype, device=device)
        qpruner_cached = evaluate_model(
            qpruner_result.model,
            batch,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            iters=args.iters,
            warmup=args.warmup,
            device=device,
        )
        qpruner_cached.update(
            {
                "status": "PASS",
                "cache_modules": cache_modules,
                "latency_speedup": (
                    round(baseline["latency_ms"] / qpruner_cached["latency_ms"], 3)
                    if qpruner_cached.get("latency_ms")
                    else None
                ),
                "loss_delta": qpruner_cached["loss"] - baseline["loss"],
            }
        )
    synchronize_device(device)

    report = {
        "status": "PASS",
        "model_id": args.model_id,
        "model_path": str(model_path),
        "device": str(device),
        "dtype": str(torch_dtype),
        "download_attempted": False,
        "inference_cache_enabled": bool(args.enable_inference_cache),
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "baseline": baseline,
        "cap": cap,
        "qpruner": qpruner,
        "peak_mem_mb": peak_memory_mb(device),
        "platform": platform.platform(),
    }
    if cap_cached is not None:
        report["cap_cached"] = cap_cached
    if qpruner_cached is not None:
        report["qpruner_cached"] = qpruner_cached
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark TinyQwen CAP/QPruner compressed inference")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--model-id", default="TinyQwen3-Offline")
    parser.add_argument("--model-path", default="/mnt/nvme/622/models/TinyQwen3-Offline")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--cap-budget", type=int, default=8192)
    parser.add_argument("--cap-max-iter", type=int, default=6)
    parser.add_argument("--cap-policy-steps", type=int, default=1)
    parser.add_argument("--cap-samples-per-step", type=int, default=1)
    parser.add_argument("--cap-rpca-backend", default="numpy")
    parser.add_argument("--qpruner-average-bits", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-label", default="tiny_qwen3_compression_npu")
    parser.add_argument("--enable-inference-cache", action="store_true")
    args = parser.parse_args()

    try:
        report = run_benchmark(args)
    except Exception as exc:
        report = {
            "status": "FAIL",
            "model_id": args.model_id,
            "model_path": args.model_path,
            "device": args.device,
            "download_attempted": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
            "platform": platform.platform(),
        }
    write_outputs(report, Path(args.demo_root), args.run_label)
    print("TINY_QWEN_COMPRESSION_BENCH " + json.dumps(report, sort_keys=True))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
