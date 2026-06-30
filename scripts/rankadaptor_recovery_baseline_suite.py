#!/usr/bin/env python
"""Run RankAdaptor recovery baselines under one controlled setup."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from ascend_inference_benchmark import peak_memory_mb, reset_peak_memory_stats, synchronize_device
from rankadaptor_lora_sync_smoke import LoRALinear, apply_lora_adapters, trainable_parameter_count
from tidal.device import resolve_device, resolve_dtype
from tidal.methods.rankadaptor.core import collect_linear_profiles, config_cost, search_rank_allocation
from tiny_qwen_lora_finetune import (
    evaluate_loss,
    generate_text,
    load_model,
    load_tokenizer,
    qwen_lora_filter,
    seeded_token_batch,
    train_lora,
)


METHOD_ORDER = ["no_recovery", "lora", "adalora_style"]
PAPER_BASELINES = ["without recovery", "LoRA", "AdaLoRA-style"]
ADALORA_SCOPE = "controlled AdaLoRA-style approximation, not official AdaLoRA implementation"


def rankadaptor_allocation(
    model: nn.Module,
    *,
    budget: int,
    max_rank: int,
    target_module_limit: int,
) -> tuple[dict[str, int], int]:
    profiles = collect_linear_profiles(
        model,
        min_rank=1,
        max_rank=max_rank,
        rank_step=1,
        name_filter=qwen_lora_filter,
    )
    profiles = sorted(profiles, key=lambda profile: profile.name)
    total = len(profiles)
    if target_module_limit > 0:
        profiles = profiles[:target_module_limit]
    if not profiles:
        return {}, total
    min_cost = config_cost({profile.name: profile.min_rank for profile in profiles}, profiles)
    effective_budget = max(int(budget), int(min_cost))
    result = search_rank_allocation(profiles, budget=effective_budget)
    return {name: rank for name, rank in result.config.items() if rank > 0}, total


def adalora_style_allocation(rank_allocation: dict[str, int], *, max_rank: int) -> dict[str, int]:
    if not rank_allocation:
        return {}
    expanded = {name: min(max(int(rank) + 1, 1), max(int(max_rank), 1)) for name, rank in rank_allocation.items()}
    # Deterministic budget trimming approximates AdaLoRA's rank redistribution without importing PEFT/AdaLoRA.
    ranked_names = sorted(expanded, key=lambda name: (-expanded[name], name))
    trimmed: dict[str, int] = {}
    for index, name in enumerate(ranked_names):
        rank = expanded[name]
        if index % 2 == 1:
            rank = max(1, rank - 1)
        trimmed[name] = rank
    return dict(sorted(trimmed.items()))


def run_no_recovery(
    *,
    model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    eval_tokens: torch.Tensor,
    max_new_tokens: int,
) -> dict[str, Any]:
    reset_peak_memory_stats(device)
    initial_loss = evaluate_loss(model, eval_tokens)
    before_generate = generate_text(model, tokenizer, device, max_new_tokens=max_new_tokens)
    synchronize_device(device)
    final_loss = evaluate_loss(model, eval_tokens)
    after_generate = generate_text(model, tokenizer, device, max_new_tokens=max_new_tokens)
    synchronize_device(device)
    return {
        "status": "PASS",
        "method": "no_recovery",
        "paper_baseline": "without recovery",
        "rank_allocation": {},
        "target_module_count": 0,
        "trainable_adapter_params": 0,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_delta": final_loss - initial_loss,
        "losses": [],
        "train_seconds": 0.0,
        "tokens_trained": 0,
        "peak_mem_mb": peak_memory_mb(device),
        "before_generate": before_generate,
        "after_generate": after_generate,
    }


def run_adapter_recovery(
    *,
    method: str,
    paper_baseline: str,
    model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    eval_tokens: torch.Tensor,
    rank_allocation: dict[str, int],
    batch_size: int,
    seq_len: int,
    steps: int,
    lr: float,
    seed: int,
    max_new_tokens: int,
    claim_scope: str | None = None,
) -> dict[str, Any]:
    apply_lora_adapters(model, rank_allocation)
    model = model.to(device)
    trainable_params = trainable_parameter_count(model)
    reset_peak_memory_stats(device)
    initial_loss = evaluate_loss(model, eval_tokens)
    before_generate = generate_text(model, tokenizer, device, max_new_tokens=max_new_tokens)
    train_result = train_lora(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        seq_len=seq_len,
        steps=steps,
        lr=lr,
        seed=seed,
    )
    final_loss = evaluate_loss(model, eval_tokens)
    after_generate = generate_text(model, tokenizer, device, max_new_tokens=max_new_tokens)
    synchronize_device(device)
    result = {
        "status": "PASS",
        "method": method,
        "paper_baseline": paper_baseline,
        "rank_allocation": rank_allocation,
        "target_module_count": len(rank_allocation),
        "trainable_adapter_params": trainable_params,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_delta": final_loss - initial_loss,
        "losses": train_result["losses"],
        "train_seconds": train_result["train_seconds"],
        "tokens_trained": train_result["tokens_trained"],
        "peak_mem_mb": peak_memory_mb(device),
        "before_generate": before_generate,
        "after_generate": after_generate,
    }
    if claim_scope:
        result["claim_scope"] = claim_scope
    return result


def fmt(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def build_summary(
    *,
    runs: list[dict[str, Any]],
    model_id: str,
    model_path: str,
    device: str,
    dtype: str,
    run_label: str,
    target_modules_total: int = 0,
    started_at: float | None = None,
) -> dict[str, Any]:
    methods = {run["method"]: run for run in runs}
    all_methods_passed = all(methods.get(name, {}).get("status") == "PASS" for name in METHOD_ORDER)
    best = min(
        (run for run in runs if run.get("status") == "PASS"),
        key=lambda run: float(run.get("final_loss", float("inf"))),
        default={},
    )
    summary = {
        "status": "PASS" if all_methods_passed else "FAIL",
        "method": "rankadaptor_recovery_baseline_suite",
        "model_id": model_id,
        "model_path": model_path,
        "device": device,
        "dtype": dtype,
        "run_label": run_label,
        "paper_baselines": PAPER_BASELINES,
        "claim_scope": "Controlled RankAdaptor recovery comparison on one model/prompt setup; AdaLoRA-style is an approximation unless official AdaLoRA is ported.",
        "all_methods_passed": all_methods_passed,
        "best_recovery_method": best.get("method", "missing"),
        "target_modules_total": target_modules_total,
        "methods": methods,
        "runs": runs,
        "platform": platform.platform(),
    }
    for name in METHOD_ORDER:
        summary[f"{name}_loss_delta"] = methods.get(name, {}).get("loss_delta")
    if started_at is not None:
        summary["elapsed_seconds"] = time.perf_counter() - started_at
    return summary


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# RankAdaptor Recovery Baseline Suite",
        "",
        f"- Overall status: `{summary.get('status')}`",
        f"- Model: `{summary.get('model_id')}`",
        f"- Device: `{summary.get('device')}`",
        f"- Paper baselines: `{', '.join(summary.get('paper_baselines', []))}`",
        f"- Best recovery method: `{summary.get('best_recovery_method')}`",
        f"- Claim scope: {summary.get('claim_scope')}",
        "",
        "| Method | Paper baseline | Status | Initial loss | Final loss | Loss delta | Trainable adapter params | Peak MB |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for name in METHOD_ORDER:
        row = summary.get("methods", {}).get(name, {})
        lines.append(
            "| {method} | {baseline} | {status} | {initial} | {final} | {delta} | {params} | {peak} |".format(
                method=row.get("method", name),
                baseline=row.get("paper_baseline", "missing"),
                status=row.get("status", "missing"),
                initial=fmt(row.get("initial_loss")),
                final=fmt(row.get("final_loss")),
                delta=fmt(row.get("loss_delta")),
                params=fmt(row.get("trainable_adapter_params")),
                peak=fmt(row.get("peak_mem_mb")),
            )
        )
    lines.extend(["", "## Generation Samples", ""])
    for name in METHOD_ORDER:
        row = summary.get("methods", {}).get(name, {})
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Before: `{row.get('before_generate', 'missing')}`",
                f"- After: `{row.get('after_generate', 'missing')}`",
                "",
            ]
        )
        if row.get("claim_scope"):
            lines.append(f"- Claim scope: {row['claim_scope']}")
            lines.append("")
    lines.extend(["## Rank Allocations", ""])
    for name in ("lora", "adalora_style"):
        row = summary.get("methods", {}).get(name, {})
        lines.extend(["### " + name, "", "```json", json.dumps(row.get("rank_allocation", {}), indent=2, sort_keys=True), "```", ""])
    return "\n".join(lines)


def write_outputs(summary: dict[str, Any], demo_root: Path, run_label: str) -> None:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    (artifacts / f"rankadaptor_recovery_baseline_suite_{run_label}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    (reports / f"rankadaptor-recovery-baseline-suite-{run_label}.md").write_text(markdown_report(summary))


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    torch_dtype = dtype if isinstance(dtype, torch.dtype) else torch.float32
    model_path = Path(args.model_path)
    tokenizer = load_tokenizer(model_path)
    eval_tokens = seeded_token_batch(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        seed=args.seed,
        device=device,
    )

    base_model = load_model(model_path, device, torch_dtype)
    rank_allocation, target_modules_total = rankadaptor_allocation(
        base_model,
        budget=args.budget,
        max_rank=args.max_rank,
        target_module_limit=args.target_module_limit,
    )
    adalora_allocation = adalora_style_allocation(rank_allocation, max_rank=max(args.max_rank, args.adalora_max_rank))
    del base_model

    runs: list[dict[str, Any]] = []
    no_recovery_model = load_model(model_path, device, torch_dtype)
    runs.append(
        run_no_recovery(
            model=no_recovery_model,
            tokenizer=tokenizer,
            device=device,
            eval_tokens=eval_tokens,
            max_new_tokens=args.max_new_tokens,
        )
    )
    del no_recovery_model

    lora_model = load_model(model_path, device, torch_dtype)
    runs.append(
        run_adapter_recovery(
            method="lora",
            paper_baseline="LoRA",
            model=lora_model,
            tokenizer=tokenizer,
            device=device,
            eval_tokens=eval_tokens,
            rank_allocation=rank_allocation,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            steps=args.steps,
            lr=args.lr,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
        )
    )
    del lora_model

    adalora_model = load_model(model_path, device, torch_dtype)
    runs.append(
        run_adapter_recovery(
            method="adalora_style",
            paper_baseline="AdaLoRA-style",
            model=adalora_model,
            tokenizer=tokenizer,
            device=device,
            eval_tokens=eval_tokens,
            rank_allocation=adalora_allocation,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            steps=args.steps,
            lr=args.lr,
            seed=args.seed + 17,
            max_new_tokens=args.max_new_tokens,
            claim_scope=ADALORA_SCOPE,
        )
    )
    del adalora_model

    return build_summary(
        runs=runs,
        model_id=args.model_id,
        model_path=str(model_path),
        device=str(device),
        dtype=str(torch_dtype),
        run_label=args.run_label,
        target_modules_total=target_modules_total,
        started_at=started,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RankAdaptor recovery baselines under one setup")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-path", default="/mnt/nvme/622/models/Qwen3-0.6B")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--max-rank", type=int, default=2)
    parser.add_argument("--adalora-max-rank", type=int, default=3)
    parser.add_argument("--target-module-limit", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--run-label", default="qwen3_06b_npu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run_suite(args)
    except Exception as exc:
        summary = {
            "status": "FAIL",
            "method": "rankadaptor_recovery_baseline_suite",
            "model_id": args.model_id,
            "model_path": args.model_path,
            "device": args.device,
            "dtype": args.dtype,
            "run_label": args.run_label,
            "paper_baselines": PAPER_BASELINES,
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "platform": platform.platform(),
        }
    write_outputs(summary, Path(args.demo_root), args.run_label)
    print("RANKADAPTOR_RECOVERY_BASELINE_SUITE " + json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
