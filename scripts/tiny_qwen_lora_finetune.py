#!/usr/bin/env python
"""Offline TinyQwen RankAdaptor LoRA fine-tuning benchmark for Ascend demos."""
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
from rankadaptor_lora_sync_smoke import LoRALinear, apply_lora_adapters, trainable_parameter_count
from tidal.device import resolve_device, resolve_dtype
from tidal.methods.rankadaptor.core import collect_linear_profiles, config_cost, search_rank_allocation


PROMPT = "hello world"


def load_model(model_path: Path, device: torch.device, dtype: torch.dtype) -> nn.Module:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=dtype,
    )
    return model.to(device)


def load_tokenizer(model_path: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)


def qwen_lora_filter(name: str) -> bool:
    return name.endswith("_proj")


def rankadaptor_qwen_allocation(model: nn.Module, *, budget: int, max_rank: int) -> dict[str, int]:
    profiles = collect_linear_profiles(
        model,
        min_rank=1,
        max_rank=max_rank,
        rank_step=1,
        name_filter=qwen_lora_filter,
    )
    min_cost = config_cost({profile.name: profile.min_rank for profile in profiles}, profiles)
    effective_budget = max(int(budget), int(min_cost))
    result = search_rank_allocation(profiles, budget=effective_budget)
    return {name: rank for name, rank in result.config.items() if rank > 0}


def seeded_token_batch(
    *,
    tokenizer: Any,
    batch_size: int,
    seq_len: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    vocab_size = int(getattr(tokenizer, "vocab_size", len(tokenizer)))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    high = max(4, vocab_size)
    tokens = torch.randint(4, high, (batch_size, seq_len), generator=generator)
    if getattr(tokenizer, "bos_token_id", None) is not None:
        tokens[:, 0] = int(tokenizer.bos_token_id)
    return tokens.to(device)


@torch.no_grad()
def evaluate_loss(model: nn.Module, tokens: torch.Tensor) -> float:
    targets = torch.roll(tokens, shifts=-1, dims=1)
    logits = model(input_ids=tokens).logits
    return float(torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)).cpu())


@torch.no_grad()
def generate_text(model: nn.Module, tokenizer: Any, device: torch.device, *, max_new_tokens: int) -> str:
    encoded = tokenizer([PROMPT], return_tensors="pt", padding=True)
    batch = {key: value.to(device) for key, value in encoded.items() if key in {"input_ids", "attention_mask"}}
    output = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output[0].detach().cpu().tolist(), skip_special_tokens=True)


def train_lora(
    *,
    model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    steps: int,
    lr: float,
    seed: int,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=lr)
    losses: list[float] = []
    start = time.perf_counter()
    model.train()
    for step in range(steps):
        tokens = seeded_token_batch(
            tokenizer=tokenizer,
            batch_size=batch_size,
            seq_len=seq_len,
            seed=seed + step,
            device=device,
        )
        targets = torch.roll(tokens, shifts=-1, dims=1)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=tokens).logits
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        synchronize_device(device)
        losses.append(float(loss.detach().cpu()))
    train_seconds = time.perf_counter() - start
    model.eval()
    return {
        "losses": losses,
        "train_seconds": train_seconds,
        "tokens_trained": int(batch_size * seq_len * steps),
    }


def markdown_report(report: dict[str, Any]) -> str:
    rows = [
        ("Status", report.get("status")),
        ("Model", report.get("model_id")),
        ("Device", report.get("device")),
        ("Dtype", report.get("dtype")),
        ("Trainable adapter params", report.get("trainable_adapter_params")),
        ("Initial loss", report.get("initial_loss")),
        ("Final loss", report.get("final_loss")),
        ("Loss delta", report.get("loss_delta")),
        ("Train seconds", report.get("train_seconds")),
        ("Tokens trained", report.get("tokens_trained")),
        ("Peak MB", report.get("peak_mem_mb")),
    ]
    lines = [
        "# TinyQwen RankAdaptor LoRA Fine-Tune",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in rows:
        if value is not None:
            lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Rank Allocation",
            "",
            "```json",
            json.dumps(report.get("rank_allocation", {}), indent=2, sort_keys=True),
            "```",
            "",
            "## Generation Snapshot",
            "",
            f"- Before: `{report.get('before_generate', '')}`",
            f"- After: `{report.get('after_generate', '')}`",
        ]
    )
    if report.get("error"):
        lines.extend(["", f"Error: `{report.get('error_type')}: {report.get('error')}`"])
    return "\n".join(lines) + "\n"


def write_outputs(report: dict[str, Any], demo_root: Path, run_label: str) -> None:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    (artifacts / f"tiny_qwen_lora_finetune_{run_label}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    (reports / f"tiny-qwen-lora-finetune-{run_label}.md").write_text(markdown_report(report))


def run_finetune(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    torch_dtype = dtype if isinstance(dtype, torch.dtype) else torch.float32
    model_path = Path(args.model_path)
    tokenizer = load_tokenizer(model_path)
    model = load_model(model_path, device, torch_dtype)
    model.eval()

    rank_allocation = rankadaptor_qwen_allocation(model, budget=args.budget, max_rank=args.max_rank)
    apply_lora_adapters(model, rank_allocation)
    model = model.to(device)
    trainable_params = trainable_parameter_count(model)
    eval_tokens = seeded_token_batch(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        seed=args.seed,
        device=device,
    )

    reset_peak_memory_stats(device)
    initial_loss = evaluate_loss(model, eval_tokens)
    before_generate = generate_text(model, tokenizer, device, max_new_tokens=args.max_new_tokens)
    train_result = train_lora(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        steps=args.steps,
        lr=args.lr,
        seed=args.seed,
    )
    final_loss = evaluate_loss(model, eval_tokens)
    after_generate = generate_text(model, tokenizer, device, max_new_tokens=args.max_new_tokens)
    synchronize_device(device)

    return {
        "status": "PASS",
        "model_id": args.model_id,
        "model_path": str(model_path),
        "device": str(device),
        "dtype": str(torch_dtype),
        "download_attempted": False,
        "rank_allocation": rank_allocation,
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
        "platform": platform.platform(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune TinyQwen with RankAdaptor LoRA")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--model-id", default="TinyQwen3-Offline")
    parser.add_argument("--model-path", default="/mnt/nvme/622/models/TinyQwen3-Offline")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--max-rank", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--run-label", default="tiny_qwen3_lora_npu")
    args = parser.parse_args()

    try:
        report = run_finetune(args)
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
    print("TINY_QWEN_LORA_FINETUNE " + json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
