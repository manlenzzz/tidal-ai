#!/usr/bin/env python
"""Offline TinyQwen BSLoRA-style shared LoRA fine-tuning benchmark."""
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
from rankadaptor_lora_sync_smoke import replace_module
from tidal.device import resolve_device, resolve_dtype
from tiny_qwen_lora_finetune import (
    PROMPT,
    evaluate_loss,
    generate_text,
    load_model,
    load_tokenizer,
    seeded_token_batch,
)


def projection_suffix(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def shared_group_name(name: str, module: nn.Linear) -> str:
    suffix = projection_suffix(name)
    shape = f"{module.in_features}x{module.out_features}"
    if suffix in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return f"attn_hidden_{shape}"
    if suffix in {"gate_proj", "up_proj"}:
        return f"mlp_up_{shape}"
    if suffix == "down_proj":
        return f"mlp_down_{shape}"
    return f"{suffix}_{shape}"


def qwen_shared_lora_allocation(model: nn.Module, *, rank: int) -> dict[str, dict[str, Any]]:
    if rank <= 0:
        raise ValueError("rank must be positive")
    allocation: dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        if not name.endswith("_proj") or not isinstance(module, nn.Linear):
            continue
        allocation[name] = {
            "rank": int(rank),
            "group": shared_group_name(name, module),
            "in_features": int(module.in_features),
            "out_features": int(module.out_features),
        }
    return allocation


class SharedLoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        *,
        lora_a: nn.Parameter,
        lora_b: nn.Parameter,
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.lora_a = lora_a
        self.lora_b = lora_b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        adapted = torch.nn.functional.linear(torch.nn.functional.linear(x, self.lora_a), self.lora_b)
        return base_out + adapted * self.scaling


def new_adapter_pair(*, rank: int, in_features: int, out_features: int) -> tuple[nn.Parameter, nn.Parameter]:
    lora_a = nn.Parameter(torch.empty(rank, in_features))
    lora_b = nn.Parameter(torch.zeros(out_features, rank))
    nn.init.kaiming_uniform_(lora_a, a=5**0.5)
    return lora_a, lora_b


def apply_shared_lora_adapters(
    model: nn.Module,
    allocation: dict[str, dict[str, Any]],
    *,
    alpha: float | None = None,
) -> dict[str, Any]:
    for param in model.parameters():
        param.requires_grad = False

    modules = dict(model.named_modules())
    adapters: dict[str, tuple[nn.Parameter, nn.Parameter]] = {}
    shared_groups: dict[str, dict[str, Any]] = {}
    unshared_adapter_params = 0

    for name, entry in allocation.items():
        target = modules[name]
        if not isinstance(target, nn.Linear):
            raise TypeError(f"{name} is not a Linear module")
        rank = int(entry["rank"])
        group = str(entry["group"])
        in_features = int(target.in_features)
        out_features = int(target.out_features)
        group_meta = shared_groups.setdefault(
            group,
            {
                "rank": rank,
                "in_features": in_features,
                "out_features": out_features,
                "adapter_params": rank * (in_features + out_features),
                "modules": [],
            },
        )
        if (
            int(group_meta["rank"]) != rank
            or int(group_meta["in_features"]) != in_features
            or int(group_meta["out_features"]) != out_features
        ):
            raise ValueError(f"shared group {group} has incompatible adapter shapes")
        group_meta["modules"].append(name)
        unshared_adapter_params += rank * (in_features + out_features)
        if group not in adapters:
            adapters[group] = new_adapter_pair(rank=rank, in_features=in_features, out_features=out_features)
        lora_a, lora_b = adapters[group]
        replace_module(
            model,
            name,
            SharedLoRALinear(
                target,
                lora_a=lora_a,
                lora_b=lora_b,
                rank=rank,
                alpha=float(alpha if alpha is not None else rank),
            ),
        )

    trainable_adapter_params = unique_adapter_parameter_count(model)
    return {
        "shared_groups": shared_groups,
        "target_module_count": len(allocation),
        "unique_adapter_tensors": len(unique_adapter_tensors(model)),
        "trainable_adapter_params": trainable_adapter_params,
        "unshared_adapter_params": unshared_adapter_params,
    }


def unique_adapter_tensors(model: nn.Module) -> list[nn.Parameter]:
    seen: set[int] = set()
    tensors: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, SharedLoRALinear):
            for param in (module.lora_a, module.lora_b):
                key = id(param)
                if key not in seen:
                    seen.add(key)
                    tensors.append(param)
    return tensors


def unique_adapter_parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in unique_adapter_tensors(model) if param.requires_grad)


def adapter_checksum(model: nn.Module, device: torch.device) -> torch.Tensor:
    checksum = torch.zeros((), device=device, dtype=torch.float32)
    for param in unique_adapter_tensors(model):
        checksum = checksum + param.detach().float().sum()
    return checksum


def train_shared_lora(
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
        ("Method", report.get("method")),
        ("Model", report.get("model_id")),
        ("Device", report.get("device")),
        ("Dtype", report.get("dtype")),
        ("Trainable adapter params", report.get("trainable_adapter_params")),
        ("Unshared adapter params", report.get("unshared_adapter_params")),
        ("Unique adapter tensors", report.get("unique_adapter_tensors")),
        ("Target modules", report.get("target_module_count")),
        ("Initial loss", report.get("initial_loss")),
        ("Final loss", report.get("final_loss")),
        ("Loss delta", report.get("loss_delta")),
        ("Train seconds", report.get("train_seconds")),
        ("Tokens trained", report.get("tokens_trained")),
        ("Peak MB", report.get("peak_mem_mb")),
    ]
    lines = [
        "# TinyQwen BSLoRA-Style Shared LoRA Fine-Tune",
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
            "## Shared Adapter Groups",
            "",
            "```json",
            json.dumps(report.get("shared_groups", {}), indent=2, sort_keys=True),
            "```",
            "",
            "## Generation Snapshot",
            "",
            f"- Prompt: `{PROMPT}`",
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
    (artifacts / f"tiny_qwen_bslora_finetune_{run_label}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    (reports / f"tiny-qwen-bslora-finetune-{run_label}.md").write_text(markdown_report(report))


def run_finetune(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    torch_dtype = dtype if isinstance(dtype, torch.dtype) else torch.float32
    model_path = Path(args.model_path)
    tokenizer = load_tokenizer(model_path)
    model = load_model(model_path, device, torch_dtype)
    model.eval()

    allocation = qwen_shared_lora_allocation(model, rank=args.rank)
    metadata = apply_shared_lora_adapters(model, allocation, alpha=args.alpha)
    model = model.to(device)
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
    train_result = train_shared_lora(
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
        "method": "bslora_shared_lora",
        "model_id": args.model_id,
        "model_path": str(model_path),
        "device": str(device),
        "dtype": str(torch_dtype),
        "download_attempted": False,
        "rank": int(args.rank),
        "alpha": args.alpha,
        "shared_groups": metadata["shared_groups"],
        "target_module_count": metadata["target_module_count"],
        "unique_adapter_tensors": metadata["unique_adapter_tensors"],
        "trainable_adapter_params": metadata["trainable_adapter_params"],
        "unshared_adapter_params": metadata["unshared_adapter_params"],
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
    parser = argparse.ArgumentParser(description="Fine-tune TinyQwen with BSLoRA-style shared LoRA")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--model-id", default="TinyQwen3-Offline")
    parser.add_argument("--model-path", default="/mnt/nvme/622/models/TinyQwen3-Offline")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--run-label", default="tiny_qwen3_bslora_npu")
    args = parser.parse_args()

    try:
        report = run_finetune(args)
    except Exception as exc:
        report = {
            "status": "FAIL",
            "method": "bslora_shared_lora",
            "model_id": args.model_id,
            "model_path": args.model_path,
            "device": args.device,
            "download_attempted": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
            "platform": platform.platform(),
        }
    write_outputs(report, Path(args.demo_root), args.run_label)
    print("TINY_QWEN_BSLORA_FINETUNE " + json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
