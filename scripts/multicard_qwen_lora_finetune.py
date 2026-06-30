#!/usr/bin/env python
"""Run Qwen3 RankAdaptor LoRA fine-tuning on synchronized cards."""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from ascend_inference_benchmark import peak_memory_mb, reset_peak_memory_stats, synchronize_device
from ascend_multicard_sync_smoke import apply_hccl_port_defaults, default_backend, parse_cards
from rankadaptor_lora_sync_smoke import apply_lora_adapters, trainable_parameter_count
from tidal.device import resolve_dtype
from tidal.methods.rankadaptor.core import collect_linear_profiles, config_cost, search_rank_allocation
from tiny_qwen_lora_finetune import evaluate_loss, load_model, load_tokenizer, qwen_lora_filter, seeded_token_batch


TRAIN_SAMPLES = [
    {
        "prompt": "Explain CAP compression in one sentence.",
        "target": "CAP accelerates inference by replacing selected dense weights with compact low-rank factors.",
    },
    {
        "prompt": "What does RankAdaptor choose?",
        "target": "RankAdaptor assigns adapter ranks under a parameter budget so LoRA capacity goes to useful layers.",
    },
]
VALIDATION_SAMPLES = [
    {
        "prompt": "What does LoRA train?",
        "target": "LoRA freezes the base model and trains small adapter matrices.",
    }
]


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def set_device(device_name: str, local_rank: int, cards: list[int]) -> torch.device:
    normalized = device_name.lower()
    if normalized.startswith("npu"):
        import torch_npu  # noqa: F401

        card = cards[local_rank]
        if hasattr(torch.npu, "set_device"):
            torch.npu.set_device(card)
        return torch.device(f"npu:{card}")
    if normalized.startswith("cuda"):
        card = cards[local_rank]
        torch.cuda.set_device(card)
        return torch.device(f"cuda:{card}")
    return torch.device("cpu")


def rankadaptor_lora_allocation(
    model: nn.Module,
    *,
    budget: int,
    max_rank: int,
    target_module_limit: int,
) -> tuple[dict[str, int], int]:
    all_profiles = collect_linear_profiles(
        model,
        min_rank=1,
        max_rank=max_rank,
        rank_step=1,
        name_filter=qwen_lora_filter,
    )
    profiles = sorted(all_profiles, key=lambda profile: profile.name)
    if target_module_limit > 0:
        profiles = profiles[:target_module_limit]
    min_cost = config_cost({profile.name: profile.min_rank for profile in profiles}, profiles) if profiles else 0
    effective_budget = max(int(budget), int(min_cost))
    result = search_rank_allocation(profiles, budget=effective_budget) if profiles else None
    allocation = {} if result is None else {name: rank for name, rank in result.config.items() if rank > 0}
    return allocation, len(all_profiles)


def adapter_checksum(model: nn.Module, device: torch.device) -> torch.Tensor:
    checksum = torch.zeros((), device=device, dtype=torch.float32)
    for name, param in model.named_parameters():
        if "lora_" in name:
            checksum = checksum + param.detach().float().sum()
    return checksum


def sample_text(sample: dict[str, str]) -> str:
    return f"{sample['prompt']} {sample['target']}"


def sample_prompt_text(sample: dict[str, str]) -> str:
    return sample["prompt"]


def encode_instruction_batch(
    tokenizer: Any,
    samples: list[dict[str, str]],
    *,
    device: torch.device,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    texts = [sample_text(sample) for sample in samples]
    prompts = [sample_prompt_text(sample) for sample in samples]
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    prompt_encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    prompt_lengths = prompt_encoded.get("attention_mask", torch.ones_like(prompt_encoded["input_ids"])).sum(dim=1)
    padding_side = getattr(tokenizer, "padding_side", "right")
    for row, prompt_len_tensor in enumerate(prompt_lengths):
        prompt_len = int(prompt_len_tensor.item())
        if padding_side == "left":
            nonpad = int(attention_mask[row].sum().item())
            start = input_ids.shape[1] - nonpad
            labels[row, start : min(start + prompt_len, input_ids.shape[1])] = -100
        else:
            labels[row, : min(prompt_len, input_ids.shape[1])] = -100
    if not torch.any(labels != -100):
        labels[attention_mask == 1] = input_ids[attention_mask == 1]
    return input_ids.to(device), attention_mask.to(device), labels.to(device)


@torch.no_grad()
def evaluate_instruction_loss(
    model: nn.Module,
    tokenizer: Any,
    samples: list[dict[str, str]],
    *,
    device: torch.device,
    max_length: int,
) -> float:
    input_ids, attention_mask, labels = encode_instruction_batch(
        tokenizer,
        samples,
        device=device,
        max_length=max_length,
    )
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    return float(torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)).cpu())


@torch.no_grad()
def generate_instruction_text(
    model: nn.Module,
    tokenizer: Any,
    *,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
) -> str:
    encoded = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True, max_length=128)
    batch = {key: value.to(device) for key, value in encoded.items() if key in {"input_ids", "attention_mask"}}
    output = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output[0].detach().cpu().tolist(), skip_special_tokens=True)


def train_ddp_lora(
    *,
    ddp_model: nn.parallel.DistributedDataParallel,
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    steps: int,
    lr: float,
    seed: int,
) -> list[float]:
    optimizer = torch.optim.AdamW((param for param in ddp_model.parameters() if param.requires_grad), lr=lr)
    losses: list[float] = []
    ddp_model.train()
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
        logits = ddp_model(input_ids=tokens).logits
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        synchronize_device(device)
        losses.append(float(loss.detach().cpu()))
    ddp_model.eval()
    return losses


def train_ddp_instruction_lora(
    *,
    ddp_model: nn.parallel.DistributedDataParallel,
    tokenizer: Any,
    samples: list[dict[str, str]],
    device: torch.device,
    max_length: int,
    steps: int,
    lr: float,
) -> list[float]:
    optimizer = torch.optim.AdamW((param for param in ddp_model.parameters() if param.requires_grad), lr=lr)
    losses: list[float] = []
    ddp_model.train()
    for step in range(steps):
        sample = samples[step % len(samples)]
        input_ids, attention_mask, labels = encode_instruction_batch(
            tokenizer,
            [sample],
            device=device,
            max_length=max_length,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = ddp_model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        loss.backward()
        optimizer.step()
        synchronize_device(device)
        losses.append(float(loss.detach().cpu()))
    ddp_model.eval()
    return losses


def all_reduce_totals(
    *,
    initial_loss: float,
    final_loss: float,
    validation_initial_loss: float | None,
    validation_final_loss: float | None,
    peak_mb: float,
    status_pass: bool,
    adapter_sync_pass: bool,
    device: torch.device,
) -> dict[str, Any]:
    values = torch.tensor(
        [
            float(initial_loss),
            float(final_loss),
            0.0 if validation_initial_loss is None else float(validation_initial_loss),
            0.0 if validation_final_loss is None else float(validation_final_loss),
            float(peak_mb),
            1.0 if status_pass else 0.0,
            1.0 if adapter_sync_pass else 0.0,
        ],
        device=device,
        dtype=torch.float32,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values_cpu = values.detach().cpu().tolist()
    return {
        "initial_loss_sum": float(values_cpu[0]),
        "final_loss_sum": float(values_cpu[1]),
        "validation_initial_loss_sum": float(values_cpu[2]),
        "validation_final_loss_sum": float(values_cpu[3]),
        "peak_mem_mb_sum": float(values_cpu[4]),
        "pass_count": int(round(values_cpu[5])),
        "adapter_sync_pass_count": int(round(values_cpu[6])),
    }


def train_one_rank(rank: int, config: dict[str, Any]) -> None:
    artifacts = Path(config["artifacts_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)
    rank_json = artifacts / f"multicard_qwen_lora_finetune_{config['run_label']}_rank{rank}.json"
    report: dict[str, Any] = {
        "status": "STARTED",
        "rank": rank,
        "world_size": config["world_size"],
        "backend": config["backend"],
        "device_request": config["device_name"],
        "card": config["cards"][rank] if rank < len(config["cards"]) else None,
        "method": "rankadaptor_lora",
        "target_module_limit": config["target_module_limit"],
        "train_mode": config["train_mode"],
    }
    try:
        os.environ["MASTER_ADDR"] = config["master_addr"]
        os.environ["MASTER_PORT"] = str(config["master_port"])
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(config["world_size"])
        if str(config["device_name"]).lower().startswith("npu"):
            apply_hccl_port_defaults(config["device_name"])
            os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(str(card) for card in config["cards"])
            os.environ.setdefault("HCCL_CONNECT_TIMEOUT", str(max(int(config["timeout_seconds"]), 120)))

        device = set_device(config["device_name"], rank, config["cards"])
        dtype = resolve_dtype(config["dtype"], device)
        torch_dtype = dtype if isinstance(dtype, torch.dtype) else torch.float32
        dist.init_process_group(
            backend=config["backend"],
            rank=rank,
            world_size=config["world_size"],
            timeout=timedelta(seconds=int(config["timeout_seconds"])),
        )
        torch.manual_seed(int(config["seed"]))
        tokenizer = load_tokenizer(Path(config["model_path"]))
        model = load_model(Path(config["model_path"]), device, torch_dtype)
        rank_allocation, targeted_modules_total = rankadaptor_lora_allocation(
            model,
            budget=int(config["budget"]),
            max_rank=int(config["max_rank"]),
            target_module_limit=int(config["target_module_limit"]),
        )
        apply_lora_adapters(model, rank_allocation)
        model = model.to(device)
        ddp_model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=None if device.type == "cpu" else [device.index],
        )

        reset_peak_memory_stats(device)
        validation_initial_loss = None
        validation_final_loss = None
        before_generate = None
        after_generate = None
        if config["train_mode"] == "instruction":
            initial_loss = evaluate_instruction_loss(
                ddp_model.module,
                tokenizer,
                TRAIN_SAMPLES,
                device=device,
                max_length=int(config["instruction_max_length"]),
            )
            validation_initial_loss = evaluate_instruction_loss(
                ddp_model.module,
                tokenizer,
                VALIDATION_SAMPLES,
                device=device,
                max_length=int(config["instruction_max_length"]),
            )
            if rank == 0:
                before_generate = generate_instruction_text(
                    ddp_model.module,
                    tokenizer,
                    device=device,
                    prompt=VALIDATION_SAMPLES[0]["prompt"],
                    max_new_tokens=int(config["max_new_tokens"]),
                )
            losses = train_ddp_instruction_lora(
                ddp_model=ddp_model,
                tokenizer=tokenizer,
                samples=TRAIN_SAMPLES,
                device=device,
                max_length=int(config["instruction_max_length"]),
                steps=int(config["steps"]),
                lr=float(config["lr"]),
            )
            final_loss = evaluate_instruction_loss(
                ddp_model.module,
                tokenizer,
                TRAIN_SAMPLES,
                device=device,
                max_length=int(config["instruction_max_length"]),
            )
            validation_final_loss = evaluate_instruction_loss(
                ddp_model.module,
                tokenizer,
                VALIDATION_SAMPLES,
                device=device,
                max_length=int(config["instruction_max_length"]),
            )
            if rank == 0:
                after_generate = generate_instruction_text(
                    ddp_model.module,
                    tokenizer,
                    device=device,
                    prompt=VALIDATION_SAMPLES[0]["prompt"],
                    max_new_tokens=int(config["max_new_tokens"]),
                )
        else:
            eval_tokens = seeded_token_batch(
                tokenizer=tokenizer,
                batch_size=int(config["batch_size"]),
                seq_len=int(config["seq_len"]),
                seed=int(config["seed"]),
                device=device,
            )
            initial_loss = evaluate_loss(ddp_model.module, eval_tokens)
            losses = train_ddp_lora(
                ddp_model=ddp_model,
                tokenizer=tokenizer,
                device=device,
                batch_size=int(config["batch_size"]),
                seq_len=int(config["seq_len"]),
                steps=int(config["steps"]),
                lr=float(config["lr"]),
                seed=int(config["seed"]),
            )
            final_loss = evaluate_loss(ddp_model.module, eval_tokens)
        synchronize_device(device)
        checksum = adapter_checksum(ddp_model.module, device)
        avg_checksum = checksum.detach().clone()
        dist.all_reduce(avg_checksum, op=dist.ReduceOp.SUM)
        avg_checksum /= int(config["world_size"])
        max_adapter_delta = float((checksum - avg_checksum).abs().cpu().item())
        adapter_sync_pass = max_adapter_delta <= float(config["adapter_sync_tolerance"])
        peak_mb = peak_memory_mb(device)
        totals = all_reduce_totals(
            initial_loss=initial_loss,
            final_loss=final_loss,
            validation_initial_loss=validation_initial_loss,
            validation_final_loss=validation_final_loss,
            peak_mb=0.0 if peak_mb is None else float(peak_mb),
            status_pass=True,
            adapter_sync_pass=adapter_sync_pass,
            device=device,
        )
        report.update(
            {
                "status": "PASS",
                "model_id": config["model_id"],
                "model_path": config["model_path"],
                "device": str(device),
                "dtype": str(torch_dtype),
                "download_attempted": False,
                "rank_allocation": rank_allocation,
                "target_module_limit": int(config["target_module_limit"]),
                "target_module_count": len(rank_allocation),
                "targeted_modules_total": targeted_modules_total,
                "trainable_adapter_params": trainable_parameter_count(ddp_model.module),
                "train_mode": config["train_mode"],
                "train_samples": TRAIN_SAMPLES if config["train_mode"] == "instruction" else [],
                "validation_samples": VALIDATION_SAMPLES if config["train_mode"] == "instruction" else [],
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "loss_delta": final_loss - initial_loss,
                "validation_initial_loss": validation_initial_loss,
                "validation_final_loss": validation_final_loss,
                "validation_loss_delta": (
                    None
                    if validation_initial_loss is None or validation_final_loss is None
                    else validation_final_loss - validation_initial_loss
                ),
                "before_generate": before_generate,
                "after_generate": after_generate,
                "losses": losses,
                "adapter_checksum": float(checksum.detach().cpu().item()),
                "max_adapter_delta": max_adapter_delta,
                "peak_mem_mb": peak_mb,
                "distributed_totals": totals,
            }
        )
    except Exception as exc:
        report.update({"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)[:2000]})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        rank_json.write_text(json.dumps(report, indent=2, sort_keys=True))


def fmt(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def aggregate_from_ranks(ranks: list[dict[str, Any]], *, adapter_sync_tolerance: float = 1e-5) -> dict[str, Any]:
    pass_count = sum(1 for row in ranks if row.get("status") == "PASS")
    adapter_sync_pass_count = sum(
        1 for row in ranks if row.get("status") == "PASS" and float(row.get("max_adapter_delta", 1.0)) <= adapter_sync_tolerance
    )
    totals = ranks[0].get("distributed_totals", {}) if ranks else {}
    reduced = {
        "initial_loss_sum": float(totals.get("initial_loss_sum", 0.0) or 0.0),
        "final_loss_sum": float(totals.get("final_loss_sum", 0.0) or 0.0),
        "validation_initial_loss_sum": float(totals.get("validation_initial_loss_sum", 0.0) or 0.0),
        "validation_final_loss_sum": float(totals.get("validation_final_loss_sum", 0.0) or 0.0),
        "peak_mem_mb_total": float(totals.get("peak_mem_mb_sum", 0.0) or 0.0),
        "pass_count": int(totals.get("pass_count", pass_count) or 0),
        "adapter_sync_pass_count": int(totals.get("adapter_sync_pass_count", adapter_sync_pass_count) or 0),
    }
    local_initial = sum(float(row.get("initial_loss", 0.0) or 0.0) for row in ranks)
    local_final = sum(float(row.get("final_loss", 0.0) or 0.0) for row in ranks)
    local_validation_initial = sum(float(row.get("validation_initial_loss", 0.0) or 0.0) for row in ranks)
    local_validation_final = sum(float(row.get("validation_final_loss", 0.0) or 0.0) for row in ranks)
    local_peak = sum(float(row.get("peak_mem_mb", 0.0) or 0.0) for row in ranks)
    reduced["distributed_reduce_consistent"] = (
        abs(reduced["initial_loss_sum"] - local_initial) < 1e-3
        and abs(reduced["final_loss_sum"] - local_final) < 1e-3
        and abs(reduced["validation_initial_loss_sum"] - local_validation_initial) < 1e-3
        and abs(reduced["validation_final_loss_sum"] - local_validation_final) < 1e-3
        and abs(reduced["peak_mem_mb_total"] - local_peak) < 1e-2
        and reduced["pass_count"] == pass_count
        and reduced["adapter_sync_pass_count"] == adapter_sync_pass_count
    )
    divisor = max(int(reduced["pass_count"]), 1)
    reduced["initial_loss_avg"] = round(reduced["initial_loss_sum"] / divisor, 6)
    reduced["final_loss_avg"] = round(reduced["final_loss_sum"] / divisor, 6)
    reduced["loss_delta_avg"] = round(reduced["final_loss_avg"] - reduced["initial_loss_avg"], 6)
    if ranks and any(row.get("validation_initial_loss") is not None for row in ranks):
        reduced["validation_initial_loss_avg"] = round(reduced["validation_initial_loss_sum"] / divisor, 6)
        reduced["validation_final_loss_avg"] = round(reduced["validation_final_loss_sum"] / divisor, 6)
        reduced["validation_loss_delta_avg"] = round(
            reduced["validation_final_loss_avg"] - reduced["validation_initial_loss_avg"],
            6,
        )
    reduced["adapter_sync_consistent"] = (
        bool(ranks)
        and reduced["distributed_reduce_consistent"]
        and reduced["pass_count"] == len(ranks)
        and reduced["adapter_sync_pass_count"] == len(ranks)
    )
    return reduced


def markdown_summary(summary: dict[str, Any]) -> str:
    aggregate = summary.get("aggregate", {})
    lines = [
        "# Multi-Card Qwen3 RankAdaptor LoRA Fine-Tune",
        "",
        f"- Overall status: `{summary.get('status')}`",
        f"- Method: `{summary.get('method')}`",
        f"- Model: `{summary.get('model_id')}`",
        f"- World size: `{summary.get('world_size')}`",
        f"- Backend: `{summary.get('backend')}`",
        f"- Device request: `{summary.get('device_request')}`",
        f"- Train mode: `{summary.get('train_mode')}`",
        f"- Target modules: `{summary.get('target_module_count')}` / `{summary.get('targeted_modules_total')}`",
        f"- all-reduce loss totals consistent: `{aggregate.get('distributed_reduce_consistent')}`",
        f"- adapter checksum synchronized: `{aggregate.get('adapter_sync_consistent')}`",
        "",
        "| Aggregate metric | Value |",
        "|---|---:|",
        f"| Initial loss avg | {fmt(aggregate.get('initial_loss_avg'))} |",
        f"| Final loss avg | {fmt(aggregate.get('final_loss_avg'))} |",
        f"| Loss delta avg | {fmt(aggregate.get('loss_delta_avg'))} |",
        f"| Validation initial loss avg | {fmt(aggregate.get('validation_initial_loss_avg'))} |",
        f"| Validation final loss avg | {fmt(aggregate.get('validation_final_loss_avg'))} |",
        f"| Validation loss delta avg | {fmt(aggregate.get('validation_loss_delta_avg'))} |",
        f"| Peak MB total | {fmt(aggregate.get('peak_mem_mb_total'))} |",
        f"| Passing ranks | {fmt(aggregate.get('pass_count'))} |",
        f"| Adapter sync pass ranks | {fmt(aggregate.get('adapter_sync_pass_count'))} |",
        "",
        "| Rank | Status | Device | Initial loss | Final loss | Adapter delta | Peak MB | Trainable params |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("ranks", []):
        lines.append(
            "| {rank} | {status} | {device} | {initial} | {final} | {delta} | {peak} | {params} |".format(
                rank=row.get("rank", ""),
                status=row.get("status", ""),
                device=row.get("device", ""),
                initial=fmt(row.get("initial_loss")),
                final=fmt(row.get("final_loss")),
                delta=fmt(row.get("max_adapter_delta")),
                peak=fmt(row.get("peak_mem_mb")),
                params=fmt(row.get("trainable_adapter_params")),
            )
        )
    lines.extend(
        [
            "",
            "This benchmark trains RankAdaptor-selected LoRA adapters through DistributedDataParallel and uses all-reduce to summarize loss, peak memory, and adapter synchronization across the selected cards.",
            "",
            "## Instruction Evaluation",
            "",
            f"- Prompt: `{(summary.get('validation_samples') or [{}])[0].get('prompt', 'missing')}`",
            f"- Target: `{(summary.get('validation_samples') or [{}])[0].get('target', 'missing')}`",
            f"- Before: `{summary.get('before_generate') or 'missing'}`",
            f"- After: `{summary.get('after_generate') or 'missing'}`",
            "",
            "## Rank Allocation",
            "",
            "```json",
            json.dumps(summary.get("rank_allocation", {}), indent=2, sort_keys=True),
            "```",
        ]
    )
    if summary.get("spawn_error"):
        lines.extend(["", f"Spawn error: `{summary['spawn_error']}`"])
    return "\n".join(lines) + "\n"


def write_summary(
    *,
    artifacts_dir: Path,
    reports_dir: Path,
    run_label: str,
    spawn_error: str | None,
) -> dict[str, Any]:
    ranks = [
        json.loads(path.read_text())
        for path in sorted(artifacts_dir.glob(f"multicard_qwen_lora_finetune_{run_label}_rank*.json"))
    ]
    world_size = ranks[0].get("world_size", len(ranks)) if ranks else 0
    backend = ranks[0].get("backend", "") if ranks else ""
    device_request = ranks[0].get("device_request", "") if ranks else ""
    aggregate = aggregate_from_ranks(ranks)
    status = (
        "PASS"
        if ranks
        and all(row.get("status") == "PASS" for row in ranks)
        and aggregate.get("adapter_sync_consistent")
        and not spawn_error
        else "FAIL"
    )
    first = ranks[0] if ranks else {}
    summary = {
        "status": status,
        "method": "rankadaptor_lora",
        "model_id": first.get("model_id", "Qwen/Qwen3-0.6B"),
        "model_path": first.get("model_path", ""),
        "world_size": world_size,
        "backend": backend,
        "device_request": device_request,
        "target_module_limit": first.get("target_module_limit", 0),
        "target_module_count": first.get("target_module_count", 0),
        "targeted_modules_total": first.get("targeted_modules_total", 0),
        "rank_allocation": first.get("rank_allocation", {}),
        "trainable_adapter_params": first.get("trainable_adapter_params", 0),
        "train_mode": first.get("train_mode", "random"),
        "train_samples": first.get("train_samples", []),
        "validation_samples": first.get("validation_samples", []),
        "before_generate": first.get("before_generate"),
        "after_generate": first.get("after_generate"),
        "aggregate": aggregate,
        "ranks": ranks,
        "spawn_error": spawn_error,
    }
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / f"multicard_qwen_lora_finetune_{run_label}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    (reports_dir / f"multicard-qwen-lora-finetune-{run_label}.md").write_text(markdown_summary(summary))
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3 RankAdaptor LoRA fine-tune on multiple cards")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-path", default="/mnt/nvme/622/models/Qwen3-0.6B")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--cards", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--backend", default="")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--max-rank", type=int, default=2)
    parser.add_argument("--target-module-limit", type=int, default=2)
    parser.add_argument("--train-mode", choices=["random", "instruction"], default="random")
    parser.add_argument("--instruction-max-length", type=int, default=96)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-label", default="qwen3_06b_lora_sync_npu")
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--adapter-sync-tolerance", type=float, default=1e-5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cards = parse_cards(args.cards)
    backend = args.backend or default_backend(args.device)
    demo_root = Path(args.demo_root)
    artifacts_dir = demo_root / "artifacts"
    reports_dir = demo_root / "reports"
    for directory in (artifacts_dir, reports_dir, demo_root / "logs"):
        directory.mkdir(parents=True, exist_ok=True)
    for old in artifacts_dir.glob(f"multicard_qwen_lora_finetune_{args.run_label}_rank*.json"):
        old.unlink()

    config = {
        "world_size": len(cards),
        "cards": cards,
        "backend": backend,
        "device_name": args.device,
        "dtype": args.dtype,
        "model_id": args.model_id,
        "model_path": args.model_path,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "steps": args.steps,
        "budget": args.budget,
        "max_rank": args.max_rank,
        "target_module_limit": args.target_module_limit,
        "train_mode": args.train_mode,
        "instruction_max_length": args.instruction_max_length,
        "max_new_tokens": args.max_new_tokens,
        "lr": args.lr,
        "seed": args.seed,
        "artifacts_dir": str(artifacts_dir),
        "master_addr": args.master_addr,
        "master_port": args.master_port or free_port(),
        "timeout_seconds": args.timeout_seconds,
        "run_label": args.run_label,
        "adapter_sync_tolerance": args.adapter_sync_tolerance,
    }
    spawn_error = None
    try:
        mp.spawn(train_one_rank, args=(config,), nprocs=len(cards), join=True)
    except Exception as exc:
        spawn_error = f"{type(exc).__name__}: {exc}"
    summary = write_summary(
        artifacts_dir=artifacts_dir,
        reports_dir=reports_dir,
        run_label=args.run_label,
        spawn_error=spawn_error,
    )
    print("MULTICARD_QWEN_LORA_FINETUNE " + json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
