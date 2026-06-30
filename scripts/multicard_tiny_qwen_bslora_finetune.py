#!/usr/bin/env python
"""Run TinyQwen BSLoRA-style shared LoRA fine-tuning on synchronized cards."""
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

from ascend_inference_benchmark import synchronize_device
from ascend_multicard_sync_smoke import apply_hccl_port_defaults, default_backend, parse_cards
from tidal.device import resolve_dtype
from tiny_qwen_bslora_finetune import (
    adapter_checksum,
    apply_shared_lora_adapters,
    evaluate_loss,
    load_model,
    load_tokenizer,
    qwen_shared_lora_allocation,
    seeded_token_batch,
)


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


def train_ddp_shared_lora(
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


def all_reduce_totals(
    *,
    initial_loss: float,
    final_loss: float,
    status_pass: bool,
    adapter_sync_pass: bool,
    device: torch.device,
) -> dict[str, Any]:
    values = torch.tensor(
        [
            float(initial_loss),
            float(final_loss),
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
        "pass_count": int(round(values_cpu[2])),
        "adapter_sync_pass_count": int(round(values_cpu[3])),
    }


def train_one_rank(rank: int, config: dict[str, Any]) -> None:
    artifacts = Path(config["artifacts_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)
    rank_json = artifacts / f"multicard_tiny_qwen_bslora_finetune_{config['run_label']}_rank{rank}.json"
    report: dict[str, Any] = {
        "status": "STARTED",
        "rank": rank,
        "world_size": config["world_size"],
        "backend": config["backend"],
        "device_request": config["device_name"],
        "card": config["cards"][rank] if rank < len(config["cards"]) else None,
        "method": "bslora_shared_lora",
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
        allocation = qwen_shared_lora_allocation(model, rank=int(config["adapter_rank"]))
        metadata = apply_shared_lora_adapters(model, allocation, alpha=config["alpha"])
        model = model.to(device)
        ddp_model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=None if device.type == "cpu" else [device.index],
        )

        eval_tokens = seeded_token_batch(
            tokenizer=tokenizer,
            batch_size=int(config["batch_size"]),
            seq_len=int(config["seq_len"]),
            seed=int(config["seed"]),
            device=device,
        )
        initial_loss = evaluate_loss(ddp_model.module, eval_tokens)
        losses = train_ddp_shared_lora(
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
        checksum = adapter_checksum(ddp_model.module, device)
        avg_checksum = checksum.detach().clone()
        dist.all_reduce(avg_checksum, op=dist.ReduceOp.SUM)
        avg_checksum /= int(config["world_size"])
        max_adapter_delta = float((checksum - avg_checksum).abs().cpu().item())
        adapter_sync_pass = max_adapter_delta <= float(config["adapter_sync_tolerance"])
        totals = all_reduce_totals(
            initial_loss=initial_loss,
            final_loss=final_loss,
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
                "rank_config": int(config["adapter_rank"]),
                "alpha": config["alpha"],
                "shared_groups": metadata["shared_groups"],
                "target_module_count": metadata["target_module_count"],
                "unique_adapter_tensors": metadata["unique_adapter_tensors"],
                "trainable_adapter_params": metadata["trainable_adapter_params"],
                "unshared_adapter_params": metadata["unshared_adapter_params"],
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "loss_delta": final_loss - initial_loss,
                "losses": losses,
                "adapter_checksum": float(checksum.detach().cpu().item()),
                "max_adapter_delta": max_adapter_delta,
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
        "pass_count": int(totals.get("pass_count", pass_count) or 0),
        "adapter_sync_pass_count": int(totals.get("adapter_sync_pass_count", adapter_sync_pass_count) or 0),
    }
    local_initial = sum(float(row.get("initial_loss", 0.0) or 0.0) for row in ranks)
    local_final = sum(float(row.get("final_loss", 0.0) or 0.0) for row in ranks)
    reduced["distributed_reduce_consistent"] = (
        abs(reduced["initial_loss_sum"] - local_initial) < 1e-3
        and abs(reduced["final_loss_sum"] - local_final) < 1e-3
        and reduced["pass_count"] == pass_count
        and reduced["adapter_sync_pass_count"] == adapter_sync_pass_count
    )
    divisor = max(int(reduced["pass_count"]), 1)
    reduced["initial_loss_avg"] = round(reduced["initial_loss_sum"] / divisor, 6)
    reduced["final_loss_avg"] = round(reduced["final_loss_sum"] / divisor, 6)
    reduced["loss_delta_avg"] = round(reduced["final_loss_avg"] - reduced["initial_loss_avg"], 6)
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
        "# Multi-Card TinyQwen BSLoRA Fine-Tune",
        "",
        f"- Overall status: `{summary.get('status')}`",
        f"- World size: `{summary.get('world_size')}`",
        f"- Backend: `{summary.get('backend')}`",
        f"- Device request: `{summary.get('device_request')}`",
        f"- all-reduce loss totals consistent: `{aggregate.get('distributed_reduce_consistent')}`",
        f"- adapter checksum synchronized: `{aggregate.get('adapter_sync_consistent')}`",
        "",
        "| Aggregate metric | Value |",
        "|---|---:|",
        f"| Initial loss avg | {fmt(aggregate.get('initial_loss_avg'))} |",
        f"| Final loss avg | {fmt(aggregate.get('final_loss_avg'))} |",
        f"| Loss delta avg | {fmt(aggregate.get('loss_delta_avg'))} |",
        f"| Passing ranks | {fmt(aggregate.get('pass_count'))} |",
        f"| Adapter sync pass ranks | {fmt(aggregate.get('adapter_sync_pass_count'))} |",
        "",
        "| Rank | Status | Device | Initial loss | Final loss | Adapter delta | Trainable params | Shared tensors |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("ranks", []):
        lines.append(
            "| {rank} | {status} | {device} | {initial} | {final} | {delta} | {params} | {tensors} |".format(
                rank=row.get("rank", ""),
                status=row.get("status", ""),
                device=row.get("device", ""),
                initial=fmt(row.get("initial_loss")),
                final=fmt(row.get("final_loss")),
                delta=fmt(row.get("max_adapter_delta")),
                params=fmt(row.get("trainable_adapter_params")),
                tensors=fmt(row.get("unique_adapter_tensors")),
            )
        )
    lines.extend(
        [
            "",
            "This benchmark trains the TinyQwen shared LoRA adapters through DistributedDataParallel and uses all-reduce to summarize loss and adapter synchronization across the selected cards.",
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
        for path in sorted(artifacts_dir.glob(f"multicard_tiny_qwen_bslora_finetune_{run_label}_rank*.json"))
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
    summary = {
        "status": status,
        "method": "bslora_shared_lora",
        "world_size": world_size,
        "backend": backend,
        "device_request": device_request,
        "aggregate": aggregate,
        "shared_groups": ranks[0].get("shared_groups", {}) if ranks else {},
        "trainable_adapter_params": ranks[0].get("trainable_adapter_params", 0) if ranks else 0,
        "unshared_adapter_params": ranks[0].get("unshared_adapter_params", 0) if ranks else 0,
        "ranks": ranks,
        "spawn_error": spawn_error,
    }
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / f"multicard_tiny_qwen_bslora_finetune_{run_label}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    (reports_dir / f"multicard-tiny-qwen-bslora-finetune-{run_label}.md").write_text(
        markdown_summary(summary)
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TinyQwen BSLoRA shared LoRA fine-tune on multiple cards")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--model-id", default="TinyQwen3-Offline")
    parser.add_argument("--model-path", default="/mnt/nvme/622/models/TinyQwen3-Offline")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--cards", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--backend", default="")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-label", default="tiny_qwen3_bslora_sync_npu")
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=300)
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
    for old in artifacts_dir.glob(f"multicard_tiny_qwen_bslora_finetune_{args.run_label}_rank*.json"):
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
        "adapter_rank": args.rank,
        "alpha": args.alpha,
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
    print("MULTICARD_TINY_QWEN_BSLORA_FINETUNE " + json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
