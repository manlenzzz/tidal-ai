#!/usr/bin/env python
"""Run a tiny RankAdaptor-driven LoRA distributed training smoke."""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from tidal.methods.rankadaptor.core import collect_linear_profiles, search_rank_allocation


def parse_cards(raw: str) -> list[int]:
    cards = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not cards:
        raise ValueError("at least one card is required")
    return cards


def default_backend(device: str) -> str:
    return "hccl" if str(device).lower().startswith("npu") else "gloo"


def apply_hccl_port_defaults(device: str) -> None:
    if str(device).lower().startswith("npu"):
        os.environ.setdefault("HCCL_HOST_SOCKET_PORT_RANGE", "auto")
        os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def set_device(device_name: str, local_rank: int, cards: list[int]) -> torch.device:
    normalized = device_name.lower()
    if normalized.startswith("npu"):
        import torch_npu  # noqa: F401

        card = cards[local_rank]
        torch.npu.set_device(card)
        return torch.device(f"npu:{card}")
    if normalized.startswith("cuda"):
        card = cards[local_rank]
        torch.cuda.set_device(card)
        return torch.device(f"cuda:{card}")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "npu":
        torch.npu.synchronize()


def tensor_checksum(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().sum().cpu().item())


def seeded_tokens(*, batch_size: int, seq_len: int, vocab_size: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, vocab_size, (batch_size, seq_len), generator=generator).to(device)


class TinyLoraCausalLM(nn.Module):
    def __init__(self, *, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.down_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        attn = self.o_proj(torch.tanh(self.q_proj(x) + self.k_proj(x) + self.v_proj(x)))
        hidden = self.down_proj(torch.relu(self.gate_proj(attn)))
        return self.lm_head(hidden)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, *, rank: int, alpha: int = 2) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        self.rank = int(rank)
        self.scaling = float(alpha * rank) / float(rank)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        adapted = torch.nn.functional.linear(torch.nn.functional.linear(x, self.lora_a), self.lora_b)
        return base_out + adapted * self.scaling


def rankadaptor_allocation(model: nn.Module, *, budget: int, max_rank: int) -> dict[str, int]:
    profiles = collect_linear_profiles(
        model,
        min_rank=1,
        max_rank=max_rank,
        rank_step=1,
        name_filter=lambda name: name.endswith("_proj"),
    )
    result = search_rank_allocation(profiles, budget=budget)
    return {name: rank for name, rank in result.config.items() if rank > 0}


def replace_module(root: nn.Module, name: str, module: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def apply_lora_adapters(model: nn.Module, allocation: dict[str, int]) -> nn.Module:
    for param in model.parameters():
        param.requires_grad = False
    modules = dict(model.named_modules())
    for name, rank in allocation.items():
        target = modules[name]
        if not isinstance(target, nn.Linear):
            raise TypeError(f"{name} is not a Linear module")
        replace_module(model, name, LoRALinear(target, rank=rank))
    return model


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def adapter_checksum(model: nn.Module, device: torch.device) -> torch.Tensor:
    checksum = torch.zeros((), device=device, dtype=torch.float32)
    for name, param in model.named_parameters():
        if "lora_" in name:
            checksum = checksum + param.detach().float().sum()
    return checksum


def train_one_rank(rank: int, config: dict[str, Any]) -> None:
    artifacts = Path(config["artifacts_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)
    rank_json = artifacts / f"rankadaptor_lora_sync_rank{rank}.json"
    report: dict[str, Any] = {
        "status": "STARTED",
        "rank": rank,
        "world_size": config["world_size"],
        "backend": config["backend"],
        "device_request": config["device_name"],
        "card": config["cards"][rank],
        "steps": config["steps"],
        "seed": config["seed"],
    }
    try:
        os.environ["MASTER_ADDR"] = config["master_addr"]
        os.environ["MASTER_PORT"] = str(config["master_port"])
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(config["world_size"])
        if config["device_name"].lower().startswith("npu"):
            apply_hccl_port_defaults(config["device_name"])
            os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(str(card) for card in config["cards"])
            os.environ.setdefault("HCCL_CONNECT_TIMEOUT", str(max(config["timeout_seconds"], 120)))

        device = set_device(config["device_name"], rank, config["cards"])
        torch.manual_seed(config["seed"])
        dist.init_process_group(
            backend=config["backend"],
            rank=rank,
            world_size=config["world_size"],
            timeout=timedelta(seconds=config["timeout_seconds"]),
        )

        model = TinyLoraCausalLM(vocab_size=config["vocab_size"], hidden_size=config["hidden_size"])
        allocation = rankadaptor_allocation(model, budget=config["budget"], max_rank=config["max_rank"])
        apply_lora_adapters(model, allocation)
        model = model.to(device)
        ddp_model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=None if device.type == "cpu" else [device.index],
        )
        optimizer = torch.optim.AdamW((p for p in ddp_model.parameters() if p.requires_grad), lr=config["lr"])

        losses: list[float] = []
        for step in range(config["steps"]):
            tokens = seeded_tokens(
                batch_size=config["batch_size"],
                seq_len=config["seq_len"],
                vocab_size=config["vocab_size"],
                seed=config["seed"] + step,
                device=device,
            )
            targets = torch.roll(tokens, shifts=-1, dims=1)
            optimizer.zero_grad(set_to_none=True)
            logits = ddp_model(tokens)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, config["vocab_size"]), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            synchronize(device)
            losses.append(float(loss.detach().cpu().item()))

        checksum = adapter_checksum(ddp_model.module, device)
        avg_checksum = checksum.detach().clone()
        dist.all_reduce(avg_checksum, op=dist.ReduceOp.SUM)
        avg_checksum /= config["world_size"]
        max_adapter_delta = (checksum - avg_checksum).abs()
        report.update(
            {
                "status": "PASS",
                "device": str(device),
                "rank_allocation": allocation,
                "trainable_adapter_params": trainable_parameter_count(ddp_model.module),
                "losses": losses,
                "initial_loss": losses[0] if losses else None,
                "final_loss": losses[-1] if losses else None,
                "adapter_checksum": tensor_checksum(checksum),
                "max_adapter_delta": float(max_adapter_delta.cpu().item()),
            }
        )
    except Exception as exc:
        report.update({"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        rank_json.write_text(json.dumps(report, indent=2, sort_keys=True))


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# RankAdaptor LoRA Sync Smoke Summary",
        "",
        f"- Overall status: `{summary['status']}`",
        f"- World size: `{summary.get('world_size', 0)}`",
        f"- Backend: `{summary.get('backend', '')}`",
        f"- Trainable adapter params: `{summary.get('trainable_adapter_params', 0)}`",
        "",
        "| Rank | Status | Device | Initial loss | Final loss | Adapter delta |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for row in summary["ranks"]:
        lines.append(
            "| {rank} | {status} | {device} | {initial} | {final} | {delta} |".format(
                rank=row.get("rank", ""),
                status=row.get("status", ""),
                device=row.get("device", ""),
                initial=fmt(row.get("initial_loss")),
                final=fmt(row.get("final_loss")),
                delta=fmt(row.get("max_adapter_delta")),
            )
        )
    lines.append("")
    lines.append("This smoke uses RankAdaptor rank allocation and trains only local LoRA adapter matrices.")
    return "\n".join(lines) + "\n"


def write_summary(*, artifacts_dir: Path, reports_dir: Path) -> dict[str, Any]:
    ranks = [json.loads(path.read_text()) for path in sorted(artifacts_dir.glob("rankadaptor_lora_sync_rank*.json"))]
    first = ranks[0] if ranks else {}
    summary = {
        "status": "PASS" if ranks and all(row.get("status") == "PASS" for row in ranks) else "FAIL",
        "world_size": first.get("world_size", len(ranks)),
        "backend": first.get("backend", ""),
        "device_request": first.get("device_request", ""),
        "trainable_adapter_params": first.get("trainable_adapter_params", 0),
        "rank_allocation": first.get("rank_allocation", {}),
        "ranks": ranks,
    }
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "rankadaptor_lora_sync_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (reports_dir / "rankadaptor-lora-sync-summary.md").write_text(markdown_summary(summary))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="RankAdaptor LoRA distributed training smoke")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--cards", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--backend", default="")
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--max-rank", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()

    cards = parse_cards(args.cards)
    backend = args.backend or default_backend(args.device)
    demo_root = Path(args.demo_root)
    artifacts_dir = demo_root / "artifacts"
    reports_dir = demo_root / "reports"
    logs_dir = demo_root / "logs"
    for directory in (artifacts_dir, reports_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for old in artifacts_dir.glob("rankadaptor_lora_sync_rank*.json"):
        old.unlink()

    config = {
        "world_size": len(cards),
        "cards": cards,
        "backend": backend,
        "device_name": args.device,
        "vocab_size": args.vocab_size,
        "hidden_size": args.hidden_size,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "steps": args.steps,
        "budget": args.budget,
        "max_rank": args.max_rank,
        "lr": args.lr,
        "seed": args.seed,
        "artifacts_dir": str(artifacts_dir),
        "master_addr": args.master_addr,
        "master_port": args.master_port or free_port(),
        "timeout_seconds": args.timeout_seconds,
    }
    mp.spawn(train_one_rank, args=(config,), nprocs=len(cards), join=True)
    summary = write_summary(artifacts_dir=artifacts_dir, reports_dir=reports_dir)
    print("RANKADAPTOR_LORA_SYNC_SMOKE " + json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
