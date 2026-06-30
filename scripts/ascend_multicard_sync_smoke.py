#!/usr/bin/env python
"""Run a tiny distributed sync smoke for Ascend multi-card training demos."""
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


def tensor_checksum(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().sum().cpu().item())


def seeded_batch(*, batch_size: int, hidden_size: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(batch_size, hidden_size, generator=generator).to(device)


class TinySyncModel(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "npu":
        torch.npu.synchronize()


def train_one_rank(
    rank: int,
    *,
    world_size: int,
    cards: list[int],
    backend: str,
    device_name: str,
    hidden_size: int,
    batch_size: int,
    steps: int,
    seed: int,
    artifacts_dir: str,
    master_addr: str,
    master_port: int,
    timeout_seconds: int,
) -> None:
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    rank_json = artifacts / f"multicard_sync_rank{rank}.json"
    report: dict[str, Any] = {
        "status": "STARTED",
        "rank": rank,
        "world_size": world_size,
        "backend": backend,
        "device_request": device_name,
        "card": cards[rank] if rank < len(cards) else None,
        "seed": seed,
        "steps": steps,
    }
    try:
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        if device_name.lower().startswith("npu"):
            apply_hccl_port_defaults(device_name)
            os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(str(card) for card in cards)
            os.environ.setdefault("HCCL_CONNECT_TIMEOUT", str(max(timeout_seconds, 120)))

        device = set_device(device_name, rank, cards)
        torch.manual_seed(seed)
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=timeout_seconds),
        )
        model = TinySyncModel(hidden_size).to(device)
        ddp_model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=None if device.type == "cpu" else [device.index],
        )
        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

        losses: list[float] = []
        for step in range(steps):
            x = seeded_batch(batch_size=batch_size, hidden_size=hidden_size, seed=seed + step, device=device)
            target = torch.zeros_like(x)
            optimizer.zero_grad(set_to_none=True)
            y = ddp_model(x)
            loss = torch.nn.functional.mse_loss(y, target)
            loss.backward()
            optimizer.step()
            synchronize(device)
            losses.append(float(loss.detach().cpu().item()))

        with torch.no_grad():
            param_vector = torch.cat([param.detach().flatten().float() for param in ddp_model.module.parameters()])
            param_sum = param_vector.sum()
            avg_param_sum = param_sum.detach().clone()
            dist.all_reduce(avg_param_sum, op=dist.ReduceOp.SUM)
            avg_param_sum /= world_size
            max_param_delta = (param_sum - avg_param_sum).abs()

            grad_sum = torch.zeros((), device=device, dtype=torch.float32)
            for param in ddp_model.module.parameters():
                if param.grad is not None:
                    grad_sum = grad_sum + param.grad.detach().float().sum()
            avg_grad_sum = grad_sum.detach().clone()
            dist.all_reduce(avg_grad_sum, op=dist.ReduceOp.SUM)
            avg_grad_sum /= world_size
            max_grad_delta = (grad_sum - avg_grad_sum).abs()

        report.update(
            {
                "status": "PASS",
                "device": str(device),
                "loss": losses[-1] if losses else None,
                "losses": losses,
                "grad_checksum": tensor_checksum(grad_sum),
                "param_checksum": tensor_checksum(param_sum),
                "max_grad_delta": float(max_grad_delta.cpu().item()),
                "max_param_delta": float(max_param_delta.cpu().item()),
            }
        )
    except Exception as exc:
        report.update({"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        rank_json.write_text(json.dumps(report, indent=2, sort_keys=True))


def train_one_rank_from_config(rank: int, config: dict[str, Any]) -> None:
    train_one_rank(rank, **config)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Multi-Card Sync Smoke Summary",
        "",
        f"- Overall status: `{summary['status']}`",
        f"- World size: `{summary.get('world_size', 0)}`",
        f"- Backend: `{summary.get('backend', '')}`",
        f"- Device request: `{summary.get('device_request', '')}`",
        "",
        "| Rank | Status | Device | Loss | Param delta | Grad delta |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for row in summary["ranks"]:
        lines.append(
            "| {rank} | {status} | {device} | {loss} | {param_delta} | {grad_delta} |".format(
                rank=row.get("rank", ""),
                status=row.get("status", ""),
                device=row.get("device", ""),
                loss=fmt(row.get("loss")),
                param_delta=fmt(row.get("max_param_delta")),
                grad_delta=fmt(row.get("max_grad_delta")),
            )
        )
    lines.extend(
        [
            "",
            "This smoke proves the demo environment can launch one distributed process per card and synchronize a training step.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_summary(*, artifacts_dir: Path, reports_dir: Path) -> dict[str, Any]:
    ranks = [json.loads(path.read_text()) for path in sorted(artifacts_dir.glob("multicard_sync_rank*.json"))]
    world_size = ranks[0].get("world_size", len(ranks)) if ranks else 0
    backend = ranks[0].get("backend", "") if ranks else ""
    device_request = ranks[0].get("device_request", "") if ranks else ""
    summary = {
        "status": "PASS" if ranks and all(row.get("status") == "PASS" for row in ranks) else "FAIL",
        "world_size": world_size,
        "backend": backend,
        "device_request": device_request,
        "ranks": ranks,
    }
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "multicard_sync_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (reports_dir / "multicard-sync-summary.md").write_text(markdown_summary(summary))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Ascend multi-card distributed sync smoke")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--cards", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--backend", default="")
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1)
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

    for old in artifacts_dir.glob("multicard_sync_rank*.json"):
        old.unlink()

    master_port = args.master_port or free_port()
    kwargs = {
        "world_size": len(cards),
        "cards": cards,
        "backend": backend,
        "device_name": args.device,
        "hidden_size": args.hidden_size,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "seed": args.seed,
        "artifacts_dir": str(artifacts_dir),
        "master_addr": args.master_addr,
        "master_port": master_port,
        "timeout_seconds": args.timeout_seconds,
    }
    mp.spawn(train_one_rank_from_config, args=(kwargs,), nprocs=len(cards), join=True)
    summary = write_summary(artifacts_dir=artifacts_dir, reports_dir=reports_dir)
    print("ASCEND_MULTICARD_SYNC_SMOKE " + json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
