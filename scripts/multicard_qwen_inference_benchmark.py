#!/usr/bin/env python
"""Run a synchronized torch inference benchmark for Qwen-family models on multiple cards."""
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

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from ascend_inference_benchmark import run_torch_backend
from ascend_multicard_sync_smoke import apply_hccl_port_defaults, default_backend, parse_cards, set_device
from tidal.device import resolve_dtype


DEFAULT_PROMPTS = (
    "Explain low-rank model compression in one concise paragraph.",
    "Summarize why Ascend NPU inference benchmarking needs synchronization.",
)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def metric(report: dict[str, Any], name: str) -> float:
    value = report.get(name)
    if value is None:
        return 0.0
    return float(value)


def int_metric(report: dict[str, Any], name: str) -> int:
    value = report.get(name)
    if value is None:
        return 0
    return int(value)


def all_reduce_totals(report: dict[str, Any], device: torch.device) -> dict[str, Any]:
    values = torch.tensor(
        [
            metric(report, "tokens_per_s"),
            float(int_metric(report, "generated_tokens")),
            metric(report, "peak_mem_mb"),
            1.0 if report.get("status") == "PASS" else 0.0,
        ],
        device=device,
        dtype=torch.float32,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values_cpu = values.detach().cpu().tolist()
    return {
        "tokens_per_s": float(values_cpu[0]),
        "generated_tokens": int(round(values_cpu[1])),
        "peak_mem_mb": float(values_cpu[2]),
        "pass_count": int(round(values_cpu[3])),
    }


def rank_prompts(base_prompts: Sequence[str], rank: int) -> list[str]:
    return [f"[rank {rank}] {prompt}" for prompt in base_prompts]


def run_one_rank(rank: int, config: dict[str, Any]) -> None:
    artifacts = Path(config["artifacts_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)
    rank_json = artifacts / f"multicard_qwen_inference_{config['run_label']}_rank{rank}.json"
    report: dict[str, Any] = {
        "status": "STARTED",
        "rank": rank,
        "world_size": config["world_size"],
        "backend": config["backend"],
        "device_request": config["device_name"],
        "card": config["cards"][rank] if rank < len(config["cards"]) else None,
        "model_id": config["model_id"],
        "model_path": config["model_path"],
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
        torch_dtype = dtype if isinstance(dtype, torch.dtype) else torch.float16
        dist.init_process_group(
            backend=config["backend"],
            rank=rank,
            world_size=config["world_size"],
            timeout=timedelta(seconds=int(config["timeout_seconds"])),
        )
        dist.barrier()
        benchmark = run_torch_backend(
            model_path=Path(config["model_path"]),
            model_id=config["model_id"],
            device=device,
            dtype=torch_dtype,
            prompts=rank_prompts(config["prompts"], rank),
            max_new_tokens=int(config["max_new_tokens"]),
            iters=int(config["iters"]),
            warmup=int(config["warmup"]),
        )
        dist.barrier()
        totals = all_reduce_totals(benchmark, device)
        dist.barrier()
        report.update(
            {
                "status": "PASS" if benchmark.get("status") == "PASS" else "FAIL",
                "device": str(device),
                "dtype": str(torch_dtype),
                "benchmark": benchmark,
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


def aggregate_from_ranks(ranks: list[dict[str, Any]]) -> dict[str, Any]:
    pass_count = sum(1 for row in ranks if row.get("status") == "PASS")
    totals = ranks[0].get("distributed_totals", {}) if ranks else {}
    reduced = {
        "tokens_per_s_total": float(totals.get("tokens_per_s", 0.0) or 0.0),
        "generated_tokens_total": int(totals.get("generated_tokens", 0) or 0),
        "peak_mem_mb_total": float(totals.get("peak_mem_mb", 0.0) or 0.0),
        "pass_count": int(totals.get("pass_count", pass_count) or 0),
    }
    local = {
        "tokens_per_s_total": sum(metric(row.get("benchmark", {}), "tokens_per_s") for row in ranks),
        "generated_tokens_total": sum(int_metric(row.get("benchmark", {}), "generated_tokens") for row in ranks),
        "peak_mem_mb_total": sum(metric(row.get("benchmark", {}), "peak_mem_mb") for row in ranks),
        "pass_count": pass_count,
    }
    reduced["distributed_reduce_consistent"] = (
        abs(float(reduced["tokens_per_s_total"]) - float(local["tokens_per_s_total"])) < 1e-3
        and int(reduced["generated_tokens_total"]) == int(local["generated_tokens_total"])
        and abs(float(reduced["peak_mem_mb_total"]) - float(local["peak_mem_mb_total"])) < 1e-3
        and int(reduced["pass_count"]) == int(local["pass_count"])
    )
    return reduced


def markdown_summary(summary: dict[str, Any]) -> str:
    aggregate = summary.get("aggregate", {})
    lines = [
        "# Multi-Card Qwen Inference Benchmark",
        "",
        f"- Overall status: `{summary.get('status')}`",
        f"- Model: `{summary.get('model_id')}`",
        f"- Model path: `{summary.get('model_path')}`",
        f"- World size: `{summary.get('world_size')}`",
        f"- Backend: `{summary.get('backend')}`",
        f"- Device request: `{summary.get('device_request')}`",
        f"- all-reduce totals consistent: `{aggregate.get('distributed_reduce_consistent')}`",
        "",
        "| Aggregate metric | Value |",
        "|---|---:|",
        f"| Tokens/s total | {fmt(aggregate.get('tokens_per_s_total'))} |",
        f"| Generated tokens total | {fmt(aggregate.get('generated_tokens_total'))} |",
        f"| Peak MB total | {fmt(aggregate.get('peak_mem_mb_total'))} |",
        f"| Passing ranks | {fmt(aggregate.get('pass_count'))} |",
        "",
        "| Rank | Status | Device | Latency ms | Tokens/s | Generated tokens | Peak MB |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in summary.get("ranks", []):
        benchmark = row.get("benchmark", {}) if isinstance(row.get("benchmark"), dict) else {}
        lines.append(
            "| {rank} | {status} | {device} | {latency} | {tps} | {tokens} | {peak} |".format(
                rank=row.get("rank", ""),
                status=row.get("status", ""),
                device=row.get("device", ""),
                latency=fmt(benchmark.get("latency_ms")),
                tps=fmt(benchmark.get("tokens_per_s")),
                tokens=fmt(benchmark.get("generated_tokens")),
                peak=fmt(benchmark.get("peak_mem_mb")),
            )
        )
    lines.extend(
        [
            "",
            "This benchmark proves the Qwen-family inference path can be launched on every selected card, synchronized with the distributed backend, and summarized via all-reduce, not extrapolated from one card.",
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
        for path in sorted(artifacts_dir.glob(f"multicard_qwen_inference_{run_label}_rank*.json"))
    ]
    world_size = ranks[0].get("world_size", len(ranks)) if ranks else 0
    backend = ranks[0].get("backend", "") if ranks else ""
    device_request = ranks[0].get("device_request", "") if ranks else ""
    model_id = ranks[0].get("model_id", "") if ranks else ""
    model_path = ranks[0].get("model_path", "") if ranks else ""
    status = "PASS" if ranks and all(row.get("status") == "PASS" for row in ranks) and not spawn_error else "FAIL"
    summary = {
        "status": status,
        "model_id": model_id,
        "model_path": model_path,
        "world_size": world_size,
        "backend": backend,
        "device_request": device_request,
        "aggregate": aggregate_from_ranks(ranks),
        "ranks": ranks,
        "spawn_error": spawn_error,
    }
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / f"multicard_qwen_inference_{run_label}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    (reports_dir / f"multicard-qwen-inference-{run_label}.md").write_text(markdown_summary(summary))
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synchronized Qwen-family torch inference on multiple cards")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-path", default="/mnt/nvme/622/models/Qwen3-0.6B")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--cards", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--backend", default="")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--run-label", default="qwen3_06b_npu")
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=420)
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
    for old in artifacts_dir.glob(f"multicard_qwen_inference_{args.run_label}_rank*.json"):
        old.unlink()

    config = {
        "world_size": len(cards),
        "cards": cards,
        "backend": backend,
        "device_name": args.device,
        "dtype": args.dtype,
        "artifacts_dir": str(artifacts_dir),
        "master_addr": args.master_addr,
        "master_port": args.master_port or free_port(),
        "timeout_seconds": args.timeout_seconds,
        "run_label": args.run_label,
        "model_id": args.model_id,
        "model_path": args.model_path,
        "max_new_tokens": args.max_new_tokens,
        "iters": args.iters,
        "warmup": args.warmup,
        "prompts": list(DEFAULT_PROMPTS),
    }
    spawn_error = None
    try:
        mp.spawn(run_one_rank, args=(config,), nprocs=len(cards), join=True)
    except Exception as exc:
        spawn_error = f"{type(exc).__name__}: {exc}"
    summary = write_summary(
        artifacts_dir=artifacts_dir,
        reports_dir=reports_dir,
        run_label=args.run_label,
        spawn_error=spawn_error,
    )
    print("MULTICARD_QWEN_INFERENCE " + json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
