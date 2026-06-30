#!/usr/bin/env python
"""Run TinyQwen compressed generate benchmark on multiple synchronized cards."""
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

from ascend_multicard_sync_smoke import apply_hccl_port_defaults, default_backend, parse_cards
from tiny_qwen_compression_generate_benchmark import run_benchmark


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


def per_rank_args(base: dict[str, Any], device: torch.device, rank: int) -> argparse.Namespace:
    args = argparse.Namespace(**base)
    args.device = str(device)
    args.seed = int(base["seed"]) + rank
    args.run_label = f"{base['run_label']}_rank{rank}"
    return args


def metric(report: dict[str, Any], section: str, name: str) -> float:
    value = report.get(section, {}).get(name)
    if value is None:
        return 0.0
    return float(value)


def all_reduce_totals(report: dict[str, Any], device: torch.device) -> dict[str, Any]:
    values = torch.tensor(
        [
            metric(report, "baseline", "tokens_per_s"),
            metric(report, "cap", "tokens_per_s"),
            metric(report, "qpruner", "tokens_per_s"),
            1.0 if report.get("status") == "PASS" else 0.0,
        ],
        device=device,
        dtype=torch.float32,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values_cpu = values.detach().cpu().tolist()
    return {
        "baseline_tokens_per_s": float(values_cpu[0]),
        "cap_tokens_per_s": float(values_cpu[1]),
        "qpruner_tokens_per_s": float(values_cpu[2]),
        "pass_count": int(round(values_cpu[3])),
    }


def run_one_rank(rank: int, config: dict[str, Any]) -> None:
    artifacts = Path(config["artifacts_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)
    rank_json = artifacts / f"multicard_tiny_qwen_generate_{config['run_label']}_rank{rank}.json"
    report: dict[str, Any] = {
        "status": "STARTED",
        "rank": rank,
        "world_size": config["world_size"],
        "backend": config["backend"],
        "device_request": config["device_name"],
        "card": config["cards"][rank] if rank < len(config["cards"]) else None,
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
        dist.init_process_group(
            backend=config["backend"],
            rank=rank,
            world_size=config["world_size"],
            timeout=timedelta(seconds=int(config["timeout_seconds"])),
        )
        dist.barrier()
        benchmark = run_benchmark(per_rank_args(config["benchmark_args"], device, rank))
        dist.barrier()
        totals = all_reduce_totals(benchmark, device)
        dist.barrier()
        report.update(
            {
                "status": "PASS" if benchmark.get("status") == "PASS" else "FAIL",
                "device": str(device),
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


def section(row: dict[str, Any], name: str) -> dict[str, Any]:
    return row.get("benchmark", {}).get(name, {}) if isinstance(row.get("benchmark"), dict) else {}


def aggregate_from_ranks(ranks: list[dict[str, Any]]) -> dict[str, Any]:
    pass_count = sum(1 for row in ranks if row.get("status") == "PASS")
    totals = ranks[0].get("distributed_totals", {}) if ranks else {}
    reduced = {
        "baseline_tokens_per_s_total": float(totals.get("baseline_tokens_per_s", 0.0) or 0.0),
        "cap_tokens_per_s_total": float(totals.get("cap_tokens_per_s", 0.0) or 0.0),
        "qpruner_tokens_per_s_total": float(totals.get("qpruner_tokens_per_s", 0.0) or 0.0),
        "pass_count": int(totals.get("pass_count", pass_count) or 0),
    }
    local = {
        "baseline_tokens_per_s_total": sum(metric(row.get("benchmark", {}), "baseline", "tokens_per_s") for row in ranks),
        "cap_tokens_per_s_total": sum(metric(row.get("benchmark", {}), "cap", "tokens_per_s") for row in ranks),
        "qpruner_tokens_per_s_total": sum(
            metric(row.get("benchmark", {}), "qpruner", "tokens_per_s") for row in ranks
        ),
        "pass_count": pass_count,
    }
    reduced["distributed_reduce_consistent"] = all(
        abs(float(reduced[key]) - float(local[key])) < 1e-3
        for key in ("baseline_tokens_per_s_total", "cap_tokens_per_s_total", "qpruner_tokens_per_s_total")
    ) and reduced["pass_count"] == local["pass_count"]
    return reduced


def markdown_summary(summary: dict[str, Any]) -> str:
    aggregate = summary.get("aggregate", {})
    lines = [
        "# Multi-Card TinyQwen Compressed Generate Benchmark",
        "",
        f"- Overall status: `{summary.get('status')}`",
        f"- World size: `{summary.get('world_size')}`",
        f"- Backend: `{summary.get('backend')}`",
        f"- Device request: `{summary.get('device_request')}`",
        f"- all-reduce throughput totals consistent: `{aggregate.get('distributed_reduce_consistent')}`",
        "",
        "| Aggregate metric | Value |",
        "|---|---:|",
        f"| Baseline tokens/s total | {fmt(aggregate.get('baseline_tokens_per_s_total'))} |",
        f"| CAP tokens/s total | {fmt(aggregate.get('cap_tokens_per_s_total'))} |",
        f"| QPruner tokens/s total | {fmt(aggregate.get('qpruner_tokens_per_s_total'))} |",
        f"| Passing ranks | {fmt(aggregate.get('pass_count'))} |",
        "",
        "| Rank | Status | Device | Baseline tok/s | CAP tok/s | QPruner tok/s | Peak MB |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in summary.get("ranks", []):
        baseline = section(row, "baseline")
        cap = section(row, "cap")
        qpruner = section(row, "qpruner")
        lines.append(
            "| {rank} | {status} | {device} | {baseline} | {cap} | {qpruner} | {peak} |".format(
                rank=row.get("rank", ""),
                status=row.get("status", ""),
                device=row.get("device", ""),
                baseline=fmt(baseline.get("tokens_per_s")),
                cap=fmt(cap.get("tokens_per_s")),
                qpruner=fmt(qpruner.get("tokens_per_s")),
                peak=fmt(row.get("benchmark", {}).get("peak_mem_mb") if isinstance(row.get("benchmark"), dict) else None),
            )
        )
    lines.extend(
        [
            "",
            "This benchmark proves the compressed generation path can be launched on every selected card, synchronized with the distributed backend, and summarized via all-reduce rather than extrapolated from one card.",
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
        for path in sorted(artifacts_dir.glob(f"multicard_tiny_qwen_generate_{run_label}_rank*.json"))
    ]
    world_size = ranks[0].get("world_size", len(ranks)) if ranks else 0
    backend = ranks[0].get("backend", "") if ranks else ""
    device_request = ranks[0].get("device_request", "") if ranks else ""
    status = "PASS" if ranks and all(row.get("status") == "PASS" for row in ranks) and not spawn_error else "FAIL"
    summary = {
        "status": status,
        "world_size": world_size,
        "backend": backend,
        "device_request": device_request,
        "aggregate": aggregate_from_ranks(ranks),
        "ranks": ranks,
        "spawn_error": spawn_error,
    }
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / f"multicard_tiny_qwen_generate_{run_label}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    (reports_dir / f"multicard-tiny-qwen-generate-{run_label}.md").write_text(markdown_summary(summary))
    return summary


def benchmark_arg_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "demo_root": args.demo_root,
        "model_id": args.model_id,
        "model_path": args.model_path,
        "device": args.device,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "iters": args.iters,
        "warmup": args.warmup,
        "cap_budget": args.cap_budget,
        "cap_max_iter": args.cap_max_iter,
        "cap_policy_steps": args.cap_policy_steps,
        "cap_samples_per_step": args.cap_samples_per_step,
        "cap_rpca_backend": args.cap_rpca_backend,
        "qpruner_average_bits": args.qpruner_average_bits,
        "enable_inference_cache": bool(args.enable_inference_cache),
        "export_dense_for_serving": bool(args.export_dense_for_serving),
        "seed": args.seed,
        "run_label": args.run_label,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TinyQwen compressed generate benchmark on multiple cards")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--model-id", default="TinyQwen3-Offline")
    parser.add_argument("--model-path", default="/mnt/nvme/622/models/TinyQwen3-Offline")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--cards", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--backend", default="")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--cap-budget", type=int, default=8192)
    parser.add_argument("--cap-max-iter", type=int, default=4)
    parser.add_argument("--cap-policy-steps", type=int, default=1)
    parser.add_argument("--cap-samples-per-step", type=int, default=1)
    parser.add_argument("--cap-rpca-backend", default="numpy")
    parser.add_argument("--qpruner-average-bits", type=float, default=4.0)
    parser.add_argument("--enable-inference-cache", action="store_true")
    parser.add_argument("--export-dense-for-serving", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-label", default="tiny_qwen3_multicard_generate_npu")
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=300)
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
    for old in artifacts_dir.glob(f"multicard_tiny_qwen_generate_{args.run_label}_rank*.json"):
        old.unlink()

    config = {
        "world_size": len(cards),
        "cards": cards,
        "backend": backend,
        "device_name": args.device,
        "artifacts_dir": str(artifacts_dir),
        "master_addr": args.master_addr,
        "master_port": args.master_port or free_port(),
        "timeout_seconds": args.timeout_seconds,
        "run_label": args.run_label,
        "benchmark_args": benchmark_arg_dict(args),
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
    print("MULTICARD_TINY_QWEN_GENERATE " + json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
