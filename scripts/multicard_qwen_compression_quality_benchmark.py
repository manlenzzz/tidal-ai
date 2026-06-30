#!/usr/bin/env python
"""Run Qwen-family CAP/QPruner compression quality benchmark on synchronized cards."""
from __future__ import annotations

import argparse
import json
import math
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

from ascend_multicard_sync_smoke import apply_hccl_port_defaults, default_backend, parse_cards, set_device
from qwen_compression_quality_benchmark import run_benchmark


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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


def top_metric(report: dict[str, Any], name: str) -> float:
    value = report.get(name)
    if value is None:
        return 0.0
    return float(value)


def all_reduce_totals(report: dict[str, Any], device: torch.device) -> dict[str, Any]:
    values = torch.tensor(
        [
            metric(report, "baseline", "tokens_per_s"),
            metric(report, "cap", "tokens_per_s"),
            metric(report, "wanda", "tokens_per_s"),
            metric(report, "sparsegpt", "tokens_per_s"),
            metric(report, "qpruner", "tokens_per_s"),
            top_metric(report, "peak_mem_mb"),
            1.0 if report.get("status") == "PASS" else 0.0,
            metric(report, "baseline", "loss"),
            metric(report, "cap", "loss"),
            metric(report, "wanda", "loss"),
            metric(report, "sparsegpt", "loss"),
            metric(report, "qpruner", "loss"),
            metric(report, "cap", "loss_delta"),
            metric(report, "wanda", "loss_delta"),
            metric(report, "sparsegpt", "loss_delta"),
            metric(report, "qpruner", "loss_delta"),
        ],
        device=device,
        dtype=torch.float32,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values_cpu = values.detach().cpu().tolist()
    return {
        "baseline_tokens_per_s": float(values_cpu[0]),
        "cap_tokens_per_s": float(values_cpu[1]),
        "wanda_tokens_per_s": float(values_cpu[2]),
        "sparsegpt_tokens_per_s": float(values_cpu[3]),
        "qpruner_tokens_per_s": float(values_cpu[4]),
        "peak_mem_mb": float(values_cpu[5]),
        "pass_count": int(round(values_cpu[6])),
        "baseline_loss_sum": float(values_cpu[7]),
        "cap_loss_sum": float(values_cpu[8]),
        "wanda_loss_sum": float(values_cpu[9]),
        "sparsegpt_loss_sum": float(values_cpu[10]),
        "qpruner_loss_sum": float(values_cpu[11]),
        "cap_loss_delta_sum": float(values_cpu[12]),
        "wanda_loss_delta_sum": float(values_cpu[13]),
        "sparsegpt_loss_delta_sum": float(values_cpu[14]),
        "qpruner_loss_delta_sum": float(values_cpu[15]),
    }


def run_one_rank(rank: int, config: dict[str, Any]) -> None:
    artifacts = Path(config["artifacts_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)
    rank_json = artifacts / f"multicard_qwen_compression_quality_{config['run_label']}_rank{rank}.json"
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


def _avg(total: float, pass_count: int) -> float | None:
    if pass_count <= 0:
        return None
    return float(total) / float(pass_count)


def aggregate_from_ranks(ranks: list[dict[str, Any]]) -> dict[str, Any]:
    pass_count = sum(1 for row in ranks if row.get("status") == "PASS")
    totals = ranks[0].get("distributed_totals", {}) if ranks else {}
    reduced = {
        "baseline_tokens_per_s_total": float(totals.get("baseline_tokens_per_s", 0.0) or 0.0),
        "cap_tokens_per_s_total": float(totals.get("cap_tokens_per_s", 0.0) or 0.0),
        "wanda_tokens_per_s_total": float(totals.get("wanda_tokens_per_s", 0.0) or 0.0),
        "sparsegpt_tokens_per_s_total": float(totals.get("sparsegpt_tokens_per_s", 0.0) or 0.0),
        "qpruner_tokens_per_s_total": float(totals.get("qpruner_tokens_per_s", 0.0) or 0.0),
        "peak_mem_mb_total": float(totals.get("peak_mem_mb", 0.0) or 0.0),
        "pass_count": int(totals.get("pass_count", pass_count) or 0),
    }
    reduced_pass_count = int(reduced["pass_count"])
    reduced.update(
        {
            "baseline_loss_avg": _avg(float(totals.get("baseline_loss_sum", 0.0) or 0.0), reduced_pass_count),
            "cap_loss_avg": _avg(float(totals.get("cap_loss_sum", 0.0) or 0.0), reduced_pass_count),
            "wanda_loss_avg": _avg(float(totals.get("wanda_loss_sum", 0.0) or 0.0), reduced_pass_count),
            "sparsegpt_loss_avg": _avg(
                float(totals.get("sparsegpt_loss_sum", 0.0) or 0.0), reduced_pass_count
            ),
            "qpruner_loss_avg": _avg(float(totals.get("qpruner_loss_sum", 0.0) or 0.0), reduced_pass_count),
            "cap_loss_delta_avg": _avg(float(totals.get("cap_loss_delta_sum", 0.0) or 0.0), reduced_pass_count),
            "wanda_loss_delta_avg": _avg(
                float(totals.get("wanda_loss_delta_sum", 0.0) or 0.0), reduced_pass_count
            ),
            "sparsegpt_loss_delta_avg": _avg(
                float(totals.get("sparsegpt_loss_delta_sum", 0.0) or 0.0), reduced_pass_count
            ),
            "qpruner_loss_delta_avg": _avg(
                float(totals.get("qpruner_loss_delta_sum", 0.0) or 0.0), reduced_pass_count
            ),
        }
    )
    local = {
        "baseline_tokens_per_s_total": sum(metric(row.get("benchmark", {}), "baseline", "tokens_per_s") for row in ranks),
        "cap_tokens_per_s_total": sum(metric(row.get("benchmark", {}), "cap", "tokens_per_s") for row in ranks),
        "wanda_tokens_per_s_total": sum(metric(row.get("benchmark", {}), "wanda", "tokens_per_s") for row in ranks),
        "sparsegpt_tokens_per_s_total": sum(
            metric(row.get("benchmark", {}), "sparsegpt", "tokens_per_s") for row in ranks
        ),
        "qpruner_tokens_per_s_total": sum(
            metric(row.get("benchmark", {}), "qpruner", "tokens_per_s") for row in ranks
        ),
        "peak_mem_mb_total": sum(top_metric(row.get("benchmark", {}), "peak_mem_mb") for row in ranks),
        "pass_count": pass_count,
        "baseline_loss_sum": sum(metric(row.get("benchmark", {}), "baseline", "loss") for row in ranks),
        "cap_loss_sum": sum(metric(row.get("benchmark", {}), "cap", "loss") for row in ranks),
        "wanda_loss_sum": sum(metric(row.get("benchmark", {}), "wanda", "loss") for row in ranks),
        "sparsegpt_loss_sum": sum(metric(row.get("benchmark", {}), "sparsegpt", "loss") for row in ranks),
        "qpruner_loss_sum": sum(metric(row.get("benchmark", {}), "qpruner", "loss") for row in ranks),
        "cap_loss_delta_sum": sum(metric(row.get("benchmark", {}), "cap", "loss_delta") for row in ranks),
        "wanda_loss_delta_sum": sum(metric(row.get("benchmark", {}), "wanda", "loss_delta") for row in ranks),
        "sparsegpt_loss_delta_sum": sum(
            metric(row.get("benchmark", {}), "sparsegpt", "loss_delta") for row in ranks
        ),
        "qpruner_loss_delta_sum": sum(metric(row.get("benchmark", {}), "qpruner", "loss_delta") for row in ranks),
    }
    throughput_consistent = all(
        math.isclose(float(reduced[key]), float(local[key]), rel_tol=1e-6, abs_tol=1e-3)
        for key in (
            "baseline_tokens_per_s_total",
            "cap_tokens_per_s_total",
            "wanda_tokens_per_s_total",
            "sparsegpt_tokens_per_s_total",
            "qpruner_tokens_per_s_total",
        )
    )
    peak_consistent = math.isclose(
        float(reduced["peak_mem_mb_total"]), float(local["peak_mem_mb_total"]), rel_tol=1e-6, abs_tol=1e-2
    )
    loss_consistent = all(
        math.isclose(float(totals.get(key, 0.0) or 0.0), float(local[key]), rel_tol=1e-6, abs_tol=1e-3)
        for key in (
            "baseline_loss_sum",
            "cap_loss_sum",
            "wanda_loss_sum",
            "sparsegpt_loss_sum",
            "qpruner_loss_sum",
            "cap_loss_delta_sum",
            "wanda_loss_delta_sum",
            "sparsegpt_loss_delta_sum",
            "qpruner_loss_delta_sum",
        )
    )
    reduced["distributed_reduce_consistent"] = (
        throughput_consistent
        and peak_consistent
        and loss_consistent
        and reduced["pass_count"] == local["pass_count"]
    )
    return reduced


def markdown_summary(summary: dict[str, Any]) -> str:
    aggregate = summary.get("aggregate", {})
    lines = [
        "# Multi-Card Qwen CAP/QPruner Compression Quality Benchmark",
        "",
        f"- Overall status: `{summary.get('status')}`",
        f"- Model: `{summary.get('model_id')}`",
        f"- Model path: `{summary.get('model_path')}`",
        f"- World size: `{summary.get('world_size')}`",
        f"- Backend: `{summary.get('backend')}`",
        f"- Device request: `{summary.get('device_request')}`",
        f"- Target layers: `{summary.get('target_layer_limit')} / {summary.get('targeted_layers_total')}`",
        f"- Target layer pattern: `{summary.get('target_layer_pattern')}`",
        f"- all-reduce quality and throughput totals consistent: `{aggregate.get('distributed_reduce_consistent')}`",
        "",
        "| Aggregate metric | Value |",
        "|---|---:|",
        f"| Baseline loss avg | {fmt(aggregate.get('baseline_loss_avg'))} |",
        f"| CAP loss avg | {fmt(aggregate.get('cap_loss_avg'))} |",
        f"| WANDA loss avg | {fmt(aggregate.get('wanda_loss_avg'))} |",
        f"| SparseGPT loss avg | {fmt(aggregate.get('sparsegpt_loss_avg'))} |",
        f"| QPruner loss avg | {fmt(aggregate.get('qpruner_loss_avg'))} |",
        f"| CAP loss delta avg | {fmt(aggregate.get('cap_loss_delta_avg'))} |",
        f"| WANDA loss delta avg | {fmt(aggregate.get('wanda_loss_delta_avg'))} |",
        f"| SparseGPT loss delta avg | {fmt(aggregate.get('sparsegpt_loss_delta_avg'))} |",
        f"| QPruner loss delta avg | {fmt(aggregate.get('qpruner_loss_delta_avg'))} |",
        f"| Baseline tokens/s total | {fmt(aggregate.get('baseline_tokens_per_s_total'))} |",
        f"| CAP tokens/s total | {fmt(aggregate.get('cap_tokens_per_s_total'))} |",
        f"| WANDA tokens/s total | {fmt(aggregate.get('wanda_tokens_per_s_total'))} |",
        f"| SparseGPT tokens/s total | {fmt(aggregate.get('sparsegpt_tokens_per_s_total'))} |",
        f"| QPruner tokens/s total | {fmt(aggregate.get('qpruner_tokens_per_s_total'))} |",
        f"| Peak MB total | {fmt(aggregate.get('peak_mem_mb_total'))} |",
        f"| Passing ranks | {fmt(aggregate.get('pass_count'))} |",
        "",
        "| Rank | Status | Device | Baseline loss | CAP loss delta | WANDA loss delta | SparseGPT loss delta | QPruner loss delta | Baseline tok/s | CAP tok/s | WANDA tok/s | SparseGPT tok/s | QPruner tok/s |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("ranks", []):
        baseline = section(row, "baseline")
        cap = section(row, "cap")
        wanda = section(row, "wanda")
        sparsegpt = section(row, "sparsegpt")
        qpruner = section(row, "qpruner")
        lines.append(
            "| {rank} | {status} | {device} | {baseline_loss} | {cap_delta} | {wanda_delta} | {sparsegpt_delta} | {q_delta} | {baseline_tps} | {cap_tps} | {wanda_tps} | {sparsegpt_tps} | {q_tps} |".format(
                rank=row.get("rank", ""),
                status=row.get("status", ""),
                device=row.get("device", ""),
                baseline_loss=fmt(baseline.get("loss")),
                cap_delta=fmt(cap.get("loss_delta")),
                wanda_delta=fmt(wanda.get("loss_delta")),
                sparsegpt_delta=fmt(sparsegpt.get("loss_delta")),
                q_delta=fmt(qpruner.get("loss_delta")),
                baseline_tps=fmt(baseline.get("tokens_per_s")),
                cap_tps=fmt(cap.get("tokens_per_s")),
                wanda_tps=fmt(wanda.get("tokens_per_s")),
                sparsegpt_tps=fmt(sparsegpt.get("tokens_per_s")),
                q_tps=fmt(qpruner.get("tokens_per_s")),
            )
        )
    lines.extend(
        [
            "",
            "This benchmark proves the Qwen-family compression quality path can be launched on every selected card, synchronized with the distributed backend, and summarized via all-reduce loss averages and throughput totals, not extrapolated from one card.",
        ]
    )
    if summary.get("spawn_error"):
        lines.extend(["", f"Spawn error: `{summary['spawn_error']}`"])
    return "\n".join(lines) + "\n"


def _first_benchmark_value(ranks: list[dict[str, Any]], name: str) -> Any:
    for row in ranks:
        benchmark = row.get("benchmark", {}) if isinstance(row.get("benchmark"), dict) else {}
        if benchmark.get(name) is not None:
            return benchmark.get(name)
    return None


def write_summary(
    *,
    artifacts_dir: Path,
    reports_dir: Path,
    run_label: str,
    spawn_error: str | None,
) -> dict[str, Any]:
    ranks = [
        json.loads(path.read_text())
        for path in sorted(artifacts_dir.glob(f"multicard_qwen_compression_quality_{run_label}_rank*.json"))
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
        "target_layer_limit": _first_benchmark_value(ranks, "target_layer_limit"),
        "target_layer_pattern": _first_benchmark_value(ranks, "target_layer_pattern"),
        "targeted_layers_total": _first_benchmark_value(ranks, "targeted_layers_total"),
        "aggregate": aggregate_from_ranks(ranks),
        "ranks": ranks,
        "spawn_error": spawn_error,
    }
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / f"multicard_qwen_compression_quality_{run_label}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    (reports_dir / f"multicard-qwen-compression-quality-{run_label}.md").write_text(markdown_summary(summary))
    return summary


def benchmark_arg_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "demo_root": args.demo_root,
        "model_id": args.model_id,
        "model_path": args.model_path,
        "device": args.device,
        "dtype": args.dtype,
        "max_length": args.max_length,
        "iters": args.iters,
        "warmup": args.warmup,
        "target_layer_limit": args.target_layer_limit,
        "target_layer_pattern": args.target_layer_pattern,
        "cap_budget": args.cap_budget,
        "cap_max_iter": args.cap_max_iter,
        "cap_policy_steps": args.cap_policy_steps,
        "cap_samples_per_step": args.cap_samples_per_step,
        "cap_rpca_backend": args.cap_rpca_backend,
        "wanda_sparsity": args.wanda_sparsity,
        "sparsegpt_sparsity": args.sparsegpt_sparsity,
        "qpruner_average_bits": args.qpruner_average_bits,
        "enable_inference_cache": bool(args.enable_inference_cache),
        "inplace_compression": bool(args.inplace_compression),
        "seed": args.seed,
        "run_label": args.run_label,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen-family compression quality benchmark on multiple cards")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-path", default="/mnt/nvme/622/models/Qwen3-0.6B")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--cards", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--backend", default="")
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--target-layer-limit", type=int, default=2)
    parser.add_argument("--target-layer-pattern")
    parser.add_argument("--cap-budget", type=int, default=1048576)
    parser.add_argument("--cap-max-iter", type=int, default=1)
    parser.add_argument("--cap-policy-steps", type=int, default=1)
    parser.add_argument("--cap-samples-per-step", type=int, default=1)
    parser.add_argument("--cap-rpca-backend", default="numpy")
    parser.add_argument("--wanda-sparsity", type=float, default=0.5)
    parser.add_argument("--sparsegpt-sparsity", type=float, default=0.5)
    parser.add_argument("--qpruner-average-bits", type=float, default=4.0)
    parser.add_argument("--enable-inference-cache", action="store_true")
    parser.add_argument("--inplace-compression", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-label", default="qwen3_06b_quality_sync_npu")
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=900)
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
    for old in artifacts_dir.glob(f"multicard_qwen_compression_quality_{args.run_label}_rank*.json"):
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
        "model_id": args.model_id,
        "model_path": args.model_path,
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
    print("MULTICARD_QWEN_COMPRESSION_QUALITY " + json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
