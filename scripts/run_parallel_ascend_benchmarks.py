#!/usr/bin/env python
"""Run CAP benchmarks on multiple Ascend cards and aggregate demo artifacts."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


BENCHMARK_RE = re.compile(r"^(?P<benchmark>.+_benchmark)(?:_(?P<label>[^_]+))?_npu(?P<card>\d+)\.json$")


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return ""
    return f"{number:.3f}"


def collect_benchmark_reports(artifacts_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(artifacts_dir.glob("*_benchmark_npu*.json")):
        match = BENCHMARK_RE.match(path.name)
        if match is None:
            continue
        payload = json.loads(path.read_text())
        benchmark = match.group("benchmark")
        card = int(match.group("card"))
        rows.append(
            {
                "benchmark": benchmark,
                "run_label": match.group("label") or "",
                "card": card,
                "status": payload.get("status", "PASS"),
                "device": payload.get("device"),
                "baseline_latency_ms": payload.get("baseline_latency_ms"),
                "compressed_latency_ms": payload.get("compressed_latency_ms"),
                "baseline_tokens_per_s": payload.get("baseline_tokens_per_s"),
                "compressed_tokens_per_s": payload.get("compressed_tokens_per_s"),
                "latency_speedup": payload.get("latency_speedup"),
                "targeted_compression_ratio": payload.get("targeted_compression_ratio"),
                "targeted_param_reduction_pct": payload.get("targeted_param_reduction_pct"),
                "compression_time_s": payload.get("compression_time_s"),
                "peak_mem_mb": payload.get("peak_mem_mb"),
            }
        )
    return rows


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Parallel Ascend Benchmark Summary",
        "",
        "| Benchmark | Label | Card | Status | Device | Speedup | Compression | Tokens/s | Peak MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {benchmark} | {label} | {card} | {status} | {device} | {speedup} | {ratio} | {tokens} | {peak} |".format(
                benchmark=row.get("benchmark", ""),
                label=row.get("run_label", ""),
                card=row.get("card", ""),
                status=row.get("status", ""),
                device=row.get("device", ""),
                speedup=fmt(row.get("latency_speedup")),
                ratio=fmt(row.get("targeted_compression_ratio")),
                tokens=fmt(row.get("compressed_tokens_per_s")),
                peak=fmt(row.get("peak_mem_mb")),
            )
        )
    lines.extend(
        [
            "",
            "Use this table as the quick video overlay: one row per card, with the raw JSON and terminal log kept beside it.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_aggregate(
    rows: list[dict[str, Any]], *, artifacts_dir: Path, reports_dir: Path
) -> dict[str, Any]:
    aggregate = {
        "status": "PASS" if rows and all(row.get("status") == "PASS" for row in rows) else "FAIL",
        "benchmarks": rows,
    }
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "parallel_ascend_benchmarks.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True)
    )
    (reports_dir / "parallel-ascend-benchmarks.md").write_text(markdown_table(rows))
    return aggregate


def run_one(
    *,
    project_root: Path,
    demo_root: Path,
    card: int,
    benchmark: str,
    run_label: str,
    device: str,
    dtype: str,
    budget: int,
    seq_len: int,
    batch_size: int,
    iters: int,
    warmup: int,
    max_iter: int,
    policy_steps: int,
    samples_per_step: int,
    seed: int,
) -> subprocess.Popen[bytes]:
    if benchmark != "cap_benchmark":
        raise ValueError(f"unsupported benchmark: {benchmark}")
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(card)
    env["PYTHONPATH"] = str(project_root)
    name = f"{benchmark}_{run_label}_npu{card}" if run_label else f"{benchmark}_npu{card}"
    log_path = demo_root / "logs" / f"{name}.log"
    json_path = demo_root / "artifacts" / f"{name}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("wb")
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "cap_benchmark.py"),
        "--device",
        device,
        "--dtype",
        dtype,
        "--budget",
        str(budget),
        "--seq-len",
        str(seq_len),
        "--batch-size",
        str(batch_size),
        "--iters",
        str(iters),
        "--warmup",
        str(warmup),
        "--max-iter",
        str(max_iter),
        "--policy-steps",
        str(policy_steps),
        "--samples-per-step",
        str(samples_per_step),
        "--seed",
        str(seed),
        "--summary-json",
        str(json_path),
    ]
    return subprocess.Popen(cmd, cwd=project_root, env=env, stdout=log_file, stderr=subprocess.STDOUT)


def parse_cards(raw: str) -> list[int]:
    cards = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not cards:
        raise ValueError("at least one card is required")
    return cards


def main() -> int:
    parser = argparse.ArgumentParser(description="Run parallel Ascend benchmarks")
    parser.add_argument("--project-root", default="/mnt/nvme/622/tidal-ai")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--cards", default="3,4,5,6")
    parser.add_argument("--run-label", default="")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--budget", type=int, default=8000)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument("--policy-steps", type=int, default=1)
    parser.add_argument("--samples-per-step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    project_root = Path(args.project_root)
    demo_root = Path(args.demo_root)
    artifacts_dir = demo_root / "artifacts"
    reports_dir = demo_root / "reports"
    for subdir in (artifacts_dir, demo_root / "logs", reports_dir, demo_root / "scripts"):
        subdir.mkdir(parents=True, exist_ok=True)

    processes = []
    for offset, card in enumerate(parse_cards(args.cards)):
        process = run_one(
            project_root=project_root,
            demo_root=demo_root,
            card=card,
            benchmark="cap_benchmark",
            run_label=args.run_label,
            device=args.device,
            dtype=args.dtype,
            budget=args.budget,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            iters=args.iters,
            warmup=args.warmup,
            max_iter=args.max_iter,
            policy_steps=args.policy_steps,
            samples_per_step=args.samples_per_step,
            seed=args.seed + offset,
        )
        processes.append((card, process))

    status = 0
    for card, process in processes:
        rc = process.wait()
        if rc == 0:
            print(f"PASS cap_benchmark npu{card}")
        else:
            print(f"FAIL cap_benchmark npu{card}")
            status = 1

    rows = collect_benchmark_reports(artifacts_dir)
    aggregate = write_aggregate(rows, artifacts_dir=artifacts_dir, reports_dir=reports_dir)
    print("PARALLEL_ASCEND_BENCHMARKS " + json.dumps(aggregate, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
