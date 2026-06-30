#!/usr/bin/env python
"""Summarize TinyQwen compression inference sweep artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.3f}{suffix}"
    return f"{value}{suffix}"


def load_artifact(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    report["_path"] = str(path)
    report["_label"] = path.stem
    prefix = "tiny_qwen_compression_tiny_qwen3_compression_"
    if report["_label"].startswith(prefix):
        report["_label"] = report["_label"][len(prefix) :]
    if report["_label"].endswith("_npu"):
        report["_label"] = report["_label"][: -len("_npu")]
    return report


def sorted_reports(paths: list[Path]) -> list[dict[str, Any]]:
    reports = [load_artifact(path) for path in paths]
    return sorted(reports, key=lambda item: (int(item.get("batch_size", 0)), int(item.get("seq_len", 0)), item["_label"]))


def bottleneck_note(reports: list[dict[str, Any]]) -> str:
    if not reports:
        return "No artifacts were provided."
    cap_speedups = [float(item.get("cap", {}).get("latency_speedup", 0.0) or 0.0) for item in reports]
    qpruner_speedups = [float(item.get("qpruner", {}).get("latency_speedup", 0.0) or 0.0) for item in reports]
    cap_cached_speedups = [
        float(item.get("cap_cached", {}).get("latency_speedup", 0.0) or 0.0) for item in reports
    ]
    qpruner_cached_speedups = [
        float(item.get("qpruner_cached", {}).get("latency_speedup", 0.0) or 0.0) for item in reports
    ]
    best_cap = max(cap_speedups) if cap_speedups else 0.0
    best_qpruner = max(qpruner_speedups) if qpruner_speedups else 0.0
    best_cap_cached = max(cap_cached_speedups) if cap_cached_speedups else 0.0
    best_qpruner_cached = max(qpruner_cached_speedups) if qpruner_cached_speedups else 0.0
    if best_cap_cached >= 1.0 or best_qpruner_cached >= 1.0:
        return (
            "the cached compressed path is latency-positive for at least one setting. "
            "This confirms the prior bottleneck was repeated dense reconstruction or dequantization in forward, "
            "and makes inference-cache export the next deployment path to preserve compression benefits."
        )
    if best_cap < 1.0 and best_qpruner < 1.0:
        return (
            "Across this sweep the compressed path is still latency-negative. "
            "The compression ratios are real, but the current CAP/QPruner layers add extra small-matmul "
            "or dequantization overhead on Ascend for this tiny model."
        )
    return (
        "At least one compressed path is latency-positive in this sweep. "
        "Use the best batch/sequence setting as the next reproduction target."
    )


def markdown_summary(paths: list[Path]) -> str:
    reports = sorted_reports(paths)
    lines = [
        "# TinyQwen Compression Sweep",
        "",
        "| Label | Batch | Seq | Baseline ms | CAP ms | CAP speedup | CAP cached speedup | CAP ratio | QPruner ms | QPruner speedup | QPruner cached speedup | QPruner bits | Peak MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        baseline = report.get("baseline", {})
        cap = report.get("cap", {})
        cap_cached = report.get("cap_cached", {})
        qpruner = report.get("qpruner", {})
        qpruner_cached = report.get("qpruner_cached", {})
        lines.append(
            "| {label} | {batch} | {seq} | {base_ms} | {cap_ms} | {cap_speedup} | {cap_cached_speedup} | {cap_ratio} | {q_ms} | {q_speedup} | {q_cached_speedup} | {q_bits} | {peak} |".format(
                label=report.get("_label", "unknown"),
                batch=report.get("batch_size", "missing"),
                seq=report.get("seq_len", "missing"),
                base_ms=fmt(baseline.get("latency_ms")),
                cap_ms=fmt(cap.get("latency_ms")),
                cap_speedup=fmt(cap.get("latency_speedup")),
                cap_cached_speedup=fmt(cap_cached.get("latency_speedup")),
                cap_ratio=fmt(cap.get("targeted_compression_ratio"), "x"),
                q_ms=fmt(qpruner.get("latency_ms")),
                q_speedup=fmt(qpruner.get("latency_speedup")),
                q_cached_speedup=fmt(qpruner_cached.get("latency_speedup")),
                q_bits=fmt(qpruner.get("average_bits")),
                peak=fmt(report.get("peak_mem_mb")),
            )
        )
    if any(report.get("inference_cache_enabled") for report in reports):
        lines.extend(
            [
                "",
                "Cache modules are counted in each raw artifact under `cap_cached.cache_modules` and `qpruner_cached.cache_modules`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Bottleneck Read",
            "",
            bottleneck_note(reports),
            "",
            "## Artifact Inputs",
            "",
            *[f"- `{report['_path']}`" for report in reports],
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize TinyQwen compression sweep artifacts")
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    text = markdown_summary(args.artifacts)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"TINY_QWEN_COMPRESSION_SWEEP_SUMMARY {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
