#!/usr/bin/env python
"""Summarize Qwen-family compressed generate benchmark artifacts."""
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
    label = str(report.get("run_label") or path.stem)
    prefix = "qwen_compression_generate_"
    if label.startswith(prefix):
        label = label[len(prefix) :]
    if label.endswith("_npu"):
        label = label[: -len("_npu")]
    report["_label"] = label
    return report


def sorted_reports(paths: list[Path]) -> list[dict[str, Any]]:
    reports = [load_artifact(path) for path in paths]
    return sorted(
        reports,
        key=lambda item: (
            int(item.get("target_layer_limit") or item.get("cap", {}).get("targeted_layers") or 0),
            item.get("_label", ""),
        ),
    )


def _target_layers(report: dict[str, Any]) -> tuple[Any, Any]:
    limit = report.get("target_layer_limit")
    if limit is None:
        limit = report.get("cap", {}).get("targeted_layers")
    return limit, report.get("targeted_layers_total")


def _best_by_speedup(reports: list[dict[str, Any]], section: str) -> dict[str, Any] | None:
    candidates = [
        report
        for report in reports
        if report.get("status") == "PASS" and report.get(section, {}).get("latency_speedup") is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: float(item.get(section, {}).get("latency_speedup") or 0.0))


def scaling_note(reports: list[dict[str, Any]]) -> str:
    if not reports:
        return "No artifacts were provided."

    best_cap = _best_by_speedup(reports, "cap")
    best_qpruner = _best_by_speedup(reports, "qpruner")
    notes: list[str] = []
    if best_cap:
        limit, total = _target_layers(best_cap)
        notes.append(
            "Best CAP speedup is {speedup}x at {limit}/{total} target layers.".format(
                speedup=fmt(best_cap.get("cap", {}).get("latency_speedup")),
                limit=limit,
                total=total,
            )
        )
    if best_qpruner:
        limit, total = _target_layers(best_qpruner)
        notes.append(
            "Best QPruner speedup is {speedup}x at {limit}/{total} target layers.".format(
                speedup=fmt(best_qpruner.get("qpruner", {}).get("latency_speedup")),
                limit=limit,
                total=total,
            )
        )
    if not notes:
        return "No passing CAP or QPruner speedup results were found in these artifacts."
    notes.append(
        "Treat this as a reproducible target-layer sweep until the target layer limit covers the full model."
    )
    return " ".join(notes)


def markdown_summary(paths: list[Path]) -> str:
    reports = sorted_reports(paths)
    lines = [
        "# Qwen Compression Generate Sweep",
        "",
        "| Label | Status | Model | Target layers | Baseline ms | CAP ms | CAP speedup | CAP ratio | QPruner ms | QPruner speedup | QPruner bits | Peak MB |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        baseline = report.get("baseline", {})
        cap = report.get("cap", {})
        qpruner = report.get("qpruner", {})
        limit, total = _target_layers(report)
        lines.append(
            "| {label} | {status} | {model} | {limit} / {total} | {base_ms} | {cap_ms} | {cap_speedup} | {cap_ratio} | {q_ms} | {q_speedup} | {q_bits} | {peak} |".format(
                label=report.get("_label", "unknown"),
                status=report.get("status", "missing"),
                model=report.get("model_id", "missing"),
                limit=limit,
                total=total,
                base_ms=fmt(baseline.get("latency_ms")),
                cap_ms=fmt(cap.get("latency_ms")),
                cap_speedup=fmt(cap.get("latency_speedup")),
                cap_ratio=fmt(cap.get("targeted_compression_ratio"), "x"),
                q_ms=fmt(qpruner.get("latency_ms")),
                q_speedup=fmt(qpruner.get("latency_speedup")),
                q_bits=fmt(qpruner.get("average_bits")),
                peak=fmt(report.get("peak_mem_mb")),
            )
        )
    lines.extend(
        [
            "",
            "## Scaling Read",
            "",
            scaling_note(reports),
            "",
            "## Artifact Inputs",
            "",
            *[f"- `{report['_path']}`" for report in reports],
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Qwen compressed generate sweep artifacts")
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    text = markdown_summary(args.artifacts)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"QWEN_COMPRESSION_GENERATE_SWEEP_SUMMARY {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
