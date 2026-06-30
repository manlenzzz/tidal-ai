#!/usr/bin/env python
"""Summarize Qwen compression quality-memory sweep artifacts."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DENSE_WEIGHT_BITS = 16.0


def fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.3f}{suffix}"
    return f"{value}{suffix}"


def fmt_bits_word(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "missing-bit"
    if number.is_integer():
        return f"{int(number)}-bit"
    return f"{number:.3f}-bit"


def normalize_label(report: dict[str, Any], path: Path) -> str:
    label = str(report.get("run_label") or path.stem)
    for prefix in ("qwen_compression_quality_",):
        if label.startswith(prefix):
            label = label[len(prefix) :]
    if label.endswith("_npu"):
        label = label[: -len("_npu")]
    return label


def load_artifact(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    report["_path"] = str(path)
    report["_label"] = normalize_label(report, path)
    return report


def _target_layers(report: dict[str, Any]) -> tuple[Any, Any]:
    limit = report.get("target_layer_limit")
    if limit is None:
        limit = report.get("baseline", {}).get("targeted_layers")
    return limit, report.get("targeted_layers_total")


def qpruner_memory_reduction_pct(report: dict[str, Any], *, dense_weight_bits: float = DENSE_WEIGHT_BITS) -> float | None:
    qpruner = report.get("qpruner", {}) if isinstance(report.get("qpruner"), dict) else {}
    average_bits = qpruner.get("average_bits")
    if average_bits is not None:
        try:
            return round((1.0 - float(average_bits) / dense_weight_bits) * 100.0, 3)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    memory_bits = qpruner.get("memory_bits")
    baseline_params = report.get("baseline", {}).get("targeted_params")
    if memory_bits is None or baseline_params is None:
        return None
    try:
        dense_bits = float(baseline_params) * dense_weight_bits
        return round((1.0 - float(memory_bits) / dense_bits) * 100.0, 3)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def bitwidth_counts(report: dict[str, Any]) -> str:
    qpruner = report.get("qpruner", {}) if isinstance(report.get("qpruner"), dict) else {}
    bitwidths = qpruner.get("bitwidths") if isinstance(qpruner.get("bitwidths"), dict) else {}
    counts = Counter(int(value) for value in bitwidths.values())
    if not counts:
        return "missing"
    return ", ".join(f"{bit}-bit:{counts[bit]}" for bit in sorted(counts))


def sorted_reports(paths: list[Path]) -> list[dict[str, Any]]:
    reports = [load_artifact(path) for path in paths]
    return sorted(
        reports,
        key=lambda item: (
            float(item.get("qpruner", {}).get("average_bits") or 0.0),
            int(item.get("target_layer_limit") or item.get("baseline", {}).get("targeted_layers") or 0),
            item.get("_label", ""),
        ),
    )


def _passing_qpruner_reports(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        report
        for report in reports
        if report.get("status") == "PASS" and report.get("qpruner", {}).get("status", "PASS") == "PASS"
    ]


def best_memory_quality_point(reports: list[dict[str, Any]], *, max_loss_delta: float) -> dict[str, Any] | None:
    candidates = []
    for report in _passing_qpruner_reports(reports):
        qpruner = report.get("qpruner", {})
        loss_delta = qpruner.get("loss_delta")
        reduction = qpruner_memory_reduction_pct(report)
        if loss_delta is None or reduction is None:
            continue
        if float(loss_delta) <= max_loss_delta:
            candidates.append(report)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            qpruner_memory_reduction_pct(item) or -1.0,
            -(float(item.get("qpruner", {}).get("loss_delta") or 0.0)),
        ),
    )


def frontier_note(reports: list[dict[str, Any]], *, max_loss_delta: float) -> str:
    if not reports:
        return "No artifacts were provided."
    best = best_memory_quality_point(reports, max_loss_delta=max_loss_delta)
    notes: list[str] = []
    if best is None:
        notes.append(f"No QPruner point met loss delta <= {max_loss_delta:.3f}.")
    else:
        qpruner = best.get("qpruner", {})
        notes.append(
            "Best memory-quality point under loss delta <= {threshold} is {label}: "
            "{reduction} targeted memory reduction, loss delta {delta}, average bits {bits}.".format(
                threshold=fmt(max_loss_delta),
                label=best.get("_label", "unknown"),
                reduction=fmt(qpruner_memory_reduction_pct(best), "%"),
                delta=fmt(qpruner.get("loss_delta")),
                bits=fmt(qpruner.get("average_bits")),
            )
        )
    lowest_bits = min(
        _passing_qpruner_reports(reports),
        key=lambda item: float(item.get("qpruner", {}).get("average_bits") or DENSE_WEIGHT_BITS),
        default=None,
    )
    if lowest_bits is not None:
        qpruner = lowest_bits.get("qpruner", {})
        if qpruner.get("loss_delta") is not None and float(qpruner["loss_delta"]) > max_loss_delta:
            notes.append(
                "{bits} saves the most memory but exceeds the quality threshold with loss delta {delta}.".format(
                    bits=fmt_bits_word(qpruner.get("average_bits")),
                    delta=fmt(qpruner.get("loss_delta")),
                )
            )
    notes.append(
        "Use this table as the demo memory-quality frontier; QPruner storage savings are computed against 16-bit dense targeted weights."
    )
    return " ".join(notes)


def markdown_summary(paths: list[Path], *, max_loss_delta: float = 0.5) -> str:
    reports = sorted_reports(paths)
    lines = [
        "# Qwen Compression Quality-Memory Sweep",
        "",
        "| Label | Status | Target layers | QPruner bits | Target memory reduction | QPruner loss delta | QPruner tokens/s | CAP delta | WANDA delta | Bitwidth mix | Peak MB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for report in reports:
        qpruner = report.get("qpruner", {}) if isinstance(report.get("qpruner"), dict) else {}
        cap = report.get("cap", {}) if isinstance(report.get("cap"), dict) else {}
        wanda = report.get("wanda", {}) if isinstance(report.get("wanda"), dict) else {}
        limit, total = _target_layers(report)
        lines.append(
            "| {label} | {status} | {limit} / {total} | {bits} | {reduction} | {loss_delta} | {tokens} | {cap_delta} | {wanda_delta} | {mix} | {peak} |".format(
                label=report.get("_label", "unknown"),
                status=report.get("status", "missing"),
                limit=limit,
                total=total,
                bits=fmt(qpruner.get("average_bits")),
                reduction=fmt(qpruner_memory_reduction_pct(report), "%"),
                loss_delta=fmt(qpruner.get("loss_delta")),
                tokens=fmt(qpruner.get("tokens_per_s")),
                cap_delta=fmt(cap.get("loss_delta")),
                wanda_delta=fmt(wanda.get("loss_delta")),
                mix=bitwidth_counts(report),
                peak=fmt(report.get("peak_mem_mb")),
            )
        )
    lines.extend(
        [
            "",
            "## Frontier Read",
            "",
            frontier_note(reports, max_loss_delta=max_loss_delta),
            "",
            "## Artifact Inputs",
            "",
            *[f"- `{report['_path']}`" for report in reports],
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Qwen quality-memory sweep artifacts")
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-loss-delta", type=float, default=0.5)
    args = parser.parse_args()

    text = markdown_summary(args.artifacts, max_loss_delta=args.max_loss_delta)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"QWEN_COMPRESSION_QUALITY_MEMORY_SWEEP_SUMMARY {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
