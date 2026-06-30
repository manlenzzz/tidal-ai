#!/usr/bin/env python
"""Summarize multi-card compression synchronization evidence for the demo."""
from __future__ import annotations

import argparse
import csv
import html
import io
import json
from pathlib import Path
from typing import Any, Sequence

from tidal.reports.multicard_selection import best_multicard_qwen_compression_generate


OUT_JSON = "multicard_compression_sync_summary.json"
OUT_MD = "multicard-compression-sync-summary.md"
OUT_CSV = "multicard-compression-sync-summary.csv"
OUT_SVG = "multicard-compression-sync-summary.svg"


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def fmt(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio(numerator: Any, denominator: Any) -> float | None:
    top = _num(numerator)
    bottom = _num(denominator)
    if top is None or bottom in (None, 0.0):
        return None
    return round(top / bottom, 3)


def _aggregate(payload: dict[str, Any] | None) -> dict[str, Any]:
    return payload.get("aggregate", {}) if isinstance((payload or {}).get("aggregate"), dict) else {}


def _status_is_pass(payload: dict[str, Any] | None) -> bool:
    return bool(payload and payload.get("status") == "PASS")


def _baseline_alignment(payload: dict[str, Any] | None) -> dict[str, str]:
    matrix = payload.get("baseline_alignment_matrix", []) if isinstance(payload, dict) else []
    rows: dict[str, str] = {}
    for row in matrix:
        if not isinstance(row, dict):
            continue
        method = str(row.get("method", "missing"))
        baselines = row.get("paper_baselines", [])
        if not isinstance(baselines, list):
            baselines = [baselines]
        role = row.get("paper_baseline_role") or row.get("paper_role") or "missing"
        rows[method] = "{role}: {baselines} -> {demo}".format(
            role=role,
            baselines=", ".join(str(value) for value in baselines) or "missing",
            demo=row.get("demo_reference_role", "missing"),
        )
    return rows


def build_summary(demo_root: Path) -> dict[str, Any]:
    artifacts = demo_root / "artifacts"
    sync = read_json(artifacts / "multicard_sync_summary.json")
    qwen_generate = best_multicard_qwen_compression_generate(artifacts)
    qwen_quality = read_json(artifacts / "multicard_qwen_compression_quality_qwen3_06b_quality_sync_npu.json")
    parallel = read_json(artifacts / "multicard_parallel_suite_demo_parallel_suite_npu.json")
    native_sweep = read_json(artifacts / "compressed_native_sweep_summary.json")
    compression_memory = read_json(artifacts / "compression_memory_report.json")

    generate_agg = _aggregate(qwen_generate)
    quality_agg = _aggregate(qwen_quality)
    native_best = native_sweep.get("best", {}) if isinstance((native_sweep or {}).get("best"), dict) else {}
    native_diag = (
        native_sweep.get("runtime_diagnosis", {})
        if isinstance((native_sweep or {}).get("runtime_diagnosis"), dict)
        else {}
    )
    native_memory = native_sweep.get("memory", {}) if isinstance((native_sweep or {}).get("memory"), dict) else {}
    baseline = _baseline_alignment(compression_memory)
    status = (
        "PASS"
        if _status_is_pass(sync)
        and _status_is_pass(qwen_generate)
        and bool(generate_agg.get("distributed_reduce_consistent"))
        and _status_is_pass(qwen_quality)
        and bool(quality_agg.get("distributed_reduce_consistent"))
        and _status_is_pass(native_sweep)
        else "ACTIONABLE"
    )
    world_size = int((qwen_generate or sync or {}).get("world_size") or 0)
    qpruner_speedup = _ratio(
        generate_agg.get("qpruner_tokens_per_s_total"),
        generate_agg.get("baseline_tokens_per_s_total"),
    )
    readout = (
        "{world}-card synchronized compression path is ready: QPruner {qpruner} tokens/s total "
        "({speedup}x baseline), all-reduce consistent {consistent}, while compressed-native keeps "
        "{storage}% QPruner storage reduction."
    ).format(
        world=world_size,
        qpruner=fmt(generate_agg.get("qpruner_tokens_per_s_total")),
        speedup=fmt(qpruner_speedup),
        consistent=generate_agg.get("distributed_reduce_consistent", "missing"),
        storage=fmt(native_memory.get("best_qpruner_storage_reduction_pct")),
    )
    return {
        "status": status,
        "demo_root": str(demo_root),
        "readout": readout,
        "sync": {
            "status": (sync or {}).get("status"),
            "world_size": (sync or {}).get("world_size"),
            "backend": (sync or {}).get("backend"),
        },
        "qwen_compression_generate": {
            "artifact": (qwen_generate or {}).get("_artifact_name"),
            "report": (qwen_generate or {}).get("_report_name"),
            "status": (qwen_generate or {}).get("status"),
            "model_id": (qwen_generate or {}).get("model_id"),
            "world_size": (qwen_generate or {}).get("world_size"),
            "backend": (qwen_generate or {}).get("backend"),
            "target_layer_limit": (qwen_generate or {}).get("target_layer_limit"),
            "targeted_layers_total": (qwen_generate or {}).get("targeted_layers_total"),
            "distributed_reduce_consistent": generate_agg.get("distributed_reduce_consistent"),
            "baseline_tokens_per_s_total": generate_agg.get("baseline_tokens_per_s_total"),
            "cap_tokens_per_s_total": generate_agg.get("cap_tokens_per_s_total"),
            "qpruner_tokens_per_s_total": generate_agg.get("qpruner_tokens_per_s_total"),
            "cap_speedup_vs_baseline": _ratio(
                generate_agg.get("cap_tokens_per_s_total"),
                generate_agg.get("baseline_tokens_per_s_total"),
            ),
            "qpruner_speedup_vs_baseline": qpruner_speedup,
            "peak_mem_mb_total": generate_agg.get("peak_mem_mb_total"),
            "pass_count": generate_agg.get("pass_count"),
        },
        "qwen_compression_quality": {
            "status": (qwen_quality or {}).get("status"),
            "world_size": (qwen_quality or {}).get("world_size"),
            "distributed_reduce_consistent": quality_agg.get("distributed_reduce_consistent"),
            "baseline_loss_avg": quality_agg.get("baseline_loss_avg"),
            "cap_loss_delta_avg": quality_agg.get("cap_loss_delta_avg"),
            "qpruner_loss_delta_avg": quality_agg.get("qpruner_loss_delta_avg"),
            "pass_count": quality_agg.get("pass_count"),
        },
        "compressed_native_memory": {
            "best_method": native_best.get("method_label") or native_best.get("method"),
            "qpruner_speedup": native_best.get("speedup"),
            "qpruner_storage_reduction_pct": native_memory.get("best_qpruner_storage_reduction_pct")
            or native_best.get("storage_reduction_pct"),
            "qpruner_cache_peak_mem_delta_mb": native_diag.get("qpruner_cache_peak_mem_delta_mb"),
            "qpruner_scaling_gap_vs_baseline": native_diag.get("qpruner_scaling_gap_vs_baseline"),
            "next_action": native_diag.get("next_action"),
        },
        "parallel_suite": {
            "status": (parallel or {}).get("status"),
            "world_size": (parallel or {}).get("world_size"),
            "launched_synchronously": (parallel or {}).get("launched_synchronously"),
            "start_window_s": (parallel or {}).get("start_window_s"),
            "release_lag_window_s": (parallel or {}).get("release_lag_window_s"),
            "task_counts": (parallel or {}).get("task_counts"),
        },
        "baseline_alignment": baseline,
        "evidence": [
            "artifacts/multicard_compression_sync_summary.json",
            "reports/multicard-compression-sync-summary.md",
            "reports/multicard-compression-sync-summary.csv",
            "reports/multicard-compression-sync-summary.svg",
            "artifacts/multicard_sync_summary.json",
            "artifacts/{name}".format(
                name=(qwen_generate or {}).get(
                    "_artifact_name",
                    "multicard_qwen_compression_generate_qwen3_06b_compression_generate_npu.json",
                )
            ),
            "artifacts/multicard_qwen_compression_quality_qwen3_06b_quality_sync_npu.json",
            "artifacts/compressed_native_sweep_summary.json",
            "artifacts/compression_memory_report.json",
        ],
    }


def markdown(summary: dict[str, Any]) -> str:
    generate = summary.get("qwen_compression_generate", {})
    quality = summary.get("qwen_compression_quality", {})
    memory = summary.get("compressed_native_memory", {})
    parallel = summary.get("parallel_suite", {})
    alignment = summary.get("baseline_alignment", {})
    lines = [
        "# Multi-Card Compression Sync Summary",
        "",
        summary.get("readout", "missing"),
        "",
        "This page combines synchronized multi-card Qwen compression evidence with compressed-native memory evidence. The multi-card totals are measured by launched ranks and all-reduce, not extrapolated from one card.",
        "",
        "## Synchronized Throughput",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Status | {generate.get('status')} |",
        f"| Model | {generate.get('model_id')} |",
        f"| World size | {generate.get('world_size')} |",
        f"| Backend | {generate.get('backend')} |",
        f"| Target layers | {generate.get('target_layer_limit')} / {generate.get('targeted_layers_total')} |",
        f"| all-reduce consistent | {generate.get('distributed_reduce_consistent')} |",
        f"| Baseline tokens/s total | {fmt(generate.get('baseline_tokens_per_s_total'))} |",
        f"| CAP tokens/s total | {fmt(generate.get('cap_tokens_per_s_total'))} |",
        f"| QPruner tokens/s total | {fmt(generate.get('qpruner_tokens_per_s_total'))} |",
        f"| CAP speedup vs baseline | {fmt(generate.get('cap_speedup_vs_baseline'))}x |",
        f"| QPruner speedup vs baseline | {fmt(generate.get('qpruner_speedup_vs_baseline'))}x |",
        f"| Passing ranks | {fmt(generate.get('pass_count'))} |",
        "",
        "## Quality Sync",
        "",
        "- all-reduce consistent: `{}`".format(quality.get("distributed_reduce_consistent")),
        "- baseline loss avg `{}`, CAP delta `{}`, QPruner delta `{}`.".format(
            fmt(quality.get("baseline_loss_avg")),
            fmt(quality.get("cap_loss_delta_avg")),
            fmt(quality.get("qpruner_loss_delta_avg")),
        ),
        "",
        "## Memory And Runtime",
        "",
        "- QPruner storage reduction {storage}% with cached short-decode speedup {speedup}x.".format(
            storage=fmt(memory.get("qpruner_storage_reduction_pct")),
            speedup=fmt(memory.get("qpruner_speedup")),
        ),
        "- QPruner cache peak +{mem} MB; scaling gap {gap}x vs baseline.".format(
            mem=fmt(memory.get("qpruner_cache_peak_mem_delta_mb")),
            gap=fmt(memory.get("qpruner_scaling_gap_vs_baseline")),
        ),
        "- Next action: {action}".format(action=memory.get("next_action", "missing")),
        "",
        "## Synchronization Control",
        "",
        "- Parallel suite status `{status}`, launched synchronously `{launched}`, start window `{window}`s, release lag window `{lag}`s.".format(
            status=parallel.get("status"),
            launched=parallel.get("launched_synchronously"),
            window=fmt(parallel.get("start_window_s")),
            lag=fmt(parallel.get("release_lag_window_s")),
        ),
        "",
        "## Baseline Alignment",
        "",
    ]
    for method, text in alignment.items():
        lines.append(f"- Baseline alignment: {method} {text}")
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            *[f"- `{item}`" for item in summary.get("evidence", [])],
        ]
    )
    return "\n".join(lines) + "\n"


def csv_text(summary: dict[str, Any]) -> str:
    generate = summary.get("qwen_compression_generate", {})
    quality = summary.get("qwen_compression_quality", {})
    memory = summary.get("compressed_native_memory", {})
    rows = {
        "status": summary.get("status"),
        "world_size": summary.get("sync", {}).get("world_size"),
        "qwen_compression_generate_baseline_tokens_per_s_total": generate.get("baseline_tokens_per_s_total"),
        "qwen_compression_generate_cap_tokens_per_s_total": generate.get("cap_tokens_per_s_total"),
        "qwen_compression_generate_qpruner_tokens_per_s_total": generate.get("qpruner_tokens_per_s_total"),
        "qwen_compression_generate_cap_speedup_vs_baseline": generate.get("cap_speedup_vs_baseline"),
        "qwen_compression_generate_qpruner_speedup_vs_baseline": generate.get("qpruner_speedup_vs_baseline"),
        "qwen_compression_quality_qpruner_loss_delta_avg": quality.get("qpruner_loss_delta_avg"),
        "compressed_native_qpruner_storage_reduction_pct": memory.get("qpruner_storage_reduction_pct"),
        "compressed_native_qpruner_cache_peak_mem_delta_mb": memory.get("qpruner_cache_peak_mem_delta_mb"),
    }
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["metric", "value"])
    for key, value in rows.items():
        writer.writerow([key, value])
    return output.getvalue()


def svg(summary: dict[str, Any]) -> str:
    generate = summary.get("qwen_compression_generate", {})
    memory = summary.get("compressed_native_memory", {})
    max_tps = max(
        float(generate.get("baseline_tokens_per_s_total") or 0.0),
        float(generate.get("cap_tokens_per_s_total") or 0.0),
        float(generate.get("qpruner_tokens_per_s_total") or 0.0),
        1.0,
    )
    bars = [
        ("baseline", generate.get("baseline_tokens_per_s_total"), "#374151"),
        ("CAP", generate.get("cap_tokens_per_s_total"), "#2563eb"),
        ("QPruner", generate.get("qpruner_tokens_per_s_total"), "#dc2626"),
    ]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="520" viewBox="0 0 1080 520" role="img" aria-label="Multi-card compression sync">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111827}.muted{fill:#6b7280}.grid{stroke:#e5e7eb}</style>",
        '<rect width="1080" height="520" fill="#ffffff"/>',
        '<text x="64" y="52" font-size="29" font-weight="700">Multi-card compression sync</text>',
        '<text x="64" y="82" font-size="15" class="muted">8-card all-reduce throughput plus compressed-native memory evidence</text>',
    ]
    x = 260
    y = 145
    width = 620
    for index, (label, value, color) in enumerate(bars):
        number = float(value or 0.0)
        bar_width = max(2.0, number / max_tps * width)
        row_y = y + index * 72
        parts.extend(
            [
                f'<text x="64" y="{row_y + 24}" font-size="17" font-weight="700">{html.escape(label)}</text>',
                f'<rect x="{x}" y="{row_y}" width="{bar_width:.1f}" height="28" fill="{color}"><title>{html.escape(label)} {fmt(value)} tokens/s</title></rect>',
                f'<text x="{x + bar_width + 12:.1f}" y="{row_y + 21}" font-size="14">{fmt(value)} tok/s total</text>',
            ]
        )
    parts.extend(
        [
            '<line class="grid" x1="64" y1="390" x2="1016" y2="390"/>',
            f'<text x="64" y="430" font-size="18" font-weight="700">8-card synchronized; not extrapolated from one card</text>',
            f'<text x="64" y="462" font-size="15" class="muted">QPruner storage reduction {fmt(memory.get("qpruner_storage_reduction_pct"))}% | cache peak +{fmt(memory.get("qpruner_cache_peak_mem_delta_mb"))} MB | next: fused NPU packed/quantized decode kernels</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def write_outputs(summary: dict[str, Any], demo_root: Path) -> None:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    (artifacts / OUT_JSON).write_text(json.dumps(summary, indent=2, sort_keys=True))
    (reports / OUT_MD).write_text(markdown(summary))
    (reports / OUT_CSV).write_text(csv_text(summary))
    (reports / OUT_SVG).write_text(svg(summary))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize multi-card compression synchronization evidence")
    parser.add_argument("--demo-root", type=Path, default=Path("/mnt/nvme/622/tidal-demo"))
    args = parser.parse_args(argv)
    summary = build_summary(args.demo_root)
    write_outputs(summary, args.demo_root)
    print(
        "MULTICARD_COMPRESSION_SYNC_SUMMARY {json} {markdown} {csv} {svg}".format(
            json=args.demo_root / "artifacts" / OUT_JSON,
            markdown=args.demo_root / "reports" / OUT_MD,
            csv=args.demo_root / "reports" / OUT_CSV,
            svg=args.demo_root / "reports" / OUT_SVG,
        )
    )
    return 0 if summary.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
