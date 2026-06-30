#!/usr/bin/env python
"""Summarize Qwen QPruner grouped-MLP full-model sweep artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


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


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _round(value: Any, digits: int = 3) -> float | None:
    number = _float(value)
    if number is None:
        return None
    return round(number, digits)


def _nested(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _profile_paths(artifacts: Path, run_group: str) -> list[Path]:
    return sorted(artifacts.glob(f"qwen_qpruner_native_profile_{run_group}*.json"))


def _monitor_artifact_name(monitor_label: str | None, run_group: str) -> str:
    return f"npu_monitor_{monitor_label or (run_group + '_utilmon')}.json"


def _row(path: Path, profile: dict[str, Any]) -> dict[str, Any]:
    baseline = _nested(profile, "baseline")
    qpruner = _nested(profile, "qpruner")
    memory = _nested(profile, "memory") or _nested(profile, "memory_reference")
    paired = _nested(profile, "paired")
    paired_summary = _nested(profile, "paired_summary")
    paired_baseline = _nested(paired_summary, "baseline")
    paired_qpruner = _nested(paired_summary, "qpruner")
    runtime_profile = _nested(profile, "runtime_profile") or _nested(qpruner, "runtime_profile")
    speedups = _nested(profile, "speedups")
    return {
        "label": profile.get("run_label") or path.stem.removeprefix("qwen_qpruner_native_profile_"),
        "artifact": str(path),
        "status": profile.get("status"),
        "device": profile.get("device"),
        "average_bits": _round(profile.get("qpruner_average_bits") or qpruner.get("average_bits")),
        "target_layer_limit": _int(profile.get("target_layer_limit")),
        "targeted_layers_total": _int(profile.get("targeted_layers_total")),
        "target_layer_coverage_pct": _round(profile.get("target_layer_coverage_pct")),
        "quantized_layers": _int(qpruner.get("targeted_layers") or profile.get("targeted_layers")),
        "cache_mode": profile.get("qpruner_cache_mode"),
        "runtime_strategy": qpruner.get("runtime_strategy") or profile.get("qpruner_runtime_strategy"),
        "grouped_mlp_strategy": (
            profile.get("qpruner_grouped_mlp_strategy")
            or qpruner.get("grouped_mlp_strategy")
            or runtime_profile.get("grouped_mlp_strategy")
        ),
        "grouped_mlp_pairs": _int(profile.get("grouped_mlp_pairs") or runtime_profile.get("grouped_mlp_pairs")),
        "grouped_attention_kv_pairs": _int(
            profile.get("grouped_attention_kv_pairs")
            or qpruner.get("grouped_attention_kv_pairs")
            or runtime_profile.get("grouped_attention_kv_pairs")
        ),
        "prebuild_scaled_code_dtype_cache": profile.get("qpruner_prebuild_scaled_code_dtype_cache"),
        "scaled_code_dtype_cache_budget_bytes": _int(
            _first_present(
                profile.get("qpruner_scaled_code_dtype_cache_budget_bytes"),
                qpruner.get("scaled_code_dtype_cache_budget_bytes"),
            )
        ),
        "scaled_code_dtype_cache_selection_policy": (
            profile.get("qpruner_scaled_code_dtype_cache_selection_policy")
            or qpruner.get("scaled_code_dtype_cache_selection_policy")
            or _nested(qpruner, "scaled_code_dtype_cache_budget_plan").get("selection_policy")
        ),
        "dense_cache_budget_bytes": _int(profile.get("qpruner_dense_cache_budget_bytes")),
        "dense_cache_selection_policy": (
            profile.get("qpruner_dense_cache_selection_policy")
            or qpruner.get("dense_cache_selection_policy")
            or _nested(qpruner, "dense_cache_budget_plan").get("selection_policy")
        ),
        "baseline_latency_ms_median": _round(
            paired_baseline.get("latency_ms_median")
            or baseline.get("latency_ms_median")
            or baseline.get("latency_ms")
        ),
        "qpruner_latency_ms_median": _round(
            paired_qpruner.get("latency_ms_median")
            or qpruner.get("latency_ms_median")
            or qpruner.get("latency_ms")
        ),
        "baseline_tokens_per_s_median": _round(
            paired_baseline.get("tokens_per_s_median")
            or baseline.get("tokens_per_s_median")
            or baseline.get("tokens_per_s")
        ),
        "qpruner_tokens_per_s_median": _round(
            paired_qpruner.get("tokens_per_s_median")
            or qpruner.get("tokens_per_s_median")
            or qpruner.get("tokens_per_s")
        ),
        "paired_latency_speedup": _round(
            paired.get("latency_speedup")
            if paired.get("latency_speedup") is not None
            else paired_summary.get("qpruner_vs_baseline_latency_median_speedup")
            if paired_summary.get("qpruner_vs_baseline_latency_median_speedup") is not None
            else speedups.get("qpruner_vs_baseline")
        ),
        "paired_tokens_ratio": _round(
            paired.get("tokens_ratio") or paired_summary.get("qpruner_vs_baseline_tokens_median_ratio")
        ),
        "targeted_storage_reduction_pct": _round(
            qpruner.get("targeted_storage_reduction_pct")
            if qpruner.get("targeted_storage_reduction_pct") is not None
            else memory.get("qpruner_targeted_storage_reduction_pct")
        ),
        "code_cache_storage_reduction_pct": _round(memory.get("qpruner_code_cache_storage_reduction_pct")),
        "compressed_payload_storage_bytes": _int(memory.get("qpruner_compressed_payload_storage_bytes")),
        "live_compressed_payload_storage_bytes": _int(
            memory.get("qpruner_live_compressed_payload_storage_bytes")
            or qpruner.get("live_compressed_payload_storage_bytes")
        ),
        "code_cache_storage_bytes": _int(memory.get("qpruner_code_cache_storage_bytes")),
        "cached_code_bytes": _int(profile.get("qpruner_cached_code_bytes") or qpruner.get("cached_code_bytes")),
        "cached_scaled_code_bytes": _int(
            profile.get("qpruner_cached_scaled_code_bytes") or qpruner.get("cached_scaled_code_bytes")
        ),
        "cached_dense_weight_bytes": _int(
            profile.get("qpruner_cached_dense_weight_bytes") or qpruner.get("cached_dense_weight_bytes")
        ),
        "cached_aux_bytes": _int(profile.get("qpruner_cached_aux_bytes") or qpruner.get("cached_aux_bytes")),
        "runtime_forward_calls": _int(runtime_profile.get("total_forward_calls")),
        "runtime_forward_time_ms": _round(runtime_profile.get("total_forward_time_ms")),
        "runtime_estimated_forward_share_pct": _round(runtime_profile.get("estimated_forward_share_pct")),
    }


def _best(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get(key) is not None and row.get("status") == "PASS"]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row[key]))


def build_summary(
    demo_root: Path,
    *,
    run_group: str,
    expected_profiles: int = 8,
    monitor_label: str | None = None,
) -> dict[str, Any]:
    artifacts = demo_root / "artifacts"
    rows: list[dict[str, Any]] = []
    for path in _profile_paths(artifacts, run_group):
        payload = read_json(path)
        if isinstance(payload, dict):
            rows.append(_row(path, payload))
    monitor_artifact = _monitor_artifact_name(monitor_label, run_group)
    monitor = read_json(artifacts / monitor_artifact) or {}
    pass_count = sum(1 for row in rows if row.get("status") == "PASS")
    best_speed = _best(rows, "paired_latency_speedup") or {}
    best_memory = _best(rows, "code_cache_storage_reduction_pct") or {}
    monitor_status = monitor.get("status")
    max_active_cards = _int(monitor.get("max_active_process_cards"))
    status = (
        "PASS"
        if pass_count >= expected_profiles
        and len(rows) >= expected_profiles
        and (monitor_status in {None, "PASS"})
        and (max_active_cards is None or max_active_cards >= min(8, expected_profiles))
        else "ACTIONABLE"
    )
    return {
        "status": status,
        "run_group": run_group,
        "demo_root": str(demo_root),
        "profile_count": len(rows),
        "expected_profiles": expected_profiles,
        "pass_count": pass_count,
        "monitor_artifact": str(artifacts / monitor_artifact),
        "monitor_status": monitor_status,
        "monitor_samples": monitor.get("samples"),
        "max_active_process_cards": max_active_cards,
        "samples_with_8_active_process_cards": monitor.get("samples_with_8_active_process_cards"),
        "samples_with_8_nonzero_aicore": monitor.get("samples_with_8_nonzero_aicore"),
        "max_aicore_by_card": monitor.get("max_aicore_by_card"),
        "best_speed": best_speed,
        "best_code_cache_reduction": best_memory,
        "rows": rows,
        "evidence": [
            f"artifacts/{run_group}.json",
            f"reports/{run_group}.md",
            f"artifacts/{monitor_artifact}",
            *(f"artifacts/{Path(row['artifact']).name}" for row in rows),
        ],
    }


def markdown(summary: dict[str, Any]) -> str:
    best_speed = summary.get("best_speed", {}) if isinstance(summary.get("best_speed"), dict) else {}
    best_memory = (
        summary.get("best_code_cache_reduction", {})
        if isinstance(summary.get("best_code_cache_reduction"), dict)
        else {}
    )
    lines = [
        "# Qwen3 Grouped-MLP Full-Model Sweep",
        "",
        (
            "Qwen3 grouped-MLP full-model sweep: {status}, profiles {passes}/{profiles}, "
            "max active cards {cards}, all-8 AICore samples {aicore}, grouped MLP pairs {pairs}, "
            "grouped K/V pairs {kv_pairs}, best paired speedup {speedup}x, best code-cache storage reduction {storage}%"
        ).format(
            status=summary.get("status", "missing"),
            passes=fmt(summary.get("pass_count")),
            profiles=fmt(summary.get("profile_count")),
            cards=fmt(summary.get("max_active_process_cards")),
            aicore=fmt(summary.get("samples_with_8_nonzero_aicore")),
            pairs=fmt(max((row.get("grouped_mlp_pairs") or 0 for row in summary.get("rows", [])), default=None)),
            kv_pairs=fmt(
                max((row.get("grouped_attention_kv_pairs") or 0 for row in summary.get("rows", [])), default=None)
            ),
            speedup=fmt(best_speed.get("paired_latency_speedup")),
            storage=fmt(best_memory.get("code_cache_storage_reduction_pct")),
        ),
        "",
        "This sweep runs full-target Qwen3-0.6B QPruner native profiles across cards. Grouped-MLP combines each gate/up pair without duplicating the activation tensor; code-cache storage is reported separately from packed payload storage.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Status | {summary.get('status')} |",
        f"| Profiles passing | {summary.get('pass_count')} / {summary.get('profile_count')} |",
        f"| Monitor status | {summary.get('monitor_status')} |",
        f"| Max active process cards | {summary.get('max_active_process_cards')} |",
        f"| Samples with 8 active process cards | {fmt(summary.get('samples_with_8_active_process_cards'))} |",
        f"| Samples with 8 nonzero AICore cards | {fmt(summary.get('samples_with_8_nonzero_aicore'))} |",
        f"| Best speed label | {best_speed.get('label', 'missing')} |",
        f"| Best paired speedup | {fmt(best_speed.get('paired_latency_speedup'))}x |",
        f"| Best memory label | {best_memory.get('label', 'missing')} |",
        f"| Best code-cache storage reduction | {fmt(best_memory.get('code_cache_storage_reduction_pct'))}% |",
        "",
        "## Profiles",
        "",
        "| Label | Status | Bits | Cache | Runtime strategy | Grouped MLP strategy | Scaled dtype-cache budget | Scaled dtype-cache policy | Dense-cache budget | Dense-cache policy | Target layers | Grouped pairs | Grouped K/V pairs | Paired speedup | Code-cache storage | Packed storage | Forward calls | Forward time ms |",
        "|---|---:|---:|---|---|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("rows", []):
        lines.append(
            "| {label} | {status} | {bits} | {cache} | {strategy} | {grouped_strategy} | {scaled_budget} | {scaled_policy} | {dense_budget} | {dense_policy} | {target} | {pairs} | {kv_pairs} | {speedup} | {code_storage}% | {storage}% | {calls} | {time} |".format(
                label=row.get("label"),
                status=row.get("status"),
                bits=fmt(row.get("average_bits")),
                cache=row.get("cache_mode"),
                strategy=row.get("runtime_strategy"),
                grouped_strategy=fmt(row.get("grouped_mlp_strategy")),
                scaled_budget=fmt(row.get("scaled_code_dtype_cache_budget_bytes")),
                scaled_policy=fmt(row.get("scaled_code_dtype_cache_selection_policy")),
                dense_budget=fmt(row.get("dense_cache_budget_bytes")),
                dense_policy=fmt(row.get("dense_cache_selection_policy")),
                target=fmt(row.get("target_layer_limit")),
                pairs=fmt(row.get("grouped_mlp_pairs")),
                kv_pairs=fmt(row.get("grouped_attention_kv_pairs")),
                speedup=fmt(row.get("paired_latency_speedup")),
                code_storage=fmt(row.get("code_cache_storage_reduction_pct")),
                storage=fmt(row.get("targeted_storage_reduction_pct")),
                calls=fmt(row.get("runtime_forward_calls")),
                time=fmt(row.get("runtime_forward_time_ms")),
            )
        )
    lines.extend(["", "## Evidence", ""])
    lines.extend(f"- `{item}`" for item in summary.get("evidence", []))
    return "\n".join(lines) + "\n"


def write_outputs(summary: dict[str, Any], demo_root: Path, run_group: str) -> None:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    (artifacts / f"{run_group}.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (reports / f"{run_group}.md").write_text(markdown(summary))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a Qwen grouped-MLP native profile sweep")
    parser.add_argument("--demo-root", type=Path, default=Path("/mnt/nvme/622/tidal-demo"))
    parser.add_argument("--run-group", required=True)
    parser.add_argument("--expected-profiles", type=int, default=8)
    parser.add_argument("--monitor-label")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_summary(
        args.demo_root,
        run_group=args.run_group,
        expected_profiles=args.expected_profiles,
        monitor_label=args.monitor_label,
    )
    write_outputs(summary, args.demo_root, args.run_group)
    print(
        "QWEN_GROUPED_MLP_SWEEP_SUMMARY {json} {markdown}".format(
            json=args.demo_root / "artifacts" / f"{args.run_group}.json",
            markdown=args.demo_root / "reports" / f"{args.run_group}.md",
        )
    )
    return 0 if summary.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
