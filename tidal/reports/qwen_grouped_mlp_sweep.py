"""Helpers for Qwen QPruner grouped-MLP full-model sweep evidence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


QWEN_GROUPED_MLP_SWEEP_PATTERNS = (
    "qwen3_06b_native_profile_8card_grouped_mlp_sweep_*.json",
    "qwen3_06b_native_profile_8card_grouped_mlp_nocopy_sweep_*.json",
)


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


def int_score(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def float_score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def qwen_grouped_mlp_sweep_report_name(sweep: dict[str, Any] | None) -> str:
    if isinstance(sweep, dict) and sweep.get("_report_name"):
        return str(sweep["_report_name"])
    artifact = str((sweep or {}).get("_artifact_name") or "")
    if artifact.endswith(".json"):
        return artifact[:-5] + ".md"
    run_label = str((sweep or {}).get("run_label") or "qwen3_06b_native_profile_8card_grouped_mlp_sweep_missing")
    return f"{run_label}.md"


def best_qwen_grouped_mlp_sweep(artifacts: Path) -> dict[str, Any] | None:
    candidates: list[tuple[tuple[int, int, int, int, int, float, float], dict[str, Any]]] = []
    paths = sorted({path for pattern in QWEN_GROUPED_MLP_SWEEP_PATTERNS for path in artifacts.glob(pattern)})
    for path in paths:
        payload = read_json(path)
        if not isinstance(payload, dict):
            continue
        rows = payload.get("rows", [])
        if not isinstance(rows, list):
            continue
        payload["_artifact_name"] = path.name
        payload["_report_name"] = qwen_grouped_mlp_sweep_report_name({"_artifact_name": path.name})
        best_speed = payload.get("best_speed", {}) if isinstance(payload.get("best_speed"), dict) else {}
        candidates.append(
            (
                (
                    1 if payload.get("status") == "PASS" else 0,
                    int_score(payload.get("pass_count")),
                    int_score(payload.get("max_active_process_cards")),
                    float_score(best_speed.get("paired_latency_speedup")),
                    int_score(payload.get("samples_with_8_nonzero_aicore")),
                    int_score(payload.get("samples_with_8_active_process_cards")),
                    path.stat().st_mtime,
                ),
                payload,
            )
        )
    return max(candidates, key=lambda row: row[0])[1] if candidates else None


def best_qwen_grouped_mlp_memory_tradeoff(
    artifacts: Path,
    *,
    min_code_cache_storage_reduction_pct: float = 49.0,
) -> dict[str, Any] | None:
    candidates: list[tuple[tuple[int, int, int, float, float, int, int, float], dict[str, Any]]] = []
    paths = sorted({path for pattern in QWEN_GROUPED_MLP_SWEEP_PATTERNS for path in artifacts.glob(pattern)})
    for path in paths:
        payload = read_json(path)
        if not isinstance(payload, dict):
            continue
        rows = payload.get("rows", [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict) or row.get("status") != "PASS":
                continue
            storage_reduction = float_score(row.get("code_cache_storage_reduction_pct"))
            speedup = float_score(row.get("paired_latency_speedup"))
            if storage_reduction < min_code_cache_storage_reduction_pct or speedup <= 0:
                continue
            candidate = dict(payload)
            candidate["_artifact_name"] = path.name
            candidate["_report_name"] = qwen_grouped_mlp_sweep_report_name({"_artifact_name": path.name})
            candidate["_memory_tradeoff_row"] = row
            candidates.append(
                (
                    (
                        1 if payload.get("status") == "PASS" else 0,
                        int_score(payload.get("pass_count")),
                        int_score(payload.get("max_active_process_cards")),
                        speedup,
                        storage_reduction,
                        int_score(payload.get("samples_with_8_nonzero_aicore")),
                        int_score(payload.get("samples_with_8_active_process_cards")),
                        path.stat().st_mtime,
                    ),
                    candidate,
                )
            )
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def qwen_grouped_mlp_sweep_readout(sweep: dict[str, Any] | None) -> str:
    if not isinstance(sweep, dict) or not sweep:
        return "Qwen3 grouped-MLP full-model sweep missing"
    best_speed = sweep.get("best_speed", {}) if isinstance(sweep.get("best_speed"), dict) else {}
    best_memory = (
        sweep.get("best_code_cache_reduction", {})
        if isinstance(sweep.get("best_code_cache_reduction"), dict)
        else {}
    )
    rows = sweep.get("rows", []) if isinstance(sweep.get("rows"), list) else []
    grouped_pairs = [row.get("grouped_mlp_pairs") for row in rows if isinstance(row, dict)]
    grouped_kv_pairs = [
        row.get("grouped_attention_kv_pairs")
        for row in rows
        if isinstance(row, dict) and row.get("grouped_attention_kv_pairs") is not None
    ]
    pair_text = fmt(max(grouped_pairs)) if grouped_pairs else "missing"
    kv_text = f"grouped K/V pairs {fmt(max(grouped_kv_pairs))}, " if grouped_kv_pairs else ""
    return (
        "Qwen3 grouped-MLP full-model sweep: {status}, profiles {passes}/{profiles}, "
        "max active cards {cards}, all-8 AICore samples {aicore}, grouped MLP pairs {pairs}, "
        "{kv_text}"
        "best paired speedup {speedup}x, best code-cache storage reduction {storage}%"
    ).format(
        status=sweep.get("status", "missing"),
        passes=fmt(sweep.get("pass_count")),
        profiles=fmt(sweep.get("profile_count")),
        cards=fmt(sweep.get("max_active_process_cards")),
        aicore=fmt(sweep.get("samples_with_8_nonzero_aicore")),
        pairs=pair_text,
        kv_text=kv_text,
        speedup=fmt(best_speed.get("paired_latency_speedup")),
        storage=fmt(best_memory.get("code_cache_storage_reduction_pct")),
    )


def qwen_grouped_mlp_memory_tradeoff_readout(sweep: dict[str, Any] | None) -> str:
    if not isinstance(sweep, dict) or not sweep:
        return "Qwen3 grouped-MLP memory-first tradeoff missing"
    row = sweep.get("_memory_tradeoff_row") if isinstance(sweep.get("_memory_tradeoff_row"), dict) else {}
    scaled_budget = row.get("scaled_code_dtype_cache_budget_bytes")
    if scaled_budget is None:
        scaled_budget = 0
    dense_budget = row.get("dense_cache_budget_bytes")
    if dense_budget is None:
        dense_budget = 0
    return (
        "Qwen3 grouped-MLP memory-first tradeoff: {status}, label {label}, max active cards {cards}, "
        "all-8 AICore samples {aicore}, paired speedup {speedup}x, code-cache storage reduction {storage}%, "
        "scaled-code dtype-cache budget {scaled_budget} bytes, dense-cache budget {dense_budget} bytes"
    ).format(
        status=sweep.get("status", "missing"),
        label=row.get("label", "missing"),
        cards=fmt(sweep.get("max_active_process_cards")),
        aicore=fmt(sweep.get("samples_with_8_nonzero_aicore")),
        speedup=fmt(row.get("paired_latency_speedup")),
        storage=fmt(row.get("code_cache_storage_reduction_pct")),
        scaled_budget=fmt(scaled_budget),
        dense_budget=fmt(dense_budget),
    )


def qwen_grouped_mlp_sweep_evidence(sweep: dict[str, Any] | None) -> list[str]:
    if not isinstance(sweep, dict) or not sweep:
        return []
    artifact = sweep.get("_artifact_name")
    report = qwen_grouped_mlp_sweep_report_name(sweep)
    evidence: list[str] = []
    if artifact:
        evidence.append(f"artifacts/{artifact}")
    if report:
        evidence.append(f"reports/{report}")
    monitor_artifact = Path(str(sweep.get("monitor_artifact") or "")).name
    if monitor_artifact:
        evidence.append(f"artifacts/{monitor_artifact}")
    return evidence
