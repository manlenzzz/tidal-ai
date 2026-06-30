"""Helpers for Qwen QPruner native bits tradeoff demo readouts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

QWEN_NATIVE_8CARD_PROFILE_SUMMARY_PATTERN = "qwen_qpruner_native_profile_8card*_summary_*.json"


def fmt(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def qwen_native_profile_report_name(artifact_name: str | None) -> str | None:
    if not artifact_name:
        return None
    prefix = "qwen_qpruner_native_profile_"
    if not artifact_name.startswith(prefix) or not artifact_name.endswith(".json"):
        return None
    run_label = artifact_name[len(prefix) : -len(".json")]
    return f"qwen-qpruner-native-profile-{run_label}.md"


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def qwen_native_8card_profile_report_name(artifact_name: str | None) -> str | None:
    if not artifact_name:
        return None
    prefix = "qwen_qpruner_native_profile_"
    if not artifact_name.startswith(prefix) or not artifact_name.endswith(".json"):
        return None
    run_label = artifact_name[len(prefix) : -len(".json")]
    if run_label.startswith("8card_"):
        run_label = run_label.replace("_", "-")
    return f"qwen-qpruner-native-profile-{run_label}.md"


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


def qwen_native_8card_profile_aggregate(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {}
    aggregate = summary.get("aggregate")
    return aggregate if isinstance(aggregate, dict) else {}


def qwen_native_8card_profile_value(summary: dict[str, Any], *keys: str) -> Any:
    aggregate = qwen_native_8card_profile_aggregate(summary)
    for key in keys:
        if summary.get(key) is not None:
            return summary.get(key)
        if aggregate.get(key) is not None:
            return aggregate.get(key)
    return None


def best_qwen_native_8card_profile_summary(artifacts: Path) -> dict[str, Any] | None:
    candidates: list[tuple[tuple[int, int, int, int, int, float, float], dict[str, Any]]] = []
    for path in artifacts.glob(QWEN_NATIVE_8CARD_PROFILE_SUMMARY_PATTERN):
        payload = read_json(path)
        if payload is None:
            continue
        monitor = payload.get("monitor", {}) if isinstance(payload.get("monitor"), dict) else {}
        payload = {**payload, "_artifact_name": path.name}
        candidates.append(
            (
                (
                    int(payload.get("status") == "PASS"),
                    int_score(qwen_native_8card_profile_value(payload, "memory_native_worker_count")),
                    int_score(payload.get("pass_count")),
                    int_score(monitor.get("samples_with_8_nonzero_aicore")),
                    int_score(monitor.get("samples_with_8_active_process_cards")),
                    float_score(qwen_native_8card_profile_value(payload, "min_code_cache_storage_reduction_pct")),
                    path.stat().st_mtime,
                ),
                payload,
            )
        )
    return max(candidates, key=lambda row: row[0])[1] if candidates else None


def qwen_native_8card_profile_artifact_name(summary: dict[str, Any] | None) -> str | None:
    if not summary:
        return None
    artifact = summary.get("_artifact_name") or summary.get("artifact")
    return str(artifact) if artifact else None


def qwen_native_8card_profile_monitor_artifact_name(summary: dict[str, Any] | None) -> str | None:
    if not summary:
        return None
    monitor = summary.get("monitor", {}) if isinstance(summary.get("monitor"), dict) else {}
    artifact = monitor.get("artifact") or summary.get("monitor_artifact")
    if not artifact:
        return None
    artifact_text = str(artifact)
    prefix = "artifacts/"
    return artifact_text[len(prefix) :] if artifact_text.startswith(prefix) else artifact_text


def qwen_native_8card_profile_readout(summary: dict[str, Any] | None) -> str:
    if not summary:
        return ""
    monitor = summary.get("monitor", {}) if isinstance(summary.get("monitor"), dict) else {}
    memory_workers = qwen_native_8card_profile_value(summary, "memory_native_worker_count")
    target_limit = qwen_native_8card_profile_value(summary, "target_layer_limit", "target_layer_limit_min")
    targeted_total = qwen_native_8card_profile_value(
        summary,
        "targeted_layers_total",
        "targeted_layers_total_max",
    )
    storage = qwen_native_8card_profile_value(summary, "min_code_cache_storage_reduction_pct")
    released = qwen_native_8card_profile_value(summary, "released_packed_code_bytes_total")
    live_payload = qwen_native_8card_profile_value(summary, "live_compressed_payload_storage_bytes_total")
    return (
        "Qwen3 8-card memory-native native profile {status}, workers {memory}/{passes}, "
        "target layers {limit}/{total}, max_new_tokens {tokens}, "
        "code-cache storage {storage}% reduction, released packed-code bytes {released}, "
        "live compressed payload bytes {payload}, monitor {monitor_status}, "
        "{process_samples} all-8 process samples, {aicore_samples} all-8 AICore samples"
    ).format(
        status=summary.get("status", "missing"),
        memory=fmt(memory_workers),
        passes=fmt(summary.get("pass_count")),
        limit=fmt(target_limit),
        total=fmt(targeted_total),
        tokens=fmt(summary.get("max_new_tokens")),
        storage=fmt(storage),
        released=fmt(float(released))
        if released is not None
        else fmt(None),
        payload=fmt(float(live_payload))
        if live_payload is not None
        else fmt(None),
        monitor_status=monitor.get("status", "missing"),
        process_samples=fmt(monitor.get("samples_with_8_active_process_cards")),
        aicore_samples=fmt(monitor.get("samples_with_8_nonzero_aicore")),
    )


def qwen_native_8card_profile_evidence(summary: dict[str, Any] | None) -> list[str]:
    evidence: list[str] = []
    artifact = qwen_native_8card_profile_artifact_name(summary)
    monitor_artifact = qwen_native_8card_profile_monitor_artifact_name(summary)
    report = qwen_native_8card_profile_report_name(artifact)
    if artifact:
        evidence.append(f"artifacts/{artifact}")
    if monitor_artifact:
        evidence.append(f"artifacts/{monitor_artifact}")
    if report:
        evidence.append(f"reports/{report}")
    return evidence


def qwen_native_bits_tradeoff_rows(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not report:
        return []
    rows = report.get("qwen_qpruner_native_bits_tradeoff")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def qwen_native_bits_tradeoff_readout(report: dict[str, Any] | None) -> str:
    parts: list[str] = []
    for row in qwen_native_bits_tradeoff_rows(report):
        speedups = row.get("speedups", {}) if isinstance(row.get("speedups"), dict) else {}
        parts.append(
            "{bits} {speedup}x/{packed}% packed storage/{code}% code-cache storage".format(
                bits=row.get("bits_label", "bits?"),
                speedup=fmt(speedups.get("qpruner_vs_baseline")),
                packed=fmt(row.get("qpruner_storage_reduction_pct")),
                code=fmt(row.get("qpruner_code_cache_storage_reduction_pct")),
            )
        )
    if not parts:
        return ""
    return "Qwen3 native bits tradeoff: " + "; ".join(parts)


def qwen_native_profile_readout(report: dict[str, Any] | None) -> str:
    if not report:
        return ""
    profile = report.get("qwen_qpruner_native_profile")
    if not isinstance(profile, dict):
        return ""
    speedups = profile.get("speedups", {}) if isinstance(profile.get("speedups"), dict) else {}
    paired_rounds = profile.get("paired_rounds")
    measurement_basis = profile.get("measurement_basis")
    label = "paired native profile" if paired_rounds else "native profile"
    return (
        "Qwen3 {label}: {speedup}x baseline, measurement {measurement}, "
        "paired_rounds {paired}, code-cache storage {code_storage}%"
    ).format(
        label=label,
        speedup=fmt(speedups.get("qpruner_vs_baseline")),
        measurement=fmt(measurement_basis),
        paired=fmt(paired_rounds),
        code_storage=fmt(profile.get("qpruner_code_cache_storage_reduction_pct")),
    )


def qwen_native_bits_tradeoff_artifacts(report: dict[str, Any] | None) -> list[str]:
    artifacts: list[str] = []
    seen: set[str] = set()
    for row in qwen_native_bits_tradeoff_rows(report):
        artifact = row.get("artifact")
        if not isinstance(artifact, str) or not artifact or artifact in seen:
            continue
        seen.add(artifact)
        artifacts.append(artifact)
    return artifacts


def qwen_native_bits_tradeoff_reports(report: dict[str, Any] | None) -> list[str]:
    reports: list[str] = []
    seen: set[str] = set()
    for artifact in qwen_native_bits_tradeoff_artifacts(report):
        report_name = qwen_native_profile_report_name(artifact)
        if report_name is None or report_name in seen:
            continue
        seen.add(report_name)
        reports.append(report_name)
    return reports


def qwen_native_bits_tradeoff_evidence(report: dict[str, Any] | None) -> list[str]:
    return [
        *(f"artifacts/{name}" for name in qwen_native_bits_tradeoff_artifacts(report)),
        *(f"reports/{name}" for name in qwen_native_bits_tradeoff_reports(report)),
    ]
