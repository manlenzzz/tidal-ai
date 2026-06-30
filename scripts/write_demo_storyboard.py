#!/usr/bin/env python
"""Write a recording-oriented storyboard from preserved Ascend demo artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from tidal.reports.baselines import engineering_baseline_note, paper_baseline_note
from tidal.reports.multicard_selection import (
    best_multicard_qwen_lora_finetune,
    multicard_qwen_lora_finetune_report_name,
)
from tidal.reports.npu_monitor_selection import (
    best_npu_utilization_monitor,
    npu_monitor_peak_score,
)
from tidal.reports.qwen_grouped_mlp_sweep import (
    best_qwen_grouped_mlp_memory_tradeoff,
    best_qwen_grouped_mlp_sweep,
    qwen_grouped_mlp_memory_tradeoff_readout,
    qwen_grouped_mlp_sweep_evidence,
    qwen_grouped_mlp_sweep_readout,
    qwen_grouped_mlp_sweep_report_name,
)
from tidal.reports.qwen_grouped_replay_selection import (
    best_multicard_qwen_qpruner_grouped_replay,
    multicard_qwen_qpruner_grouped_replay_report_name,
)
from tidal.reports.qwen_native_tradeoff import (
    best_qwen_native_8card_profile_summary,
    qwen_native_8card_profile_evidence,
    qwen_native_8card_profile_readout,
    qwen_native_bits_tradeoff_artifacts,
    qwen_native_bits_tradeoff_readout,
    qwen_native_bits_tradeoff_reports,
    qwen_native_profile_readout,
)


QPRUNER_PACKED_DECODE_ARTIFACT = "qpruner_packed_decode_benchmark_tiny_qwen3_qpruner_decode_npu.json"
QPRUNER_PACKED_DECODE_REPORT = "qpruner-packed-decode-benchmark-tiny_qwen3_qpruner_decode_npu.md"
QPRUNER_SHAPE_SWEEP_ARTIFACT = "qpruner_packed_decode_benchmark_qwen3_06b_shape_sweep_npu.json"
QPRUNER_SHAPE_SWEEP_REPORT = "qpruner-packed-decode-benchmark-qwen3_06b_shape_sweep_npu.md"
QPRUNER_DTYPE_SHAPE_SWEEP_ARTIFACT = "qpruner_packed_decode_benchmark_qwen3_06b_shape_sweep_dtype_preserving_npu.json"
QPRUNER_DTYPE_SHAPE_SWEEP_REPORT = "qpruner-packed-decode-benchmark-qwen3_06b_shape_sweep_dtype_preserving_npu.md"
CHOICE_ACCURACY_ARTIFACT = "qwen_compression_choice_accuracy_qwen3_06b_choice_accuracy_npu.json"
CHOICE_ACCURACY_REPORT = "qwen-compression-choice-accuracy-qwen3_06b_choice_accuracy_npu.md"
MULTITASK_CHOICE_ACCURACY_ARTIFACT = "qwen_compression_choice_accuracy_qwen3_06b_multitask_choice_npu.json"
MULTITASK_CHOICE_ACCURACY_REPORT = "qwen-compression-choice-accuracy-qwen3_06b_multitask_choice_npu.md"
FULLTARGET_QPRUNER_CHOICE_ARTIFACT = (
    "qwen_compression_choice_accuracy_qwen3_06b_multitask_choice_fulltarget_qpruner_npu.json"
)
FULLTARGET_QPRUNER_CHOICE_REPORT = (
    "qwen-compression-choice-accuracy-qwen3_06b_multitask_choice_fulltarget_qpruner_npu.md"
)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def first_existing_json(artifacts: Path, names: Sequence[str]) -> dict[str, Any] | None:
    for name in names:
        payload = read_json(artifacts / name)
        if payload is not None:
            return payload
    return None


def artifact_from_payload(payload: dict[str, Any] | None, default: str) -> str:
    return str((payload or {}).get("_artifact_name") or default)


def report_from_payload(payload: dict[str, Any] | None, default: str) -> str:
    return str((payload or {}).get("_report_name") or default)


def qpruner_shape_sweep_report_name(artifact_name: str | None) -> str | None:
    if not artifact_name:
        return None
    prefix = "qpruner_packed_decode_benchmark_"
    if not artifact_name.startswith(prefix) or not artifact_name.endswith(".json"):
        return None
    run_label = artifact_name[len(prefix) : -len(".json")]
    return f"qpruner-packed-decode-benchmark-{run_label}.md"


def qwen_native_profile_report_name(artifact_name: str | None) -> str | None:
    if not artifact_name:
        return None
    prefix = "qwen_qpruner_native_profile_"
    if not artifact_name.startswith(prefix) or not artifact_name.endswith(".json"):
        return None
    run_label = artifact_name[len(prefix) : -len(".json")]
    return f"qwen-qpruner-native-profile-{run_label}.md"


def selected_qpruner_shape_sweep(inference_bottleneck: dict[str, Any] | None) -> dict[str, Any]:
    if not inference_bottleneck:
        return {}
    sweep = inference_bottleneck.get("qpruner_packed_decode_shape_sweep")
    return sweep if isinstance(sweep, dict) else {}


def selected_qwen_native_profile(inference_bottleneck: dict[str, Any] | None) -> dict[str, Any]:
    if not inference_bottleneck:
        return {}
    profile = inference_bottleneck.get("qwen_qpruner_native_profile")
    return profile if isinstance(profile, dict) else {}


def selected_qwen_speed_first_profile(inference_bottleneck: dict[str, Any] | None) -> dict[str, Any]:
    if not inference_bottleneck:
        return {}
    profile = inference_bottleneck.get("qwen_qpruner_native_speed_first_profile")
    return profile if isinstance(profile, dict) else {}


def best_vllm_parallel_suite(artifacts: Path) -> dict[str, Any] | None:
    candidates: list[tuple[tuple[int, int, float], dict[str, Any]]] = []
    for path in artifacts.glob("multicard_parallel_suite_vllm_metadata_sync*_metrics.json"):
        payload = read_json(path)
        if payload is None:
            continue
        payload["_artifact_name"] = path.name
        status_score = 1 if payload.get("status") == "PASS" else 0
        try:
            world_size = int(payload.get("world_size") or 0)
        except (TypeError, ValueError):
            world_size = 0
        candidates.append(((status_score, world_size, path.stat().st_mtime), payload))
    return max(candidates, key=lambda row: row[0])[1] if candidates else None


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


def multicard_qwen_qpruner_grouped_replay_text(replay: dict[str, Any] | None) -> str:
    if not isinstance(replay, dict) or not replay:
        return "8-card grouped replay: missing; Qwen3 QPruner grouped replay memory-native: missing"
    aggregate = replay.get("aggregate", {}) if isinstance(replay.get("aggregate"), dict) else {}
    released_bytes = aggregate.get("released_packed_code_bytes_total")
    live_payload_bytes = aggregate.get(
        "live_compressed_payload_storage_bytes_total",
        aggregate.get("live_compressed_payload_bytes_total"),
    )
    return (
        "8-card grouped replay: {status}; Qwen3 QPruner grouped replay memory-native: world {world}, "
        "memory-native workers {memory}/{passes}, code-cache storage {storage}%, "
        "best grouped speedup {speedup}x, released packed-code bytes {released}, "
        "live compressed payload bytes {payload}"
    ).format(
        status=replay.get("status", "missing"),
        world=replay.get("world_size", "missing"),
        memory=fmt(aggregate.get("memory_native_worker_count")),
        passes=fmt(aggregate.get("pass_count")),
        storage=fmt(aggregate.get("min_code_cache_storage_reduction_pct")),
        speedup=fmt(aggregate.get("best_grouped_speedup")),
        released=fmt(float(released_bytes)) if released_bytes is not None else fmt(None),
        payload=fmt(float(live_payload_bytes)) if live_payload_bytes is not None else fmt(None),
    )


def npu_monitor_artifact_name(monitor: dict[str, Any] | None) -> str | None:
    if not monitor:
        return None
    artifact = monitor.get("_artifact_name")
    if artifact:
        return str(artifact)
    run_label = monitor.get("run_label")
    return f"npu_monitor_{run_label}.json" if run_label else None


def npu_monitor_log_path(monitor: dict[str, Any] | None, demo_root: Path) -> str | None:
    if not monitor or not monitor.get("monitor_log"):
        return None
    log_path = Path(str(monitor["monitor_log"]))
    try:
        return str(log_path.relative_to(demo_root))
    except ValueError:
        pass
    try:
        return str(log_path.resolve().relative_to(demo_root.resolve()))
    except (OSError, ValueError):
        return str(log_path)


def npu_monitor_aicore_peaks(monitor: dict[str, Any] | None) -> str:
    if not monitor or not isinstance(monitor.get("max_aicore_by_card"), dict):
        return "missing"
    peaks = monitor["max_aicore_by_card"]

    def key_order(key: Any) -> tuple[int, str]:
        try:
            return int(str(key)), str(key)
        except ValueError:
            return 9999, str(key)

    return ", ".join(f"{key}={peaks[key]}" for key in sorted(peaks, key=key_order)) or "missing"


def npu_monitor_readout(monitor: dict[str, Any] | None) -> str:
    if not monitor:
        return "NPU utilization monitor: missing"
    return (
        "NPU utilization monitor: {status}, {samples} samples, max active process cards {active}, "
        "{process_samples} samples with all 8 process cards, {aicore_samples} samples with all 8 AICore nonzero, "
        "per-card AICore peaks {peaks}"
    ).format(
        status=monitor.get("status", "missing"),
        samples=monitor.get("samples", "missing"),
        active=monitor.get("max_active_process_cards", "missing"),
        process_samples=monitor.get("samples_with_8_active_process_cards", "missing"),
        aicore_samples=monitor.get("samples_with_8_nonzero_aicore", "missing"),
        peaks=npu_monitor_aicore_peaks(monitor),
    )


def parallel_suite_artifact_name(parallel_suite: dict[str, Any] | None, fallback: str) -> str:
    return str((parallel_suite or {}).get("_artifact_name") or fallback)


def parallel_suite_report_name(parallel_suite: dict[str, Any] | None, fallback: str) -> str:
    run_label = (parallel_suite or {}).get("run_label")
    if not run_label:
        artifact_name = str((parallel_suite or {}).get("_artifact_name") or "")
        prefix = "multicard_parallel_suite_"
        suffix = ".json"
        if artifact_name.startswith(prefix) and artifact_name.endswith(suffix):
            run_label = artifact_name[len(prefix) : -len(suffix)]
    return f"multicard-parallel-suite-{run_label}.md" if run_label else fallback


def fmt(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def qwen_speed_first_readout(inference_bottleneck: dict[str, Any] | None) -> str | None:
    profile = selected_qwen_speed_first_profile(inference_bottleneck)
    if not profile:
        return None
    return (
        "Qwen3 speed-first native profile: {speedup}x baseline, measurement {measurement}, "
        "paired_rounds {paired}, code-cache storage {code_storage}%, scaled-code dtype-cache bytes "
        "{scaled_bytes}, dense-cache bytes {dense_bytes}, dense-cache budget bytes {dense_budget}, "
        "prebuild dtype cache {prebuild}"
    ).format(
        speedup=fmt((profile.get("speedups") or {}).get("qpruner_vs_baseline")),
        measurement=fmt(profile.get("measurement_basis")),
        paired=fmt(profile.get("paired_rounds")),
        code_storage=fmt(profile.get("qpruner_code_cache_storage_reduction_pct")),
        scaled_bytes=fmt(profile.get("qpruner_cached_scaled_code_bytes")),
        dense_bytes=fmt(profile.get("qpruner_cached_dense_weight_bytes")),
        dense_budget=fmt(profile.get("qpruner_dense_cache_budget_bytes")),
        prebuild=profile.get("qpruner_prebuild_scaled_code_dtype_cache", "missing"),
    )


def fmt_accuracy(value: Any) -> str:
    if value is None:
        return "missing"
    try:
        return f"{float(value) * 100.0:.3f}%"
    except (TypeError, ValueError):
        return str(value)


def task_counts_text(counts: dict[str, Any] | None) -> str:
    if not counts:
        return "missing"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def issue_classes(probe: dict[str, Any] | None) -> str:
    if not probe:
        return "missing"
    classes = [
        str(row.get("id"))
        for row in probe.get("issue_classes", [])
        if isinstance(row, dict) and row.get("id")
    ]
    return ", ".join(classes) if classes else "none"


def missing_backend_parameters(probe: dict[str, Any] | None) -> str:
    if not probe:
        return "missing"
    api = probe.get("api_diagnostics", {})
    if not isinstance(api, dict):
        return "missing"
    values = api.get("missing_patch_get_attn_backend_parameters") or []
    return ", ".join(str(value) for value in values) if values else "none"


def first_diagnosis_title(report: dict[str, Any] | None) -> str:
    diagnoses = report.get("diagnoses", []) if report else []
    for item in diagnoses:
        if isinstance(item, dict) and item.get("title"):
            return str(item["title"])
    return "none" if report else "missing"


def diagnosis_details_text(report: dict[str, Any] | None) -> str:
    diagnoses = report.get("diagnoses", []) if report else []
    parts: list[str] = []
    for item in diagnoses:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        evidence = item.get("evidence")
        if title and evidence:
            parts.append(f"{title}: {evidence}")
        elif title:
            parts.append(str(title))
        elif evidence:
            parts.append(str(evidence))
    return "; ".join(parts) if parts else "no detailed diagnosis"


def next_optimization_title(report: dict[str, Any] | None) -> str:
    target = report.get("next_optimization_target", {}) if report else {}
    if isinstance(target, dict) and target.get("title"):
        return str(target["title"])
    return "missing"


def resource_blocker_text(report: dict[str, Any] | None) -> str:
    blocker = report.get("resource_blocker", {}) if report else {}
    if not isinstance(blocker, dict) or blocker.get("status") != "BLOCKED":
        return "resource blocker none"
    return (
        "Resource blocker {status}: {occupied}/{total} cards occupied by {container} running {model} "
        "with tensor_parallel_size={tp}; policy {policy}"
    ).format(
        status=blocker.get("status", "missing"),
        occupied=fmt(blocker.get("occupied_cards")),
        total=fmt(blocker.get("total_cards")),
        container=blocker.get("container_name", "missing"),
        model=blocker.get("service_model", "missing"),
        tp=fmt(blocker.get("tensor_parallel_size")),
        policy=blocker.get("policy", "missing"),
    )


def current_vllm_hbm_rerun_text(report: dict[str, Any] | None) -> str:
    current = report.get("current_vllm_rerun", {}) if report else {}
    if not isinstance(current, dict) or not current:
        return "current vLLM HBM rerun missing"
    cards = current.get("method_cards")
    card_text = ",".join(str(card) for card in cards) if isinstance(cards, list) else "missing"
    artifact = current.get("artifact")
    artifact_text = f"artifacts/{artifact}" if artifact else "missing"
    root_cause = current.get("root_cause") or "none"
    speedups = current.get("speedups", {}) if isinstance(current.get("speedups"), dict) else {}
    return (
        "Current vLLM HBM rerun: {status}, parallel={parallel}, cards {cards}, "
        "max_new_tokens={tokens}, best {best} {best_tps} tokens/s, "
        "qpruner_vs_baseline {q_speedup}x, artifact {artifact}, root cause {root_cause}"
    ).format(
        status=current.get("status", "missing"),
        parallel=current.get("parallel_methods", "missing"),
        cards=card_text,
        tokens=current.get("max_new_tokens", "missing"),
        best=current.get("best_method", "missing"),
        best_tps=fmt(current.get("best_tokens_per_s")),
        q_speedup=fmt(speedups.get("qpruner_vs_baseline")),
        artifact=artifact_text,
        root_cause=root_cause,
    )


def join_values(values: Any) -> str:
    if not isinstance(values, list):
        return "missing"
    return ", ".join(str(value) for value in values) or "missing"


def paper_baseline_evidence_text(report: dict[str, Any]) -> str:
    rows = []
    for item in report.get("paper_baseline_evidence", []):
        if not isinstance(item, dict):
            continue
        rows.append(
            "paper baseline evidence {method} {baseline} {status}".format(
                method=item.get("method", "missing"),
                baseline=item.get("paper_baseline", "missing"),
                status=item.get("status", "missing"),
            )
        )
    return "; ".join(rows) if rows else "paper baseline evidence missing"


def paper_baseline_audit_text(report: dict[str, Any] | None) -> str:
    if not report:
        return "Paper baseline audit missing"
    evidence = report.get("evidence", {}) if isinstance(report.get("evidence"), dict) else {}
    qpruner = evidence.get("qpruner_llm_pruner_full196", {})
    cap = evidence.get("cap_wanda_sparsegpt_full196", {})
    recovery = evidence.get("rankadaptor_recovery_controlled", {})
    rows = [f"Paper baseline audit {report.get('status', 'missing')}"]
    if isinstance(qpruner, dict) and qpruner:
        rows.append(
            "QPruner LLM-Pruner-style {status}, target layers {layers}/{total}, prune {prune}%, "
            "loss delta {loss}, speedup {speedup}x".format(
                status=qpruner.get("status", "missing"),
                layers=fmt(qpruner.get("targeted_layers")),
                total=fmt(qpruner.get("targeted_layers_total")),
                prune=fmt(qpruner.get("targeted_param_reduction_pct")),
                loss=fmt(qpruner.get("loss_delta")),
                speedup=fmt(qpruner.get("latency_speedup")),
            )
        )
    if isinstance(cap, dict) and cap:
        target_layers = "{layers}/{total}".format(
            layers=fmt(cap.get("targeted_layers")),
            total=fmt(cap.get("targeted_layers_total")),
        )
        rows.append(
            "CAP Wanda {status}, target layers {target_layers}, sparsity {sparsity}%, loss delta {loss}".format(
                status=cap.get("status", "missing"),
                target_layers=target_layers,
                sparsity=fmt(cap.get("wanda_targeted_param_reduction_pct")),
                loss=fmt(cap.get("wanda_loss_delta")),
            )
        )
        rows.append(
            "CAP SparseGPT {status}, target layers {target_layers}, sparsity {sparsity}%, loss delta {loss}".format(
                status=cap.get("status", "missing"),
                target_layers=target_layers,
                sparsity=fmt(cap.get("sparsegpt_targeted_param_reduction_pct")),
                loss=fmt(cap.get("sparsegpt_loss_delta")),
            )
        )
    if isinstance(recovery, dict) and recovery:
        rows.append(
            "RankAdaptor recovery baselines {status}, best {best}, no-recovery delta {none}, "
            "LoRA delta {lora}, AdaLoRA-style delta {adalora}".format(
                status=recovery.get("status", "missing"),
                best=recovery.get("best_recovery_method", "missing"),
                none=fmt(recovery.get("no_recovery_loss_delta")),
                lora=fmt(recovery.get("lora_loss_delta")),
                adalora=fmt(recovery.get("adalora_style_loss_delta")),
            )
        )
    return "; ".join(rows)


def compression_memory_video_readout(report: dict[str, Any]) -> str:
    if report.get("video_readout"):
        return str(report["video_readout"])
    memory = report.get("compression_memory", {}) if isinstance(report.get("compression_memory"), dict) else {}
    gap = report.get("memory_gap", {}) if isinstance(report.get("memory_gap"), dict) else {}
    exports = report.get("serving_export_memory", {}) if isinstance(report.get("serving_export_memory"), dict) else {}
    cap_memory = memory.get("cap", {}) if isinstance(memory.get("cap"), dict) else {}
    q_memory = memory.get("qpruner", {}) if isinstance(memory.get("qpruner"), dict) else {}
    cap_export = exports.get("cap", {}) if isinstance(exports.get("cap"), dict) else {}
    q_export = exports.get("qpruner", {}) if isinstance(exports.get("qpruner"), dict) else {}
    cap_ratio = cap_export.get("size_ratio_vs_baseline") or cap_export.get("size_ratio_vs_engineering_reference")
    q_ratio = q_export.get("size_ratio_vs_baseline") or q_export.get("size_ratio_vs_engineering_reference")
    if gap.get("dense_export_erases_storage_savings"):
        cap_ratio = 1.0 if cap_ratio is None else cap_ratio
        q_ratio = 1.0 if q_ratio is None else q_ratio
    return (
        "Memory-first compression readout: CAP targeted memory -{cap}%, QPruner targeted memory -{q}%, "
        "target layers {limit}/{total}; dense HF export ratio CAP/QPruner {cap_ratio}x/{q_ratio}x, "
        "so memory savings require the compressed-native runtime path."
    ).format(
        cap=fmt(cap_memory.get("targeted_param_reduction_pct")),
        q=fmt(q_memory.get("targeted_param_reduction_pct")),
        limit=fmt(memory.get("target_layer_limit")),
        total=fmt(memory.get("targeted_layers_total")),
        cap_ratio=fmt(cap_ratio),
        q_ratio=fmt(q_ratio),
    )


def compression_memory_text(
    report: dict[str, Any] | None, *, include_legacy_baseline_evidence: bool = True
) -> str:
    if not report:
        return "Compression memory report missing"
    paper = report.get("paper_baselines", {}) if isinstance(report.get("paper_baselines"), dict) else {}
    memory = report.get("compression_memory", {}) if isinstance(report.get("compression_memory"), dict) else {}
    gap = report.get("memory_gap", {}) if isinstance(report.get("memory_gap"), dict) else {}
    qpruner = paper.get("qpruner", {}) if isinstance(paper.get("qpruner"), dict) else {}
    cap = paper.get("cap", {}) if isinstance(paper.get("cap"), dict) else {}
    return (
        "Compression memory report {status}: {readout}; QPruner paper baseline {q_base}; "
        "CAP paper baselines {cap_base}; memory reductions CAP {cap_mem}%, QPruner {q_mem}%; "
        "dense export erases storage savings {erases}; {evidence}; {alignment}"
    ).format(
        status=report.get("status", "missing"),
        readout=compression_memory_video_readout(report),
        q_base=join_values(qpruner.get("primary")),
        cap_base=join_values(cap.get("pruning_baselines")),
        cap_mem=fmt((memory.get("cap", {}) if isinstance(memory.get("cap"), dict) else {}).get("targeted_param_reduction_pct")),
        q_mem=fmt(
            (memory.get("qpruner", {}) if isinstance(memory.get("qpruner"), dict) else {}).get(
                "targeted_param_reduction_pct"
            )
        ),
        erases=gap.get("dense_export_erases_storage_savings", "missing"),
        evidence=paper_baseline_evidence_text(report) if include_legacy_baseline_evidence else "paper baseline audit supersedes legacy evidence",
        alignment=baseline_alignment_text(report),
    )


def baseline_alignment_text(report: dict[str, Any]) -> str:
    rows = []
    for item in report.get("baseline_alignment_matrix", []):
        if not isinstance(item, dict):
            continue
        rows.append(
            "baseline alignment {method} {role} {baselines} -> {demo_role}".format(
                method=item.get("method", "missing"),
                role=item.get("paper_baseline_role", "missing"),
                baselines=join_values(item.get("paper_baselines")),
                demo_role=item.get("demo_reference_role", "missing"),
            )
        )
    return "; ".join(rows) if rows else "baseline alignment missing"


def quality_memory_sweep_text(demo_root: Path) -> str:
    path = demo_root / "reports" / "qwen-compression-quality-memory-sweep.md"
    if not path.exists():
        return "Quality-memory sweep missing"
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("Best memory-quality point"):
            return f"Quality-memory sweep: {line}"
    return "Quality-memory sweep present without frontier summary"


def qpruner_quality_memory_sweep_text(demo_root: Path) -> str:
    path = demo_root / "reports" / "qwen-qpruner-quality-memory-sweep.md"
    if not path.exists():
        return "QPruner-only quality-memory sweep missing"
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("Best memory-quality point"):
            return f"QPruner-only quality-memory sweep: {line}"
    return "QPruner-only quality-memory sweep present without frontier summary"


def qpruner_scale_quality_text(demo_root: Path) -> str:
    payload = read_json(demo_root / "artifacts" / "qwen_qpruner_scale_quality_summary.json")
    if isinstance(payload, dict) and payload.get("readout"):
        return str(payload["readout"])
    path = demo_root / "reports" / "qwen-qpruner-scale-quality-summary.md"
    if not path.exists():
        return "QPruner scale-quality summary missing"
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("QPruner scale-quality sweep"):
            return line
    return "QPruner scale-quality summary present without readout"


def compressed_native_text(report: dict[str, Any] | None) -> str:
    if not report:
        return "Compressed-native torch_npu reference missing"
    baseline = report.get("baseline", {}) if isinstance(report.get("baseline"), dict) else {}
    cap = report.get("cap", {}) if isinstance(report.get("cap"), dict) else {}
    qpruner = report.get("qpruner", {}) if isinstance(report.get("qpruner"), dict) else {}
    summary = report.get("summary", {}) if isinstance(report.get("summary"), dict) else {}
    memory = report.get("memory_reference", {}) if isinstance(report.get("memory_reference"), dict) else {}
    return (
        "Compressed-native torch_npu reference {status}: baseline {baseline} tokens/s, "
        "CAP {cap_tps} tokens/s, QPruner {q_tps} tokens/s; packed {packed} / quantized {quantized}; "
        "dense export {dense}; qpruner_vs_baseline {speedup}x; "
        "storage bytes baseline {baseline_storage}, CAP {cap_storage}, QPruner {q_storage}; "
        "storage reductions CAP {cap_storage_reduction}%, QPruner {q_storage_reduction}%"
    ).format(
        status=report.get("status", "missing"),
        baseline=fmt(baseline.get("tokens_per_s")),
        cap_tps=fmt(cap.get("tokens_per_s")),
        q_tps=fmt(qpruner.get("tokens_per_s")),
        packed=fmt(cap.get("packed_layers")),
        quantized=fmt(qpruner.get("quantized_layers")),
        dense=report.get("serving_dense_export", "missing"),
        speedup=fmt(summary.get("qpruner_vs_baseline_speedup")),
        baseline_storage=fmt(memory.get("baseline_targeted_storage_bytes")),
        cap_storage=fmt(memory.get("cap_targeted_storage_bytes")),
        q_storage=fmt(memory.get("qpruner_targeted_storage_bytes")),
        cap_storage_reduction=fmt(memory.get("cap_targeted_storage_reduction_pct")),
        q_storage_reduction=fmt(memory.get("qpruner_targeted_storage_reduction_pct")),
    )


def inference_acceleration_summary_text(summary: dict[str, Any] | None) -> str:
    if not summary:
        return "Inference acceleration summary missing"
    payload = summary.get("summary", {}) if isinstance(summary.get("summary"), dict) else {}
    best = payload.get("best_throughput", {}) if isinstance(payload.get("best_throughput"), dict) else {}
    native = payload.get("best_native_memory", {}) if isinstance(payload.get("best_native_memory"), dict) else {}
    bottleneck = summary.get("vllm_bottleneck", {}) if isinstance(summary.get("vllm_bottleneck"), dict) else {}
    return (
        "Inference acceleration summary {status}: best {track} / {method} {tokens} tokens/s; "
        "compressed-native preserves quantized/pruned modules with {native_method} memory {memory}%, "
        "storage {storage}%; bottleneck {bottleneck}; {engineering}; {paper}"
    ).format(
        status=summary.get("status", "missing"),
        track=best.get("track", "missing"),
        method=best.get("method_label") or best.get("method", "missing"),
        tokens=fmt(best.get("tokens_per_s")),
        native_method=native.get("method_label") or native.get("method", "missing"),
        memory=fmt(native.get("memory_reduction_pct")),
        storage=fmt(native.get("storage_reduction_pct")),
        bottleneck=bottleneck.get("title", "missing"),
        engineering=engineering_baseline_note(),
        paper=paper_baseline_note(),
    )


def compressed_native_sweep_text(summary: dict[str, Any] | None) -> str:
    if not summary:
        return "Compressed-native cache/long-decode sweep missing"
    best = summary.get("best", {}) if isinstance(summary.get("best"), dict) else {}
    cache = summary.get("cache_effect", {}) if isinstance(summary.get("cache_effect"), dict) else {}
    long_decode = (
        summary.get("long_decode_effect", {}) if isinstance(summary.get("long_decode_effect"), dict) else {}
    )
    diagnosis = summary.get("runtime_diagnosis", {}) if isinstance(summary.get("runtime_diagnosis"), dict) else {}
    return (
        "Compressed-native cache/long-decode sweep {status}: {readout}; "
        "cache qpruner delta {cache_delta}x; long decode qpruner delta {long_delta}x; "
        "runtime diagnosis QPruner scaling gap {gap}x, QPruner cache peak +{mem} MB, next {action}"
    ).format(
        status=summary.get("status", "missing"),
        readout=summary.get("readout", "missing"),
        cache_delta=fmt(cache.get("qpruner_speedup_delta")),
        long_delta=fmt(long_decode.get("qpruner_speedup_delta")),
        gap=fmt(diagnosis.get("qpruner_scaling_gap_vs_baseline")),
        mem=fmt(diagnosis.get("qpruner_cache_peak_mem_delta_mb")),
        action=diagnosis.get("next_action", "missing"),
    )


def qpruner_packed_decode_text(report: dict[str, Any] | None) -> str:
    if not report:
        return "QPruner packed decode benchmark missing"
    uncached = report.get("uncached", {}) if isinstance(report.get("uncached"), dict) else {}
    code_cached = report.get("code_cached", {}) if isinstance(report.get("code_cached"), dict) else {}
    scaled_code = report.get("scaled_code_matmul", {}) if isinstance(report.get("scaled_code_matmul"), dict) else {}
    cached = report.get("cached", {}) if isinstance(report.get("cached"), dict) else {}
    dense = report.get("dense", {}) if isinstance(report.get("dense"), dict) else {}
    return (
        "QPruner packed decode benchmark {status} on {device}: runtime {runtime}, storage {storage}%; "
        "code-cache storage {code_storage}%; uncached {uncached} ms, cache {cache} ms, dense {dense} ms; "
        "code-cache {code_latency} ms; code-cache speedup {code_speedup}x; "
        "scaled-code matmul {scaled_latency} ms; scaled-code speedup {scaled_speedup}x; "
        "scaled-code peak {scaled_peak} MB; cache speedup {speedup}x; next {action}"
    ).format(
        status=report.get("status", "missing"),
        device=report.get("device", "missing"),
        runtime=report.get("runtime_storage_format", "missing"),
        storage=fmt(report.get("storage_reduction_pct")),
        code_storage=fmt(report.get("code_cache_storage_reduction_pct")),
        uncached=fmt(uncached.get("latency_ms")),
        code_latency=fmt(code_cached.get("latency_ms")),
        cache=fmt(cached.get("latency_ms")),
        dense=fmt(dense.get("latency_ms")),
        code_speedup=fmt(code_cached.get("speedup_vs_uncached")),
        scaled_latency=fmt(scaled_code.get("latency_ms")),
        scaled_speedup=fmt(scaled_code.get("speedup_vs_uncached")),
        scaled_peak=fmt(scaled_code.get("peak_mem_mb")),
        speedup=fmt(cached.get("speedup_vs_uncached")),
        action=report.get("next_action", "missing"),
    )


def multicard_compression_sync_text(summary: dict[str, Any] | None) -> str:
    if not summary:
        return "Multi-card compression sync missing"
    generate = (
        summary.get("qwen_compression_generate", {})
        if isinstance(summary.get("qwen_compression_generate"), dict)
        else {}
    )
    memory = (
        summary.get("compressed_native_memory", {})
        if isinstance(summary.get("compressed_native_memory"), dict)
        else {}
    )
    parallel = summary.get("parallel_suite", {}) if isinstance(summary.get("parallel_suite"), dict) else {}
    artifact = generate.get("artifact")
    report = generate.get("report")
    return (
        "Multi-card compression sync {status}: {readout}; "
        "target layers {limit}/{total}, artifact {artifact}, report {report}; "
        "QPruner storage reduction {storage}%, cache peak +{mem} MB; "
        "sync start window {window}s"
    ).format(
        status=summary.get("status", "missing"),
        readout=summary.get("readout", "missing"),
        limit=fmt(generate.get("target_layer_limit")),
        total=fmt(generate.get("targeted_layers_total")),
        artifact=f"artifacts/{artifact}" if artifact else "missing",
        report=f"reports/{report}" if report else "missing",
        storage=fmt(memory.get("qpruner_storage_reduction_pct")),
        mem=fmt(memory.get("qpruner_cache_peak_mem_delta_mb")),
        window=fmt(parallel.get("start_window_s")),
    )


def choice_accuracy_text(report: dict[str, Any] | None) -> str:
    if not report:
        return "Choice accuracy missing"
    methods = report.get("methods", {}) if isinstance(report.get("methods"), dict) else {}
    baseline = methods.get("baseline", {}) if isinstance(methods.get("baseline"), dict) else {}
    cap = methods.get("cap", {}) if isinstance(methods.get("cap"), dict) else {}
    qpruner = methods.get("qpruner", {}) if isinstance(methods.get("qpruner"), dict) else {}
    prefix = "multi-task choice accuracy" if int(report.get("task_count") or 0) > 1 else "choice accuracy"
    sample_count = baseline.get("total") or len(report.get("samples", []))
    sample_suffix = f"; samples {sample_count}" if sample_count else ""
    task_suffix = f"; tasks {report.get('task_count')}" if report.get("task_count") else ""
    if "cap" not in methods and "qpruner" in methods:
        qpruner_layers = qpruner.get("targeted_layers") or report.get("target_layer_limit", "missing")
        compression_time = qpruner.get("compression_time_s")
        compression_suffix = (
            f"; QPruner compression time {float(compression_time):.3f}s" if compression_time is not None else ""
        )
        return (
            "full-target QPruner choice accuracy baseline {baseline}, QPruner {qpruner}; "
            "choice accuracy target layers {limit}/{total}{sample_suffix}{task_suffix}{compression_suffix}"
        ).format(
            baseline=fmt_accuracy(baseline.get("accuracy")),
            qpruner=fmt_accuracy(qpruner.get("accuracy")),
            limit=qpruner_layers,
            total=report.get("targeted_layers_total", "missing"),
            sample_suffix=sample_suffix,
            task_suffix=task_suffix,
            compression_suffix=compression_suffix,
        )
    return (
        "{prefix} baseline {baseline}, CAP {cap}, QPruner {qpruner}; "
        "choice accuracy target layers {limit}/{total}{sample_suffix}{task_suffix}"
    ).format(
        prefix=prefix,
        baseline=fmt_accuracy(baseline.get("accuracy")),
        cap=fmt_accuracy(cap.get("accuracy")),
        qpruner=fmt_accuracy(qpruner.get("accuracy")),
        limit=report.get("target_layer_limit", "missing"),
        total=report.get("targeted_layers_total", "missing"),
        sample_suffix=sample_suffix,
        task_suffix=task_suffix,
    )


def objective_coverage_text(audit: dict[str, Any] | None) -> str:
    if not audit:
        return "Objective coverage audit missing"
    section_texts = []
    for item in audit.get("sections", []):
        if not isinstance(item, dict):
            continue
        section_texts.append(
            "{title} {status}: {summary}; gap {gap}".format(
                title=item.get("title", "missing"),
                status=item.get("status", "missing"),
                summary=item.get("evidence_summary", "missing"),
                gap=item.get("remaining_gap", "missing"),
            )
        )
    return "Objective coverage audit {status}: {sections}".format(
        status=audit.get("status", "missing"),
        sections=" | ".join(section_texts) if section_texts else "no sections",
    )


def markdown(demo_root: Path) -> str:
    artifacts = demo_root / "artifacts"
    inventory = read_json(artifacts / "model_inventory.json")
    qwen3_snapshot = read_json(artifacts / "model_snapshot_qwen3_06b.json")
    sync = read_json(artifacts / "multicard_sync_summary.json")
    qwen3_inference = read_json(artifacts / "ascend_inference_qwen3_06b_torch_npu.json")
    qwen3_multicard_inference = read_json(artifacts / "multicard_qwen_inference_qwen3_06b_npu.json")
    qwen3_quality = read_json(artifacts / "qwen_compression_quality_qwen3_06b_quality_npu.json")
    qwen3_pattern_quality = read_json(
        artifacts / "multicard_qwen_compression_quality_qwen3_06b_quality_pattern_2card_npu.json"
    )
    qwen3_pattern_generate = read_json(
        artifacts / "multicard_qwen_compression_generate_qwen3_06b_generate_pattern_2card_npu.json"
    )
    choice_accuracy = (
        read_json(artifacts / FULLTARGET_QPRUNER_CHOICE_ARTIFACT)
        or read_json(artifacts / MULTITASK_CHOICE_ACCURACY_ARTIFACT)
        or read_json(artifacts / CHOICE_ACCURACY_ARTIFACT)
    )
    compression_memory = read_json(artifacts / "compression_memory_report.json")
    paper_baseline_coverage_audit = read_json(artifacts / "paper_baseline_coverage_audit.json")
    qwen3_lora = best_multicard_qwen_lora_finetune(artifacts)
    qwen3_lora_report = report_from_payload(
        qwen3_lora,
        multicard_qwen_lora_finetune_report_name(
            artifact_from_payload(
                qwen3_lora,
                "multicard_qwen_lora_finetune_qwen3_06b_lora_sync_npu.json",
            )
        )
        or "multicard-qwen-lora-finetune-qwen3_06b_lora_sync_npu.md",
    )
    qwen3_bslora = read_json(artifacts / "tiny_qwen_bslora_finetune_qwen3_06b_bslora_npu.json")
    fallback = read_json(artifacts / "torch_serving_fallback_tiny_qwen3_serving_fallback_npu.json")
    compressed_native = read_json(artifacts / "compressed_native_torch_serving_tiny_qwen3_native_serving_npu.json")
    inference_summary = read_json(artifacts / "inference_acceleration_summary.json")
    compressed_native_sweep = read_json(artifacts / "compressed_native_sweep_summary.json")
    qpruner_packed_decode = read_json(artifacts / QPRUNER_PACKED_DECODE_ARTIFACT)
    multicard_compression_sync = read_json(artifacts / "multicard_compression_sync_summary.json")
    multicard_qwen_qpruner_grouped_replay = best_multicard_qwen_qpruner_grouped_replay(artifacts)
    qwen_grouped_mlp_sweep = best_qwen_grouped_mlp_sweep(artifacts)
    qwen_grouped_mlp_memory_tradeoff = best_qwen_grouped_mlp_memory_tradeoff(artifacts)
    qwen_native_8card_profile = best_qwen_native_8card_profile_summary(artifacts)
    npu_monitor = best_npu_utilization_monitor(artifacts)
    objective_coverage_audit = read_json(artifacts / "objective_coverage_audit.json")
    parallel = read_json(artifacts / "multicard_parallel_suite_demo_parallel_suite_npu.json")
    vllm_parallel = best_vllm_parallel_suite(artifacts)
    vllm_probe = read_json(artifacts / "vllm_serving_probe_tiny_qwen3_serving_export_npu.json")
    vllm_selector_shim_probe = read_json(
        artifacts / "vllm_serving_probe_tiny_qwen3_serving_export_npu_selector_shim.json"
    )
    vllm_serving_benchmark = first_existing_json(
        artifacts,
        [
            "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu.json",
            "vllm_serving_benchmark_tiny_qwen3_serving_vllm_selector_shim_npu.json",
        ],
    )
    inference_bottleneck = read_json(artifacts / "inference_bottleneck_report.json")
    qpruner_shape_sweep = selected_qpruner_shape_sweep(inference_bottleneck)
    qpruner_shape_sweep_artifact = qpruner_shape_sweep.get("artifact")
    qpruner_shape_sweep_report = qpruner_shape_sweep_report_name(qpruner_shape_sweep_artifact)
    qwen_native_profile = selected_qwen_native_profile(inference_bottleneck)
    qwen_native_profile_artifact = qwen_native_profile.get("artifact")
    qwen_native_profile_report = qwen_native_profile_report_name(qwen_native_profile_artifact)
    qwen_speed_first_profile = selected_qwen_speed_first_profile(inference_bottleneck)
    qwen_speed_first_artifact = qwen_speed_first_profile.get("artifact")
    qwen_speed_first_report = qwen_native_profile_report_name(qwen_speed_first_artifact)
    qwen_speed_first_readout_text = qwen_speed_first_readout(inference_bottleneck)
    qwen_native_profile_readout_text = qwen_native_profile_readout(inference_bottleneck)
    qwen_native_bits_readout = qwen_native_bits_tradeoff_readout(inference_bottleneck)
    qwen_native_bits_artifacts = qwen_native_bits_tradeoff_artifacts(inference_bottleneck)
    qwen_native_bits_reports = qwen_native_bits_tradeoff_reports(inference_bottleneck)
    qwen_native_8card_profile_readout_text = qwen_native_8card_profile_readout(qwen_native_8card_profile)
    qwen_native_8card_profile_evidence_paths = qwen_native_8card_profile_evidence(qwen_native_8card_profile)
    multicard_qwen_qpruner_grouped_replay_artifact = str(
        (multicard_qwen_qpruner_grouped_replay or {}).get("_artifact_name")
        or "multicard_qwen_qpruner_grouped_replay_missing.json"
    )
    multicard_qwen_qpruner_grouped_replay_report = multicard_qwen_qpruner_grouped_replay_report_name(
        multicard_qwen_qpruner_grouped_replay
    )
    multicard_qwen_qpruner_grouped_replay_readout = multicard_qwen_qpruner_grouped_replay_text(
        multicard_qwen_qpruner_grouped_replay
    )
    qwen_grouped_mlp_sweep_readout_text = qwen_grouped_mlp_sweep_readout(qwen_grouped_mlp_sweep)
    qwen_grouped_mlp_sweep_artifact = str(
        (qwen_grouped_mlp_sweep or {}).get("_artifact_name")
        or "qwen3_06b_native_profile_8card_grouped_mlp_sweep_missing.json"
    )
    qwen_grouped_mlp_sweep_report = qwen_grouped_mlp_sweep_report_name(qwen_grouped_mlp_sweep)
    qwen_grouped_mlp_memory_tradeoff_readout_text = qwen_grouped_mlp_memory_tradeoff_readout(
        qwen_grouped_mlp_memory_tradeoff
    )
    npu_monitor_artifact = npu_monitor_artifact_name(npu_monitor)
    npu_monitor_log = npu_monitor_log_path(npu_monitor, demo_root)

    target_models = []
    if inventory:
        for row in inventory.get("target_candidates", []):
            if isinstance(row, dict):
                target_models.append(f"{row.get('model_id', 'missing')} {row.get('status', 'missing')}")
    fallback_exports = fallback.get("exports", {}) if fallback else {}
    fallback_summary = fallback.get("summary", {}) if fallback else {}
    baseline = fallback_exports.get("baseline", {}) if isinstance(fallback_exports.get("baseline"), dict) else {}
    cap = fallback_exports.get("cap", {}) if isinstance(fallback_exports.get("cap"), dict) else {}
    qpruner = fallback_exports.get("qpruner", {}) if isinstance(fallback_exports.get("qpruner"), dict) else {}
    qwen3_multi_aggregate = (
        qwen3_multicard_inference.get("aggregate", {}) if qwen3_multicard_inference else {}
    )
    quality_cap = qwen3_quality.get("cap", {}) if qwen3_quality else {}
    quality_qpruner = qwen3_quality.get("qpruner", {}) if qwen3_quality else {}
    pattern_quality_aggregate = qwen3_pattern_quality.get("aggregate", {}) if qwen3_pattern_quality else {}
    pattern_generate_aggregate = qwen3_pattern_generate.get("aggregate", {}) if qwen3_pattern_generate else {}
    lora_aggregate = qwen3_lora.get("aggregate", {}) if qwen3_lora else {}
    vllm_parallel_aggregate = vllm_parallel.get("aggregate", {}) if vllm_parallel else {}
    vllm_parallel_artifact = parallel_suite_artifact_name(
        vllm_parallel,
        "multicard_parallel_suite_vllm_metadata_sync_npu_metrics.json",
    )
    vllm_parallel_report = parallel_suite_report_name(
        vllm_parallel,
        "multicard-parallel-suite-vllm_metadata_sync_npu_metrics.md",
    )
    vllm_benchmark_exports = vllm_serving_benchmark.get("exports", {}) if vllm_serving_benchmark else {}
    vllm_benchmark_summary = vllm_serving_benchmark.get("summary", {}) if vllm_serving_benchmark else {}
    vllm_baseline = (
        vllm_benchmark_exports.get("baseline", {})
        if isinstance(vllm_benchmark_exports.get("baseline"), dict)
        else {}
    )
    vllm_cap = (
        vllm_benchmark_exports.get("cap", {}) if isinstance(vllm_benchmark_exports.get("cap"), dict) else {}
    )
    vllm_qpruner = (
        vllm_benchmark_exports.get("qpruner", {})
        if isinstance(vllm_benchmark_exports.get("qpruner"), dict)
        else {}
    )

    lines = [
        "# TIDAL-AI Ascend 910B Demo Storyboard",
        "",
        f"Demo root: `{demo_root}`",
        "",
        "Use this as the terminal recording script. Keep raw artifacts and regenerated reports under the demo root.",
        "",
        "## Shot List",
        "",
        "| # | Shot | Open / run | What to say |",
        "|---:|---|---|---|",
        "| 1 | Hardware proof | `npu-smi info` | 8 x Ascend 910B visible |",
        "| 2 | Evidence bundle | `find /mnt/nvme/622/tidal-demo -maxdepth 2 -type f | sort` | Artifacts, logs, reports, and recordings are preserved |",
        "| 3 | 8-card sync | `reports/ascend-910b-demo-progress.md` | HCCL status {sync_status}, world size {world}, backend {backend} |".format(
            sync_status=(sync or {}).get("status", "missing"),
            world=(sync or {}).get("world_size", "missing"),
            backend=(sync or {}).get("backend", "missing"),
        ),
        "| 4 | Modern model path | `reports/model-inventory.md` | {models}; Qwen3-0.6B snapshot {snapshot_status} via {provider}, {mb} MB |".format(
            models="; ".join(target_models) or "model inventory missing",
            snapshot_status=(qwen3_snapshot or {}).get("status", "missing"),
            provider=(qwen3_snapshot or {}).get("provider", "missing"),
            mb=fmt((qwen3_snapshot or {}).get("parameter_mb")),
        ),
        "| 5 | Qwen3-0.6B runnable model | `reports/ascend-inference-qwen3_06b_torch_npu.md` | torch_npu PASS at {tps} tokens/s, peak {peak} MB |".format(
            tps=fmt((qwen3_inference or {}).get("tokens_per_s")),
            peak=fmt((qwen3_inference or {}).get("peak_mem_mb")),
        ),
        "| 6 | Qwen3-0.6B 8-card inference | `reports/multicard-qwen-inference-qwen3_06b_npu.md` | {tps} tokens/s total, all-reduce consistent {consistent} |".format(
            tps=fmt(qwen3_multi_aggregate.get("tokens_per_s_total")),
            consistent=qwen3_multi_aggregate.get("distributed_reduce_consistent", "missing"),
        ),
        "| 7 | Compression quality | `reports/qwen-compression-quality-qwen3_06b_quality_npu.md` | target layers {limit}/{total}, baseline loss {loss}, CAP delta {cap_delta}, QPruner delta {q_delta} |".format(
            limit=(qwen3_quality or {}).get("target_layer_limit", "missing"),
            total=(qwen3_quality or {}).get("targeted_layers_total", "missing"),
            loss=fmt((qwen3_quality or {}).get("baseline", {}).get("loss") if qwen3_quality else None),
            cap_delta=fmt(quality_cap.get("loss_delta")),
            q_delta=fmt(quality_qpruner.get("loss_delta")),
        ),
        "| 8 | Compression memory report | `reports/compression-memory-report.md` | {memory} |".format(
            memory=compression_memory_text(
                compression_memory,
                include_legacy_baseline_evidence=paper_baseline_coverage_audit is None,
            ),
        ),
        "| 9 | Paper baseline audit | `reports/paper-baseline-coverage-audit.md` | {audit} |".format(
            audit=paper_baseline_audit_text(paper_baseline_coverage_audit),
        ),
        "| 10 | Patterned Qwen3 compression sync | `reports/multicard-qwen-compression-quality-qwen3_06b_quality_pattern_2card_npu.md` | pattern {pattern}, quality {limit}/{total} on {world} cards, all-reduce {consistent}; WANDA delta {wanda_delta}; SparseGPT delta {sparsegpt_delta}; generate CAP {cap} tokens/s, QPruner {qpruner} tokens/s |".format(
            pattern=(qwen3_pattern_quality or {}).get("target_layer_pattern", "missing"),
            limit=(qwen3_pattern_quality or {}).get("target_layer_limit", "missing"),
            total=(qwen3_pattern_quality or {}).get("targeted_layers_total", "missing"),
            world=(qwen3_pattern_quality or {}).get("world_size", "missing"),
            consistent=pattern_quality_aggregate.get("distributed_reduce_consistent", "missing"),
            wanda_delta=fmt(pattern_quality_aggregate.get("wanda_loss_delta_avg")),
            sparsegpt_delta=fmt(pattern_quality_aggregate.get("sparsegpt_loss_delta_avg")),
            cap=fmt(pattern_generate_aggregate.get("cap_tokens_per_s_total")),
            qpruner=fmt(pattern_generate_aggregate.get("qpruner_tokens_per_s_total")),
        ),
        "| 10 | Quality-memory sweep | `reports/qwen-compression-quality-memory-sweep.md` | {sweep} |".format(
            sweep=quality_memory_sweep_text(demo_root),
        ),
        "| 11 | QPruner-only quality-memory sweep | `reports/qwen-qpruner-quality-memory-sweep.md` | {sweep} |".format(
            sweep=qpruner_quality_memory_sweep_text(demo_root),
        ),
        "| 12 | QPruner scale-quality summary | `reports/qwen-qpruner-scale-quality-summary.md` | {summary} |".format(
            summary=qpruner_scale_quality_text(demo_root),
        ),
        "| 13 | Choice accuracy | `reports/{report}` | {summary} |".format(
            report=CHOICE_ACCURACY_REPORT,
            summary=choice_accuracy_text(choice_accuracy),
        ),
        "| 14 | QPruner frontier chart | `reports/qwen-qpruner-quality-memory-frontier.svg` | Visual memory-quality frontier for the 9-point QPruner sweep |",
        "| 11 | Fine-tuning path | `reports/{report}` | RankAdaptor/LoRA world size {world}, adapter sync {sync}, validation delta {delta} |".format(
            report=qwen3_lora_report,
            world=(qwen3_lora or {}).get("world_size", "missing"),
            sync=lora_aggregate.get("adapter_sync_consistent", "missing"),
            delta=fmt(lora_aggregate.get("validation_loss_delta_avg")),
        ),
        "| 10 | Qwen3 BSLoRA shared adapters | `reports/tiny-qwen-bslora-finetune-qwen3_06b_bslora_npu.md` | Qwen3 BSLoRA target modules {modules}, shared adapter params {trainable} vs {unshared}, loss delta {delta} |".format(
            modules=(qwen3_bslora or {}).get("target_module_count", "missing"),
            trainable=(qwen3_bslora or {}).get("trainable_adapter_params", "missing"),
            unshared=(qwen3_bslora or {}).get("unshared_adapter_params", "missing"),
            delta=fmt((qwen3_bslora or {}).get("loss_delta")),
        ),
        "| 10 | Inference acceleration fallback | `reports/torch-serving-fallback-tiny_qwen3_serving_fallback_npu.md` | baseline {baseline_tps}, CAP {cap_tps}, QPruner {q_tps}; speedups CAP {cap_speedup}x / QPruner {q_speedup}x |".format(
            baseline_tps=fmt(baseline.get("tokens_per_s")),
            cap_tps=fmt(cap.get("tokens_per_s")),
            q_tps=fmt(qpruner.get("tokens_per_s")),
            cap_speedup=fmt(fallback_summary.get("cap_vs_baseline_speedup")),
            q_speedup=fmt(fallback_summary.get("qpruner_vs_baseline_speedup")),
        ),
        "| 11 | Compressed-native torch_npu reference | `reports/compressed-native-torch-serving-tiny_qwen3_native_serving_npu.md` | {native} |".format(
            native=compressed_native_text(compressed_native),
        ),
        "| 12 | Inference acceleration summary | `reports/inference-acceleration-summary.svg` | {summary} |".format(
            summary=inference_acceleration_summary_text(inference_summary),
        ),
        "| 12 | Compressed-native cache/long-decode sweep | `reports/compressed-native-cache-long-decode-sweep.svg` | {summary} |".format(
            summary=compressed_native_sweep_text(compressed_native_sweep),
        ),
        "| 12 | QPruner packed decode bottleneck | `reports/{report}` | {summary} |".format(
            report=QPRUNER_PACKED_DECODE_REPORT,
            summary=qpruner_packed_decode_text(qpruner_packed_decode),
        ),
        "| 12 | Multi-card compression sync | `reports/multicard-compression-sync-summary.svg` | {summary} |".format(
            summary=multicard_compression_sync_text(multicard_compression_sync),
        ),
        "| 13 | 8-card grouped replay | `reports/{report}` | {summary} |".format(
            report=multicard_qwen_qpruner_grouped_replay_report,
            summary=multicard_qwen_qpruner_grouped_replay_readout,
        ),
        "| 13 | 8-card grouped-MLP full-model sweep | `reports/{report}` | {summary} |".format(
            report=qwen_grouped_mlp_sweep_report,
            summary=qwen_grouped_mlp_sweep_readout_text,
        ),
        "| 13 | 8-card grouped-MLP memory-first tradeoff | `reports/{report}` | {summary} |".format(
            report=qwen_grouped_mlp_sweep_report_name(qwen_grouped_mlp_memory_tradeoff),
            summary=qwen_grouped_mlp_memory_tradeoff_readout_text,
        ),
        "| 13 | 8-card utilization proof | `artifacts/{artifact}` | {summary} |".format(
            artifact=npu_monitor_artifact or "npu_monitor_*.json",
            summary=npu_monitor_readout(npu_monitor),
        ),
        "| 12 | Objective coverage audit | `reports/objective-coverage-audit.md` | {summary} |".format(
            summary=objective_coverage_text(objective_coverage_audit),
        ),
        "| 10 | Synchronized multi-card parallel suite | `reports/multicard-parallel-suite-demo_parallel_suite_npu.md` | workers {workers}, cards {cards}, sync gate {gate}, start window {window}s, release lag window {release_window}s, {tasks} |".format(
            workers=(parallel or {}).get("world_size", "missing"),
            cards=",".join(str(card) for card in (parallel or {}).get("cards", [])) or "missing",
            gate=fmt((parallel or {}).get("sync_start_target_ts")),
            window=fmt((parallel or {}).get("start_window_s")),
            release_window=fmt((parallel or {}).get("release_lag_window_s")),
            tasks=task_counts_text((parallel or {}).get("task_counts")),
        ),
        "| 11 | Synchronized vLLM serving slice | `reports/{report}` | workers {workers}, cards {cards}, start window {window}s, best vLLM {best} tokens/s, {tasks} |".format(
            report=vllm_parallel_report,
            workers=(vllm_parallel or {}).get("world_size", "missing"),
            cards=",".join(str(card) for card in (vllm_parallel or {}).get("cards", [])) or "missing",
            window=fmt((vllm_parallel or {}).get("start_window_s")),
            best=fmt(vllm_parallel_aggregate.get("best_vllm_tokens_per_s")),
            tasks=task_counts_text((vllm_parallel or {}).get("task_counts")),
        ),
        "| 12 | vLLM boundary and shim | `reports/vllm-serving-probe-tiny_qwen3_serving_export_npu_selector_shim.md` | default {issues}; missing attention parameters {params}; selector shim {shim_status}; torch_npu path remains runnable |".format(
            issues=issue_classes(vllm_probe),
            params=missing_backend_parameters(vllm_probe),
            shim_status=(vllm_selector_shim_probe or {}).get("status", "missing"),
        ),
        "| 13 | vLLM metadata-shim benchmark | `reports/vllm-serving-benchmark-tiny_qwen3_serving_vllm_metadata_shim_npu.md` | baseline {baseline}, CAP {cap}, QPruner {qpruner}; qpruner_vs_baseline {speedup}x |".format(
            baseline=fmt(vllm_baseline.get("tokens_per_s")),
            cap=fmt(vllm_cap.get("tokens_per_s")),
            qpruner=fmt(vllm_qpruner.get("tokens_per_s")),
            speedup=fmt(vllm_benchmark_summary.get("qpruner_vs_baseline_speedup")),
        ),
        "| 14 | Inference bottleneck diagnosis | `reports/inference-bottleneck-report.md` | {diagnosis}; details {details}; next target {target}; {blocker} |".format(
            diagnosis=first_diagnosis_title(inference_bottleneck),
            details="{details}; {current}".format(
                details=diagnosis_details_text(inference_bottleneck),
                current=current_vllm_hbm_rerun_text(inference_bottleneck),
            ),
            target=next_optimization_title(inference_bottleneck),
            blocker=resource_blocker_text(inference_bottleneck),
        ),
        "| 15 | Close | `reports/compatibility-issues.md` | Separate ready evidence from backend blockers and point to preserved artifacts |",
        "",
        "## Key Metrics To Say Out Loud",
        "",
        "- Qwen3-0.6B runnable model: {single} tokens/s on torch_npu, {multi} tokens/s total across 8 cards.".format(
            single=fmt((qwen3_inference or {}).get("tokens_per_s")),
            multi=fmt(qwen3_multi_aggregate.get("tokens_per_s_total")),
        ),
        "- Serving fallback: baseline {baseline} tokens/s, CAP {cap_tps} tokens/s, QPruner {q_tps} tokens/s; best `{best}` at {best_tps} tokens/s.".format(
            baseline=fmt(baseline.get("tokens_per_s")),
            cap_tps=fmt(cap.get("tokens_per_s")),
            q_tps=fmt(qpruner.get("tokens_per_s")),
            best=fallback_summary.get("best_method", "missing"),
            best_tps=fmt(fallback_summary.get("best_tokens_per_s")),
        ),
        f"- {compressed_native_text(compressed_native)}.",
        f"- {inference_acceleration_summary_text(inference_summary)}.",
        f"- {compressed_native_sweep_text(compressed_native_sweep)}.",
        f"- {qpruner_packed_decode_text(qpruner_packed_decode)}.",
        f"- {multicard_compression_sync_text(multicard_compression_sync)}.",
        f"- {multicard_qwen_qpruner_grouped_replay_readout}.",
        f"- {qwen_grouped_mlp_sweep_readout_text}.",
        f"- {qwen_grouped_mlp_memory_tradeoff_readout_text}.",
        f"- {npu_monitor_readout(npu_monitor)}.",
        f"- {objective_coverage_text(objective_coverage_audit)}.",
        f"- {paper_baseline_audit_text(paper_baseline_coverage_audit)}.",
        "- Parallel suite: workers {workers}, sync gate {gate}, start window {window}s, release lag window {release_window}s, task mix {tasks}.".format(
            workers=(parallel or {}).get("world_size", "missing"),
            gate=fmt((parallel or {}).get("sync_start_target_ts")),
            window=fmt((parallel or {}).get("start_window_s")),
            release_window=fmt((parallel or {}).get("release_lag_window_s")),
            tasks=task_counts_text((parallel or {}).get("task_counts")),
        ),
        "- Synchronized vLLM serving slice: workers {workers}, cards {cards}, start window {window}s, best vLLM {best} tokens/s.".format(
            workers=(vllm_parallel or {}).get("world_size", "missing"),
            cards=",".join(str(card) for card in (vllm_parallel or {}).get("cards", [])) or "missing",
            window=fmt((vllm_parallel or {}).get("start_window_s")),
            best=fmt(vllm_parallel_aggregate.get("best_vllm_tokens_per_s")),
        ),
        "- vLLM-Ascend blocker: {issues}; missing `{params}` in attention backend API; selector shim {shim_status}.".format(
            issues=issue_classes(vllm_probe),
            params=missing_backend_parameters(vllm_probe),
            shim_status=(vllm_selector_shim_probe or {}).get("status", "missing"),
        ),
        "- vLLM metadata-shim benchmark: baseline {baseline} tokens/s, CAP {cap_tps} tokens/s, QPruner {q_tps} tokens/s; qpruner_vs_baseline {speedup}x.".format(
            baseline=fmt(vllm_baseline.get("tokens_per_s")),
            cap_tps=fmt(vllm_cap.get("tokens_per_s")),
            q_tps=fmt(vllm_qpruner.get("tokens_per_s")),
            speedup=fmt(vllm_benchmark_summary.get("qpruner_vs_baseline_speedup")),
        ),
        "- Patterned Qwen3 compression sync: pattern {pattern}, quality {limit}/{total} on {world} cards, WANDA delta {wanda_delta}, SparseGPT delta {sparsegpt_delta}, generate CAP {cap} tokens/s, QPruner {qpruner} tokens/s.".format(
            pattern=(qwen3_pattern_quality or {}).get("target_layer_pattern", "missing"),
            limit=(qwen3_pattern_quality or {}).get("target_layer_limit", "missing"),
            total=(qwen3_pattern_quality or {}).get("targeted_layers_total", "missing"),
            world=(qwen3_pattern_quality or {}).get("world_size", "missing"),
            wanda_delta=fmt(pattern_quality_aggregate.get("wanda_loss_delta_avg")),
            sparsegpt_delta=fmt(pattern_quality_aggregate.get("sparsegpt_loss_delta_avg")),
            cap=fmt(pattern_generate_aggregate.get("cap_tokens_per_s_total")),
            qpruner=fmt(pattern_generate_aggregate.get("qpruner_tokens_per_s_total")),
        ),
        f"- {quality_memory_sweep_text(demo_root)}.",
        f"- {qpruner_quality_memory_sweep_text(demo_root)}.",
        f"- QPruner scale-quality summary: {qpruner_scale_quality_text(demo_root)}.",
        *([] if not qwen_native_profile_readout_text else [f"- {qwen_native_profile_readout_text}."]),
        *([] if not qwen_speed_first_readout_text else [f"- {qwen_speed_first_readout_text}."]),
        *([] if not qwen_native_bits_readout else [f"- {qwen_native_bits_readout}."]),
        *([] if not qwen_native_8card_profile_readout_text else [f"- {qwen_native_8card_profile_readout_text}."]),
        "- Qwen3 BSLoRA: target modules {modules}, shared adapter params {trainable} vs {unshared}, loss delta {delta}.".format(
            modules=(qwen3_bslora or {}).get("target_module_count", "missing"),
            trainable=(qwen3_bslora or {}).get("trainable_adapter_params", "missing"),
            unshared=(qwen3_bslora or {}).get("unshared_adapter_params", "missing"),
            delta=fmt((qwen3_bslora or {}).get("loss_delta")),
        ),
        f"- {compression_memory_text(compression_memory, include_legacy_baseline_evidence=paper_baseline_coverage_audit is None)}.",
        "- Inference bottleneck diagnosis: {status}, {diagnosis}; details {details}; next target {target}; {blocker}.".format(
            status=(inference_bottleneck or {}).get("status", "missing"),
            diagnosis=first_diagnosis_title(inference_bottleneck),
            details="{details}; {current}".format(
                details=diagnosis_details_text(inference_bottleneck),
                current=current_vllm_hbm_rerun_text(inference_bottleneck),
            ),
            target=next_optimization_title(inference_bottleneck),
            blocker=resource_blocker_text(inference_bottleneck),
        ),
        "",
        "## Recording Checklist",
        "",
        "- Start from `/mnt/nvme/622/tidal-ai` inside container `3ee`.",
        "- Keep one terminal pane on `/mnt/nvme/622/tidal-demo` paths so evidence provenance stays visible.",
        "- Show tables, not full logs; open JSON only when proving exact raw numbers.",
        "- Preserve the latest terminal recording at `recordings/ascend-910b-demo-playback-latest.typescript`.",
        "- Primary progress report: `reports/ascend-910b-demo-progress.md`.",
        "- Objective readiness report: `reports/demo-readiness-report.md`.",
        "",
        "## Evidence Paths",
        "",
        "- `reports/ascend-910b-demo-progress.md`",
        "- `reports/demo-readiness-report.md`",
        "- `reports/torch-serving-fallback-tiny_qwen3_serving_fallback_npu.md`",
        "- `reports/compressed-native-torch-serving-tiny_qwen3_native_serving_npu.md`",
        "- `reports/inference-acceleration-summary.md`",
        "- `reports/inference-acceleration-summary.csv`",
        "- `reports/inference-acceleration-summary.svg`",
        "- `reports/compressed-native-cache-long-decode-sweep.md`",
        "- `reports/compressed-native-cache-long-decode-sweep.csv`",
        "- `reports/compressed-native-cache-long-decode-sweep.svg`",
        f"- `reports/{QPRUNER_PACKED_DECODE_REPORT}`",
        f"- `reports/{QPRUNER_SHAPE_SWEEP_REPORT}`",
        *([] if not qpruner_shape_sweep_report else [f"- `reports/{qpruner_shape_sweep_report}`"]),
        *([] if not qwen_native_profile_report else [f"- `reports/{qwen_native_profile_report}`"]),
        *([] if not qwen_speed_first_report else [f"- `reports/{qwen_speed_first_report}`"]),
        *(f"- `reports/{report}`" for report in qwen_native_bits_reports),
        *(f"- `{path}`" for path in qwen_native_8card_profile_evidence_paths if path.startswith("reports/")),
        f"- `reports/{QPRUNER_DTYPE_SHAPE_SWEEP_REPORT}`",
        "- `reports/multicard-compression-sync-summary.md`",
        "- `reports/multicard-compression-sync-summary.csv`",
        "- `reports/multicard-compression-sync-summary.svg`",
        f"- `reports/{multicard_qwen_qpruner_grouped_replay_report}`",
        f"- `reports/{qwen_grouped_mlp_sweep_report}`",
        "- `reports/objective-coverage-audit.md`",
        "- `reports/multicard-parallel-suite-demo_parallel_suite_npu.md`",
        f"- `reports/{vllm_parallel_report}`",
        "- `reports/compression-memory-report.md`",
        "- `reports/qwen-llm-pruner-baseline-qwen3_06b_llm_pruner_npu.md`",
        "- `reports/qwen-compression-quality-memory-sweep.md`",
        "- `reports/qwen-qpruner-quality-memory-sweep.md`",
        "- `reports/qwen-qpruner-scale-quality-summary.md`",
        "- `reports/qwen-qpruner-scale-quality-summary.csv`",
        f"- `reports/{CHOICE_ACCURACY_REPORT}`",
        f"- `reports/{MULTITASK_CHOICE_ACCURACY_REPORT}`",
        f"- `reports/{FULLTARGET_QPRUNER_CHOICE_REPORT}`",
        "- `reports/qwen-qpruner-quality-memory-frontier.svg`",
        "- `reports/qwen-qpruner-quality-memory-frontier.md`",
        "- `reports/multicard-qwen-compression-quality-qwen3_06b_quality_pattern_2card_npu.md`",
        "- `reports/multicard-qwen-compression-generate-qwen3_06b_generate_pattern_2card_npu.md`",
        "- `reports/tiny-qwen-bslora-finetune-qwen3_06b_bslora_npu.md`",
        "- `reports/vllm-serving-probe-tiny_qwen3_serving_export_npu.md`",
        "- `reports/vllm-serving-probe-tiny_qwen3_serving_export_npu_selector_shim.md`",
        "- `reports/vllm-serving-benchmark-tiny_qwen3_serving_vllm_metadata_shim_npu.md`",
        "- `reports/inference-bottleneck-report.md`",
        "- `reports/compatibility-issues.md`",
        "- `recordings/ascend-910b-demo-playback-latest.typescript`",
        "- `artifacts/torch_serving_fallback_tiny_qwen3_serving_fallback_npu.json`",
        "- `artifacts/compressed_native_torch_serving_tiny_qwen3_native_serving_npu.json`",
        "- `artifacts/inference_acceleration_summary.json`",
        "- `artifacts/compressed_native_sweep_summary.json`",
        f"- `artifacts/{QPRUNER_PACKED_DECODE_ARTIFACT}`",
        f"- `artifacts/{QPRUNER_SHAPE_SWEEP_ARTIFACT}`",
        *([] if not qpruner_shape_sweep_artifact else [f"- `artifacts/{qpruner_shape_sweep_artifact}`"]),
        *([] if not qwen_native_profile_artifact else [f"- `artifacts/{qwen_native_profile_artifact}`"]),
        *([] if not qwen_speed_first_artifact else [f"- `artifacts/{qwen_speed_first_artifact}`"]),
        *(f"- `artifacts/{artifact}`" for artifact in qwen_native_bits_artifacts),
        *(f"- `{path}`" for path in qwen_native_8card_profile_evidence_paths if path.startswith("artifacts/")),
        f"- `artifacts/{QPRUNER_DTYPE_SHAPE_SWEEP_ARTIFACT}`",
        "- `artifacts/multicard_compression_sync_summary.json`",
        f"- `artifacts/{multicard_qwen_qpruner_grouped_replay_artifact}`",
        f"- `artifacts/{qwen_grouped_mlp_sweep_artifact}`",
        *(f"- `{item}`" for item in qwen_grouped_mlp_sweep_evidence(qwen_grouped_mlp_sweep)),
        *(f"- `{item}`" for item in qwen_grouped_mlp_sweep_evidence(qwen_grouped_mlp_memory_tradeoff)),
        *([] if not npu_monitor_artifact else [f"- `artifacts/{npu_monitor_artifact}`"]),
        *([] if not npu_monitor_log else [f"- `{npu_monitor_log}`"]),
        "- `artifacts/objective_coverage_audit.json`",
        "- `artifacts/vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu.json`",
        "- `artifacts/compression_memory_report.json`",
        "- `artifacts/paper_baseline_coverage_audit.json`",
        "- `reports/paper-baseline-coverage-audit.md`",
        f"- `artifacts/{CHOICE_ACCURACY_ARTIFACT}`",
        f"- `artifacts/{MULTITASK_CHOICE_ACCURACY_ARTIFACT}`",
        f"- `artifacts/{FULLTARGET_QPRUNER_CHOICE_ARTIFACT}`",
        "- `artifacts/multicard_qwen_compression_quality_qwen3_06b_quality_pattern_2card_npu.json`",
        "- `artifacts/multicard_qwen_compression_generate_qwen3_06b_generate_pattern_2card_npu.json`",
        "- `artifacts/tiny_qwen_bslora_finetune_qwen3_06b_bslora_npu.json`",
        "- `artifacts/multicard_parallel_suite_demo_parallel_suite_npu.json`",
        f"- `artifacts/{vllm_parallel_artifact}`",
    ]
    return "\n".join(lines) + "\n"


def write_demo_storyboard(demo_root: Path) -> Path:
    reports = demo_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = reports / "demo-storyboard.md"
    output.write_text(markdown(demo_root))
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write an Ascend 910B demo recording storyboard")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = write_demo_storyboard(Path(args.demo_root))
    print(f"DEMO_STORYBOARD {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
