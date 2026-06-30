#!/usr/bin/env python
"""Write a concise demo progress report from preserved Ascend artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from tidal.reports.baselines import engineering_baseline_note, paper_baseline_note
from tidal.reports.multicard_selection import best_multicard_qwen_compression_generate
from tidal.reports.npu_monitor_selection import best_npu_utilization_monitor
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
    qwen_native_bits_tradeoff_artifacts,
    qwen_native_bits_tradeoff_readout,
    qwen_native_bits_tradeoff_reports,
    qwen_native_profile_readout,
)


QPRUNER_PACKED_DECODE_ARTIFACT = "qpruner_packed_decode_benchmark_tiny_qwen3_qpruner_decode_npu.json"
QPRUNER_PACKED_DECODE_REPORT = "qpruner-packed-decode-benchmark-tiny_qwen3_qpruner_decode_npu.md"
QPRUNER_DTYPE_SHAPE_SWEEP_ARTIFACT = "qpruner_packed_decode_benchmark_qwen3_06b_shape_sweep_dtype_preserving_npu.json"
QPRUNER_DTYPE_SHAPE_SWEEP_REPORT = "qpruner-packed-decode-benchmark-qwen3_06b_shape_sweep_dtype_preserving_npu.md"
CHOICE_ACCURACY_ARTIFACT = "qwen_compression_choice_accuracy_qwen3_06b_choice_accuracy_npu.json"
CHOICE_ACCURACY_REPORT = "qwen-compression-choice-accuracy-qwen3_06b_choice_accuracy_npu.md"
CHOICE_ACCURACY_CSV = "qwen-compression-choice-accuracy-qwen3_06b_choice_accuracy_npu.csv"
MULTITASK_CHOICE_ACCURACY_ARTIFACT = "qwen_compression_choice_accuracy_qwen3_06b_multitask_choice_npu.json"
MULTITASK_CHOICE_ACCURACY_REPORT = "qwen-compression-choice-accuracy-qwen3_06b_multitask_choice_npu.md"
MULTITASK_CHOICE_ACCURACY_CSV = "qwen-compression-choice-accuracy-qwen3_06b_multitask_choice_npu.csv"
FULLTARGET_QPRUNER_CHOICE_ARTIFACT = (
    "qwen_compression_choice_accuracy_qwen3_06b_multitask_choice_fulltarget_qpruner_npu.json"
)
FULLTARGET_QPRUNER_CHOICE_REPORT = (
    "qwen-compression-choice-accuracy-qwen3_06b_multitask_choice_fulltarget_qpruner_npu.md"
)
FULLTARGET_QPRUNER_CHOICE_CSV = (
    "qwen-compression-choice-accuracy-qwen3_06b_multitask_choice_fulltarget_qpruner_npu.csv"
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


def fmt_accuracy(value: Any) -> str:
    if value is None:
        return "missing"
    try:
        return f"{float(value) * 100.0:.3f}%"
    except (TypeError, ValueError):
        return str(value)


def join_values(values: Any) -> str:
    if not isinstance(values, list):
        return "missing"
    return ", ".join(str(value) for value in values) or "missing"


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


def qpruner_grouped_projection_plan_readout(inference_bottleneck: dict[str, Any] | None) -> str | None:
    if not inference_bottleneck:
        return None
    plan = inference_bottleneck.get("qpruner_grouped_projection_plan")
    if not isinstance(plan, dict) or not plan:
        return None
    best = plan.get("best_candidate", {}) if isinstance(plan.get("best_candidate"), dict) else {}
    full_model = plan.get("full_model", {}) if isinstance(plan.get("full_model"), dict) else {}
    return (
        "QPruner grouped projection plan: {label} {speedup}x grouped speedup, preset {preset}, "
        "candidates {count}, top-module hits {hits}, full-model forwards {calls}, "
        "forward share {share}%, code-cache storage {code_storage}%, cached scaled-code bytes {scaled_bytes}"
    ).format(
        label=best.get("label", "missing"),
        speedup=fmt(best.get("speedup_vs_sequential_scaled_code")),
        preset=fmt(plan.get("shape_preset")),
        count=fmt(plan.get("candidate_count")),
        hits=fmt(best.get("profile_top_module_hits")),
        calls=fmt(full_model.get("total_forward_calls")),
        share=fmt(full_model.get("forward_share_pct")),
        code_storage=fmt(full_model.get("code_cache_storage_reduction_pct")),
        scaled_bytes=fmt(full_model.get("cached_scaled_code_bytes")),
    )


def qwen_qpruner_grouped_replay_readout(inference_bottleneck: dict[str, Any] | None) -> str | None:
    if not inference_bottleneck:
        return None
    replay = inference_bottleneck.get("qwen_qpruner_grouped_replay")
    if not isinstance(replay, dict) or not replay:
        return None
    best = replay.get("best_group", {}) if isinstance(replay.get("best_group"), dict) else {}
    grouped_code_bytes = best.get("grouped_code_cache_bytes")
    grouped_scaled_code_bytes = best.get("grouped_scaled_code_cache_bytes")
    return (
        "Qwen3 real grouped replay: role {role}, {speedup}x grouped speedup, modules {modules}/{available}, "
        "groups {groups}, target layers {layers}/{total}, quantized layers {quantized}, sequential {seq} ms, "
        "grouped {grouped} ms, code-cache storage {code_storage}%, grouped code bytes {codes}, "
        "grouped scaled-code bytes {scaled}, max_abs_diff {diff}"
    ).format(
        role=best.get("role", "missing"),
        speedup=fmt(best.get("speedup_vs_sequential_scaled_code")),
        modules=fmt(best.get("module_count")),
        available=fmt(best.get("available_module_count")),
        groups=fmt(replay.get("group_count")),
        layers=fmt(replay.get("target_layer_limit")),
        total=fmt(replay.get("targeted_layers_total")),
        quantized=fmt(replay.get("quantized_layers")),
        seq=fmt(best.get("sequential_latency_ms")),
        grouped=fmt(best.get("grouped_latency_ms")),
        code_storage=fmt(replay.get("code_cache_storage_reduction_pct")),
        codes=fmt(float(grouped_code_bytes)) if grouped_code_bytes is not None else fmt(None),
        scaled=fmt(float(grouped_scaled_code_bytes)) if grouped_scaled_code_bytes is not None else fmt(None),
        diff=fmt(best.get("max_abs_diff_vs_sequential_scaled_code")),
    )


def multicard_qwen_qpruner_grouped_replay_readout(replay: dict[str, Any] | None) -> str | None:
    if not isinstance(replay, dict) or not replay:
        return None
    aggregate = replay.get("aggregate", {}) if isinstance(replay.get("aggregate"), dict) else {}
    released_bytes = aggregate.get("released_packed_code_bytes_total")
    live_payload_bytes = aggregate.get(
        "live_compressed_payload_storage_bytes_total",
        aggregate.get("live_compressed_payload_bytes_total"),
    )
    return (
        "Multi-card Qwen3 real grouped replay: best role {role}, {best}x best grouped speedup, "
        "mean {mean}x, min {min_speedup}x, groups {groups}, target layers {layers}/{total}, "
        "quantized layers {quantized}, memory-native workers {memory}/{passes}, "
        "code-cache storage {code_storage}%, grouped code bytes {codes}, grouped scaled-code bytes {scaled}, "
        "released packed-code bytes {released}, live compressed payload bytes {payload}, max_abs_diff {diff}"
    ).format(
        role=aggregate.get("best_role", "missing"),
        best=fmt(aggregate.get("best_grouped_speedup")),
        mean=fmt(aggregate.get("mean_grouped_speedup")),
        min_speedup=fmt(aggregate.get("min_grouped_speedup")),
        groups=fmt(aggregate.get("max_group_count")),
        layers=fmt(aggregate.get("target_layer_limit")),
        total=fmt(aggregate.get("targeted_layers_total")),
        quantized=fmt(aggregate.get("quantized_layers")),
        memory=fmt(aggregate.get("memory_native_worker_count")),
        passes=fmt(aggregate.get("pass_count")),
        code_storage=fmt(aggregate.get("min_code_cache_storage_reduction_pct")),
        codes=fmt(float(aggregate.get("max_grouped_code_cache_bytes")))
        if aggregate.get("max_grouped_code_cache_bytes") is not None
        else fmt(None),
        scaled=fmt(float(aggregate.get("max_grouped_scaled_code_cache_bytes")))
        if aggregate.get("max_grouped_scaled_code_cache_bytes") is not None
        else fmt(None),
        released=fmt(float(released_bytes)) if released_bytes is not None else fmt(None),
        payload=fmt(float(live_payload_bytes)) if live_payload_bytes is not None else fmt(None),
        diff=fmt(aggregate.get("max_abs_diff")),
    )


def paper_baseline_evidence_metrics(metrics: Any) -> str:
    if not isinstance(metrics, dict) or not metrics:
        return "missing"
    return ", ".join(f"{key} {fmt(value)}" for key, value in metrics.items())


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


def first_existing_json(artifacts: Path, names: list[str]) -> dict[str, Any] | None:
    for name in names:
        payload = read_json(artifacts / name)
        if payload is not None:
            return payload
    return None


def artifact_with_default(payload: dict[str, Any] | None, default: str) -> str:
    return str((payload or {}).get("_artifact_name") or default)


def report_with_default(payload: dict[str, Any] | None, default: str) -> str:
    return str((payload or {}).get("_report_name") or default)


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


def paper_baseline_audit_rows(report: dict[str, Any] | None) -> list[str]:
    if not report:
        return ["| Paper baseline audit | missing |"]

    rows = [f"| Paper baseline audit {report.get('status', 'missing')} | full196 paper-baseline compatibility evidence |"]
    evidence = report.get("evidence", {}) if isinstance(report.get("evidence"), dict) else {}

    qpruner = evidence.get("qpruner_llm_pruner_full196", {})
    if isinstance(qpruner, dict) and qpruner:
        rows.append(
            "| Paper baseline audit QPruner | QPruner LLM-Pruner-style {status}, target layers {layers}/{total}; "
            "prune {prune}%, loss delta {loss}, speedup {speedup}x |".format(
                status=qpruner.get("status", "missing"),
                layers=fmt(qpruner.get("targeted_layers")),
                total=fmt(qpruner.get("targeted_layers_total")),
                prune=fmt(qpruner.get("targeted_param_reduction_pct")),
                loss=fmt(qpruner.get("loss_delta")),
                speedup=fmt(qpruner.get("latency_speedup")),
            )
        )

    cap = evidence.get("cap_wanda_sparsegpt_full196", {})
    if isinstance(cap, dict) and cap:
        target_layers = "{layers}/{total}".format(
            layers=fmt(cap.get("targeted_layers")),
            total=fmt(cap.get("targeted_layers_total")),
        )
        rows.append(
            "| Paper baseline audit CAP Wanda | CAP Wanda {status}, target layers {target_layers}; "
            "sparsity {sparsity}%, loss delta {loss}, speedup {speedup}x |".format(
                status=cap.get("status", "missing"),
                target_layers=target_layers,
                sparsity=fmt(cap.get("wanda_targeted_param_reduction_pct")),
                loss=fmt(cap.get("wanda_loss_delta")),
                speedup=fmt(cap.get("wanda_latency_speedup")),
            )
        )
        rows.append(
            "| Paper baseline audit CAP SparseGPT | CAP SparseGPT {status}, target layers {target_layers}; "
            "sparsity {sparsity}%, loss delta {loss}, speedup {speedup}x |".format(
                status=cap.get("status", "missing"),
                target_layers=target_layers,
                sparsity=fmt(cap.get("sparsegpt_targeted_param_reduction_pct")),
                loss=fmt(cap.get("sparsegpt_loss_delta")),
                speedup=fmt(cap.get("sparsegpt_latency_speedup")),
            )
        )

    rankadaptor = evidence.get("rankadaptor_recovery_controlled", {})
    if isinstance(rankadaptor, dict) and rankadaptor:
        rows.append(
            "| Paper baseline audit RankAdaptor recovery | RankAdaptor recovery baselines {status}, best {best}; "
            "no-recovery delta {none}, LoRA delta {lora}, AdaLoRA-style delta {adalora} |".format(
                status=rankadaptor.get("status", "missing"),
                best=rankadaptor.get("best_recovery_method", "missing"),
                none=fmt(rankadaptor.get("no_recovery_loss_delta")),
                lora=fmt(rankadaptor.get("lora_loss_delta")),
                adalora=fmt(rankadaptor.get("adalora_style_loss_delta")),
            )
        )

    return rows


def compression_memory_rows(
    report: dict[str, Any] | None, *, include_legacy_baseline_evidence: bool = True
) -> list[str]:
    if not report:
        return [
            "| Status | missing |",
            "| Engineering reference baseline | missing |",
            "| QPruner paper baseline | missing |",
            "| CAP paper baselines | missing |",
            "| RankAdaptor paper recovery baselines | missing |",
            "| Memory reductions | missing |",
            "| Serving memory gap | missing |",
        ]
    paper = report.get("paper_baselines", {}) if isinstance(report.get("paper_baselines"), dict) else {}
    memory = report.get("compression_memory", {}) if isinstance(report.get("compression_memory"), dict) else {}
    gap = report.get("memory_gap", {}) if isinstance(report.get("memory_gap"), dict) else {}
    reference = report.get("engineering_reference", {}) if isinstance(report.get("engineering_reference"), dict) else {}
    qpruner = paper.get("qpruner", {}) if isinstance(paper.get("qpruner"), dict) else {}
    cap = paper.get("cap", {}) if isinstance(paper.get("cap"), dict) else {}
    rankadaptor = paper.get("rankadaptor", {}) if isinstance(paper.get("rankadaptor"), dict) else {}
    cap_memory = memory.get("cap", {}) if isinstance(memory.get("cap"), dict) else {}
    q_memory = memory.get("qpruner", {}) if isinstance(memory.get("qpruner"), dict) else {}
    rows = [
        f"| Status | {report.get('status', 'missing')} |",
        f"| Memory-first video readout | {compression_memory_video_readout(report)} |",
        "| Engineering reference baseline | {role}, not paper baseline {not_paper} |".format(
            role=reference.get("serving_baseline_role", "missing"),
            not_paper=reference.get("not_paper_baseline", "missing"),
        ),
        f"| QPruner paper baseline | {join_values(qpruner.get('primary'))} |",
        f"| QPruner recovery baselines | {join_values(qpruner.get('recovery_baselines'))} |",
        f"| CAP paper baselines | {join_values(cap.get('pruning_baselines'))} |",
        f"| RankAdaptor paper recovery baselines | {join_values(rankadaptor.get('recovery_baselines'))} |",
        " | Memory reductions | CAP {cap}%, QPruner {qpruner}%, target coverage {coverage}% |".format(
            cap=fmt(cap_memory.get("targeted_param_reduction_pct")),
            qpruner=fmt(q_memory.get("targeted_param_reduction_pct")),
            coverage=fmt(memory.get("target_layer_coverage_pct")),
        ).strip(),
        "| Serving memory gap | dense export {dense}, erases storage savings {erases} |".format(
            dense=gap.get("serving_dense_export", "missing"),
            erases=gap.get("dense_export_erases_storage_savings", "missing"),
        ),
        f"| Next action | {report.get('next_action', 'missing')} |",
    ]
    for item in report.get("baseline_alignment_matrix", []):
        if not isinstance(item, dict):
            continue
        rows.append(
            "| Baseline alignment {method} | {role} {baselines} -> {demo_role} |".format(
                method=item.get("method", "missing"),
                role=item.get("paper_baseline_role", "missing"),
                baselines=join_values(item.get("paper_baselines")),
                demo_role=item.get("demo_reference_role", "missing"),
            )
        )
    if include_legacy_baseline_evidence:
        for item in report.get("paper_baseline_evidence", []):
            if not isinstance(item, dict):
                continue
            rows.append(
                "| Paper baseline evidence {method} | {baseline} {status}; metrics {metrics}; artifacts {artifacts} |".format(
                    method=item.get("method", "missing"),
                    baseline=item.get("paper_baseline", "missing"),
                    status=item.get("status", "missing"),
                    metrics=paper_baseline_evidence_metrics(item.get("metrics")),
                    artifacts=join_values(item.get("artifacts")),
                )
            )
    return rows


def first_diagnosis_title(report: dict[str, Any] | None) -> str:
    diagnoses = report.get("diagnoses", []) if report else []
    for item in diagnoses:
        if isinstance(item, dict) and item.get("title"):
            return str(item["title"])
    return "none" if report else "missing"


def inference_bottleneck_diagnosis_rows(report: dict[str, Any] | None) -> list[str]:
    diagnoses = report.get("diagnoses", []) if report else []
    rows: list[str] = []
    for index, item in enumerate(diagnoses, start=1):
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        evidence = item.get("evidence")
        if not title and not evidence:
            continue
        rows.append(f"| Diagnosis {index} | {title or 'missing'} |")
        if evidence:
            rows.append(f"| Diagnosis {index} evidence | {evidence} |")
    return rows or ["| Diagnosis details | none |"]


def current_vllm_hbm_rerun_text(report: dict[str, Any] | None) -> str | None:
    current = report.get("current_vllm_rerun", {}) if report else {}
    if not isinstance(current, dict) or not current:
        return None
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


def next_optimization_title(report: dict[str, Any] | None) -> str:
    target = report.get("next_optimization_target", {}) if report else {}
    if isinstance(target, dict) and target.get("title"):
        return str(target["title"])
    return "missing"


def inference_acceleration_summary_rows(summary: dict[str, Any] | None) -> list[str]:
    if not summary:
        return [
            "| Status | missing |",
            "| Best throughput point | missing |",
            "| Native memory point | missing |",
            "| 8-card grouped replay | missing |",
            "| vLLM bottleneck | missing |",
        ]
    payload = summary.get("summary", {}) if isinstance(summary.get("summary"), dict) else {}
    best = payload.get("best_throughput", {}) if isinstance(payload.get("best_throughput"), dict) else {}
    native = payload.get("best_native_memory", {}) if isinstance(payload.get("best_native_memory"), dict) else {}
    grouped = payload.get("grouped_replay", {}) if isinstance(payload.get("grouped_replay"), dict) else {}
    monitor = grouped.get("monitor", {}) if isinstance(grouped.get("monitor"), dict) else {}
    bottleneck = summary.get("vllm_bottleneck", {}) if isinstance(summary.get("vllm_bottleneck"), dict) else {}
    return [
        f"| Status | {summary.get('status', 'missing')} |",
        "| Best throughput point | {track} / {method}, {tokens} tokens/s |".format(
            track=best.get("track", "missing"),
            method=best.get("method_label") or best.get("method", "missing"),
            tokens=fmt(best.get("tokens_per_s")),
        ),
        "| Native memory point | {method}, memory {memory}%, storage {storage}% |".format(
            method=native.get("method_label") or native.get("method", "missing"),
            memory=fmt(native.get("memory_reduction_pct")),
            storage=fmt(native.get("storage_reduction_pct")),
        ),
        (
            "| 8-card grouped replay | {status}, memory-native workers {memory}/{passes}, "
            "code-cache storage {storage}%, best grouped speedup {speedup}x, "
            "{aicore_samples} samples with all 8 AICore nonzero, released packed-code bytes {released} |"
        ).format(
            status=grouped.get("status", "missing"),
            memory=fmt(grouped.get("memory_native_worker_count")),
            passes=fmt(grouped.get("pass_count")),
            storage=fmt(grouped.get("code_cache_storage_reduction_pct")),
            speedup=fmt(grouped.get("best_grouped_speedup")),
            aicore_samples=fmt(monitor.get("samples_with_8_nonzero_aicore")),
            released=fmt(grouped.get("released_packed_code_bytes_total")),
        ),
        f"| vLLM bottleneck | {bottleneck.get('title', 'missing')} |",
    ]


def compressed_native_sweep_rows(summary: dict[str, Any] | None) -> list[str]:
    if not summary:
        return [
            "| Status | missing |",
            "| Readout | missing |",
            "| Best compressed-native point | missing |",
            "| Cache effect | missing |",
            "| Long decode effect | missing |",
            "| Runtime diagnosis | missing |",
            "| Memory point | missing |",
        ]
    best = summary.get("best", {}) if isinstance(summary.get("best"), dict) else {}
    cache = summary.get("cache_effect", {}) if isinstance(summary.get("cache_effect"), dict) else {}
    long_decode = (
        summary.get("long_decode_effect", {}) if isinstance(summary.get("long_decode_effect"), dict) else {}
    )
    diagnosis = summary.get("runtime_diagnosis", {}) if isinstance(summary.get("runtime_diagnosis"), dict) else {}
    memory = summary.get("memory", {}) if isinstance(summary.get("memory"), dict) else {}
    return [
        f"| Status | {summary.get('status', 'missing')} |",
        f"| Readout | {summary.get('readout', 'missing')} |",
        "| Best compressed-native point | {method} {speedup}x baseline, max_new_tokens={tokens}, cache={cache} |".format(
            method=best.get("method_label") or best.get("method", "missing"),
            speedup=fmt(best.get("speedup")),
            tokens=best.get("max_new_tokens", "missing"),
            cache=best.get("inference_cache_enabled", "missing"),
        ),
        "| Cache effect | max_new_tokens {tokens}, cache qpruner delta {delta}x |".format(
            tokens=cache.get("max_new_tokens", "missing"),
            delta=fmt(cache.get("qpruner_speedup_delta")),
        ),
        "| Long decode effect | {short} -> {long} tokens, long decode qpruner delta {delta}x |".format(
            short=long_decode.get("from_max_new_tokens", "missing"),
            long=long_decode.get("to_max_new_tokens", "missing"),
            delta=fmt(long_decode.get("qpruner_speedup_delta")),
        ),
        "| Runtime diagnosis | QPruner scaling gap {gap}x, QPruner cache peak +{mem} MB, next {action} |".format(
            gap=fmt(diagnosis.get("qpruner_scaling_gap_vs_baseline")),
            mem=fmt(diagnosis.get("qpruner_cache_peak_mem_delta_mb")),
            action=diagnosis.get("next_action", "missing"),
        ),
        "| Memory point | QPruner storage reduction {storage}% |".format(
            storage=fmt(memory.get("best_qpruner_storage_reduction_pct")),
        ),
    ]


def qpruner_packed_decode_rows(report: dict[str, Any] | None) -> list[str]:
    if not report:
        return [
            "| Status | missing |",
            "| Runtime storage | missing |",
            "| Storage reduction | missing |",
            "| Decode latency | missing |",
            "| Next action | missing |",
        ]
    uncached = report.get("uncached", {}) if isinstance(report.get("uncached"), dict) else {}
    code_cached = report.get("code_cached", {}) if isinstance(report.get("code_cached"), dict) else {}
    scaled_code = report.get("scaled_code_matmul", {}) if isinstance(report.get("scaled_code_matmul"), dict) else {}
    cached = report.get("cached", {}) if isinstance(report.get("cached"), dict) else {}
    dense = report.get("dense", {}) if isinstance(report.get("dense"), dict) else {}
    sequential_group = (
        report.get("sequential_scaled_code_matmul_group", {})
        if isinstance(report.get("sequential_scaled_code_matmul_group"), dict)
        else {}
    )
    grouped = (
        report.get("grouped_scaled_code_matmul", {})
        if isinstance(report.get("grouped_scaled_code_matmul"), dict)
        else {}
    )
    grouped_rows = []
    if grouped:
        grouped_rows.append(
            "| Grouped scaled code | grouped scaled-code {grouped_ms} ms vs sequential {seq_ms} ms, "
            "grouped speedup {speedup}x, grouped code-cache bytes {codes}, aux-cache bytes {aux} |".format(
                grouped_ms=fmt(grouped.get("latency_ms")),
                seq_ms=fmt(sequential_group.get("latency_ms")),
                speedup=fmt(grouped.get("speedup_vs_sequential_scaled_code")),
                codes=fmt(grouped.get("grouped_code_cache_bytes")),
                aux=fmt(grouped.get("grouped_aux_cache_bytes")),
            )
        )
    return [
        "| Summary | {status} on {device}, runtime {runtime}, storage {storage}% |".format(
            status=report.get("status", "missing"),
            device=report.get("device", "missing"),
            runtime=report.get("runtime_storage_format", "missing"),
            storage=fmt(report.get("storage_reduction_pct")),
        ),
        "| Payload | {payload} bytes packed payload vs {dense} dense bytes |".format(
            payload=fmt(report.get("quantized_payload_bytes")),
            dense=fmt(report.get("dense_payload_bytes")),
        ),
        "| Code-cache payload | code-cache storage {storage}%, {payload} bytes vs dense {dense} bytes |".format(
            storage=fmt(report.get("code_cache_storage_reduction_pct")),
            payload=fmt(report.get("code_cache_payload_bytes")),
            dense=fmt(report.get("dense_payload_bytes")),
        ),
        "| Decode latency | uncached {uncached} ms, cache {cache} ms, dense {dense} ms |".format(
            uncached=fmt(uncached.get("latency_ms")),
            cache=fmt(cached.get("latency_ms")),
            dense=fmt(dense.get("latency_ms")),
        ),
        "| Code-cache latency | code-cache {latency} ms, code-cache speedup {speedup}x |".format(
            latency=fmt(code_cached.get("latency_ms")),
            speedup=fmt(code_cached.get("speedup_vs_uncached")),
        ),
        "| Scaled code matmul | scaled-code matmul {latency} ms, scaled-code speedup {speedup}x, scaled-code peak {peak} MB |".format(
            latency=fmt(scaled_code.get("latency_ms")),
            speedup=fmt(scaled_code.get("speedup_vs_uncached")),
            peak=fmt(scaled_code.get("peak_mem_mb")),
        ),
        *grouped_rows,
        "| Cache readout | cache speedup {speedup}x |".format(
            speedup=fmt(cached.get("speedup_vs_uncached")),
        ),
        f"| Next action | {report.get('next_action', 'missing')} |",
    ]


def multicard_compression_sync_rows(summary: dict[str, Any] | None) -> list[str]:
    if not summary:
        return [
            "| Status | missing |",
            "| Readout | missing |",
            "| Qwen compressed all-reduce | missing |",
            "| Memory | missing |",
            "| Sync window | missing |",
        ]
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
    sync = summary.get("sync", {}) if isinstance(summary.get("sync"), dict) else {}
    parallel = summary.get("parallel_suite", {}) if isinstance(summary.get("parallel_suite"), dict) else {}
    return [
        f"| Status | {summary.get('status', 'missing')} |",
        f"| Readout | {summary.get('readout', 'missing')} |",
        "| Sync | world size {world}, backend {backend} |".format(
            world=sync.get("world_size", "missing"),
            backend=sync.get("backend", "missing"),
        ),
        "| Qwen compressed all-reduce | QPruner {qpruner} tokens/s total, {speedup}x baseline, all-reduce consistent {consistent} |".format(
            qpruner=fmt(generate.get("qpruner_tokens_per_s_total")),
            speedup=fmt(generate.get("qpruner_speedup_vs_baseline")),
            consistent=generate.get("distributed_reduce_consistent", "missing"),
        ),
        "| CAP all-reduce | {cap} tokens/s total |".format(
            cap=fmt(generate.get("cap_tokens_per_s_total")),
        ),
        "| Memory | QPruner storage reduction {storage}%, QPruner cache peak +{mem} MB |".format(
            storage=fmt(memory.get("qpruner_storage_reduction_pct")),
            mem=fmt(memory.get("qpruner_cache_peak_mem_delta_mb")),
        ),
        "| Sync window | start {start}s, release lag {lag}s |".format(
            start=fmt(parallel.get("start_window_s")),
            lag=fmt(parallel.get("release_lag_window_s")),
        ),
        f"| Next action | {memory.get('next_action', 'missing')} |",
    ]


def objective_coverage_rows(audit: dict[str, Any] | None) -> list[str]:
    if not audit:
        return [
            "| Overall status | missing |",
            "| Objective slice | missing |",
        ]
    rows = [f"| Overall status | {audit.get('status', 'missing')} |"]
    for item in audit.get("sections", []):
        if not isinstance(item, dict):
            continue
        rows.append(
            "| {title} | {status}: {summary}; gap {gap} |".format(
                title=item.get("title", "missing"),
                status=item.get("status", "missing"),
                summary=item.get("evidence_summary", "missing"),
                gap=item.get("remaining_gap", "missing"),
            )
        )
    return rows


def sync_rows(sync: dict[str, Any] | None) -> list[str]:
    if not sync:
        return ["| missing | missing | missing | missing | missing |"]
    rows = []
    for rank in sync.get("ranks", []):
        rows.append(
            "| {rank} | {status} | {device} | {grad_delta} | {param_delta} |".format(
                rank=rank.get("rank", ""),
                status=rank.get("status", ""),
                device=rank.get("device", ""),
                grad_delta=fmt(rank.get("max_grad_delta")),
                param_delta=fmt(rank.get("max_param_delta")),
            )
        )
    return rows or ["| missing | missing | missing | missing | missing |"]


def lora_rows(lora: dict[str, Any] | None) -> list[str]:
    if not lora:
        return ["| missing | missing | missing | missing | missing |"]
    rows = []
    for rank in lora.get("ranks", []):
        rows.append(
            "| {rank} | {status} | {device} | {initial} | {final} | {delta} |".format(
                rank=rank.get("rank", ""),
                status=rank.get("status", ""),
                device=rank.get("device", ""),
                initial=fmt(rank.get("initial_loss")),
                final=fmt(rank.get("final_loss")),
                delta=fmt(rank.get("max_adapter_delta")),
            )
        )
    return rows or ["| missing | missing | missing | missing | missing |"]


def bslora_rows(bslora_sync: dict[str, Any] | None) -> list[str]:
    if not bslora_sync:
        return ["| missing | missing | missing | missing | missing |"]
    rows = []
    for rank in bslora_sync.get("ranks", []):
        rows.append(
            "| {rank} | {status} | {device} | {initial} | {final} | {delta} |".format(
                rank=rank.get("rank", ""),
                status=rank.get("status", ""),
                device=rank.get("device", ""),
                initial=fmt(rank.get("initial_loss")),
                final=fmt(rank.get("final_loss")),
                delta=fmt(rank.get("max_adapter_delta")),
            )
        )
    return rows or ["| missing | missing | missing | missing | missing |"]


def multicard_generate_rows(multicard: dict[str, Any] | None) -> list[str]:
    if not multicard:
        return ["| missing | missing | missing | missing | missing | missing |"]
    rows = []
    for rank in multicard.get("ranks", []):
        benchmark = rank.get("benchmark", {}) if isinstance(rank.get("benchmark"), dict) else {}
        rows.append(
            "| {rank} | {status} | {device} | {baseline} | {cap} | {qpruner} |".format(
                rank=rank.get("rank", ""),
                status=rank.get("status", ""),
                device=rank.get("device", ""),
                baseline=fmt(benchmark.get("baseline", {}).get("tokens_per_s")),
                cap=fmt(benchmark.get("cap", {}).get("tokens_per_s")),
                qpruner=fmt(benchmark.get("qpruner", {}).get("tokens_per_s")),
            )
        )
    return rows or ["| missing | missing | missing | missing | missing | missing |"]


def multicard_qwen_inference_rows(multicard: dict[str, Any] | None) -> list[str]:
    if not multicard:
        return ["| missing | missing | missing | missing | missing | missing | missing |"]
    rows = []
    for rank in multicard.get("ranks", []):
        benchmark = rank.get("benchmark", {}) if isinstance(rank.get("benchmark"), dict) else {}
        rows.append(
            "| {rank} | {status} | {device} | {latency} | {tps} | {tokens} | {peak} |".format(
                rank=rank.get("rank", ""),
                status=rank.get("status", ""),
                device=rank.get("device", ""),
                latency=fmt(benchmark.get("latency_ms")),
                tps=fmt(benchmark.get("tokens_per_s")),
                tokens=fmt(benchmark.get("generated_tokens")),
                peak=fmt(benchmark.get("peak_mem_mb")),
            )
        )
    return rows or ["| missing | missing | missing | missing | missing | missing | missing |"]


def multicard_qwen_lora_rows(multicard: dict[str, Any] | None) -> list[str]:
    if not multicard:
        return ["| missing | missing | missing | missing | missing | missing | missing | missing |"]
    rows = []
    for rank in multicard.get("ranks", []):
        rows.append(
            "| {rank} | {status} | {device} | {initial} | {final} | {delta} | {peak} | {params} |".format(
                rank=rank.get("rank", ""),
                status=rank.get("status", ""),
                device=rank.get("device", ""),
                initial=fmt(rank.get("initial_loss")),
                final=fmt(rank.get("final_loss")),
                delta=fmt(rank.get("max_adapter_delta")),
                peak=fmt(rank.get("peak_mem_mb")),
                params=fmt(rank.get("trainable_adapter_params")),
            )
        )
    return rows or ["| missing | missing | missing | missing | missing | missing | missing | missing |"]


def multicard_quality_rows(multicard: dict[str, Any] | None) -> list[str]:
    if not multicard:
        return ["| missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing |"]
    rows = []
    for rank in multicard.get("ranks", []):
        benchmark = rank.get("benchmark", {}) if isinstance(rank.get("benchmark"), dict) else {}
        baseline = benchmark.get("baseline", {}) if isinstance(benchmark.get("baseline"), dict) else {}
        cap = benchmark.get("cap", {}) if isinstance(benchmark.get("cap"), dict) else {}
        wanda = benchmark.get("wanda", {}) if isinstance(benchmark.get("wanda"), dict) else {}
        sparsegpt = benchmark.get("sparsegpt", {}) if isinstance(benchmark.get("sparsegpt"), dict) else {}
        qpruner = benchmark.get("qpruner", {}) if isinstance(benchmark.get("qpruner"), dict) else {}
        rows.append(
            "| {rank} | {status} | {device} | {baseline_loss} | {cap_delta} | {wanda_delta} | {sparsegpt_delta} | {q_delta} | {baseline_tps} | {cap_tps} | {wanda_tps} | {sparsegpt_tps} | {q_tps} |".format(
                rank=rank.get("rank", ""),
                status=rank.get("status", ""),
                device=rank.get("device", ""),
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
    return rows or ["| missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing |"]


def probe_issue_classes(probe: dict[str, Any] | None) -> str:
    if not probe:
        return "missing"
    classes = [
        str(row.get("id"))
        for row in probe.get("issue_classes", [])
        if isinstance(row, dict) and row.get("id")
    ]
    return ", ".join(classes) if classes else "none"


def probe_export_rows(probe: dict[str, Any] | None) -> list[str]:
    if not probe:
        return ["| missing | missing | missing | missing | missing |"]
    rows = []
    exports = probe.get("exports", {})
    for method in ("cap", "qpruner"):
        payload = exports.get(method, {}) if isinstance(exports.get(method), dict) else {}
        display_method = method.upper() if method == "cap" else "QPruner"
        rows.append(
            "| {method} | {exists} | {method} {hf} / vLLM {vllm} | {hf} | {vllm} | {path} |".format(
                method=display_method,
                exists=payload.get("exists", "missing"),
                hf=payload.get("transformers_load", "missing"),
                vllm=payload.get("vllm_load", "missing"),
                path=payload.get("path", "missing"),
            )
        )
    return rows


def torch_serving_rows(fallback: dict[str, Any] | None) -> list[str]:
    if not fallback:
        return ["| missing | missing | missing | missing | missing | missing |"]
    rows = []
    exports = fallback.get("exports", {})
    for method in ("baseline", "cap", "qpruner"):
        payload = exports.get(method, {}) if isinstance(exports.get(method), dict) else {}
        display_method = {
            "baseline": "Baseline",
            "cap": "CAP",
            "qpruner": "QPruner",
        }[method]
        rows.append(
            "| {method} | {status} | {load} | {latency} | {tps} | {peak} |".format(
                method=display_method,
                status=payload.get("status", "missing"),
                load=payload.get("transformers_load", "missing"),
                latency=fmt(payload.get("latency_ms")),
                tps=fmt(payload.get("tokens_per_s")),
                peak=fmt(payload.get("peak_mem_mb")),
            )
        )
    return rows


def method_label(method: str) -> str:
    return {"baseline": "baseline", "cap": "CAP", "qpruner": "QPruner"}.get(method, method)


def first_root_cause_line(text: str) -> str | None:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for marker in ("AttributeError:", "RuntimeError:", "ValueError:", "TypeError:", "ImportError:"):
            if marker in line:
                return line[line.index(marker) :]
    return None


def vllm_failure_summary(benchmark: dict[str, Any] | None) -> str:
    if not benchmark:
        return "missing"
    exports = benchmark.get("exports", {})
    if not isinstance(exports, dict):
        return "missing"
    failures = []
    root_cause = None
    for method in ("baseline", "cap", "qpruner"):
        payload = exports.get(method, {}) if isinstance(exports.get(method), dict) else {}
        if payload.get("status") == "PASS":
            continue
        if not payload:
            continue
        failures.append((method_label(method), payload.get("error_type") or payload.get("status") or "unknown"))
        if root_cause is None:
            root_cause = first_root_cause_line(
                "\n".join(str(payload.get(key) or "") for key in ("stdout_tail", "stderr_tail", "error"))
            )
    if not failures:
        return "no failing methods"
    labels = ", ".join(label for label, _ in failures)
    error_types = {error_type for _, error_type in failures}
    error_text = failures[0][1] if len(error_types) == 1 else ", ".join(
        f"{label}={error_type}" for label, error_type in failures
    )
    summary = f"{labels}: {error_text}"
    if root_cause:
        summary += f"; {root_cause}"
    return summary


def vllm_benchmark_rows(benchmark: dict[str, Any] | None) -> list[str]:
    if not benchmark:
        return ["| missing | missing | missing | missing | missing | missing | missing | missing | missing | missing |"]
    rows = []
    exports = benchmark.get("exports", {})
    for method in ("baseline", "cap", "qpruner"):
        payload = exports.get(method, {}) if isinstance(exports.get(method), dict) else {}
        display_method = {
            "baseline": "Baseline",
            "cap": "CAP",
            "qpruner": "QPruner",
        }[method]
        rows.append(
            "| {method} | {status} | {load} | {load_time} | {latency} | {tps} | {peak} | {source} | {npu_smi} | {torch_peak} |".format(
                method=display_method,
                status=payload.get("status", "missing"),
                load=payload.get("vllm_load", "missing"),
                load_time=fmt(payload.get("load_time_s")),
                latency=fmt(payload.get("latency_ms")),
                tps=fmt(payload.get("tokens_per_s")),
                peak=fmt(payload.get("peak_mem_mb")),
                source=fmt(payload.get("memory_measurement_source")),
                npu_smi=fmt(payload.get("npu_smi_process_mem_mb")),
                torch_peak=fmt(payload.get("torch_peak_mem_mb")),
            )
        )
    return rows


def parallel_suite_rows(parallel_suite: dict[str, Any] | None) -> list[str]:
    if not parallel_suite:
        return ["| missing | missing | missing | missing | missing | missing |"]
    rows = []
    for worker in parallel_suite.get("workers", []):
        rows.append(
            "| {rank} | {card} | {task} | {status} | {payload} | {runtime} |".format(
                rank=worker.get("rank", ""),
                card=worker.get("card", ""),
                task=worker.get("task", ""),
                status=worker.get("status", ""),
                payload=worker.get("payload_status", ""),
                runtime=fmt(worker.get("runtime_s")),
            )
        )
    return rows or ["| missing | missing | missing | missing | missing | missing |"]


def vllm_parallel_suite_rows(parallel_suite: dict[str, Any] | None) -> list[str]:
    if not parallel_suite:
        return ["| missing | missing | missing | missing | missing | missing | missing |"]
    rows = []
    for worker in parallel_suite.get("workers", []):
        metrics = worker.get("metrics", {}) if isinstance(worker.get("metrics"), dict) else {}
        rows.append(
            "| {rank} | {card} | {method} | {status} | {latency} | {tps} | {shim} / {backend} |".format(
                rank=worker.get("rank", ""),
                card=worker.get("card", ""),
                method=metrics.get("method", "missing"),
                status=worker.get("status", ""),
                latency=fmt(metrics.get("latency_ms")),
                tps=fmt(metrics.get("tokens_per_s")),
                shim=metrics.get("metadata_shim_status", "missing"),
                backend=metrics.get("metadata_backend_forward_status", "missing"),
            )
        )
    return rows or ["| missing | missing | missing | missing | missing | missing | missing |"]


def multicard_qwen_qpruner_grouped_replay_rows(replay: dict[str, Any] | None) -> list[str]:
    if not replay:
        return ["| missing | missing | missing | missing | missing | missing | missing | missing |"]
    rows = []
    for worker in replay.get("workers", []):
        if not isinstance(worker, dict):
            continue
        metrics = worker.get("metrics", {}) if isinstance(worker.get("metrics"), dict) else {}
        rows.append(
            "| {rank} | {card} | {status} | {payload} | {role} | {speedup} | {storage} | {runtime} |".format(
                rank=worker.get("rank", ""),
                card=worker.get("card", ""),
                status=worker.get("status", ""),
                payload=worker.get("payload_status", ""),
                role=metrics.get("best_role", "missing"),
                speedup=fmt(metrics.get("best_grouped_speedup")),
                storage=fmt(metrics.get("code_cache_storage_reduction_pct")),
                runtime=fmt(worker.get("runtime_s")),
            )
        )
    return rows or ["| missing | missing | missing | missing | missing | missing | missing | missing |"]


def npu_monitor_artifact_name(monitor: dict[str, Any] | None) -> str:
    return str((monitor or {}).get("_artifact_name") or "npu_monitor_missing.json")


def npu_monitor_log_path(monitor: dict[str, Any] | None, demo_root: Path) -> str:
    raw = (monitor or {}).get("monitor_log")
    if not raw:
        run_label = (monitor or {}).get("run_label")
        return f"logs/npu-smi-{run_label}.log" if run_label else "logs/npu-smi-missing.log"
    path = Path(str(raw))
    try:
        return str(path.relative_to(demo_root))
    except ValueError:
        return str(path)


def npu_monitor_aicore_peaks(monitor: dict[str, Any] | None) -> str:
    peaks = (monitor or {}).get("max_aicore_by_card", {})
    if not isinstance(peaks, dict) or not peaks:
        return "missing"
    def sort_key(item: tuple[Any, Any]) -> tuple[int, str]:
        key = str(item[0])
        try:
            return (int(key), key)
        except ValueError:
            return (9999, key)

    return ", ".join(f"{card}={value}" for card, value in sorted(peaks.items(), key=sort_key))


def npu_monitor_readout(monitor: dict[str, Any] | None) -> str | None:
    if not isinstance(monitor, dict) or not monitor:
        return None
    return (
        "NPU utilization monitor: {samples} samples, max active process cards {max_active}, "
        "{process_samples} samples with all 8 process cards, {aicore_samples} samples with all 8 AICore nonzero, "
        "per-card AICore peaks {peaks}"
    ).format(
        samples=fmt(monitor.get("samples")),
        max_active=fmt(monitor.get("max_active_process_cards")),
        process_samples=fmt(monitor.get("samples_with_8_active_process_cards")),
        aicore_samples=fmt(monitor.get("samples_with_8_nonzero_aicore")),
        peaks=npu_monitor_aicore_peaks(monitor),
    )


def parallel_task_counts(parallel_suite: dict[str, Any] | None) -> str:
    if not parallel_suite:
        return "missing"
    counts = parallel_suite.get("task_counts", {})
    if not isinstance(counts, dict) or not counts:
        return "missing"
    return ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))


def model_inventory_rows(model_inventory: dict[str, Any] | None) -> list[str]:
    if not model_inventory:
        return ["| missing | missing | missing | missing |"]
    rows = []
    for row in model_inventory.get("target_candidates", []):
        rows.append(
            "| {model} | {status} | {path} | {mb} |".format(
                model=row.get("model_id", ""),
                status=row.get("status", ""),
                path=row.get("path") or "",
                mb=fmt(row.get("parameter_mb")),
            )
        )
    return rows or ["| missing | missing | missing | missing |"]


def tiny_fixture_rows(model_inventory: dict[str, Any] | None) -> list[str]:
    if not model_inventory:
        return ["| missing | missing | missing |"]
    rows = []
    for row in model_inventory.get("tiny_fixtures", []):
        rows.append(
            "| {model} | {status} | {mb} |".format(
                model=row.get("name") or row.get("model_id", ""),
                status=row.get("status", ""),
                mb=fmt(row.get("parameter_mb")),
            )
        )
    return rows or ["| missing | missing | missing |"]


def short_error(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "missing"
    error = str(payload.get("error") or "")
    if len(error) <= 180:
        return error
    return error[:177] + "..."


def qpruner_quality_memory_sweep_text(demo_root: Path) -> str:
    path = demo_root / "reports" / "qwen-qpruner-quality-memory-sweep.md"
    if not path.exists():
        return "QPruner-only quality-memory sweep missing"
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("Best memory-quality point"):
            return line
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


def choice_accuracy_rows(report: dict[str, Any] | None) -> list[str]:
    if not report:
        return ["| missing | missing | missing | missing | missing |"]
    methods = report.get("methods", {}) if isinstance(report.get("methods"), dict) else {}
    rows = []
    for method, label in (("baseline", "baseline"), ("cap", "CAP"), ("qpruner", "QPruner")):
        if method not in methods:
            continue
        payload = methods.get(method, {}) if isinstance(methods.get(method), dict) else {}
        layers = payload.get("targeted_layers") or report.get("target_layer_limit", "missing")
        rows.append(
            "| {method} | {status} | {accuracy} | {correct} / {total} | {layers} / {layer_total} |".format(
                method=label,
                status=payload.get("status", "missing"),
                accuracy=fmt_accuracy(payload.get("accuracy")),
                correct=payload.get("correct", "missing"),
                total=payload.get("total", "missing"),
                layers=layers,
                layer_total=report.get("targeted_layers_total", "missing"),
            )
        )
    return rows


def choice_accuracy_text(report: dict[str, Any] | None) -> str:
    if not report:
        return "choice accuracy table missing"
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


def markdown(
    demo_root: Path,
    model_inventory: dict[str, Any] | None,
    inference: dict[str, Any] | None,
    sync: dict[str, Any] | None,
    lora: dict[str, Any] | None,
    qwen_lora: dict[str, Any] | None,
    qwen_bslora: dict[str, Any] | None,
    multicard_bslora: dict[str, Any] | None,
    qwen_compression: dict[str, Any] | None,
    qwen_generate: dict[str, Any] | None,
    multicard_generate: dict[str, Any] | None,
    multicard_qwen_compression_generate: dict[str, Any] | None,
    torch_serving_fallback: dict[str, Any] | None,
    multicard_parallel_suite: dict[str, Any] | None,
    vllm_parallel_suite: dict[str, Any] | None,
    vllm_probe: dict[str, Any] | None,
    vllm_patch_probe: dict[str, Any] | None,
    vllm_selector_shim_probe: dict[str, Any] | None,
    vllm_serving_benchmark: dict[str, Any] | None,
    qwen35_snapshot: dict[str, Any] | None,
    qwen35_inference: dict[str, Any] | None,
    qwen3_snapshot: dict[str, Any] | None,
    qwen3_inference: dict[str, Any] | None,
    multicard_qwen_inference: dict[str, Any] | None,
    qwen_compression_quality: dict[str, Any] | None,
    multicard_qwen_compression_quality: dict[str, Any] | None,
    qwen_compression_generate: dict[str, Any] | None,
    multicard_qwen_lora_finetune: dict[str, Any] | None,
    compatibility: dict[str, Any] | None,
    compression_memory_report: dict[str, Any] | None = None,
    paper_baseline_coverage_audit: dict[str, Any] | None = None,
    inference_bottleneck_report: dict[str, Any] | None = None,
    inference_acceleration_summary: dict[str, Any] | None = None,
    compressed_native_sweep_summary: dict[str, Any] | None = None,
    qpruner_packed_decode: dict[str, Any] | None = None,
    multicard_compression_sync_summary: dict[str, Any] | None = None,
    multicard_qwen_qpruner_grouped_replay: dict[str, Any] | None = None,
    qwen_grouped_mlp_sweep: dict[str, Any] | None = None,
    qwen_grouped_mlp_memory_tradeoff: dict[str, Any] | None = None,
    npu_utilization_monitor: dict[str, Any] | None = None,
    objective_coverage_audit: dict[str, Any] | None = None,
    choice_accuracy: dict[str, Any] | None = None,
) -> str:
    model_inventory_status = model_inventory.get("status") if model_inventory else "missing"
    inference_status = inference.get("status") if inference else "missing"
    sync_status = sync.get("status") if sync else "missing"
    sync_world = sync.get("world_size") if sync else "missing"
    lora_status = lora.get("status") if lora else "missing"
    lora_world = lora.get("world_size") if lora else "missing"
    qwen_lora_status = qwen_lora.get("status") if qwen_lora else "missing"
    qwen_bslora_status = qwen_bslora.get("status") if qwen_bslora else "missing"
    multicard_bslora_status = multicard_bslora.get("status") if multicard_bslora else "missing"
    multicard_bslora_world = multicard_bslora.get("world_size") if multicard_bslora else "missing"
    multicard_bslora_aggregate = multicard_bslora.get("aggregate", {}) if multicard_bslora else {}
    qwen_compression_status = qwen_compression.get("status") if qwen_compression else "missing"
    qwen_generate_status = qwen_generate.get("status") if qwen_generate else "missing"
    qwen_generate_backend = qwen_generate.get("backend") if qwen_generate else "missing"
    multicard_status = multicard_generate.get("status") if multicard_generate else "missing"
    multicard_world = multicard_generate.get("world_size") if multicard_generate else "missing"
    multicard_aggregate = multicard_generate.get("aggregate", {}) if multicard_generate else {}
    multicard_qwen_compression_status = (
        multicard_qwen_compression_generate.get("status") if multicard_qwen_compression_generate else "missing"
    )
    multicard_qwen_compression_world = (
        multicard_qwen_compression_generate.get("world_size") if multicard_qwen_compression_generate else "missing"
    )
    multicard_qwen_compression_aggregate = (
        multicard_qwen_compression_generate.get("aggregate", {}) if multicard_qwen_compression_generate else {}
    )
    vllm_probe_status = vllm_probe.get("status") if vllm_probe else "missing"
    vllm_patch_status = vllm_patch_probe.get("status") if vllm_patch_probe else "missing"
    vllm_selector_shim_status = vllm_selector_shim_probe.get("status") if vllm_selector_shim_probe else "missing"
    vllm_benchmark_status = vllm_serving_benchmark.get("status") if vllm_serving_benchmark else "missing"
    vllm_benchmark_summary = vllm_serving_benchmark.get("summary", {}) if vllm_serving_benchmark else {}
    vllm_benchmark_failure = vllm_failure_summary(vllm_serving_benchmark)
    inference_bottleneck_status = (
        inference_bottleneck_report.get("status") if inference_bottleneck_report else "missing"
    )
    inference_bottleneck_title = first_diagnosis_title(inference_bottleneck_report)
    inference_bottleneck_next = next_optimization_title(inference_bottleneck_report)
    qpruner_shape_sweep = selected_qpruner_shape_sweep(inference_bottleneck_report)
    qpruner_shape_sweep_artifact = qpruner_shape_sweep.get("artifact")
    qpruner_shape_sweep_report = qpruner_shape_sweep_report_name(qpruner_shape_sweep_artifact)
    qwen_native_profile = selected_qwen_native_profile(inference_bottleneck_report)
    qwen_native_profile_artifact = qwen_native_profile.get("artifact")
    qwen_native_profile_report = qwen_native_profile_report_name(qwen_native_profile_artifact)
    qwen_speed_first_profile = selected_qwen_speed_first_profile(inference_bottleneck_report)
    qwen_speed_first_artifact = qwen_speed_first_profile.get("artifact")
    qwen_speed_first_report = qwen_native_profile_report_name(qwen_speed_first_artifact)
    qwen_speed_first_readout_text = qwen_speed_first_readout(inference_bottleneck_report)
    qwen_native_profile_readout_text = qwen_native_profile_readout(inference_bottleneck_report)
    qwen_native_bits_readout = qwen_native_bits_tradeoff_readout(inference_bottleneck_report)
    qwen_native_bits_artifacts = qwen_native_bits_tradeoff_artifacts(inference_bottleneck_report)
    qwen_native_bits_reports = qwen_native_bits_tradeoff_reports(inference_bottleneck_report)
    qpruner_grouped_projection_plan = (
        inference_bottleneck_report.get("qpruner_grouped_projection_plan", {})
        if inference_bottleneck_report
        else {}
    )
    qpruner_grouped_projection_plan_readout_text = qpruner_grouped_projection_plan_readout(
        inference_bottleneck_report
    )
    qpruner_grouped_projection_shape_artifact = (
        qpruner_grouped_projection_plan.get("shape_artifact")
        if isinstance(qpruner_grouped_projection_plan, dict)
        else None
    )
    qwen_qpruner_grouped_replay = (
        inference_bottleneck_report.get("qwen_qpruner_grouped_replay", {})
        if inference_bottleneck_report
        else {}
    )
    qwen_qpruner_grouped_replay_artifact = (
        qwen_qpruner_grouped_replay.get("artifact")
        if isinstance(qwen_qpruner_grouped_replay, dict)
        else None
    )
    qwen_qpruner_grouped_replay_readout_text = qwen_qpruner_grouped_replay_readout(
        inference_bottleneck_report
    )
    current_vllm_hbm_rerun = current_vllm_hbm_rerun_text(inference_bottleneck_report)
    torch_fallback_status = torch_serving_fallback.get("status") if torch_serving_fallback else "missing"
    torch_fallback_summary = torch_serving_fallback.get("summary", {}) if torch_serving_fallback else {}
    parallel_suite_status = multicard_parallel_suite.get("status") if multicard_parallel_suite else "missing"
    parallel_suite_aggregate = multicard_parallel_suite.get("aggregate", {}) if multicard_parallel_suite else {}
    vllm_parallel_status = vllm_parallel_suite.get("status") if vllm_parallel_suite else "missing"
    vllm_parallel_aggregate = vllm_parallel_suite.get("aggregate", {}) if vllm_parallel_suite else {}
    vllm_parallel_artifact = parallel_suite_artifact_name(
        vllm_parallel_suite,
        "multicard_parallel_suite_vllm_metadata_sync_npu_metrics.json",
    )
    vllm_parallel_report = parallel_suite_report_name(
        vllm_parallel_suite,
        "multicard-parallel-suite-vllm_metadata_sync_npu_metrics.md",
    )
    multicard_qwen_qpruner_grouped_replay_status = (
        multicard_qwen_qpruner_grouped_replay.get("status")
        if multicard_qwen_qpruner_grouped_replay
        else "missing"
    )
    multicard_qwen_qpruner_grouped_replay_world = (
        multicard_qwen_qpruner_grouped_replay.get("world_size")
        if multicard_qwen_qpruner_grouped_replay
        else "missing"
    )
    multicard_qwen_qpruner_grouped_replay_aggregate = (
        multicard_qwen_qpruner_grouped_replay.get("aggregate", {})
        if multicard_qwen_qpruner_grouped_replay
        else {}
    )
    multicard_qwen_qpruner_grouped_replay_readout_text = (
        multicard_qwen_qpruner_grouped_replay_readout(multicard_qwen_qpruner_grouped_replay)
    )
    multicard_qwen_qpruner_grouped_replay_artifact = str(
        (multicard_qwen_qpruner_grouped_replay or {}).get("_artifact_name")
        or "multicard_qwen_qpruner_grouped_replay_qwen3_06b_real_grouped_replay_2card_npu.json"
    )
    multicard_qwen_qpruner_grouped_replay_report = multicard_qwen_qpruner_grouped_replay_report_name(
        multicard_qwen_qpruner_grouped_replay,
        "multicard-qwen-qpruner-grouped-replay-qwen3_06b_real_grouped_replay_2card_npu.md",
    )
    qwen_grouped_mlp_sweep_status = qwen_grouped_mlp_sweep.get("status") if qwen_grouped_mlp_sweep else "missing"
    qwen_grouped_mlp_sweep_artifact = str(
        (qwen_grouped_mlp_sweep or {}).get("_artifact_name")
        or "qwen3_06b_native_profile_8card_grouped_mlp_sweep_missing.json"
    )
    qwen_grouped_mlp_sweep_report = qwen_grouped_mlp_sweep_report_name(qwen_grouped_mlp_sweep)
    qwen_grouped_mlp_sweep_readout_text = qwen_grouped_mlp_sweep_readout(qwen_grouped_mlp_sweep)
    qwen_grouped_mlp_memory_tradeoff_readout_text = qwen_grouped_mlp_memory_tradeoff_readout(
        qwen_grouped_mlp_memory_tradeoff
    )
    npu_monitor_status = npu_utilization_monitor.get("status") if npu_utilization_monitor else "missing"
    npu_monitor_max_active = (
        npu_utilization_monitor.get("max_active_process_cards") if npu_utilization_monitor else "missing"
    )
    npu_monitor_readout_text = npu_monitor_readout(npu_utilization_monitor)
    npu_monitor_artifact = npu_monitor_artifact_name(npu_utilization_monitor)
    npu_monitor_log = npu_monitor_log_path(npu_utilization_monitor, demo_root)
    vllm_packages = vllm_probe.get("package_diagnostics", {}) if vllm_probe else {}
    vllm_acl = vllm_probe.get("acl_diagnostics", {}) if vllm_probe else {}
    vllm_api = vllm_probe.get("api_diagnostics", {}) if vllm_probe else {}
    vllm_patch_api = vllm_patch_probe.get("api_diagnostics", {}) if vllm_patch_probe else {}
    vllm_selector_shim_api = (
        vllm_selector_shim_probe.get("api_diagnostics", {}) if vllm_selector_shim_probe else {}
    )
    qwen35_snapshot_status = qwen35_snapshot.get("status") if qwen35_snapshot else "missing"
    qwen35_inference_status = qwen35_inference.get("status") if qwen35_inference else "missing"
    qwen3_snapshot_status = qwen3_snapshot.get("status") if qwen3_snapshot else "missing"
    qwen3_inference_status = qwen3_inference.get("status") if qwen3_inference else "missing"
    multicard_qwen_status = multicard_qwen_inference.get("status") if multicard_qwen_inference else "missing"
    multicard_qwen_world = multicard_qwen_inference.get("world_size") if multicard_qwen_inference else "missing"
    multicard_qwen_aggregate = multicard_qwen_inference.get("aggregate", {}) if multicard_qwen_inference else {}
    qwen_compression_quality_status = qwen_compression_quality.get("status") if qwen_compression_quality else "missing"
    qwen_quality_baseline = qwen_compression_quality.get("baseline", {}) if qwen_compression_quality else {}
    qwen_quality_cap = qwen_compression_quality.get("cap", {}) if qwen_compression_quality else {}
    qwen_quality_wanda = qwen_compression_quality.get("wanda", {}) if qwen_compression_quality else {}
    qwen_quality_sparsegpt = qwen_compression_quality.get("sparsegpt", {}) if qwen_compression_quality else {}
    qwen_quality_qpruner = qwen_compression_quality.get("qpruner", {}) if qwen_compression_quality else {}
    multicard_qwen_quality_status = (
        multicard_qwen_compression_quality.get("status") if multicard_qwen_compression_quality else "missing"
    )
    multicard_qwen_quality_world = (
        multicard_qwen_compression_quality.get("world_size") if multicard_qwen_compression_quality else "missing"
    )
    multicard_qwen_quality_aggregate = (
        multicard_qwen_compression_quality.get("aggregate", {}) if multicard_qwen_compression_quality else {}
    )
    qwen_compression_generate_status = qwen_compression_generate.get("status") if qwen_compression_generate else "missing"
    qwen_generate_baseline = qwen_compression_generate.get("baseline", {}) if qwen_compression_generate else {}
    qwen_generate_cap = qwen_compression_generate.get("cap", {}) if qwen_compression_generate else {}
    qwen_generate_qpruner = qwen_compression_generate.get("qpruner", {}) if qwen_compression_generate else {}
    multicard_qwen_lora_status = (
        multicard_qwen_lora_finetune.get("status") if multicard_qwen_lora_finetune else "missing"
    )
    multicard_qwen_lora_world = (
        multicard_qwen_lora_finetune.get("world_size") if multicard_qwen_lora_finetune else "missing"
    )
    multicard_qwen_lora_aggregate = (
        multicard_qwen_lora_finetune.get("aggregate", {}) if multicard_qwen_lora_finetune else {}
    )
    compatibility_status = compatibility.get("status") if compatibility else "missing"
    compatibility_issue_count = len(compatibility.get("issues", [])) if compatibility else "missing"
    cap = qwen_compression.get("cap", {}) if qwen_compression else {}
    qpruner = qwen_compression.get("qpruner", {}) if qwen_compression else {}
    compression_baseline = qwen_compression.get("baseline", {}) if qwen_compression else {}
    generate_baseline = qwen_generate.get("baseline", {}) if qwen_generate else {}
    generate_cap = qwen_generate.get("cap", {}) if qwen_generate else {}
    generate_qpruner = qwen_generate.get("qpruner", {}) if qwen_generate else {}
    serving_dense_export = qwen_generate.get("serving_dense_export") if qwen_generate else "missing"
    cap_layers = (
        f"{fmt(generate_cap.get('exported_dense_linears'))} dense / {fmt(generate_cap.get('cache_modules'))} cached"
    )
    qpruner_layers = (
        f"{fmt(generate_qpruner.get('exported_dense_linears'))} dense / {fmt(generate_qpruner.get('cache_modules'))} cached"
    )
    lines = [
        "# Ascend 910B Demo Progress",
        "",
        "## What Is Ready For Video",
        "",
        "- Multi-card synchronization is the primary demo evidence: HCCL sync, LoRA adapter sync, BSLoRA adapter checksum sync, and compressed-generate all-reduce totals all preserve per-rank artifacts.",
        f"- Model inventory status: `{model_inventory_status}`.",
        "- CAP/QPruner/RankAdaptor tiny workflow smokes have already produced per-card artifacts under `artifacts/`.",
        "- TinyQwen3-Offline torch_npu inference is available as a no-download benchmark artifact.",
        f"- 8-card HCCL sync smoke status: `{sync_status}` with world size `{sync_world}`.",
        f"- RankAdaptor LoRA 8-card fine-tuning smoke status: `{lora_status}` with world size `{lora_world}`.",
        f"- TinyQwen RankAdaptor LoRA fine-tune status: `{qwen_lora_status}`.",
        f"- TinyQwen BSLoRA/shared-LoRA fine-tune status: `{qwen_bslora_status}`.",
        f"- TinyQwen BSLoRA 8-card synchronized fine-tune status: `{multicard_bslora_status}` with world size `{multicard_bslora_world}`.",
        f"- TinyQwen CAP/QPruner compressed inference status: `{qwen_compression_status}`.",
        f"- TinyQwen CAP/QPruner compressed generate status: `{qwen_generate_status}`.",
        f"- Multi-card TinyQwen compressed generate status: `{multicard_status}` with world size `{multicard_world}`.",
        f"- Modern Qwen3.5-0.8B snapshot status: `{qwen35_snapshot_status}`.",
        f"- Modern Qwen3.5-0.8B torch_npu inference status: `{qwen35_inference_status}`.",
        f"- Runnable Qwen3-0.6B torch_npu inference status: `{qwen3_inference_status}`.",
        f"- Multi-card Qwen3-0.6B torch_npu inference status: `{multicard_qwen_status}` with world size `{multicard_qwen_world}`.",
        f"- Qwen3-0.6B CAP/QPruner compression quality status: `{qwen_compression_quality_status}`.",
        f"- Multi-card Qwen3-0.6B CAP/QPruner compression quality status: `{multicard_qwen_quality_status}` with world size `{multicard_qwen_quality_world}`.",
        f"- Qwen3-0.6B CAP/QPruner compressed generate status: `{qwen_compression_generate_status}`.",
        f"- Multi-card Qwen3-0.6B RankAdaptor LoRA fine-tune status: `{multicard_qwen_lora_status}` with world size `{multicard_qwen_lora_world}`.",
        f"- Multi-card Qwen3-0.6B CAP/QPruner compressed generate status: `{multicard_qwen_compression_status}` with world size `{multicard_qwen_compression_world}`.",
        f"- Multi-card Qwen3 real grouped replay status: `{multicard_qwen_qpruner_grouped_replay_status}` with world size `{multicard_qwen_qpruner_grouped_replay_world}`.",
        f"- Qwen3 full-model grouped-MLP 8-card sweep status: `{qwen_grouped_mlp_sweep_status}`.",
        f"- NPU utilization monitor status: `{npu_monitor_status}`, max active process cards `{npu_monitor_max_active}`.",
        f"- torch_npu serving fallback benchmark status: `{torch_fallback_status}`.",
        f"- Synchronized multi-card parallel suite status: `{parallel_suite_status}` with start window `{fmt(multicard_parallel_suite.get('start_window_s') if multicard_parallel_suite else None)}`.",
        f"- Synchronized vLLM multi-card serving slice status: `{vllm_parallel_status}` with start window `{fmt(vllm_parallel_suite.get('start_window_s') if vllm_parallel_suite else None)}`.",
        f"- vLLM default serving probe status: `{vllm_probe_status}`.",
        f"- vLLM patch-preload serving probe status: `{vllm_patch_status}`.",
        f"- vLLM selector-shim serving probe status: `{vllm_selector_shim_status}`.",
        f"- vLLM metadata-shim serving benchmark status: `{vllm_benchmark_status}`.",
        f"- Compatibility issue report status: `{compatibility_status}` with `{compatibility_issue_count}` action items.",
        "- The HCCL launch path requires `HCCL_HOST_SOCKET_PORT_RANGE=auto` and `HCCL_NPU_SOCKET_PORT_RANGE=auto`; the sync script now sets both for NPU runs.",
        "",
        "## Model Inventory",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Status | {model_inventory_status} |",
        f"| Recommended next action | {model_inventory.get('recommended_next_action') if model_inventory else 'missing'} |",
        "",
        "| Target model | Status | Path | MB |",
        "|---|---:|---|---:|",
        *model_inventory_rows(model_inventory),
        "",
        "| Tiny fixture | Status | MB |",
        "|---|---:|---:|",
        *tiny_fixture_rows(model_inventory),
        "",
        "## Modern Model Track",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Snapshot status | {qwen35_snapshot_status} |",
        f"| Snapshot model | {qwen35_snapshot.get('model_id') if qwen35_snapshot else 'missing'} |",
        f"| Snapshot path | {qwen35_snapshot.get('model_path') if qwen35_snapshot else 'missing'} |",
        f"| Snapshot parameter MB | {fmt(qwen35_snapshot.get('parameter_mb') if qwen35_snapshot else None)} |",
        f"| torch_npu inference status | {qwen35_inference_status} |",
        f"| torch_npu inference backend | {qwen35_inference.get('backend') if qwen35_inference else 'missing'} |",
        f"| torch_npu inference error type | {qwen35_inference.get('error_type') if qwen35_inference else 'missing'} |",
        f"| torch_npu inference error | {short_error(qwen35_inference)} |",
        "",
        "## Qwen3-0.6B Runnable Track",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Snapshot status | {qwen3_snapshot_status} |",
        f"| Snapshot provider | {qwen3_snapshot.get('provider') if qwen3_snapshot else 'missing'} |",
        f"| Snapshot model | {qwen3_snapshot.get('model_id') if qwen3_snapshot else 'missing'} |",
        f"| Snapshot path | {qwen3_snapshot.get('model_path') if qwen3_snapshot else 'missing'} |",
        f"| Snapshot parameter MB | {fmt(qwen3_snapshot.get('parameter_mb') if qwen3_snapshot else None)} |",
        f"| torch_npu inference status | {qwen3_inference_status} |",
        f"| torch_npu inference backend | {qwen3_inference.get('backend') if qwen3_inference else 'missing'} |",
        f"| torch_npu inference device | {qwen3_inference.get('device') if qwen3_inference else 'missing'} |",
        f"| torch_npu inference dtype | {qwen3_inference.get('dtype') if qwen3_inference else 'missing'} |",
        f"| Batch size | {qwen3_inference.get('batch_size') if qwen3_inference else 'missing'} |",
        f"| Prompt tokens | {qwen3_inference.get('prompt_tokens') if qwen3_inference else 'missing'} |",
        f"| Generated tokens | {qwen3_inference.get('generated_tokens') if qwen3_inference else 'missing'} |",
        f"| Latency ms | {fmt(qwen3_inference.get('latency_ms') if qwen3_inference else None)} |",
        f"| Tokens/s | {fmt(qwen3_inference.get('tokens_per_s') if qwen3_inference else None)} |",
        f"| Peak MB | {fmt(qwen3_inference.get('peak_mem_mb') if qwen3_inference else None)} |",
        "",
        "## Multi-Card Qwen3-0.6B Inference",
        "",
        "This Qwen3-0.6B benchmark is measured from launched ranks synchronized with all-reduce totals, not extrapolated from one card.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Status | {multicard_qwen_status} |",
        f"| Model | {multicard_qwen_inference.get('model_id') if multicard_qwen_inference else 'missing'} |",
        f"| Model path | {multicard_qwen_inference.get('model_path') if multicard_qwen_inference else 'missing'} |",
        f"| World size | {multicard_qwen_world} |",
        f"| Backend | {multicard_qwen_inference.get('backend') if multicard_qwen_inference else 'missing'} |",
        f"| Device request | {multicard_qwen_inference.get('device_request') if multicard_qwen_inference else 'missing'} |",
        f"| all-reduce consistent | {multicard_qwen_aggregate.get('distributed_reduce_consistent', 'missing')} |",
        f"| Tokens/s total | {fmt(multicard_qwen_aggregate.get('tokens_per_s_total'))} |",
        f"| Generated tokens total | {fmt(multicard_qwen_aggregate.get('generated_tokens_total'))} |",
        f"| Peak MB total | {fmt(multicard_qwen_aggregate.get('peak_mem_mb_total'))} |",
        f"| Passing ranks | {fmt(multicard_qwen_aggregate.get('pass_count'))} |",
        "",
        "| Rank | Status | Device | Latency ms | Tokens/s | Generated tokens | Peak MB |",
        "|---:|---|---|---:|---:|---:|---:|",
        *multicard_qwen_inference_rows(multicard_qwen_inference),
        "",
        "## Qwen3-0.6B CAP/QPruner Compression Quality",
        "",
        "This modern-model quality pilot evaluates held-out forward loss and latency after bounded CAP/QPruner compression.",
        "",
        "| Metric | Baseline | CAP | WANDA | SparseGPT | QPruner |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Status | {qwen_quality_baseline.get('status', qwen_compression_quality_status)} | {qwen_quality_cap.get('status', 'missing')} | {qwen_quality_wanda.get('status', 'missing')} | {qwen_quality_sparsegpt.get('status', 'missing')} | {qwen_quality_qpruner.get('status', 'missing')} |",
        f"| Backend | {qwen_compression_quality.get('backend') if qwen_compression_quality else 'missing'} | {qwen_compression_quality.get('backend') if qwen_compression_quality else 'missing'} | {qwen_compression_quality.get('backend') if qwen_compression_quality else 'missing'} | {qwen_compression_quality.get('backend') if qwen_compression_quality else 'missing'} | {qwen_compression_quality.get('backend') if qwen_compression_quality else 'missing'} |",
        f"| Target layers | {fmt(qwen_quality_baseline.get('targeted_layers'))} / {qwen_compression_quality.get('targeted_layers_total') if qwen_compression_quality else 'missing'} | {fmt(qwen_quality_cap.get('targeted_layers'))} / {qwen_compression_quality.get('targeted_layers_total') if qwen_compression_quality else 'missing'} | {fmt(qwen_quality_wanda.get('targeted_layers'))} / {qwen_compression_quality.get('targeted_layers_total') if qwen_compression_quality else 'missing'} | {fmt(qwen_quality_sparsegpt.get('targeted_layers'))} / {qwen_compression_quality.get('targeted_layers_total') if qwen_compression_quality else 'missing'} | {fmt(qwen_quality_qpruner.get('targeted_layers'))} / {qwen_compression_quality.get('targeted_layers_total') if qwen_compression_quality else 'missing'} |",
        f"| Loss | {fmt(qwen_quality_baseline.get('loss'))} | {fmt(qwen_quality_cap.get('loss'))} | {fmt(qwen_quality_wanda.get('loss'))} | {fmt(qwen_quality_sparsegpt.get('loss'))} | {fmt(qwen_quality_qpruner.get('loss'))} |",
        f"| Loss delta | - | {fmt(qwen_quality_cap.get('loss_delta'))} | {fmt(qwen_quality_wanda.get('loss_delta'))} | {fmt(qwen_quality_sparsegpt.get('loss_delta'))} | {fmt(qwen_quality_qpruner.get('loss_delta'))} |",
        f"| Latency ms | {fmt(qwen_quality_baseline.get('latency_ms'))} | {fmt(qwen_quality_cap.get('latency_ms'))} | {fmt(qwen_quality_wanda.get('latency_ms'))} | {fmt(qwen_quality_sparsegpt.get('latency_ms'))} | {fmt(qwen_quality_qpruner.get('latency_ms'))} |",
        f"| Tokens/s | {fmt(qwen_quality_baseline.get('tokens_per_s'))} | {fmt(qwen_quality_cap.get('tokens_per_s'))} | {fmt(qwen_quality_wanda.get('tokens_per_s'))} | {fmt(qwen_quality_sparsegpt.get('tokens_per_s'))} | {fmt(qwen_quality_qpruner.get('tokens_per_s'))} |",
        f"| Speedup | - | {fmt(qwen_quality_cap.get('latency_speedup'))} | {fmt(qwen_quality_wanda.get('latency_speedup'))} | {fmt(qwen_quality_sparsegpt.get('latency_speedup'))} | {fmt(qwen_quality_qpruner.get('latency_speedup'))} |",
        f"| Compression metric | - | {fmt(qwen_quality_cap.get('targeted_compression_ratio'))}x targeted | {fmt(qwen_quality_wanda.get('targeted_param_reduction_pct'))}% sparse | {fmt(qwen_quality_sparsegpt.get('targeted_param_reduction_pct'))}% sparse | {fmt(qwen_quality_qpruner.get('average_bits'))} avg bits |",
        f"| Compression seconds | - | {fmt(qwen_quality_cap.get('compression_time_s'))} | {fmt(qwen_quality_wanda.get('compression_time_s'))} | {fmt(qwen_quality_sparsegpt.get('compression_time_s'))} | {fmt(qwen_quality_qpruner.get('compression_time_s'))} |",
        f"| Peak MB | {fmt(qwen_compression_quality.get('peak_mem_mb') if qwen_compression_quality else None)} | {fmt(qwen_compression_quality.get('peak_mem_mb') if qwen_compression_quality else None)} | {fmt(qwen_compression_quality.get('peak_mem_mb') if qwen_compression_quality else None)} | {fmt(qwen_compression_quality.get('peak_mem_mb') if qwen_compression_quality else None)} | {fmt(qwen_compression_quality.get('peak_mem_mb') if qwen_compression_quality else None)} |",
        "",
        "## Qwen3-0.6B QPruner-Only Quality-Memory Frontier",
        "",
        qpruner_quality_memory_sweep_text(demo_root),
        "",
        "## Qwen3-0.6B QPruner Scale-Quality Summary",
        "",
        qpruner_scale_quality_text(demo_root),
        "",
        "## Qwen3-0.6B Compression Choice Accuracy",
        "",
        choice_accuracy_text(choice_accuracy),
        "",
        "| Method | Status | Accuracy | Correct | Target layers |",
        "|---|---:|---:|---:|---:|",
        *choice_accuracy_rows(choice_accuracy),
        "",
        "- Markdown: `reports/qwen-qpruner-quality-memory-sweep.md`",
        "- CSV: `reports/qwen-qpruner-quality-memory-sweep.csv`",
        "- Scale-quality JSON: `artifacts/qwen_qpruner_scale_quality_summary.json`",
        "- Scale-quality Markdown: `reports/qwen-qpruner-scale-quality-summary.md`",
        "- Scale-quality CSV: `reports/qwen-qpruner-scale-quality-summary.csv`",
        f"- Choice accuracy JSON: `artifacts/{CHOICE_ACCURACY_ARTIFACT}`",
        f"- Choice accuracy Markdown: `reports/{CHOICE_ACCURACY_REPORT}`",
        f"- Choice accuracy CSV: `reports/{CHOICE_ACCURACY_CSV}`",
        f"- Multi-task choice accuracy JSON: `artifacts/{MULTITASK_CHOICE_ACCURACY_ARTIFACT}`",
        f"- Multi-task choice accuracy Markdown: `reports/{MULTITASK_CHOICE_ACCURACY_REPORT}`",
        f"- Multi-task choice accuracy CSV: `reports/{MULTITASK_CHOICE_ACCURACY_CSV}`",
        "- Frontier chart SVG: `reports/qwen-qpruner-quality-memory-frontier.svg`",
        "- Frontier chart Markdown: `reports/qwen-qpruner-quality-memory-frontier.md`",
        "",
        "## Multi-Card Qwen3-0.6B CAP/QPruner Compression Quality",
        "",
        "This Qwen3-0.6B quality benchmark is launched on selected ranks and synchronized with all-reduce loss averages and throughput totals.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Status | {multicard_qwen_quality_status} |",
        f"| Model | {multicard_qwen_compression_quality.get('model_id') if multicard_qwen_compression_quality else 'missing'} |",
        f"| Model path | {multicard_qwen_compression_quality.get('model_path') if multicard_qwen_compression_quality else 'missing'} |",
        f"| World size | {multicard_qwen_quality_world} |",
        f"| Backend | {multicard_qwen_compression_quality.get('backend') if multicard_qwen_compression_quality else 'missing'} |",
        f"| Target layers | {multicard_qwen_compression_quality.get('target_layer_limit') if multicard_qwen_compression_quality else 'missing'} / {multicard_qwen_compression_quality.get('targeted_layers_total') if multicard_qwen_compression_quality else 'missing'} |",
        f"| all-reduce consistent | {multicard_qwen_quality_aggregate.get('distributed_reduce_consistent', 'missing')} |",
        f"| Baseline loss avg | {fmt(multicard_qwen_quality_aggregate.get('baseline_loss_avg'))} |",
        f"| CAP loss delta avg | {fmt(multicard_qwen_quality_aggregate.get('cap_loss_delta_avg'))} |",
        f"| WANDA loss delta avg | {fmt(multicard_qwen_quality_aggregate.get('wanda_loss_delta_avg'))} |",
        f"| SparseGPT loss delta avg | {fmt(multicard_qwen_quality_aggregate.get('sparsegpt_loss_delta_avg'))} |",
        f"| QPruner loss delta avg | {fmt(multicard_qwen_quality_aggregate.get('qpruner_loss_delta_avg'))} |",
        f"| Baseline tokens/s total | {fmt(multicard_qwen_quality_aggregate.get('baseline_tokens_per_s_total'))} |",
        f"| CAP tokens/s total | {fmt(multicard_qwen_quality_aggregate.get('cap_tokens_per_s_total'))} |",
        f"| WANDA tokens/s total | {fmt(multicard_qwen_quality_aggregate.get('wanda_tokens_per_s_total'))} |",
        f"| SparseGPT tokens/s total | {fmt(multicard_qwen_quality_aggregate.get('sparsegpt_tokens_per_s_total'))} |",
        f"| QPruner tokens/s total | {fmt(multicard_qwen_quality_aggregate.get('qpruner_tokens_per_s_total'))} |",
        f"| Peak MB total | {fmt(multicard_qwen_quality_aggregate.get('peak_mem_mb_total'))} |",
        f"| Passing ranks | {fmt(multicard_qwen_quality_aggregate.get('pass_count'))} |",
        "",
        "| Rank | Status | Device | Baseline loss | CAP loss delta | WANDA loss delta | SparseGPT loss delta | QPruner loss delta | Baseline tok/s | CAP tok/s | WANDA tok/s | SparseGPT tok/s | QPruner tok/s |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *multicard_quality_rows(multicard_qwen_compression_quality),
        "",
        "## Qwen3-0.6B CAP/QPruner Generate",
        "",
        "This is a modern-model compression pilot. `target_layer_limit` records the intentionally bounded layer subset used to keep the demo reproducible.",
        "",
        "| Metric | Baseline | CAP | QPruner |",
        "|---|---:|---:|---:|",
        f"| Status | {qwen_generate_baseline.get('status', qwen_compression_generate_status)} | {qwen_generate_cap.get('status', 'missing')} | {qwen_generate_qpruner.get('status', 'missing')} |",
        f"| Backend | {qwen_compression_generate.get('backend') if qwen_compression_generate else 'missing'} | {qwen_compression_generate.get('backend') if qwen_compression_generate else 'missing'} | {qwen_compression_generate.get('backend') if qwen_compression_generate else 'missing'} |",
        f"| Target layers | {fmt(qwen_generate_baseline.get('targeted_layers'))} / {qwen_compression_generate.get('targeted_layers_total') if qwen_compression_generate else 'missing'} | {fmt(qwen_generate_cap.get('targeted_layers'))} / {qwen_compression_generate.get('targeted_layers_total') if qwen_compression_generate else 'missing'} | {fmt(qwen_generate_qpruner.get('targeted_layers'))} / {qwen_compression_generate.get('targeted_layers_total') if qwen_compression_generate else 'missing'} |",
        f"| Latency ms | {fmt(qwen_generate_baseline.get('latency_ms'))} | {fmt(qwen_generate_cap.get('latency_ms'))} | {fmt(qwen_generate_qpruner.get('latency_ms'))} |",
        f"| Tokens/s | {fmt(qwen_generate_baseline.get('tokens_per_s'))} | {fmt(qwen_generate_cap.get('tokens_per_s'))} | {fmt(qwen_generate_qpruner.get('tokens_per_s'))} |",
        f"| Speedup | - | {fmt(qwen_generate_cap.get('latency_speedup'))} | {fmt(qwen_generate_qpruner.get('latency_speedup'))} |",
        f"| Compression metric | - | {fmt(qwen_generate_cap.get('targeted_compression_ratio'))}x targeted | {fmt(qwen_generate_qpruner.get('average_bits'))} avg bits |",
        f"| Dense/cache modules | - | {fmt(qwen_generate_cap.get('exported_dense_linears'))} dense / {fmt(qwen_generate_cap.get('cache_modules'))} cached | {fmt(qwen_generate_qpruner.get('exported_dense_linears'))} dense / {fmt(qwen_generate_qpruner.get('cache_modules'))} cached |",
        f"| Peak MB | {fmt(qwen_compression_generate.get('peak_mem_mb') if qwen_compression_generate else None)} | {fmt(qwen_compression_generate.get('peak_mem_mb') if qwen_compression_generate else None)} | {fmt(qwen_compression_generate.get('peak_mem_mb') if qwen_compression_generate else None)} |",
        "",
        "## Multi-Card Qwen3-0.6B RankAdaptor LoRA Fine-Tune",
        "",
        "This Qwen3-0.6B fine-tune pilot trains RankAdaptor-selected LoRA adapters through DistributedDataParallel and verifies adapter checksum synchronization.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Status | {multicard_qwen_lora_status} |",
        f"| Method | {multicard_qwen_lora_finetune.get('method') if multicard_qwen_lora_finetune else 'missing'} |",
        f"| Model | {multicard_qwen_lora_finetune.get('model_id') if multicard_qwen_lora_finetune else 'missing'} |",
        f"| Model path | {multicard_qwen_lora_finetune.get('model_path') if multicard_qwen_lora_finetune else 'missing'} |",
        f"| World size | {multicard_qwen_lora_world} |",
        f"| Backend | {multicard_qwen_lora_finetune.get('backend') if multicard_qwen_lora_finetune else 'missing'} |",
        f"| Train mode | {multicard_qwen_lora_finetune.get('train_mode') if multicard_qwen_lora_finetune else 'missing'} |",
        f"| Target modules | {multicard_qwen_lora_finetune.get('target_module_count') if multicard_qwen_lora_finetune else 'missing'} / {multicard_qwen_lora_finetune.get('targeted_modules_total') if multicard_qwen_lora_finetune else 'missing'} |",
        f"| Trainable adapter params | {multicard_qwen_lora_finetune.get('trainable_adapter_params') if multicard_qwen_lora_finetune else 'missing'} |",
        f"| all-reduce consistent | {multicard_qwen_lora_aggregate.get('distributed_reduce_consistent', 'missing')} |",
        f"| adapter checksum synchronized | {multicard_qwen_lora_aggregate.get('adapter_sync_consistent', 'missing')} |",
        f"| Initial loss avg | {fmt(multicard_qwen_lora_aggregate.get('initial_loss_avg'))} |",
        f"| Final loss avg | {fmt(multicard_qwen_lora_aggregate.get('final_loss_avg'))} |",
        f"| Loss delta avg | {fmt(multicard_qwen_lora_aggregate.get('loss_delta_avg'))} |",
        f"| Validation initial loss avg | {fmt(multicard_qwen_lora_aggregate.get('validation_initial_loss_avg'))} |",
        f"| Validation final loss avg | {fmt(multicard_qwen_lora_aggregate.get('validation_final_loss_avg'))} |",
        f"| Validation loss delta avg | {fmt(multicard_qwen_lora_aggregate.get('validation_loss_delta_avg'))} |",
        f"| Peak MB total | {fmt(multicard_qwen_lora_aggregate.get('peak_mem_mb_total'))} |",
        f"| Passing ranks | {fmt(multicard_qwen_lora_aggregate.get('pass_count'))} |",
        "",
        "Generation snapshot:",
        "",
        f"- Before: `{multicard_qwen_lora_finetune.get('before_generate') if multicard_qwen_lora_finetune else 'missing'}`",
        f"- After: `{multicard_qwen_lora_finetune.get('after_generate') if multicard_qwen_lora_finetune else 'missing'}`",
        "",
        "| Rank | Status | Device | Initial loss | Final loss | Adapter delta | Peak MB | Trainable params |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
        *multicard_qwen_lora_rows(multicard_qwen_lora_finetune),
        "",
        "## Multi-Card Qwen3-0.6B CAP/QPruner Generate",
        "",
        "This Qwen3-0.6B compressed-generate benchmark is launched on selected ranks and synchronized with all-reduce throughput totals.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Status | {multicard_qwen_compression_status} |",
        f"| Model | {multicard_qwen_compression_generate.get('model_id') if multicard_qwen_compression_generate else 'missing'} |",
        f"| Model path | {multicard_qwen_compression_generate.get('model_path') if multicard_qwen_compression_generate else 'missing'} |",
        f"| World size | {multicard_qwen_compression_world} |",
        f"| Backend | {multicard_qwen_compression_generate.get('backend') if multicard_qwen_compression_generate else 'missing'} |",
        f"| Target layers | {multicard_qwen_compression_generate.get('target_layer_limit') if multicard_qwen_compression_generate else 'missing'} / {multicard_qwen_compression_generate.get('targeted_layers_total') if multicard_qwen_compression_generate else 'missing'} |",
        f"| all-reduce consistent | {multicard_qwen_compression_aggregate.get('distributed_reduce_consistent', 'missing')} |",
        f"| Baseline tokens/s total | {fmt(multicard_qwen_compression_aggregate.get('baseline_tokens_per_s_total'))} |",
        f"| CAP tokens/s total | {fmt(multicard_qwen_compression_aggregate.get('cap_tokens_per_s_total'))} |",
        f"| QPruner tokens/s total | {fmt(multicard_qwen_compression_aggregate.get('qpruner_tokens_per_s_total'))} |",
        f"| Peak MB total | {fmt(multicard_qwen_compression_aggregate.get('peak_mem_mb_total'))} |",
        f"| Passing ranks | {fmt(multicard_qwen_compression_aggregate.get('pass_count'))} |",
        "",
        "| Rank | Status | Device | Baseline tok/s | CAP tok/s | QPruner tok/s |",
        "|---:|---|---|---:|---:|---:|",
        *multicard_generate_rows(multicard_qwen_compression_generate),
        "",
        "## Tiny Qwen Torch NPU Inference",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Status | {inference_status} |",
        f"| Model | {inference.get('model_id') if inference else 'missing'} |",
        f"| Backend | {inference.get('backend') if inference else 'missing'} |",
        f"| Device | {inference.get('device') if inference else 'missing'} |",
        f"| Dtype | {inference.get('dtype') if inference else 'missing'} |",
        f"| Latency ms | {fmt(inference.get('latency_ms') if inference else None)} |",
        f"| Tokens/s | {fmt(inference.get('tokens_per_s') if inference else None)} |",
        f"| Peak MB | {fmt(inference.get('peak_mem_mb') if inference else None)} |",
        "",
        "## 8-Card HCCL Sync",
        "",
        "This table is measured from eight launched ranks and synchronized HCCL collectives, not extrapolated from one card.",
        "",
        "| Rank | Status | Device | Grad delta | Param delta |",
        "|---:|---|---|---:|---:|",
        *sync_rows(sync),
        "",
        "## RankAdaptor LoRA 8-Card Fine-Tuning",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Status | {lora_status} |",
        f"| Backend | {lora.get('backend') if lora else 'missing'} |",
        f"| World size | {lora_world} |",
        f"| Trainable adapter params | {lora.get('trainable_adapter_params') if lora else 'missing'} |",
        f"| Rank allocation | {json.dumps(lora.get('rank_allocation', {}), sort_keys=True) if lora else 'missing'} |",
        "",
        "| Rank | Status | Device | Initial loss | Final loss | Adapter delta |",
        "|---:|---|---|---:|---:|---:|",
        *lora_rows(lora),
        "",
        "## TinyQwen RankAdaptor LoRA Fine-Tune",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Status | {qwen_lora_status} |",
        f"| Model | {qwen_lora.get('model_id') if qwen_lora else 'missing'} |",
        f"| Device | {qwen_lora.get('device') if qwen_lora else 'missing'} |",
        f"| Dtype | {qwen_lora.get('dtype') if qwen_lora else 'missing'} |",
        f"| Trainable adapter params | {qwen_lora.get('trainable_adapter_params') if qwen_lora else 'missing'} |",
        f"| Initial loss | {fmt(qwen_lora.get('initial_loss') if qwen_lora else None)} |",
        f"| Final loss | {fmt(qwen_lora.get('final_loss') if qwen_lora else None)} |",
        f"| Loss delta | {fmt(qwen_lora.get('loss_delta') if qwen_lora else None)} |",
        f"| Peak MB | {fmt(qwen_lora.get('peak_mem_mb') if qwen_lora else None)} |",
        "",
        "Generation snapshot:",
        "",
        f"- Before: `{qwen_lora.get('before_generate') if qwen_lora else 'missing'}`",
        f"- After: `{qwen_lora.get('after_generate') if qwen_lora else 'missing'}`",
        "",
        "## TinyQwen BSLoRA Shared-LoRA Fine-Tune",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Status | {qwen_bslora_status} |",
        f"| Method | {qwen_bslora.get('method') if qwen_bslora else 'missing'} |",
        f"| Model | {qwen_bslora.get('model_id') if qwen_bslora else 'missing'} |",
        f"| Device | {qwen_bslora.get('device') if qwen_bslora else 'missing'} |",
        f"| Dtype | {qwen_bslora.get('dtype') if qwen_bslora else 'missing'} |",
        f"| Trainable adapter params | {qwen_bslora.get('trainable_adapter_params') if qwen_bslora else 'missing'} |",
        f"| Unshared adapter params | {qwen_bslora.get('unshared_adapter_params') if qwen_bslora else 'missing'} |",
        f"| Unique adapter tensors | {qwen_bslora.get('unique_adapter_tensors') if qwen_bslora else 'missing'} |",
        f"| Target modules | {qwen_bslora.get('target_module_count') if qwen_bslora else 'missing'} |",
        f"| Initial loss | {fmt(qwen_bslora.get('initial_loss') if qwen_bslora else None)} |",
        f"| Final loss | {fmt(qwen_bslora.get('final_loss') if qwen_bslora else None)} |",
        f"| Loss delta | {fmt(qwen_bslora.get('loss_delta') if qwen_bslora else None)} |",
        f"| Peak MB | {fmt(qwen_bslora.get('peak_mem_mb') if qwen_bslora else None)} |",
        "",
        "Generation snapshot:",
        "",
        f"- Before: `{qwen_bslora.get('before_generate') if qwen_bslora else 'missing'}`",
        f"- After: `{qwen_bslora.get('after_generate') if qwen_bslora else 'missing'}`",
        "",
        "## Multi-Card TinyQwen BSLoRA Fine-Tune",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Status | {multicard_bslora_status} |",
        f"| World size | {multicard_bslora_world} |",
        f"| Backend | {multicard_bslora.get('backend') if multicard_bslora else 'missing'} |",
        f"| all-reduce consistent | {multicard_bslora_aggregate.get('distributed_reduce_consistent', 'missing')} |",
        f"| adapter checksum synchronized | {multicard_bslora_aggregate.get('adapter_sync_consistent', 'missing')} |",
        f"| Initial loss avg | {fmt(multicard_bslora_aggregate.get('initial_loss_avg'))} |",
        f"| Final loss avg | {fmt(multicard_bslora_aggregate.get('final_loss_avg'))} |",
        f"| Loss delta avg | {fmt(multicard_bslora_aggregate.get('loss_delta_avg'))} |",
        "",
        "| Rank | Status | Device | Initial loss | Final loss | Adapter delta |",
        "|---:|---|---|---:|---:|---:|",
        *bslora_rows(multicard_bslora),
        "",
        "## TinyQwen Compression Quality / Forward",
        "",
        "| Metric | Baseline | CAP | QPruner |",
        "|---|---:|---:|---:|",
        f"| Status | {qwen_compression_status} | {cap.get('status', 'missing')} | {qpruner.get('status', 'missing')} |",
        f"| Loss | {fmt(compression_baseline.get('loss'))} | {fmt(cap.get('loss'))} | {fmt(qpruner.get('loss'))} |",
        f"| Loss delta | - | {fmt(cap.get('loss_delta'))} | {fmt(qpruner.get('loss_delta'))} |",
        f"| Latency ms | {fmt(compression_baseline.get('latency_ms'))} | {fmt(cap.get('latency_ms'))} | {fmt(qpruner.get('latency_ms'))} |",
        f"| Tokens/s | {fmt(compression_baseline.get('tokens_per_s'))} | {fmt(cap.get('tokens_per_s'))} | {fmt(qpruner.get('tokens_per_s'))} |",
        f"| Speedup | - | {fmt(cap.get('latency_speedup'))} | {fmt(qpruner.get('latency_speedup'))} |",
        f"| Compression metric | - | {fmt(cap.get('targeted_compression_ratio'))}x targeted | {fmt(qpruner.get('average_bits'))} avg bits |",
        f"| Compression seconds | - | {fmt(cap.get('compression_time_s'))} | {fmt(qpruner.get('compression_time_s'))} |",
        f"| Peak MB | {fmt(qwen_compression.get('peak_mem_mb') if qwen_compression else None)} | {fmt(qwen_compression.get('peak_mem_mb') if qwen_compression else None)} | {fmt(qwen_compression.get('peak_mem_mb') if qwen_compression else None)} |",
        "",
        "## TinyQwen Compression Generate",
        "",
        "| Metric | Baseline | CAP | QPruner |",
        "|---|---:|---:|---:|",
        f"| Status | {qwen_generate_status} | {generate_cap.get('status', 'missing')} | {generate_qpruner.get('status', 'missing')} |",
        f"| Backend | {qwen_generate_backend} | {qwen_generate_backend} | {qwen_generate_backend} |",
        f"| Latency ms | {fmt(generate_baseline.get('latency_ms'))} | {fmt(generate_cap.get('latency_ms'))} | {fmt(generate_qpruner.get('latency_ms'))} |",
        f"| Tokens/s | {fmt(generate_baseline.get('tokens_per_s'))} | {fmt(generate_cap.get('tokens_per_s'))} | {fmt(generate_qpruner.get('tokens_per_s'))} |",
        f"| Speedup | - | {fmt(generate_cap.get('latency_speedup'))} | {fmt(generate_qpruner.get('latency_speedup'))} |",
        f"| Compression metric | - | {fmt(generate_cap.get('targeted_compression_ratio'))}x targeted | {fmt(generate_qpruner.get('average_bits'))} avg bits |",
        f"| Serving dense export | {serving_dense_export} | {serving_dense_export} | {serving_dense_export} |",
        f"| Dense/cache modules | - | {cap_layers} | {qpruner_layers} |",
        f"| Peak MB | {fmt(qwen_generate.get('peak_mem_mb') if qwen_generate else None)} | {fmt(qwen_generate.get('peak_mem_mb') if qwen_generate else None)} | {fmt(qwen_generate.get('peak_mem_mb') if qwen_generate else None)} |",
        "",
        "## Multi-Card TinyQwen Compression Generate",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Status | {multicard_status} |",
        f"| World size | {multicard_world} |",
        f"| Backend | {multicard_generate.get('backend') if multicard_generate else 'missing'} |",
        f"| all-reduce consistent | {multicard_aggregate.get('distributed_reduce_consistent', 'missing')} |",
        f"| Baseline tokens/s total | {fmt(multicard_aggregate.get('baseline_tokens_per_s_total'))} |",
        f"| CAP tokens/s total | {fmt(multicard_aggregate.get('cap_tokens_per_s_total'))} |",
        f"| QPruner tokens/s total | {fmt(multicard_aggregate.get('qpruner_tokens_per_s_total'))} |",
        "",
        "| Rank | Status | Device | Baseline tok/s | CAP tok/s | QPruner tok/s |",
        "|---:|---|---|---:|---:|---:|",
        *multicard_generate_rows(multicard_generate),
        "",
        "## Torch Serving Fallback Benchmark",
        "",
        "Runnable Transformers torch_generate benchmark for the CAP/QPruner serving exports when vLLM-Ascend is blocked.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Status | {torch_fallback_status} |",
        f"| Backend | {torch_serving_fallback.get('backend') if torch_serving_fallback else 'missing'} |",
        f"| Export run label | {torch_serving_fallback.get('export_run_label') if torch_serving_fallback else 'missing'} |",
        f"| Best method | {torch_fallback_summary.get('best_method', 'missing')} |",
        f"| Best tokens/s | {fmt(torch_fallback_summary.get('best_tokens_per_s'))} |",
        f"| CAP vs baseline speedup | {fmt(torch_fallback_summary.get('cap_vs_baseline_speedup'))} |",
        f"| QPruner vs baseline speedup | {fmt(torch_fallback_summary.get('qpruner_vs_baseline_speedup'))} |",
        f"| CAP vs QPruner speedup | {fmt(torch_fallback_summary.get('cap_vs_qpruner_speedup'))} |",
        "",
        "| Method | Status | Transformers load | Latency ms | Tokens/s | Peak MB |",
        "|---|---:|---:|---:|---:|---:|",
        *torch_serving_rows(torch_serving_fallback),
        "",
        "## Inference Acceleration Summary",
        "",
        "One-screen summary for the demo video: throughput winner, compressed-native preserves quantized/pruned modules, and the vLLM runtime bottleneck.",
        "",
        engineering_baseline_note(),
        "",
        paper_baseline_note(),
        "",
        "| Metric | Value |",
        "|---|---:|",
        *inference_acceleration_summary_rows(inference_acceleration_summary),
        "",
        "- Markdown: `reports/inference-acceleration-summary.md`",
        "- CSV: `reports/inference-acceleration-summary.csv`",
        "- SVG: `reports/inference-acceleration-summary.svg`",
        "- JSON: `artifacts/inference_acceleration_summary.json`",
        "",
        "## Compressed-Native Cache/Long-Decode Sweep",
        "",
        "Tests whether cache reuse and longer decode shift the compressed-native path from storage-only evidence to latency-positive evidence.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        *compressed_native_sweep_rows(compressed_native_sweep_summary),
        "",
        "- Markdown: `reports/compressed-native-cache-long-decode-sweep.md`",
        "- CSV: `reports/compressed-native-cache-long-decode-sweep.csv`",
        "- SVG: `reports/compressed-native-cache-long-decode-sweep.svg`",
        "- JSON: `artifacts/compressed_native_sweep_summary.json`",
        "",
        "## QPruner Packed Decode Benchmark",
        "",
        "Microbenchmark isolating packed-code storage savings from per-forward decode overhead.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        *qpruner_packed_decode_rows(qpruner_packed_decode),
        "",
        f"- Markdown: `reports/{QPRUNER_PACKED_DECODE_REPORT}`",
        f"- JSON: `artifacts/{QPRUNER_PACKED_DECODE_ARTIFACT}`",
        "",
        "## Multi-Card Compression Sync Summary",
        "",
        "One-screen demo evidence tying synchronized Qwen3 compression throughput, quality synchronization, compressed-native memory savings, and paper baseline alignment together.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        *multicard_compression_sync_rows(multicard_compression_sync_summary),
        "",
        "- Markdown: `reports/multicard-compression-sync-summary.md`",
        "- CSV: `reports/multicard-compression-sync-summary.csv`",
        "- SVG: `reports/multicard-compression-sync-summary.svg`",
        "- JSON: `artifacts/multicard_compression_sync_summary.json`",
        "",
        "## Multi-Card Qwen3 Real QPruner Grouped Replay",
        "",
        "This stage starts real Qwen3 QPruner grouped-replay workers together across selected Ascend cards, preserving the compressed code-cache path while measuring module-level grouped projection replay.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Status | {multicard_qwen_qpruner_grouped_replay_status} |",
        f"| Run label | {multicard_qwen_qpruner_grouped_replay.get('run_label') if multicard_qwen_qpruner_grouped_replay else 'missing'} |",
        f"| World size | {multicard_qwen_qpruner_grouped_replay_world} |",
        f"| Cards | {','.join(str(card) for card in multicard_qwen_qpruner_grouped_replay.get('cards', [])) if multicard_qwen_qpruner_grouped_replay else 'missing'} |",
        f"| Launched synchronously | {multicard_qwen_qpruner_grouped_replay.get('launched_synchronously') if multicard_qwen_qpruner_grouped_replay else 'missing'} |",
        f"| Release lag window | {fmt(multicard_qwen_qpruner_grouped_replay.get('release_lag_window_s') if multicard_qwen_qpruner_grouped_replay else None)} |",
        f"| Best role | {multicard_qwen_qpruner_grouped_replay_aggregate.get('best_role', 'missing')} |",
        f"| Best grouped speedup | {fmt(multicard_qwen_qpruner_grouped_replay_aggregate.get('best_grouped_speedup'))}x |",
        f"| Mean grouped speedup | {fmt(multicard_qwen_qpruner_grouped_replay_aggregate.get('mean_grouped_speedup'))}x |",
        f"| Target layers | {fmt(multicard_qwen_qpruner_grouped_replay_aggregate.get('target_layer_limit'))} / {fmt(multicard_qwen_qpruner_grouped_replay_aggregate.get('targeted_layers_total'))} |",
        f"| Quantized layers | {fmt(multicard_qwen_qpruner_grouped_replay_aggregate.get('quantized_layers'))} |",
        f"| Memory-native workers | {fmt(multicard_qwen_qpruner_grouped_replay_aggregate.get('memory_native_worker_count'))} / {fmt(multicard_qwen_qpruner_grouped_replay_aggregate.get('pass_count'))} |",
        f"| Code-cache storage reduction | {fmt(multicard_qwen_qpruner_grouped_replay_aggregate.get('min_code_cache_storage_reduction_pct'))}% |",
        f"| Grouped code bytes | {fmt(multicard_qwen_qpruner_grouped_replay_aggregate.get('max_grouped_code_cache_bytes'))} |",
        f"| Grouped scaled-code bytes | {fmt(multicard_qwen_qpruner_grouped_replay_aggregate.get('max_grouped_scaled_code_cache_bytes'))} |",
        f"| Released packed-code bytes | {fmt(multicard_qwen_qpruner_grouped_replay_aggregate.get('released_packed_code_bytes_total'))} |",
        f"| Live compressed payload bytes | {fmt(multicard_qwen_qpruner_grouped_replay_aggregate.get('live_compressed_payload_storage_bytes_total', multicard_qwen_qpruner_grouped_replay_aggregate.get('live_compressed_payload_bytes_total')))} |",
        f"| Readout | {multicard_qwen_qpruner_grouped_replay_readout_text or 'missing'} |",
        "",
        "| Rank | Card | Status | Payload status | Best role | Speedup | Code-cache storage % | Runtime s |",
        "|---:|---:|---|---:|---|---:|---:|---:|",
        *multicard_qwen_qpruner_grouped_replay_rows(multicard_qwen_qpruner_grouped_replay),
        "",
        f"- Markdown: `reports/{multicard_qwen_qpruner_grouped_replay_report}`",
        f"- JSON: `artifacts/{multicard_qwen_qpruner_grouped_replay_artifact}`",
        "",
        "## NPU Utilization Monitor",
        "",
        "This section preserves the `npu-smi` sampling evidence used for video capture, separate from benchmark correctness metrics.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Status | {npu_monitor_status} |",
        f"| Run label | {npu_utilization_monitor.get('run_label') if npu_utilization_monitor else 'missing'} |",
        f"| Samples | {fmt(npu_utilization_monitor.get('samples') if npu_utilization_monitor else None)} |",
        f"| Max active process cards | {fmt(npu_utilization_monitor.get('max_active_process_cards') if npu_utilization_monitor else None)} |",
        f"| Samples with all 8 process cards | {fmt(npu_utilization_monitor.get('samples_with_8_active_process_cards') if npu_utilization_monitor else None)} |",
        f"| Samples with all 8 AICore nonzero | {fmt(npu_utilization_monitor.get('samples_with_8_nonzero_aicore') if npu_utilization_monitor else None)} |",
        f"| Per-card AICore peaks | {npu_monitor_aicore_peaks(npu_utilization_monitor)} |",
        f"| Readout | {npu_monitor_readout_text or 'missing'} |",
        "",
        f"- JSON: `artifacts/{npu_monitor_artifact}`",
        f"- Log: `{npu_monitor_log}`",
        "",
        "## Objective Coverage Audit",
        "",
        "Requirement-level audit for the current Ascend objective: 910B baseline, inference acceleration, fine-tuning, compression, and demo materials.",
        "",
        "| Metric | Value |",
        "|---|---|",
        *objective_coverage_rows(objective_coverage_audit),
        "",
        "- Markdown: `reports/objective-coverage-audit.md`",
        "- JSON: `artifacts/objective_coverage_audit.json`",
        "",
        "## Synchronized Multi-Card Parallel Suite",
        "",
        "This stage starts independent workflow and serving workers together, one per selected card, so the recording can show concurrent Ascend utilization rather than a serial demo.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Status | {parallel_suite_status} |",
        f"| Run label | {multicard_parallel_suite.get('run_label') if multicard_parallel_suite else 'missing'} |",
        f"| World size | {multicard_parallel_suite.get('world_size') if multicard_parallel_suite else 'missing'} |",
        f"| Cards | {','.join(str(card) for card in multicard_parallel_suite.get('cards', [])) if multicard_parallel_suite else 'missing'} |",
        f"| Launched synchronously | {multicard_parallel_suite.get('launched_synchronously') if multicard_parallel_suite else 'missing'} |",
        f"| start window | {fmt(multicard_parallel_suite.get('start_window_s') if multicard_parallel_suite else None)} |",
        f"| Task counts | {parallel_task_counts(multicard_parallel_suite)} |",
        f"| Passing workers | {fmt(parallel_suite_aggregate.get('pass_count'))} |",
        f"| Failed workers | {fmt(parallel_suite_aggregate.get('fail_count'))} |",
        f"| Best serving fallback tokens/s | {fmt(parallel_suite_aggregate.get('best_tokens_per_s'))} |",
        "",
        "| Rank | Card | Task | Status | Payload status | Runtime s |",
        "|---:|---:|---|---:|---:|---:|",
        *parallel_suite_rows(multicard_parallel_suite),
        "",
        "## Synchronized vLLM Multi-Card Serving Slice",
        "",
        "This stage starts baseline, CAP, and QPruner vLLM-Ascend workers at the same sync gate, so the demo can show concurrent compressed-model serving rather than a serial benchmark.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Status | {vllm_parallel_status} |",
        f"| Run label | {vllm_parallel_suite.get('run_label') if vllm_parallel_suite else 'missing'} |",
        f"| World size | {vllm_parallel_suite.get('world_size') if vllm_parallel_suite else 'missing'} |",
        f"| Cards | {','.join(str(card) for card in vllm_parallel_suite.get('cards', [])) if vllm_parallel_suite else 'missing'} |",
        f"| Launched synchronously | {vllm_parallel_suite.get('launched_synchronously') if vllm_parallel_suite else 'missing'} |",
        f"| start window | {fmt(vllm_parallel_suite.get('start_window_s') if vllm_parallel_suite else None)} |",
        f"| release lag window | {fmt(vllm_parallel_suite.get('release_lag_window_s') if vllm_parallel_suite else None)} |",
        f"| Task counts | {parallel_task_counts(vllm_parallel_suite)} |",
        f"| Passing workers | {fmt(vllm_parallel_aggregate.get('pass_count'))} |",
        f"| Failed workers | {fmt(vllm_parallel_aggregate.get('fail_count'))} |",
        f"| best_vllm_tokens_per_s | {fmt(vllm_parallel_aggregate.get('best_vllm_tokens_per_s'))} |",
        "",
        "| Rank | Card | Method | Status | Latency ms | Tokens/s | metadata_shim_status / backend_forward |",
        "|---:|---:|---|---:|---:|---:|---|",
        *vllm_parallel_suite_rows(vllm_parallel_suite),
        "",
        "## vLLM Serving Compatibility Boundary",
        "",
        "| Metric | Default probe | Patch-preload probe | Selector-shim probe |",
        "|---|---:|---:|---:|",
        f"| Status | {vllm_probe_status} | {vllm_patch_status} | {vllm_selector_shim_status} |",
        f"| Issue classes | {probe_issue_classes(vllm_probe)} | {probe_issue_classes(vllm_patch_probe)} | {probe_issue_classes(vllm_selector_shim_probe)} |",
        f"| Export run label | {vllm_probe.get('export_run_label') if vllm_probe else 'missing'} | {vllm_patch_probe.get('export_run_label') if vllm_patch_probe else 'missing'} | {vllm_selector_shim_probe.get('export_run_label') if vllm_selector_shim_probe else 'missing'} |",
        f"| gpu_memory_utilization | {fmt(vllm_probe.get('gpu_memory_utilization') if vllm_probe else None)} | {fmt(vllm_patch_probe.get('gpu_memory_utilization') if vllm_patch_probe else None)} | {fmt(vllm_selector_shim_probe.get('gpu_memory_utilization') if vllm_selector_shim_probe else None)} |",
        f"| preload_vllm_ascend_patch | {vllm_probe.get('preload_vllm_ascend_patch') if vllm_probe else 'missing'} | {vllm_patch_probe.get('preload_vllm_ascend_patch') if vllm_patch_probe else 'missing'} | {vllm_selector_shim_probe.get('preload_vllm_ascend_patch') if vllm_selector_shim_probe else 'missing'} |",
        f"| preload_vllm_ascend_selector_shim | {vllm_probe.get('preload_vllm_ascend_selector_shim') if vllm_probe else 'missing'} | {vllm_patch_probe.get('preload_vllm_ascend_selector_shim') if vllm_patch_probe else 'missing'} | {vllm_selector_shim_probe.get('preload_vllm_ascend_selector_shim') if vllm_selector_shim_probe else 'missing'} |",
        f"| ACL import | {vllm_acl.get('import_acl', 'missing')} | {(vllm_patch_probe.get('acl_diagnostics', {}) if vllm_patch_probe else {}).get('import_acl', 'missing')} | {(vllm_selector_shim_probe.get('acl_diagnostics', {}) if vllm_selector_shim_probe else {}).get('import_acl', 'missing')} |",
        f"| vLLM version | {vllm_packages.get('vllm', {}).get('version', 'missing')} | {(vllm_patch_probe.get('package_diagnostics', {}) if vllm_patch_probe else {}).get('vllm', {}).get('version', 'missing')} | {(vllm_selector_shim_probe.get('package_diagnostics', {}) if vllm_selector_shim_probe else {}).get('vllm', {}).get('version', 'missing')} |",
        f"| vLLM-Ascend version | {vllm_packages.get('vllm-ascend', {}).get('version', 'missing')} | {(vllm_patch_probe.get('package_diagnostics', {}) if vllm_patch_probe else {}).get('vllm-ascend', {}).get('version', 'missing')} | {(vllm_selector_shim_probe.get('package_diagnostics', {}) if vllm_selector_shim_probe else {}).get('vllm-ascend', {}).get('version', 'missing')} |",
        f"| Attention selector API status | {vllm_api.get('status', 'missing')} | {vllm_patch_api.get('status', 'missing')} | {vllm_selector_shim_api.get('status', 'missing')} |",
        f"| Missing patch config fields | {', '.join(vllm_api.get('missing_patch_config_fields') or []) or 'none'} | {', '.join(vllm_patch_api.get('missing_patch_config_fields') or []) or 'none'} | {', '.join(vllm_selector_shim_api.get('missing_patch_config_fields') or []) or 'none'} |",
        f"| Missing patch get_attn_backend parameters | {', '.join(vllm_api.get('missing_patch_get_attn_backend_parameters') or []) or 'none'} | {', '.join(vllm_patch_api.get('missing_patch_get_attn_backend_parameters') or []) or 'none'} | {', '.join(vllm_selector_shim_api.get('missing_patch_get_attn_backend_parameters') or []) or 'none'} |",
        "",
        "Default vLLM probe export load checks:",
        "",
        "| Method | Exists | Summary | Transformers load | vLLM load | Path |",
        "|---|---:|---|---:|---:|---|",
        *probe_export_rows(vllm_probe),
        "",
        "Patch-preload vLLM probe export load checks:",
        "",
        "| Method | Exists | Summary | Transformers load | vLLM load | Path |",
        "|---|---:|---|---:|---:|---|",
        *probe_export_rows(vllm_patch_probe),
        "",
        "Selector-shim vLLM probe export load checks:",
        "",
        "| Method | Exists | Summary | Transformers load | vLLM load | Path |",
        "|---|---:|---|---:|---:|---|",
        *probe_export_rows(vllm_selector_shim_probe),
        "",
        "## vLLM-Ascend Serving Benchmark",
        "",
        "vLLM generate benchmark on the same baseline/CAP/QPruner serving exports after preloading the project-local selector shim.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Status | {vllm_benchmark_status} |",
        f"| Backend | {vllm_serving_benchmark.get('backend') if vllm_serving_benchmark else 'missing'} |",
        f"| Export run label | {vllm_serving_benchmark.get('export_run_label') if vllm_serving_benchmark else 'missing'} |",
        f"| Dtype | {vllm_serving_benchmark.get('dtype') if vllm_serving_benchmark else 'missing'} |",
        f"| gpu_memory_utilization | {fmt(vllm_serving_benchmark.get('gpu_memory_utilization') if vllm_serving_benchmark else None)} |",
        f"| preload_vllm_ascend_patch | {vllm_serving_benchmark.get('preload_vllm_ascend_patch') if vllm_serving_benchmark else 'missing'} |",
        f"| preload_vllm_ascend_selector_shim | {vllm_serving_benchmark.get('preload_vllm_ascend_selector_shim') if vllm_serving_benchmark else 'missing'} |",
        f"| preload_vllm_ascend_metadata_shim | {vllm_serving_benchmark.get('preload_vllm_ascend_metadata_shim') if vllm_serving_benchmark else 'missing'} |",
        f"| metadata_shim_status | {(vllm_serving_benchmark.get('metadata_shim', {}) if vllm_serving_benchmark else {}).get('status', 'missing')} |",
        f"| Best method | {vllm_benchmark_summary.get('best_method', 'missing')} |",
        f"| Best tokens/s | {fmt(vllm_benchmark_summary.get('best_tokens_per_s'))} |",
        f"| CAP vs baseline speedup | {fmt(vllm_benchmark_summary.get('cap_vs_baseline_speedup'))} |",
        f"| QPruner vs baseline speedup | {fmt(vllm_benchmark_summary.get('qpruner_vs_baseline_speedup'))} |",
        f"| qpruner_vs_baseline | {fmt(vllm_benchmark_summary.get('qpruner_vs_baseline_speedup'))}x |",
        f"| Root cause | {vllm_benchmark_failure} |",
        "",
        "vLLM selector-shim speedup summary: qpruner_vs_baseline={speedup}x.".format(
            speedup=fmt(vllm_benchmark_summary.get("qpruner_vs_baseline_speedup"))
        ),
        "",
        "| Method | Status | vLLM load | Load time s | Latency ms | Tokens/s | Peak MB | Peak source | NPU process MB | Torch peak MB |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
        *vllm_benchmark_rows(vllm_serving_benchmark),
        "",
        "## Paper Baseline Alignment",
        "",
        "This section keeps paper baselines separate from the engineering `baseline` method key used for runtime ratios.",
        "",
        "| Metric | Value |",
        "|---|---|",
        *paper_baseline_audit_rows(paper_baseline_coverage_audit),
        *compression_memory_rows(
            compression_memory_report,
            include_legacy_baseline_evidence=paper_baseline_coverage_audit is None,
        ),
        "",
        "## Inference Bottleneck Diagnosis",
        "",
        "This stage separates synchronized multi-card execution from the still-open serving acceleration target.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Status | {inference_bottleneck_status} |",
        f"| Primary diagnosis | {inference_bottleneck_title} |",
        *inference_bottleneck_diagnosis_rows(inference_bottleneck_report),
        *(
            ["| Qwen paired native profile | {readout} |".format(readout=qwen_native_profile_readout_text)]
            if qwen_native_profile_readout_text
            else []
        ),
        *(
            ["| Qwen speed-first native profile | {readout} |".format(readout=qwen_speed_first_readout_text)]
            if qwen_speed_first_readout_text
            else []
        ),
        *(["| Qwen native bits tradeoff | {readout} |".format(readout=qwen_native_bits_readout)] if qwen_native_bits_readout else []),
        *(
            [
                "| QPruner grouped projection plan | {readout} |".format(
                    readout=qpruner_grouped_projection_plan_readout_text
                )
            ]
            if qpruner_grouped_projection_plan_readout_text
            else []
        ),
        *(
            [
                "| Qwen3 real grouped replay | {readout} |".format(
                    readout=qwen_qpruner_grouped_replay_readout_text
                )
            ]
            if qwen_qpruner_grouped_replay_readout_text
            else []
        ),
        *(
            [
                "| Multi-card Qwen3 real grouped replay | {readout} |".format(
                    readout=multicard_qwen_qpruner_grouped_replay_readout_text
                )
            ]
            if multicard_qwen_qpruner_grouped_replay_readout_text
            else []
        ),
        "| Qwen3 grouped-MLP full-model sweep | {readout} |".format(
            readout=qwen_grouped_mlp_sweep_readout_text
        ),
        "| Qwen3 grouped-MLP memory-first tradeoff | {readout} |".format(
            readout=qwen_grouped_mlp_memory_tradeoff_readout_text
        ),
        *(
            ["| Current vLLM HBM rerun | {readout} |".format(readout=current_vllm_hbm_rerun)]
            if current_vllm_hbm_rerun
            else []
        ),
        f"| Next optimization target | {inference_bottleneck_next} |",
        "",
        "## Compatibility Issues",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Status | {compatibility_status} |",
        f"| Action items | {compatibility_issue_count} |",
        f"| Ready evidence items | {len(compatibility.get('ready', [])) if compatibility else 'missing'} |",
        f"| Notes | {len(compatibility.get('notes', [])) if compatibility else 'missing'} |",
        "",
        "## Evidence Paths",
        "",
        f"- Demo root: `{demo_root}`",
        "- Model inventory JSON: `artifacts/model_inventory.json`",
        "- Model inventory Markdown: `reports/model-inventory.md`",
        "- Qwen3.5 snapshot JSON: `artifacts/model_snapshot_qwen35_08b.json`",
        "- Qwen3.5 snapshot Markdown: `reports/model-snapshot-qwen35_08b.md`",
        "- Qwen3.5 torch_npu inference JSON: `artifacts/ascend_inference_qwen35_08b_torch_npu.json`",
        "- Qwen3.5 torch_npu inference Markdown: `reports/ascend-inference-qwen35_08b_torch_npu.md`",
        "- Qwen3-0.6B snapshot JSON: `artifacts/model_snapshot_qwen3_06b.json`",
        "- Qwen3-0.6B snapshot Markdown: `reports/model-snapshot-qwen3_06b.md`",
        "- Qwen3-0.6B torch_npu inference JSON: `artifacts/ascend_inference_qwen3_06b_torch_npu.json`",
        "- Qwen3-0.6B torch_npu inference Markdown: `reports/ascend-inference-qwen3_06b_torch_npu.md`",
        "- Qwen3-0.6B multi-card inference JSON: `artifacts/multicard_qwen_inference_qwen3_06b_npu.json`",
        "- Qwen3-0.6B multi-card inference Markdown: `reports/multicard-qwen-inference-qwen3_06b_npu.md`",
        "- Qwen3-0.6B patterned compression quality JSON: `artifacts/qwen_compression_quality_qwen3_06b_quality_pattern_npu.json`",
        "- Qwen3-0.6B patterned compression quality Markdown: `reports/qwen-compression-quality-qwen3_06b_quality_pattern_npu.md`",
        "- Qwen3-0.6B compression quality JSON: `artifacts/qwen_compression_quality_qwen3_06b_quality_npu.json`",
        "- Qwen3-0.6B compression quality Markdown: `reports/qwen-compression-quality-qwen3_06b_quality_npu.md`",
        "- Qwen3-0.6B LLM-Pruner baseline JSON: `artifacts/qwen_llm_pruner_baseline_qwen3_06b_llm_pruner_npu.json`",
        "- Qwen3-0.6B LLM-Pruner baseline Markdown: `reports/qwen-llm-pruner-baseline-qwen3_06b_llm_pruner_npu.md`",
        "- Qwen3-0.6B LLM-Pruner-style full196 baseline JSON: `artifacts/qwen_llm_pruner_baseline_qwen3_06b_llm_pruner_full196_npu.json`",
        "- Qwen3-0.6B Wanda/SparseGPT full196 baseline JSON: `artifacts/paper_baseline_qwen3_06b_wanda_sparsegpt_full196_npu.json`",
        "- Qwen3-0.6B RankAdaptor recovery baseline JSON: `artifacts/rankadaptor_recovery_baseline_suite_qwen3_06b_recovery_baselines_npu.json`",
        "- Qwen3-0.6B patterned multi-card compression quality JSON: `artifacts/multicard_qwen_compression_quality_qwen3_06b_quality_pattern_2card_npu.json`",
        "- Qwen3-0.6B patterned multi-card compression quality Markdown: `reports/multicard-qwen-compression-quality-qwen3_06b_quality_pattern_2card_npu.md`",
        "- Qwen3-0.6B multi-card compression quality JSON: `artifacts/multicard_qwen_compression_quality_qwen3_06b_quality_sync_npu.json`",
        "- Qwen3-0.6B multi-card compression quality Markdown: `reports/multicard-qwen-compression-quality-qwen3_06b_quality_sync_npu.md`",
        "- Qwen3-0.6B quality-memory sweep Markdown: `reports/qwen-compression-quality-memory-sweep.md`",
        "- Qwen3-0.6B QPruner-only quality-memory sweep Markdown: `reports/qwen-qpruner-quality-memory-sweep.md`",
        "- Qwen3-0.6B QPruner-only quality-memory sweep CSV: `reports/qwen-qpruner-quality-memory-sweep.csv`",
        "- Qwen3-0.6B QPruner scale-quality summary JSON: `artifacts/qwen_qpruner_scale_quality_summary.json`",
        "- Qwen3-0.6B QPruner scale-quality summary Markdown: `reports/qwen-qpruner-scale-quality-summary.md`",
        "- Qwen3-0.6B QPruner scale-quality summary CSV: `reports/qwen-qpruner-scale-quality-summary.csv`",
        f"- Qwen3-0.6B compression choice accuracy JSON: `artifacts/{CHOICE_ACCURACY_ARTIFACT}`",
        f"- Qwen3-0.6B compression choice accuracy Markdown: `reports/{CHOICE_ACCURACY_REPORT}`",
        f"- Qwen3-0.6B compression choice accuracy CSV: `reports/{CHOICE_ACCURACY_CSV}`",
        f"- Qwen3-0.6B multi-task choice accuracy JSON: `artifacts/{MULTITASK_CHOICE_ACCURACY_ARTIFACT}`",
        f"- Qwen3-0.6B multi-task choice accuracy Markdown: `reports/{MULTITASK_CHOICE_ACCURACY_REPORT}`",
        f"- Qwen3-0.6B multi-task choice accuracy CSV: `reports/{MULTITASK_CHOICE_ACCURACY_CSV}`",
        f"- Qwen3-0.6B full-target QPruner choice accuracy JSON: `artifacts/{FULLTARGET_QPRUNER_CHOICE_ARTIFACT}`",
        f"- Qwen3-0.6B full-target QPruner choice accuracy Markdown: `reports/{FULLTARGET_QPRUNER_CHOICE_REPORT}`",
        f"- Qwen3-0.6B full-target QPruner choice accuracy CSV: `reports/{FULLTARGET_QPRUNER_CHOICE_CSV}`",
        "- Qwen3-0.6B QPruner-only quality-memory frontier SVG: `reports/qwen-qpruner-quality-memory-frontier.svg`",
        "- Qwen3-0.6B QPruner-only quality-memory frontier Markdown: `reports/qwen-qpruner-quality-memory-frontier.md`",
        "- Qwen3-0.6B compression generate JSON: `artifacts/qwen_compression_generate_qwen3_06b_generate_npu.json`",
        "- Qwen3-0.6B compression generate Markdown: `reports/qwen-compression-generate-qwen3_06b_generate_npu.md`",
        "- Qwen3-0.6B compression generate sweep Markdown: `reports/qwen-compression-generate-sweep.md`",
        "- Qwen3-0.6B multi-card LoRA fine-tune JSON: `artifacts/multicard_qwen_lora_finetune_qwen3_06b_lora_sync_npu.json`",
        "- Qwen3-0.6B multi-card LoRA fine-tune Markdown: `reports/multicard-qwen-lora-finetune-qwen3_06b_lora_sync_npu.md`",
        "- Qwen3-0.6B multi-card compression generate JSON: `artifacts/{}`".format(
            artifact_with_default(
                multicard_qwen_compression_generate,
                "multicard_qwen_compression_generate_qwen3_06b_compression_generate_npu.json",
            )
        ),
        "- Qwen3-0.6B multi-card compression generate Markdown: `reports/{}`".format(
            report_with_default(
                multicard_qwen_compression_generate,
                "multicard-qwen-compression-generate-qwen3_06b_compression_generate_npu.md",
            )
        ),
        "- Inference JSON: `artifacts/ascend_inference_tiny_qwen3_torch_npu.json`",
        "- Multi-card sync JSON: `artifacts/multicard_sync_summary.json`",
        "- Multi-card sync Markdown: `reports/multicard-sync-summary.md`",
        "- RankAdaptor LoRA sync JSON: `artifacts/rankadaptor_lora_sync_summary.json`",
        "- RankAdaptor LoRA sync Markdown: `reports/rankadaptor-lora-sync-summary.md`",
        "- TinyQwen LoRA fine-tune JSON: `artifacts/tiny_qwen_lora_finetune_tiny_qwen3_lora_npu.json`",
        "- TinyQwen LoRA fine-tune Markdown: `reports/tiny-qwen-lora-finetune-tiny_qwen3_lora_npu.md`",
        "- TinyQwen BSLoRA fine-tune JSON: `artifacts/tiny_qwen_bslora_finetune_tiny_qwen3_bslora_npu.json`",
        "- TinyQwen BSLoRA fine-tune Markdown: `reports/tiny-qwen-bslora-finetune-tiny_qwen3_bslora_npu.md`",
        "- Multi-card TinyQwen BSLoRA JSON: `artifacts/multicard_tiny_qwen_bslora_finetune_tiny_qwen3_bslora_sync_npu.json`",
        "- Multi-card TinyQwen BSLoRA Markdown: `reports/multicard-tiny-qwen-bslora-finetune-tiny_qwen3_bslora_sync_npu.md`",
        "- TinyQwen compression quality JSON: `artifacts/tiny_qwen_compression_tiny_qwen3_compression_npu.json`",
        "- TinyQwen compression quality Markdown: `reports/tiny-qwen-compression-tiny_qwen3_compression_npu.md`",
        "- TinyQwen compression generate JSON: `artifacts/tiny_qwen_compression_generate_tiny_qwen3_generate_serving_export_npu.json`",
        "- TinyQwen compression generate Markdown: `reports/tiny-qwen-compression-generate-tiny_qwen3_generate_serving_export_npu.md`",
        "- Multi-card TinyQwen compression JSON: `artifacts/multicard_tiny_qwen_generate_tiny_qwen3_multicard_generate_npu.json`",
        "- Multi-card TinyQwen compression Markdown: `reports/multicard-tiny-qwen-generate-tiny_qwen3_multicard_generate_npu.md`",
        "- Torch serving fallback JSON: `artifacts/torch_serving_fallback_tiny_qwen3_serving_fallback_npu.json`",
        "- Torch serving fallback Markdown: `reports/torch-serving-fallback-tiny_qwen3_serving_fallback_npu.md`",
        "- Inference acceleration summary JSON: `artifacts/inference_acceleration_summary.json`",
        "- Inference acceleration summary Markdown: `reports/inference-acceleration-summary.md`",
        "- Inference acceleration summary CSV: `reports/inference-acceleration-summary.csv`",
        "- Inference acceleration summary SVG: `reports/inference-acceleration-summary.svg`",
        "- Compressed-native sweep JSON: `artifacts/compressed_native_sweep_summary.json`",
        "- Compressed-native sweep Markdown: `reports/compressed-native-cache-long-decode-sweep.md`",
        "- Compressed-native sweep CSV: `reports/compressed-native-cache-long-decode-sweep.csv`",
        "- Compressed-native sweep SVG: `reports/compressed-native-cache-long-decode-sweep.svg`",
        f"- QPruner packed decode JSON: `artifacts/{QPRUNER_PACKED_DECODE_ARTIFACT}`",
        f"- QPruner packed decode Markdown: `reports/{QPRUNER_PACKED_DECODE_REPORT}`",
        "- QPruner Qwen3 shape sweep JSON: `artifacts/qpruner_packed_decode_benchmark_qwen3_06b_shape_sweep_npu.json`",
        "- QPruner Qwen3 shape sweep Markdown: `reports/qpruner-packed-decode-benchmark-qwen3_06b_shape_sweep_npu.md`",
        *(
            []
            if not qpruner_shape_sweep_artifact
            else [f"- QPruner selected Qwen3 shape sweep JSON: `artifacts/{qpruner_shape_sweep_artifact}`"]
        ),
        *(
            []
            if not qpruner_shape_sweep_report
            else [f"- QPruner selected Qwen3 shape sweep Markdown: `reports/{qpruner_shape_sweep_report}`"]
        ),
        *(
            []
            if not qpruner_grouped_projection_shape_artifact
            else [
                "- QPruner grouped projection actual-shape JSON: "
                f"`artifacts/{qpruner_grouped_projection_shape_artifact}`"
            ]
        ),
        *(
            []
            if not qwen_qpruner_grouped_replay_artifact
            else [
                "- Qwen3 real QPruner grouped replay JSON: "
                f"`artifacts/{qwen_qpruner_grouped_replay_artifact}`"
            ]
        ),
        *(
            []
            if not qwen_native_profile_artifact
            else [f"- QPruner selected Qwen3 native profile JSON: `artifacts/{qwen_native_profile_artifact}`"]
        ),
        *(
            []
            if not qwen_native_profile_report
            else [f"- QPruner selected Qwen3 native profile Markdown: `reports/{qwen_native_profile_report}`"]
        ),
        *(
            []
            if not qwen_speed_first_artifact
            else [f"- QPruner speed-first Qwen3 native profile JSON: `artifacts/{qwen_speed_first_artifact}`"]
        ),
        *(
            []
            if not qwen_speed_first_report
            else [f"- QPruner speed-first Qwen3 native profile Markdown: `reports/{qwen_speed_first_report}`"]
        ),
        *(f"- QPruner Qwen3 native bits tradeoff JSON: `artifacts/{artifact}`" for artifact in qwen_native_bits_artifacts),
        *(f"- QPruner Qwen3 native bits tradeoff Markdown: `reports/{report}`" for report in qwen_native_bits_reports),
        f"- QPruner Qwen3 dtype-preserving shape sweep JSON: `artifacts/{QPRUNER_DTYPE_SHAPE_SWEEP_ARTIFACT}`",
        f"- QPruner Qwen3 dtype-preserving shape sweep Markdown: `reports/{QPRUNER_DTYPE_SHAPE_SWEEP_REPORT}`",
        "- Multi-card compression sync JSON: `artifacts/multicard_compression_sync_summary.json`",
        "- Multi-card compression sync Markdown: `reports/multicard-compression-sync-summary.md`",
        "- Multi-card compression sync CSV: `reports/multicard-compression-sync-summary.csv`",
        "- Multi-card compression sync SVG: `reports/multicard-compression-sync-summary.svg`",
        f"- Multi-card Qwen3 real QPruner grouped replay JSON: `artifacts/{multicard_qwen_qpruner_grouped_replay_artifact}`",
        f"- Multi-card Qwen3 real QPruner grouped replay Markdown: `reports/{multicard_qwen_qpruner_grouped_replay_report}`",
        f"- Qwen3 grouped-MLP full-model sweep JSON: `artifacts/{qwen_grouped_mlp_sweep_artifact}`",
        f"- Qwen3 grouped-MLP full-model sweep Markdown: `reports/{qwen_grouped_mlp_sweep_report}`",
        *(f"- Qwen3 grouped-MLP full-model sweep evidence: `{item}`" for item in qwen_grouped_mlp_sweep_evidence(qwen_grouped_mlp_sweep)),
        *(
            f"- Qwen3 grouped-MLP memory-first tradeoff evidence: `{item}`"
            for item in qwen_grouped_mlp_sweep_evidence(qwen_grouped_mlp_memory_tradeoff)
        ),
        f"- NPU utilization monitor JSON: `artifacts/{npu_monitor_artifact}`",
        f"- NPU utilization monitor log: `{npu_monitor_log}`",
        "- Objective coverage audit JSON: `artifacts/objective_coverage_audit.json`",
        "- Objective coverage audit Markdown: `reports/objective-coverage-audit.md`",
        "- Synchronized parallel suite JSON: `artifacts/multicard_parallel_suite_demo_parallel_suite_npu.json`",
        "- Synchronized parallel suite Markdown: `reports/multicard-parallel-suite-demo_parallel_suite_npu.md`",
        f"- Synchronized vLLM parallel suite JSON: `artifacts/{vllm_parallel_artifact}`",
        f"- Synchronized vLLM parallel suite Markdown: `reports/{vllm_parallel_report}`",
        "- vLLM default probe JSON: `artifacts/vllm_serving_probe_tiny_qwen3_serving_export_npu.json`",
        "- vLLM default probe Markdown: `reports/vllm-serving-probe-tiny_qwen3_serving_export_npu.md`",
        "- vLLM patch-preload probe JSON: `artifacts/vllm_serving_probe_tiny_qwen3_serving_export_npu_preload_patch.json`",
        "- vLLM patch-preload probe Markdown: `reports/vllm-serving-probe-tiny_qwen3_serving_export_npu_preload_patch.md`",
        "- vLLM selector-shim probe JSON: `artifacts/vllm_serving_probe_tiny_qwen3_serving_export_npu_selector_shim.json`",
        "- vLLM selector-shim probe Markdown: `reports/vllm-serving-probe-tiny_qwen3_serving_export_npu_selector_shim.md`",
        "- vLLM metadata-shim benchmark JSON: `artifacts/vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu.json`",
        "- vLLM metadata-shim benchmark Markdown: `reports/vllm-serving-benchmark-tiny_qwen3_serving_vllm_metadata_shim_npu.md`",
        "- Compression memory JSON: `artifacts/compression_memory_report.json`",
        "- Compression memory Markdown: `reports/compression-memory-report.md`",
        "- Paper baseline coverage JSON: `artifacts/paper_baseline_coverage_audit.json`",
        "- Paper baseline coverage Markdown: `reports/paper-baseline-coverage-audit.md`",
        "- Inference bottleneck JSON: `artifacts/inference_bottleneck_report.json`",
        "- Inference bottleneck Markdown: `reports/inference-bottleneck-report.md`",
        "- Compatibility issues JSON: `artifacts/compatibility_issues.json`",
        "- Compatibility issues Markdown: `reports/compatibility-issues.md`",
        "",
        "## Demo Story",
        "",
        "1. Show `npu-smi info` and the 8 Ascend910B1 cards.",
        "2. Show the 8-card HCCL sync table first to establish that the demo is synchronized across cards.",
        "3. Show RankAdaptor LoRA 8-card fine-tuning loss and adapter synchronization.",
        "4. Show multi-card TinyQwen BSLoRA all-reduce loss averages and adapter checksum synchronization.",
        "5. Show multi-card TinyQwen compressed generate all-reduce throughput totals to prove synchronized benchmark execution across cards.",
        "6. Show parallel CAP/QPruner/RankAdaptor smokes to establish algorithm compatibility.",
        "7. Show model inventory and Qwen3.5-0.8B snapshot: the latest small model is downloaded, while current Transformers needs qwen3_5 support before inference.",
        "8. Show Qwen3-0.6B torch_npu inference as the modern runnable model path.",
        "9. Show Qwen3-0.6B 8-card inference all-reduce totals as the modern synchronized model path.",
        "10. Show Qwen3-0.6B CAP/QPruner compression quality loss deltas with its explicit target-layer limit.",
        "11. Show Qwen3-0.6B 8-card CAP/QPruner compression quality all-reduce loss averages.",
        "12. Show Qwen3-0.6B CAP/QPruner compressed generate pilot with its explicit target-layer limit.",
        "13. Show Qwen3-0.6B 8-card RankAdaptor LoRA fine-tune all-reduce loss and adapter checksum synchronization.",
        "14. Show Qwen3-0.6B 8-card CAP/QPruner compressed generate all-reduce totals as the modern synchronized compression path.",
        "15. Show the multi-card compression sync summary: synchronized compression throughput, QPruner storage reduction, and baseline alignment on one page.",
        "16. Show multi-card Qwen3 real QPruner grouped replay: synchronized workers, memory-native code-cache storage, and grouped projection speedup.",
        "17. Show the NPU utilization monitor: npu-smi samples with 8 active process cards and all-card AICore activity for the grouped replay run.",
        "18. Show the objective coverage audit so the video explicitly maps evidence to baseline, acceleration, fine-tuning, compression, and demo-material goals.",
        "19. Show TinyQwen3-Offline torch_npu inference as the offline, reproducible inference path.",
        "20. Show TinyQwen HF LoRA fine-tuning loss and before/after generation snapshot.",
        "21. Show TinyQwen BSLoRA/shared-LoRA parameter sharing and loss delta.",
        "22. Show TinyQwen CAP/QPruner compression quality metrics: loss delta, compression time, CAP targeted compression ratio, and QPruner average bitwidth.",
        "23. Show TinyQwen compressed generate metrics and serving dense export tradeoff.",
        "24. Show synchronized multi-card parallel suite: CAP, QPruner, RankAdaptor, and torch_npu serving fallback workers start together across cards.",
        "25. Show synchronized vLLM multi-card serving slice: baseline, CAP, and QPruner workers launch together with metadata shim status in each worker payload.",
        "26. Show vLLM default and patch-preload probes to isolate the upstream attention selector mismatch, then show selector-shim probe PASS without changing system packages.",
        "27. Show vLLM selector-shim latency/throughput/memory benchmark on baseline/CAP/QPruner serving exports.",
        "28. Keep Qwen3-0.6B as the modern runnable torch_npu benchmark while Qwen3.5 waits for qwen3_5 backend support.",
        "29. Show compatibility report: vLLM-Ascend backend issue is isolated from HF export and torch_npu path.",
    ]
    return "\n".join(lines) + "\n"


def write_demo_progress_report(demo_root: Path) -> Path:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    model_inventory = read_json(artifacts / "model_inventory.json")
    inference = read_json(artifacts / "ascend_inference_tiny_qwen3_torch_npu.json")
    sync = read_json(artifacts / "multicard_sync_summary.json")
    lora = read_json(artifacts / "rankadaptor_lora_sync_summary.json")
    qwen_lora = read_json(artifacts / "tiny_qwen_lora_finetune_tiny_qwen3_lora_npu.json")
    qwen_bslora = read_json(artifacts / "tiny_qwen_bslora_finetune_tiny_qwen3_bslora_npu.json")
    multicard_bslora = first_existing_json(
        artifacts,
        [
            "multicard_tiny_qwen_bslora_finetune_tiny_qwen3_bslora_sync_npu.json",
            "multicard_tiny_qwen_bslora_finetune_cpu_test.json",
        ],
    )
    qwen_compression = first_existing_json(
        artifacts,
        [
            "tiny_qwen_compression_tiny_qwen3_compression_npu.json",
            "tiny_qwen_compression_cpu_test.json",
        ],
    )
    qwen_generate = first_existing_json(
        artifacts,
        [
            "tiny_qwen_compression_generate_tiny_qwen3_generate_serving_export_npu.json",
            "tiny_qwen_compression_generate_tiny_qwen3_generate_cache_npu.json",
        ],
    )
    multicard_generate = first_existing_json(
        artifacts,
        [
            "multicard_tiny_qwen_generate_tiny_qwen3_multicard_generate_npu.json",
            "multicard_tiny_qwen_generate_cpu_test.json",
        ],
    )
    multicard_qwen_compression_generate = best_multicard_qwen_compression_generate(artifacts) or first_existing_json(
        artifacts,
        [
            "multicard_qwen_compression_generate_cpu_test.json",
        ],
    )
    vllm_probe = read_json(artifacts / "vllm_serving_probe_tiny_qwen3_serving_export_npu.json")
    vllm_patch_probe = read_json(artifacts / "vllm_serving_probe_tiny_qwen3_serving_export_npu_preload_patch.json")
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
    torch_serving_fallback = read_json(artifacts / "torch_serving_fallback_tiny_qwen3_serving_fallback_npu.json")
    multicard_parallel_suite = first_existing_json(
        artifacts,
        [
            "multicard_parallel_suite_demo_parallel_suite_npu.json",
            "multicard_parallel_suite_cpu_test.json",
        ],
    )
    vllm_parallel_suite = best_vllm_parallel_suite(artifacts)
    qwen35_snapshot = read_json(artifacts / "model_snapshot_qwen35_08b.json")
    qwen35_inference = read_json(artifacts / "ascend_inference_qwen35_08b_torch_npu.json")
    qwen3_snapshot = read_json(artifacts / "model_snapshot_qwen3_06b.json")
    qwen3_inference = read_json(artifacts / "ascend_inference_qwen3_06b_torch_npu.json")
    multicard_qwen_inference = first_existing_json(
        artifacts,
        [
            "multicard_qwen_inference_qwen3_06b_npu.json",
            "multicard_qwen_inference_cpu_test.json",
        ],
    )
    qwen_compression_quality = first_existing_json(
        artifacts,
        [
            "qwen_compression_quality_qwen3_06b_quality_pattern_npu.json",
            "qwen_compression_quality_qwen3_06b_quality_npu.json",
            "qwen_compression_quality_cpu_test.json",
        ],
    )
    multicard_qwen_compression_quality = first_existing_json(
        artifacts,
        [
            "multicard_qwen_compression_quality_qwen3_06b_quality_pattern_2card_npu.json",
            "multicard_qwen_compression_quality_qwen3_06b_quality_sync_npu.json",
            "multicard_qwen_compression_quality_cpu_test.json",
        ],
    )
    qwen_compression_generate = first_existing_json(
        artifacts,
        [
            "qwen_compression_generate_qwen3_06b_generate_npu.json",
            "qwen_compression_generate_cpu_test.json",
        ],
    )
    multicard_qwen_lora_finetune = first_existing_json(
        artifacts,
        [
            "multicard_qwen_lora_finetune_qwen3_06b_lora_sync_npu.json",
            "multicard_qwen_lora_finetune_cpu_test.json",
        ],
    )
    compatibility = read_json(artifacts / "compatibility_issues.json")
    compression_memory_report = read_json(artifacts / "compression_memory_report.json")
    paper_baseline_coverage_audit = read_json(artifacts / "paper_baseline_coverage_audit.json")
    inference_bottleneck_report = read_json(artifacts / "inference_bottleneck_report.json")
    inference_acceleration_summary = read_json(artifacts / "inference_acceleration_summary.json")
    compressed_native_sweep_summary = read_json(artifacts / "compressed_native_sweep_summary.json")
    qpruner_packed_decode = read_json(artifacts / QPRUNER_PACKED_DECODE_ARTIFACT)
    multicard_compression_sync_summary = read_json(artifacts / "multicard_compression_sync_summary.json")
    multicard_qwen_qpruner_grouped_replay = best_multicard_qwen_qpruner_grouped_replay(artifacts)
    qwen_grouped_mlp_sweep = best_qwen_grouped_mlp_sweep(artifacts)
    qwen_grouped_mlp_memory_tradeoff = best_qwen_grouped_mlp_memory_tradeoff(artifacts)
    npu_utilization_monitor = best_npu_utilization_monitor(artifacts)
    objective_coverage_audit = read_json(artifacts / "objective_coverage_audit.json")
    choice_accuracy = (
        read_json(artifacts / FULLTARGET_QPRUNER_CHOICE_ARTIFACT)
        or read_json(artifacts / MULTITASK_CHOICE_ACCURACY_ARTIFACT)
        or read_json(artifacts / CHOICE_ACCURACY_ARTIFACT)
    )
    output = reports / "ascend-910b-demo-progress.md"
    output.write_text(
        markdown(
            demo_root,
            model_inventory,
            inference,
            sync,
            lora,
            qwen_lora,
            qwen_bslora,
            multicard_bslora,
            qwen_compression,
            qwen_generate,
            multicard_generate,
            multicard_qwen_compression_generate,
            torch_serving_fallback,
            multicard_parallel_suite,
            vllm_parallel_suite,
            vllm_probe,
            vllm_patch_probe,
            vllm_selector_shim_probe,
            vllm_serving_benchmark,
            qwen35_snapshot,
            qwen35_inference,
            qwen3_snapshot,
            qwen3_inference,
            multicard_qwen_inference,
            qwen_compression_quality,
            multicard_qwen_compression_quality,
            qwen_compression_generate,
            multicard_qwen_lora_finetune,
            compatibility,
            compression_memory_report,
            paper_baseline_coverage_audit,
            inference_bottleneck_report,
            inference_acceleration_summary,
            compressed_native_sweep_summary,
            qpruner_packed_decode,
            multicard_compression_sync_summary,
            multicard_qwen_qpruner_grouped_replay,
            qwen_grouped_mlp_sweep,
            qwen_grouped_mlp_memory_tradeoff,
            npu_utilization_monitor,
            objective_coverage_audit,
            choice_accuracy,
        )
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Write Ascend 910B demo progress report")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    args = parser.parse_args()
    output = write_demo_progress_report(Path(args.demo_root))
    print(f"DEMO_PROGRESS_REPORT {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
