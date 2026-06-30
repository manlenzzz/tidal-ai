#!/usr/bin/env python
"""Write a memory-first report for CAP/QPruner serving exports."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any, Sequence


METHODS = ("baseline", "cap", "qpruner")
QUALITY_ARTIFACT = "qwen_compression_quality_qwen3_06b_quality_npu.json"
GENERATE_ARTIFACT = "qwen_compression_generate_qwen3_06b_generate_npu.json"
QPRUNER_SCALE_QUALITY_ARTIFACT = "qwen_qpruner_scale_quality_summary.json"
VLLM_LONG_DECODE_ARTIFACT = "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu_long_decode.json"
WANDA_QUALITY_ARTIFACT_CANDIDATES = (
    QUALITY_ARTIFACT,
    "qwen_compression_quality_qwen3_06b_quality_pattern_npu.json",
    "multicard_qwen_compression_quality_qwen3_06b_quality_pattern_2card_npu.json",
    "multicard_qwen_compression_quality_qwen3_06b_quality_sync_npu.json",
)
SPARSEGPT_QUALITY_ARTIFACT_CANDIDATES = (
    "multicard_qwen_compression_quality_qwen3_06b_quality_sparsegpt_2card_npu.json",
    QUALITY_ARTIFACT,
    "qwen_compression_quality_qwen3_06b_quality_sparsegpt_npu.json",
    "qwen_compression_quality_qwen3_06b_quality_pattern_npu.json",
)
LLM_PRUNER_BASELINE_ARTIFACT_CANDIDATES = (
    "qwen_llm_pruner_baseline_qwen3_06b_llm_pruner_npu.json",
)


PAPER_BASELINES: dict[str, dict[str, Any]] = {
    "qpruner": {
        "primary": ["LLM-Pruner"],
        "variants": ["QPruner1 uniform quantization", "QPruner2 mutual-information mixed precision", "QPruner3 Bayesian-refined mixed precision"],
        "recovery_baselines": ["LoRA", "LoftQ"],
        "paper_metric_focus": ["zero-shot accuracy", "peak memory", "pruning rate"],
    },
    "cap": {
        "pruning_baselines": [
            "SparseGPT",
            "Wanda",
            "DSNoT",
            "OATS",
            "OWL",
            "AlphaPruning",
        ],
        "joint_compression_baselines": ["SLiM", "JSQ", "L2QER", "LPAF"],
        "svd_structured_baselines": ["SVD-LLM v2", "Dobi-SVD", "Basis Sharing", "LoSparse"],
        "paper_metric_focus": ["perplexity", "zero-shot accuracy", "throughput", "memory footprint"],
    },
    "rankadaptor": {
        "pruning_stage_baselines": ["LLM-Pruner", "Shortened LLaMA"],
        "recovery_baselines": ["LoRA", "AdaLoRA", "without recovery"],
        "paper_metric_focus": ["zero-shot accuracy", "generation quality", "trainable adapter budget"],
    },
}


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def file_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def export_summary(demo_root: Path, export_run_label: str, method: str, baseline_bytes: int | None) -> dict[str, Any]:
    path = demo_root / "serving_exports" / export_run_label / method
    config = read_json(path / "config.json")
    model_bytes = file_size_bytes(path / "model.safetensors")
    if model_bytes == 0:
        model_bytes = file_size_bytes(path)
    ratio = round(model_bytes / baseline_bytes, 3) if baseline_bytes else None
    return {
        "path": str(path),
        "exists": path.exists(),
        "format": "dense_hf" if (path / "config.json").exists() and (path / "model.safetensors").exists() else "unknown",
        "model_bytes": model_bytes,
        "size_ratio_vs_engineering_reference": ratio,
        "size_ratio_vs_baseline": ratio,
        "architecture": (config or {}).get("architectures"),
        "model_type": (config or {}).get("model_type"),
        "torch_dtype": (config or {}).get("torch_dtype"),
    }


def engineering_reference_summary(export_run_label: str, baseline_bytes: int) -> dict[str, Any]:
    return {
        "serving_baseline_role": "uncompressed_serving_export",
        "artifact_method_key": "baseline",
        "export_run_label": export_run_label,
        "model_bytes": baseline_bytes,
        "not_paper_baseline": True,
        "description": (
            "The `baseline` method key is only the uncompressed serving export used for "
            "Ascend runtime sanity checks and speed/memory ratios."
        ),
    }


def baseline_alignment_matrix() -> list[dict[str, Any]]:
    return [
        {
            "method": "QPruner",
            "paper_baseline_role": "pruning baseline",
            "paper_baselines": PAPER_BASELINES["qpruner"]["primary"],
            "demo_reference_role": "compressed-vs-uncompressed Ascend runtime reference",
            "demo_artifacts": [
                f"artifacts/{QUALITY_ARTIFACT}",
                f"artifacts/{GENERATE_ARTIFACT}",
                "artifacts/compressed_native_sweep_summary.json",
            ],
            "current_status": "NPU algorithm evidence exists; paper-baseline rerun is not claimed.",
        },
        {
            "method": "CAP",
            "paper_baseline_role": "pruning / joint-compression / SVD baselines",
            "paper_baselines": (
                PAPER_BASELINES["cap"]["pruning_baselines"]
                + PAPER_BASELINES["cap"]["joint_compression_baselines"]
                + PAPER_BASELINES["cap"]["svd_structured_baselines"]
            ),
            "demo_reference_role": "compressed-vs-uncompressed Ascend runtime reference",
            "demo_artifacts": [
                f"artifacts/{QUALITY_ARTIFACT}",
                f"artifacts/{GENERATE_ARTIFACT}",
                "artifacts/compressed_native_sweep_summary.json",
            ],
            "current_status": (
                "Wanda paper baseline has an Ascend NPU quality artifact; "
                "SparseGPT/DSNoT/OATS/OWL/AlphaPruning are not claimed as rerun."
            ),
        },
        {
            "method": "RankAdaptor",
            "paper_baseline_role": "recovery baselines",
            "paper_baselines": PAPER_BASELINES["rankadaptor"]["recovery_baselines"],
            "demo_reference_role": "LoRA/BSLoRA recovery evidence on Ascend",
            "demo_artifacts": [
                "artifacts/multicard_qwen_lora_finetune_qwen3_06b_lora_sync_npu.json",
                "artifacts/tiny_qwen_bslora_finetune_qwen3_06b_bslora_npu.json",
                "artifacts/multicard_tiny_qwen_bslora_finetune_tiny_qwen3_bslora_sync_npu.json",
            ],
            "current_status": "NPU recovery evidence exists; full paper-baseline rerun remains future work.",
        },
        {
            "method": "Engineering runtime baseline",
            "paper_baseline_role": "not a paper baseline",
            "paper_baselines": ["uncompressed serving export"],
            "demo_reference_role": "runtime sanity, latency, throughput, and memory reference",
            "demo_artifacts": [
                "artifacts/inference_acceleration_summary.json",
                "artifacts/compressed_native_sweep_summary.json",
                "artifacts/torch_serving_fallback_tiny_qwen3_serving_fallback_npu.json",
            ],
            "current_status": "Used only for runtime ratios and not claimed as a paper baseline.",
        },
    ]


def wanda_evidence_from_quality(payload: dict[str, Any] | None, artifact_name: str) -> dict[str, Any] | None:
    if not isinstance(payload, dict) or payload.get("status") != "PASS":
        return None
    wanda = payload.get("wanda", {}) if isinstance(payload.get("wanda"), dict) else {}
    if wanda.get("status") == "PASS":
        return {
            "artifacts": [f"artifacts/{artifact_name}"],
            "metrics": {
                "loss_delta": wanda.get("loss_delta"),
                "targeted_param_reduction_pct": wanda.get("targeted_param_reduction_pct"),
                "latency_speedup": wanda.get("latency_speedup"),
                "targeted_layers": wanda.get("targeted_layers"),
            },
        }
    aggregate = payload.get("aggregate", {}) if isinstance(payload.get("aggregate"), dict) else {}
    ranks = payload.get("ranks", []) if isinstance(payload.get("ranks"), list) else []
    rank_wanda: dict[str, Any] = {}
    for row in ranks:
        if not isinstance(row, dict):
            continue
        benchmark = row.get("benchmark", {}) if isinstance(row.get("benchmark"), dict) else {}
        candidate = benchmark.get("wanda", {}) if isinstance(benchmark.get("wanda"), dict) else {}
        if candidate.get("status") == "PASS":
            rank_wanda = candidate
            break
    if aggregate.get("wanda_loss_delta_avg") is not None or aggregate.get("wanda_tokens_per_s_total") is not None:
        return {
            "artifacts": [f"artifacts/{artifact_name}"],
            "metrics": {
                "loss_delta": aggregate.get("wanda_loss_delta_avg"),
                "targeted_param_reduction_pct": rank_wanda.get("targeted_param_reduction_pct"),
                "tokens_per_s_total": aggregate.get("wanda_tokens_per_s_total"),
                "targeted_layers": rank_wanda.get("targeted_layers") or payload.get("target_layer_limit"),
                "world_size": payload.get("world_size"),
                "distributed_reduce_consistent": aggregate.get("distributed_reduce_consistent"),
            },
        }
    return None


def resolve_wanda_evidence(artifacts: Path, quality: dict[str, Any] | None) -> dict[str, Any] | None:
    for artifact_name in WANDA_QUALITY_ARTIFACT_CANDIDATES:
        payload = quality if artifact_name == QUALITY_ARTIFACT else read_json(artifacts / artifact_name)
        evidence = wanda_evidence_from_quality(payload, artifact_name)
        if evidence is not None:
            return evidence
    return None


def sparsegpt_evidence_from_quality(payload: dict[str, Any] | None, artifact_name: str) -> dict[str, Any] | None:
    if not isinstance(payload, dict) or payload.get("status") != "PASS":
        return None
    sparsegpt = payload.get("sparsegpt", {}) if isinstance(payload.get("sparsegpt"), dict) else {}
    if sparsegpt.get("status") == "PASS":
        return {
            "artifacts": [f"artifacts/{artifact_name}"],
            "metrics": {
                "sparsegpt": {
                    "loss_delta": sparsegpt.get("loss_delta"),
                    "targeted_param_reduction_pct": sparsegpt.get("targeted_param_reduction_pct"),
                    "latency_speedup": sparsegpt.get("latency_speedup"),
                    "targeted_layers": sparsegpt.get("targeted_layers"),
                }
            },
        }
    aggregate = payload.get("aggregate", {}) if isinstance(payload.get("aggregate"), dict) else {}
    ranks = payload.get("ranks", []) if isinstance(payload.get("ranks"), list) else []
    rank_sparsegpt: dict[str, Any] = {}
    for row in ranks:
        if not isinstance(row, dict):
            continue
        benchmark = row.get("benchmark", {}) if isinstance(row.get("benchmark"), dict) else {}
        candidate = benchmark.get("sparsegpt", {}) if isinstance(benchmark.get("sparsegpt"), dict) else {}
        if candidate.get("status") == "PASS":
            rank_sparsegpt = candidate
            break
    if aggregate.get("sparsegpt_loss_delta_avg") is None and aggregate.get("sparsegpt_tokens_per_s_total") is None:
        return None
    return {
        "artifacts": [f"artifacts/{artifact_name}"],
        "metrics": {
            "sparsegpt": {
                "loss_delta": aggregate.get("sparsegpt_loss_delta_avg"),
                "targeted_param_reduction_pct": rank_sparsegpt.get("targeted_param_reduction_pct"),
                "tokens_per_s_total": aggregate.get("sparsegpt_tokens_per_s_total"),
                "targeted_layers": rank_sparsegpt.get("targeted_layers") or payload.get("target_layer_limit"),
                "world_size": payload.get("world_size"),
                "distributed_reduce_consistent": aggregate.get("distributed_reduce_consistent"),
            }
        },
    }


def resolve_sparsegpt_evidence(artifacts: Path, quality: dict[str, Any] | None) -> dict[str, Any] | None:
    for artifact_name in SPARSEGPT_QUALITY_ARTIFACT_CANDIDATES:
        payload = quality if artifact_name == QUALITY_ARTIFACT else read_json(artifacts / artifact_name)
        evidence = sparsegpt_evidence_from_quality(payload, artifact_name)
        if evidence is not None:
            return evidence
    return None


def llm_pruner_evidence_from_quality(payload: dict[str, Any] | None, artifact_name: str) -> dict[str, Any] | None:
    if not isinstance(payload, dict) or payload.get("status") != "PASS":
        return None
    if payload.get("paper_baseline") != "LLM-Pruner" and payload.get("baseline_family") != "LLM-Pruner":
        return None
    llm_pruner = payload.get("llm_pruner", {}) if isinstance(payload.get("llm_pruner"), dict) else {}
    if llm_pruner.get("status") != "PASS":
        return None
    metrics = {
        "loss_delta": llm_pruner.get("loss_delta"),
        "targeted_param_reduction_pct": llm_pruner.get("targeted_param_reduction_pct"),
        "latency_speedup": llm_pruner.get("latency_speedup"),
        "tokens_per_s": llm_pruner.get("tokens_per_s"),
        "targeted_layers": llm_pruner.get("targeted_layers"),
        "compression_time_s": llm_pruner.get("compression_time_s"),
        "claim_scope": payload.get("claim_scope"),
    }
    return {
        "artifacts": [f"artifacts/{artifact_name}"],
        "metrics": {key: value for key, value in metrics.items() if value is not None},
    }


def resolve_llm_pruner_evidence(artifacts: Path) -> dict[str, Any] | None:
    for artifact_name in LLM_PRUNER_BASELINE_ARTIFACT_CANDIDATES:
        evidence = llm_pruner_evidence_from_quality(read_json(artifacts / artifact_name), artifact_name)
        if evidence is not None:
            return evidence
    return None


def paper_baseline_evidence(artifacts: Path, quality: dict[str, Any] | None) -> list[dict[str, Any]]:
    wanda_evidence = resolve_wanda_evidence(artifacts, quality)
    wanda_metrics = wanda_evidence.get("metrics", {}) if wanda_evidence else {}
    wanda_artifacts = wanda_evidence.get("artifacts", []) if wanda_evidence else []
    wanda_metrics = {
        key: value for key, value in wanda_metrics.items() if value is not None
    }
    sparsegpt_evidence = resolve_sparsegpt_evidence(artifacts, quality)
    sparsegpt_metrics = sparsegpt_evidence.get("metrics", {}) if sparsegpt_evidence else {}
    sparsegpt_artifacts = sparsegpt_evidence.get("artifacts", []) if sparsegpt_evidence else []
    sparsegpt_metrics = {
        method: {key: value for key, value in payload.items() if value is not None}
        for method, payload in sparsegpt_metrics.items()
        if isinstance(payload, dict)
    }
    llm_pruner_evidence = resolve_llm_pruner_evidence(artifacts)
    llm_pruner_metrics = llm_pruner_evidence.get("metrics", {}) if llm_pruner_evidence else {}
    llm_pruner_artifacts = llm_pruner_evidence.get("artifacts", []) if llm_pruner_evidence else []
    return [
        {
            "method": "CAP",
            "paper_baseline": "Wanda",
            "status": "RUN_ON_ASCEND" if wanda_evidence else "MISSING",
            "evidence_role": "actual paper pruning baseline",
            "artifacts": wanda_artifacts,
            "metrics": wanda_metrics,
            "next_action": (
                "Use this as the first paper-baseline comparison point; "
                "add SparseGPT/DSNoT/OATS/OWL/AlphaPruning when ported."
            ),
        },
        {
            "method": "CAP",
            "paper_baseline": "SparseGPT / DSNoT / OATS / OWL / AlphaPruning",
            "status": "PARTIAL_RUN_ON_ASCEND" if sparsegpt_evidence else "PENDING_NOT_RUN_ON_ASCEND",
            "evidence_role": "paper pruning baselines",
            "artifacts": sparsegpt_artifacts,
            "metrics": sparsegpt_metrics,
            "pending_baselines": ["DSNoT", "OATS", "OWL", "AlphaPruning"] if sparsegpt_evidence else [
                "SparseGPT",
                "DSNoT",
                "OATS",
                "OWL",
                "AlphaPruning",
            ],
            "next_action": "Port and run the remaining CAP paper pruning baselines on the same Qwen3 target-layer set before claiming full paper reproduction.",
        },
        {
            "method": "QPruner",
            "paper_baseline": "LLM-Pruner",
            "status": "PARTIAL_RUN_ON_ASCEND" if llm_pruner_evidence else "PENDING_NOT_RUN_ON_ASCEND",
            "evidence_role": (
                "LLM-Pruner-style structural pruning paper-baseline compatibility point"
                if llm_pruner_evidence
                else "paper pruning baseline not yet ported in this demo"
            ),
            "artifacts": llm_pruner_artifacts,
            "metrics": llm_pruner_metrics,
            "next_action": (
                "Replace the lightweight structural baseline with an official full LLM-Pruner port before claiming full QPruner paper-baseline reproduction."
                if llm_pruner_evidence
                else "Port LLM-Pruner on the same Ascend Qwen3 target-layer set before claiming full QPruner paper-baseline reproduction."
            ),
        },
    ]


def qpruner_scale_quality_memory(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("status") != "PASS":
        return {}
    best = payload.get("best_under_loss_delta", {})
    if not isinstance(best, dict) or not best:
        return {}
    memory_reduction = as_float(best.get("memory_reduction_pct") or best.get("targeted_memory_reduction_pct"))
    target_limit = as_float(best.get("target_layer_limit") or payload.get("max_target_layer_limit"))
    target_total = as_float(best.get("targeted_layers_total") or payload.get("targeted_layers_total"))
    coverage = as_float(best.get("coverage_pct") or payload.get("max_coverage_pct"))
    result: dict[str, Any] = {
        "scale_quality_artifact": QPRUNER_SCALE_QUALITY_ARTIFACT,
        "scale_quality_run_label": best.get("run_label"),
        "scale_quality_readout": payload.get("readout"),
    }
    if memory_reduction is not None:
        result["targeted_param_reduction_pct"] = memory_reduction
    average_bits = as_float(best.get("average_bits"))
    if average_bits is not None:
        result["average_bits"] = average_bits
    loss_delta = as_float(best.get("loss_delta"))
    if loss_delta is not None:
        result["loss_delta"] = loss_delta
    if target_limit is not None:
        result["target_layer_limit"] = int(target_limit)
    if target_total is not None:
        result["targeted_layers_total"] = int(target_total)
    if coverage is not None:
        result["target_layer_coverage_pct"] = round(coverage, 3)
    return result


def compression_memory_summary(
    quality: dict[str, Any] | None,
    generate: dict[str, Any] | None,
    qpruner_scale_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    quality = quality or {}
    generate = generate or {}
    target_limit = as_float(quality.get("target_layer_limit") or generate.get("target_layer_limit"))
    target_total = as_float(quality.get("targeted_layers_total") or generate.get("targeted_layers_total"))
    coverage = round(target_limit / target_total * 100, 2) if target_limit is not None and target_total else None
    cap = quality.get("cap", {}) if isinstance(quality.get("cap"), dict) else {}
    cap_generate = generate.get("cap", {}) if isinstance(generate.get("cap"), dict) else {}
    qpruner = quality.get("qpruner", {}) if isinstance(quality.get("qpruner"), dict) else {}
    qpruner_generate = generate.get("qpruner", {}) if isinstance(generate.get("qpruner"), dict) else {}
    cap_ratio = as_float(cap.get("targeted_compression_ratio") or cap_generate.get("targeted_compression_ratio"))
    q_bits = as_float(qpruner.get("average_bits") or qpruner_generate.get("average_bits"))
    cap_reduction = as_float(cap_generate.get("targeted_param_reduction_pct"))
    if cap_reduction is None and cap_ratio:
        cap_reduction = round((1.0 - 1.0 / cap_ratio) * 100.0, 3)
    q_reduction = as_float(qpruner_generate.get("targeted_param_reduction_pct"))
    if q_reduction is None and q_bits:
        q_reduction = round((1.0 - q_bits / 16.0) * 100.0, 3)
    summary = {
        "target_layer_limit": int(target_limit) if target_limit is not None else None,
        "targeted_layers_total": int(target_total) if target_total is not None else None,
        "target_layer_coverage_pct": coverage,
        "cap": {
            "targeted_compression_ratio": cap_ratio,
            "targeted_param_reduction_pct": cap_reduction,
            "loss_delta": cap.get("loss_delta"),
            "peak_mem_mb": cap_generate.get("peak_mem_mb"),
        },
        "qpruner": {
            "average_bits": q_bits,
            "targeted_param_reduction_pct": q_reduction,
            "loss_delta": qpruner.get("loss_delta"),
            "peak_mem_mb": qpruner_generate.get("peak_mem_mb"),
        },
    }
    qpruner_scale = qpruner_scale_quality_memory(qpruner_scale_quality)
    if qpruner_scale:
        qpruner_summary = summary["qpruner"]
        for key in (
            "targeted_param_reduction_pct",
            "average_bits",
            "loss_delta",
            "scale_quality_artifact",
            "scale_quality_run_label",
            "scale_quality_readout",
        ):
            if key in qpruner_scale:
                qpruner_summary[key] = qpruner_scale[key]
        if "target_layer_limit" in qpruner_scale:
            summary["target_layer_limit"] = qpruner_scale["target_layer_limit"]
        if "targeted_layers_total" in qpruner_scale:
            summary["targeted_layers_total"] = qpruner_scale["targeted_layers_total"]
        if "target_layer_coverage_pct" in qpruner_scale:
            summary["target_layer_coverage_pct"] = qpruner_scale["target_layer_coverage_pct"]
    return summary


def benchmark_memory_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    exports = payload.get("exports", {}) if isinstance(payload, dict) and isinstance(payload.get("exports"), dict) else {}
    summary = payload.get("summary", {}) if isinstance(payload, dict) and isinstance(payload.get("summary"), dict) else {}
    return {
        "artifact_status": payload.get("status") if payload else "MISSING",
        "backend": payload.get("backend") if payload else None,
        "max_new_tokens": payload.get("max_new_tokens") if payload else None,
        "speedups": {
            "cap_vs_baseline": summary.get("cap_vs_baseline_speedup"),
            "qpruner_vs_baseline": summary.get("qpruner_vs_baseline_speedup"),
        },
        "peak_mem_mb": {
            method: exports.get(method, {}).get("peak_mem_mb") if isinstance(exports.get(method), dict) else None
            for method in METHODS
        },
        "tokens_per_s": {
            method: exports.get(method, {}).get("tokens_per_s") if isinstance(exports.get(method), dict) else None
            for method in METHODS
        },
    }


def memory_first_summary(report: dict[str, Any]) -> dict[str, str]:
    memory = report.get("compression_memory", {}) if isinstance(report.get("compression_memory"), dict) else {}
    cap_memory = memory.get("cap", {}) if isinstance(memory.get("cap"), dict) else {}
    q_memory = memory.get("qpruner", {}) if isinstance(memory.get("qpruner"), dict) else {}
    exports = report.get("serving_export_memory", {}) if isinstance(report.get("serving_export_memory"), dict) else {}
    cap_export = exports.get("cap", {}) if isinstance(exports.get("cap"), dict) else {}
    q_export = exports.get("qpruner", {}) if isinstance(exports.get("qpruner"), dict) else {}
    cap_reduction = fmt(cap_memory.get("targeted_param_reduction_pct"))
    q_reduction = fmt(q_memory.get("targeted_param_reduction_pct"))
    target_layers = "{limit}/{total}".format(
        limit=fmt(memory.get("target_layer_limit")),
        total=fmt(memory.get("targeted_layers_total")),
    )
    cap_ratio = fmt(cap_export.get("size_ratio_vs_baseline"))
    q_ratio = fmt(q_export.get("size_ratio_vs_baseline"))
    next_action = str(report.get("next_action", "missing"))
    return {
        "headline": (
            "Compression demo is memory-first: CAP saves {cap}% targeted memory and QPruner saves "
            "{q}% targeted memory across {layers} Qwen3 target layers."
        ).format(cap=cap_reduction, q=q_reduction, layers=target_layers),
        "serving_gap": (
            "Current dense HF serving export is {cap_ratio}x/{q_ratio}x of the uncompressed reference, "
            "so the video should show compressed-native/runtime-preserved memory savings before claiming "
            "serving memory savings."
        ).format(cap_ratio=cap_ratio, q_ratio=q_ratio),
        "next_action": next_action,
    }


def memory_first_video_readout(report: dict[str, Any]) -> str:
    memory = report.get("compression_memory", {}) if isinstance(report.get("compression_memory"), dict) else {}
    cap_memory = memory.get("cap", {}) if isinstance(memory.get("cap"), dict) else {}
    q_memory = memory.get("qpruner", {}) if isinstance(memory.get("qpruner"), dict) else {}
    exports = report.get("serving_export_memory", {}) if isinstance(report.get("serving_export_memory"), dict) else {}
    cap_export = exports.get("cap", {}) if isinstance(exports.get("cap"), dict) else {}
    q_export = exports.get("qpruner", {}) if isinstance(exports.get("qpruner"), dict) else {}
    return (
        "Memory-first compression readout: CAP targeted memory -{cap}%, QPruner targeted memory -{q}%, "
        "target layers {limit}/{total}; dense HF export ratio CAP/QPruner {cap_ratio}x/{q_ratio}x, "
        "so memory savings require the compressed-native runtime path."
    ).format(
        cap=fmt(cap_memory.get("targeted_param_reduction_pct")),
        q=fmt(q_memory.get("targeted_param_reduction_pct")),
        limit=fmt(memory.get("target_layer_limit")),
        total=fmt(memory.get("targeted_layers_total")),
        cap_ratio=fmt(cap_export.get("size_ratio_vs_baseline")),
        q_ratio=fmt(q_export.get("size_ratio_vs_baseline")),
    )


def finding(identifier: str, title: str, evidence: str, recommendation: str) -> dict[str, str]:
    return {"id": identifier, "title": title, "evidence": evidence, "recommendation": recommendation}


def build_findings(report: dict[str, Any]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    memory = report["compression_memory"]
    gap = report["memory_gap"]
    cap_reduction = as_float(memory.get("cap", {}).get("targeted_param_reduction_pct"))
    q_reduction = as_float(memory.get("qpruner", {}).get("targeted_param_reduction_pct"))
    if cap_reduction or q_reduction:
        findings.append(
            finding(
                "compression_memory_savings_exist",
                "Compression has real targeted memory savings",
                "CAP targeted memory reduction {cap}% and QPruner targeted memory reduction {q}% over "
                "{limit}/{total} targeted layers.".format(
                    cap=fmt(cap_reduction),
                    q=fmt(q_reduction),
                    limit=fmt(memory.get("target_layer_limit")),
                    total=fmt(memory.get("targeted_layers_total")),
                ),
                "Lead the demo with memory reduction, not only latency.",
            )
        )
    if gap.get("dense_export_erases_storage_savings"):
        findings.append(
            finding(
                "dense_serving_export_erases_memory_savings",
                "Dense serving export erases compressed storage savings",
                "Serving export is dense HF and CAP/QPruner model file size ratios are {cap}x/{q}x vs baseline.".format(
                    cap=fmt(report["serving_export_memory"].get("cap", {}).get("size_ratio_vs_baseline")),
                    q=fmt(report["serving_export_memory"].get("qpruner", {}).get("size_ratio_vs_baseline")),
                ),
                "Keep pruned/quantized weights in the serving runtime instead of materializing dense HF weights.",
            )
        )
    peak_values = report.get("vllm_long_decode", {}).get("peak_mem_mb", {})
    if not any(value for value in peak_values.values()):
        findings.append(
            finding(
                "benchmark_peak_memory_not_yet_proving_savings",
                "Current serving benchmark peak memory does not prove savings",
                "vLLM peak_mem_mb is {values}; dense export makes storage savings invisible to serving memory metrics.".format(
                    values=", ".join(f"{method}={fmt(value)}" for method, value in peak_values.items())
                ),
                "Add HBM/process-memory sampling around compressed-native serving once the runtime preserves compressed weights.",
            )
        )
    return findings


def build_report(demo_root: Path, export_run_label: str) -> dict[str, Any]:
    artifacts = demo_root / "artifacts"
    quality = read_json(artifacts / QUALITY_ARTIFACT)
    generate = read_json(artifacts / GENERATE_ARTIFACT)
    qpruner_scale_quality = read_json(artifacts / QPRUNER_SCALE_QUALITY_ARTIFACT)
    vllm = read_json(artifacts / VLLM_LONG_DECODE_ARTIFACT)
    baseline_bytes = file_size_bytes(demo_root / "serving_exports" / export_run_label / "baseline" / "model.safetensors")
    if baseline_bytes == 0:
        baseline_bytes = file_size_bytes(demo_root / "serving_exports" / export_run_label / "baseline")
    exports = {
        method: export_summary(demo_root, export_run_label, method, baseline_bytes if baseline_bytes else None)
        for method in METHODS
    }
    baseline_arch = exports["baseline"].get("architecture")
    for method in ("cap", "qpruner"):
        exports[method]["same_architecture_as_baseline"] = exports[method].get("architecture") == baseline_arch
    cap_ratio = as_float(exports["cap"].get("size_ratio_vs_baseline"))
    q_ratio = as_float(exports["qpruner"].get("size_ratio_vs_baseline"))
    generate = generate or {}
    report = {
        "status": "UNKNOWN",
        "demo_root": str(demo_root),
        "platform": platform.platform(),
        "export_run_label": export_run_label,
        "paper_baselines": PAPER_BASELINES,
        "baseline_alignment_matrix": baseline_alignment_matrix(),
        "paper_baseline_evidence": paper_baseline_evidence(artifacts, quality),
        "engineering_reference": engineering_reference_summary(export_run_label, baseline_bytes),
        "compression_memory": compression_memory_summary(quality, generate, qpruner_scale_quality),
        "serving_export_memory": exports,
        "vllm_long_decode": benchmark_memory_summary(vllm),
        "memory_gap": {
            "serving_dense_export": bool(generate.get("serving_dense_export")),
            "inference_cache_enabled": bool(generate.get("inference_cache_enabled")),
            "dense_export_erases_storage_savings": bool(
                generate.get("serving_dense_export")
                and cap_ratio is not None
                and q_ratio is not None
                and cap_ratio >= 0.95
                and q_ratio >= 0.95
            ),
        },
        "next_action": "Keep compression in pruned/quantized form inside serving runtime so memory savings survive inference.",
    }
    report["memory_first_summary"] = memory_first_summary(report)
    report["video_readout"] = memory_first_video_readout(report)
    report["findings"] = build_findings(report)
    report["status"] = "ACTIONABLE" if report["findings"] else "INSUFFICIENT_EVIDENCE"
    return report


def markdown(report: dict[str, Any]) -> str:
    memory = report["compression_memory"]
    exports = report["serving_export_memory"]
    vllm = report["vllm_long_decode"]
    paper = report.get("paper_baselines", {})
    alignment = report.get("baseline_alignment_matrix", [])
    memory_first = report.get("memory_first_summary", {}) if isinstance(report.get("memory_first_summary"), dict) else {}
    lines = [
        "# Compression Memory Report",
        "",
        "For compression, memory is the primary compression win. This report separates algorithmic compression, paper baselines, and dense serving export behavior.",
        "",
        "## Memory-First Video Readout",
        "",
        f"- {report.get('video_readout', 'missing')}",
        f"- {memory_first.get('headline', 'missing')}",
        f"- {memory_first.get('serving_gap', 'missing')}",
        f"- Next action: {memory_first.get('next_action', report.get('next_action', 'missing'))}",
        "",
        "## Paper Baselines",
        "",
        "- Engineering reference: `baseline` is the uncompressed serving export, not the paper baseline.",
        "- QPruner paper baseline: {primary}; recovery baselines: {recovery}.".format(
            primary=", ".join(paper.get("qpruner", {}).get("primary", [])) or "missing",
            recovery=", ".join(paper.get("qpruner", {}).get("recovery_baselines", [])) or "missing",
        ),
        "- CAP paper baselines: {pruning}; joint compression/SVD baselines: {joint}.".format(
            pruning=", ".join(paper.get("cap", {}).get("pruning_baselines", [])[:2]) or "missing",
            joint=", ".join(
                (paper.get("cap", {}).get("joint_compression_baselines", []) or [])
                + (paper.get("cap", {}).get("svd_structured_baselines", []) or [])
            )
            or "missing",
        ),
        "- RankAdaptor paper baselines: pruning stage {pruning}; recovery {recovery}.".format(
            pruning=", ".join(paper.get("rankadaptor", {}).get("pruning_stage_baselines", [])) or "missing",
            recovery=", ".join(paper.get("rankadaptor", {}).get("recovery_baselines", [])) or "missing",
        ),
        "",
        "## Baseline Alignment Matrix",
        "",
        "| Method | Paper baseline role | Paper baselines | Demo reference role | Current status |",
        "|---|---|---|---|---|",
    ]
    for row in alignment:
        baselines = row.get("paper_baselines", [])
        if isinstance(baselines, list):
            baseline_text = ", ".join(str(item) for item in baselines)
        else:
            baseline_text = str(baselines)
        lines.append(
            "| {method} | {role} | {baselines} | {demo_role} | {status} |".format(
                method=row.get("method", "missing"),
                role=row.get("paper_baseline_role", "missing"),
                baselines=baseline_text or "missing",
                demo_role=row.get("demo_reference_role", "missing"),
                status=row.get("current_status", "missing"),
            )
        )
    lines.extend(
        [
            "",
            "## Paper Baseline Evidence",
            "",
            "| Method | Paper baseline | Status | Artifacts | Metrics |",
            "|---|---|---|---|---|",
        ]
    )
    for row in report.get("paper_baseline_evidence", []):
        if not isinstance(row, dict):
            continue
        metrics = row.get("metrics", {})
        if isinstance(metrics, dict) and metrics:
            has_nested_metrics = any(isinstance(value, dict) for value in metrics.values())
            chunks = []
            for key, value in metrics.items():
                if isinstance(value, dict):
                    chunks.append(
                        "{name}: {metrics}".format(
                            name=key,
                            metrics=", ".join(f"{metric_key}={fmt(metric_value)}" for metric_key, metric_value in value.items()),
                        )
                    )
                else:
                    chunks.append(f"{key}={fmt(value)}")
            pending = row.get("pending_baselines")
            if isinstance(pending, list) and pending:
                chunks.append("pending=" + ", ".join(str(item) for item in pending))
            metric_text = "; ".join(chunks) if has_nested_metrics else ", ".join(chunks)
        else:
            metric_text = "missing"
        lines.append(
            "| {method} | {baseline} | {status} | {artifacts} | {metrics} |".format(
                method=row.get("method", "missing"),
                baseline=row.get("paper_baseline", "missing"),
                status=row.get("status", "missing"),
                artifacts=", ".join(row.get("artifacts", [])) if isinstance(row.get("artifacts"), list) and row.get("artifacts") else "missing",
                metrics=metric_text,
            )
        )
    lines.extend(
        [
            "",
            "| Layer | CAP | QPruner |",
            "|---|---:|---:|",
            "| Targeted memory reduction | {cap}% | {q}% |".format(
                cap=fmt(memory.get("cap", {}).get("targeted_param_reduction_pct")),
                q=fmt(memory.get("qpruner", {}).get("targeted_param_reduction_pct")),
            ),
            "| Quality delta | {cap} | {q} |".format(
                cap=fmt(memory.get("cap", {}).get("loss_delta")),
                q=fmt(memory.get("qpruner", {}).get("loss_delta")),
            ),
            "| Target-layer coverage | {coverage}% | {coverage}% |".format(
                coverage=fmt(memory.get("target_layer_coverage_pct"))
            ),
            "",
            "| Serving export | Format | Model bytes | Size vs engineering reference | Same architecture |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        payload = exports.get(method, {})
        lines.append(
            "| {method} | {format} | {bytes} | {ratio}x | {arch} |".format(
                method=method,
                format=payload.get("format"),
                bytes=payload.get("model_bytes"),
                ratio=fmt(payload.get("size_ratio_vs_baseline")),
                arch=payload.get("same_architecture_as_baseline", True if method == "baseline" else "missing"),
            )
        )
    lines.extend(
        [
            "",
            "## Serving Gap",
            "",
            "- CAP targeted memory reduction {cap}%.".format(
                cap=fmt(memory.get("cap", {}).get("targeted_param_reduction_pct"))
            ),
            "- QPruner targeted memory reduction {q}%.".format(
                q=fmt(memory.get("qpruner", {}).get("targeted_param_reduction_pct"))
            ),
            "- QPruner scale-quality source {source}: {readout}".format(
                source=memory.get("qpruner", {}).get("scale_quality_artifact", "missing"),
                readout=memory.get("qpruner", {}).get("scale_quality_readout", "missing"),
            ),
            "- CAP dense HF export size ratio {cap}x.".format(
                cap=fmt(exports.get("cap", {}).get("size_ratio_vs_baseline"))
            ),
            "- QPruner dense HF export size ratio {q}x.".format(
                q=fmt(exports.get("qpruner", {}).get("size_ratio_vs_baseline"))
            ),
            "- vLLM long-decode speedups: CAP {cap}x, QPruner {q}x at max_new_tokens={tokens}.".format(
                cap=fmt(vllm.get("speedups", {}).get("cap_vs_baseline")),
                q=fmt(vllm.get("speedups", {}).get("qpruner_vs_baseline")),
                tokens=fmt(vllm.get("max_new_tokens")),
            ),
            "",
            "## Findings",
            "",
        ]
    )
    for item in report.get("findings", []):
        lines.extend(
            [
                f"### {item['title']}",
                "",
                f"- Evidence: {item['evidence']}",
                f"- Recommendation: {item['recommendation']}",
                "",
            ]
        )
    lines.extend(["## Next action", "", report["next_action"]])
    return "\n".join(lines) + "\n"


def write_outputs(report: dict[str, Any], demo_root: Path) -> None:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    (artifacts / "compression_memory_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    (reports / "compression-memory-report.md").write_text(markdown(report))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write memory-first compression serving report")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--export-run-label", default="tiny_qwen3_serving_export_npu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(Path(args.demo_root), args.export_run_label)
    write_outputs(report, Path(args.demo_root))
    print("COMPRESSION_MEMORY_REPORT " + json.dumps(report, sort_keys=True))
    return 0 if report.get("status") == "ACTIONABLE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
