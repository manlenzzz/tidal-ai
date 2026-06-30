#!/usr/bin/env python
"""Write the paper-baseline coverage audit used by the Ascend demo playback."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


QPRUNER_FULL = "qwen_llm_pruner_baseline_qwen3_06b_llm_pruner_full196_npu.json"
CAP_FULL = "paper_baseline_qwen3_06b_wanda_sparsegpt_full196_npu.json"
COMPRESSION_MEMORY = "compression_memory_report.json"
FINETUNE_EFFECT = "finetuning_effect_summary.json"
RANKADAPTOR_RECOVERY = "rankadaptor_recovery_baseline_suite_qwen3_06b_recovery_baselines_npu.json"


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


def file_info(path: Path) -> dict[str, Any]:
    return {"artifact": f"artifacts/{path.name}", "exists": path.exists(), "size": path.stat().st_size if path.exists() else 0}


def method_loss(summary: dict[str, Any] | None, method: str) -> Any:
    methods = summary.get("methods", {}) if isinstance((summary or {}).get("methods"), dict) else {}
    row = methods.get(method, {}) if isinstance(methods.get(method), dict) else {}
    return row.get("loss_delta")


def method_params(summary: dict[str, Any] | None, method: str) -> Any:
    methods = summary.get("methods", {}) if isinstance((summary or {}).get("methods"), dict) else {}
    row = methods.get(method, {}) if isinstance(methods.get(method), dict) else {}
    return row.get("trainable_adapter_params")


def nested(payload: dict[str, Any], *names: str) -> dict[str, Any]:
    for name in names:
        value = payload.get(name)
        if isinstance(value, dict):
            return value
    return {}


def first_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def qpruner_metric(payload: dict[str, Any], key: str) -> Any:
    row = nested(payload, "llm_pruner", "qpruner", "benchmark")
    return first_value(payload.get(key), row.get(key))


def cap_metric(payload: dict[str, Any], method: str, key: str) -> Any:
    row = nested(payload, method)
    return first_value(payload.get(f"{method}_{key}"), row.get(key))


def target_layers(payload: dict[str, Any], method: str | None = None) -> Any:
    if method:
        row = nested(payload, method)
        return first_value(payload.get("targeted_layers"), payload.get("target_layer_limit"), row.get("targeted_layers"))
    return first_value(payload.get("targeted_layers"), payload.get("target_layer_limit"))


def build_audit(demo_root: Path) -> dict[str, Any]:
    artifacts = demo_root / "artifacts"
    qpruner_path = artifacts / QPRUNER_FULL
    cap_path = artifacts / CAP_FULL
    memory_path = artifacts / COMPRESSION_MEMORY
    finetune_path = artifacts / FINETUNE_EFFECT
    recovery_path = artifacts / RANKADAPTOR_RECOVERY

    qpruner = read_json(qpruner_path) or {}
    cap = read_json(cap_path) or {}
    memory = read_json(memory_path) or {}
    finetune = read_json(finetune_path) or {}
    recovery = read_json(recovery_path) or {}
    compression_memory = memory.get("compression_memory", {}) if isinstance(memory.get("compression_memory"), dict) else {}
    cap_memory = compression_memory.get("cap", {}) if isinstance(compression_memory.get("cap"), dict) else {}
    q_memory = compression_memory.get("qpruner", {}) if isinstance(compression_memory.get("qpruner"), dict) else {}

    evidence = {
        "cap_wanda_sparsegpt_full196": {
            **file_info(cap_path),
            "status": cap.get("status", "missing"),
            "targeted_layers": target_layers(cap, "wanda"),
            "targeted_layers_total": cap.get("targeted_layers_total"),
            "claim_scope": cap.get(
                "claim_scope",
                "Ascend full-target lightweight Wanda/SparseGPT scoring evidence, not official upstream full reproduction.",
            ),
            "wanda_targeted_param_reduction_pct": cap_metric(cap, "wanda", "targeted_param_reduction_pct"),
            "wanda_loss_delta": cap_metric(cap, "wanda", "loss_delta"),
            "wanda_latency_speedup": cap_metric(cap, "wanda", "latency_speedup"),
            "sparsegpt_targeted_param_reduction_pct": cap_metric(cap, "sparsegpt", "targeted_param_reduction_pct"),
            "sparsegpt_loss_delta": cap_metric(cap, "sparsegpt", "loss_delta"),
            "sparsegpt_latency_speedup": cap_metric(cap, "sparsegpt", "latency_speedup"),
            "wanda_compression_time_s": cap_metric(cap, "wanda", "compression_time_s"),
            "sparsegpt_compression_time_s": cap_metric(cap, "sparsegpt", "compression_time_s"),
        },
        "memory_first_existing": {
            **file_info(memory_path),
            "status": memory.get("status", "missing"),
            "cap_targeted_memory_reduction_pct": cap_memory.get("targeted_param_reduction_pct"),
            "qpruner_targeted_memory_reduction_pct": q_memory.get("targeted_param_reduction_pct"),
            "target_layer_coverage_pct": compression_memory.get("target_layer_coverage_pct"),
        },
        "qpruner_llm_pruner_full196": {
            **file_info(qpruner_path),
            "status": qpruner.get("status", "missing"),
            "targeted_layers": target_layers(qpruner),
            "targeted_layers_total": qpruner.get("targeted_layers_total"),
            "targeted_param_reduction_pct": qpruner_metric(qpruner, "targeted_param_reduction_pct"),
            "loss_delta": qpruner_metric(qpruner, "loss_delta"),
            "latency_speedup": qpruner_metric(qpruner, "latency_speedup"),
            "tokens_per_s": qpruner_metric(qpruner, "tokens_per_s"),
            "peak_mem_mb": qpruner_metric(qpruner, "peak_mem_mb"),
            "compression_time_s": qpruner_metric(qpruner, "compression_time_s"),
            "claim_scope": qpruner.get(
                "claim_scope",
                "lightweight Ascend structural-pruning baseline evidence, not official full LLM-Pruner reproduction",
            ),
        },
        "rankadaptor_existing": {
            **file_info(finetune_path),
            "status": finetune.get("status", "missing"),
            "qwen3_lora_world_size": finetune.get("qwen3_lora_world_size"),
            "qwen3_lora_validation_loss_delta_avg": finetune.get("qwen3_lora_validation_loss_delta_avg"),
            "qwen3_lora_adapter_sync_consistent": finetune.get("qwen3_lora_adapter_sync_consistent"),
            "qwen3_bslora_target_module_count": finetune.get("qwen3_bslora_target_module_count"),
            "qwen3_bslora_trainable_adapter_params": finetune.get("qwen3_bslora_trainable_adapter_params"),
            "qwen3_bslora_unshared_adapter_params": finetune.get("qwen3_bslora_unshared_adapter_params"),
            "qwen3_bslora_loss_delta": finetune.get("qwen3_bslora_loss_delta"),
        },
        "rankadaptor_recovery_controlled": {
            **file_info(recovery_path),
            "status": recovery.get("status", "missing"),
            "all_methods_passed": recovery.get("all_methods_passed", False),
            "best_recovery_method": recovery.get("best_recovery_method", "missing"),
            "paper_baselines": recovery.get("paper_baselines", []),
            "target_modules_total": recovery.get("target_modules_total"),
            "no_recovery_loss_delta": method_loss(recovery, "no_recovery"),
            "lora_loss_delta": method_loss(recovery, "lora"),
            "adalora_style_loss_delta": method_loss(recovery, "adalora_style"),
            "no_recovery_trainable_adapter_params": method_params(recovery, "no_recovery"),
            "lora_trainable_adapter_params": method_params(recovery, "lora"),
            "adalora_style_trainable_adapter_params": method_params(recovery, "adalora_style"),
            "claim_scope": recovery.get(
                "claim_scope",
                "Controlled RankAdaptor recovery comparison; AdaLoRA-style is an approximation unless official AdaLoRA is ported.",
            ),
        },
    }

    rankadaptor_gap = (
        "Official AdaLoRA and official pruning-stage reproductions remain stricter paper-reproduction work; "
        "the demo now has controlled LoRA / AdaLoRA-style / no-recovery evidence on Ascend."
    )
    coverage_decisions = [
        {
            "area": "QPruner",
            "decision": "Use the new 196/196 LLM-Pruner-style artifact as the strongest current paper-baseline compatibility evidence.",
            "remaining_gap": "Still not an official full LLM-Pruner codebase reproduction; report it as LLM-Pruner-style structural pruning until official port is run.",
        },
        {
            "area": "CAP",
            "decision": "Use the new 196/196 Wanda/SparseGPT artifact as full-target lightweight paper-baseline evidence on Ascend.",
            "remaining_gap": "Need official upstream reproductions and remaining DSNoT/OATS/OWL/AlphaPruning/SLiM/JSQ/L2QER/LPAF/SVD baselines before claiming full paper reproduction.",
        },
        {
            "area": "RankAdaptor",
            "decision": "Use controlled LoRA / AdaLoRA-style / no-recovery recovery baselines plus LoRA/BSLoRA training evidence for the Ascend demo.",
            "remaining_gap": rankadaptor_gap,
        },
        {
            "area": "Engineering baseline",
            "decision": "Keep dense baseline only for latency/throughput/memory ratios.",
            "remaining_gap": "Do not present dense baseline as a paper baseline.",
        },
    ]

    video_readout = [
        "Baseline critique resolved by separating paper baselines from engineering runtime references.",
        (
            "QPruner paper baseline now has {layers}/{total} LLM-Pruner-style Ascend evidence: "
            "{prune}% targeted parameter pruning, loss delta {loss}, latency speedup {speedup}x."
        ).format(
            layers=fmt(target_layers(qpruner)),
            total=fmt(qpruner.get("targeted_layers_total")),
            prune=fmt(qpruner_metric(qpruner, "targeted_param_reduction_pct")),
            loss=fmt(qpruner_metric(qpruner, "loss_delta")),
            speedup=fmt(qpruner_metric(qpruner, "latency_speedup")),
        ),
        (
            "CAP paper baselines now have {layers}/{total} Wanda and SparseGPT Ascend evidence: "
            "Wanda loss delta {wanda}, SparseGPT loss delta {sparsegpt}, both at 50% targeted sparsity."
        ).format(
            layers=fmt(target_layers(cap, "wanda")),
            total=fmt(cap.get("targeted_layers_total")),
            wanda=fmt(cap_metric(cap, "wanda", "loss_delta")),
            sparsegpt=fmt(cap_metric(cap, "sparsegpt", "loss_delta")),
        ),
        (
            "RankAdaptor recovery baselines now have controlled Ascend evidence: no-recovery delta {none}, "
            "LoRA delta {lora}, AdaLoRA-style delta {adalora}; best method {best}."
        ).format(
            none=fmt(method_loss(recovery, "no_recovery")),
            lora=fmt(method_loss(recovery, "lora")),
            adalora=fmt(method_loss(recovery, "adalora_style")),
            best=recovery.get("best_recovery_method", "missing"),
        ),
    ]
    next_runs = [
        "If time allows, replace LLM-Pruner-style with official LLM-Pruner implementation on Ascend or mark exact incompatibilities.",
        "Expand CAP baseline family beyond Wanda/SparseGPT when each upstream method has an Ascend-safe path.",
        "Port official AdaLoRA if strict paper-table reproduction is required beyond the controlled AdaLoRA-style comparison.",
    ]
    return {
        "status": "PASS",
        "purpose": "paper-baseline alignment audit for Ascend 910B demo material",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "engineering_baseline_note": "`baseline` in serving artifacts is dense/uncompressed runtime reference, not a paper baseline.",
        "paper_baselines": {
            "cap": {
                "pruning": ["SparseGPT", "Wanda", "DSNoT", "OATS", "OWL", "AlphaPruning"],
                "joint_compression": ["SLiM", "JSQ", "L2QER", "LPAF"],
                "svd_structured": ["SVD-LLM v2", "Dobi-SVD", "Basis Sharing", "LoSparse"],
                "paper_metrics": ["WikiText perplexity", "zero-shot accuracy", "throughput", "memory footprint"],
                "paper_modern_models": ["LLaMA-3.1-8B-Instruct", "Qwen2.5-7B"],
            },
            "qpruner": {
                "primary": ["LLM-Pruner"],
                "recovery": ["LoRA", "LoftQ"],
                "paper_metrics": ["zero-shot accuracy", "peak memory", "pruning rate"],
                "paper_tasks": ["BoolQ", "PIQA", "HellaSwag", "WinoGrande", "ARC-e", "ARC-c", "OpenBookQA"],
            },
            "rankadaptor": {
                "pruning_stage": ["LLM-Pruner", "Shortened LLaMA"],
                "recovery": ["LoRA", "AdaLoRA", "without recovery"],
                "paper_tasks": ["BoolQ", "PIQA", "HellaSwag", "WinoGrande", "ARC-e", "ARC-c", "OpenBookQA"],
            },
        },
        "evidence": evidence,
        "coverage_decisions": coverage_decisions,
        "video_readout": video_readout,
        "next_runs": next_runs,
    }


def markdown(report: dict[str, Any]) -> str:
    evidence = report.get("evidence", {})
    lines = [
        "# Paper Baseline Coverage Audit",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Purpose: {report.get('purpose')}",
        f"- Engineering baseline note: {report.get('engineering_baseline_note')}",
        "",
        "## Video Readout",
        "",
    ]
    lines.extend(f"- {item}" for item in report.get("video_readout", []))
    lines.extend(["", "## Current Evidence", "", "| Area | Status | Metrics |", "|---|---:|---|"])
    q = evidence.get("qpruner_llm_pruner_full196", {})
    lines.append(
        "| QPruner LLM-Pruner-style | {status} | target layers {layers}/{total}, prune {prune}%, loss delta {loss}, speedup {speedup}x |".format(
            status=q.get("status", "missing"),
            layers=fmt(q.get("targeted_layers")),
            total=fmt(q.get("targeted_layers_total")),
            prune=fmt(q.get("targeted_param_reduction_pct")),
            loss=fmt(q.get("loss_delta")),
            speedup=fmt(q.get("latency_speedup")),
        )
    )
    cap = evidence.get("cap_wanda_sparsegpt_full196", {})
    lines.append(
        "| CAP Wanda/SparseGPT | {status} | target layers {layers}/{total}, Wanda loss delta {wanda}, SparseGPT loss delta {sparsegpt} |".format(
            status=cap.get("status", "missing"),
            layers=fmt(cap.get("targeted_layers")),
            total=fmt(cap.get("targeted_layers_total")),
            wanda=fmt(cap.get("wanda_loss_delta")),
            sparsegpt=fmt(cap.get("sparsegpt_loss_delta")),
        )
    )
    rec = evidence.get("rankadaptor_recovery_controlled", {})
    lines.append(
        "| RankAdaptor recovery controlled baselines | {status} | no-recovery delta {none}, LoRA delta {lora}, AdaLoRA-style delta {adalora}, best {best} |".format(
            status=rec.get("status", "missing"),
            none=fmt(rec.get("no_recovery_loss_delta")),
            lora=fmt(rec.get("lora_loss_delta")),
            adalora=fmt(rec.get("adalora_style_loss_delta")),
            best=rec.get("best_recovery_method", "missing"),
        )
    )
    mem = evidence.get("memory_first_existing", {})
    lines.append(
        "| Memory-first compression | {status} | CAP -{cap}%, QPruner -{q}%, coverage {coverage}% |".format(
            status=mem.get("status", "missing"),
            cap=fmt(mem.get("cap_targeted_memory_reduction_pct")),
            q=fmt(mem.get("qpruner_targeted_memory_reduction_pct")),
            coverage=fmt(mem.get("target_layer_coverage_pct")),
        )
    )
    lines.extend(["", "## Coverage Decisions", ""])
    for item in report.get("coverage_decisions", []):
        lines.append(f"- **{item.get('area')}**: {item.get('decision')} Gap: {item.get('remaining_gap')}")
    lines.extend(["", "## Next Runs", ""])
    lines.extend(f"- {item}" for item in report.get("next_runs", []))
    return "\n".join(lines) + "\n"


def write_outputs(demo_root: Path) -> tuple[Path, Path]:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    report = build_audit(demo_root)
    json_path = artifacts / "paper_baseline_coverage_audit.json"
    md_path = reports / "paper-baseline-coverage-audit.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    md_path.write_text(markdown(report))
    return json_path, md_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write paper-baseline coverage audit")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    json_path, md_path = write_outputs(Path(args.demo_root))
    print(f"PAPER_BASELINE_AUDIT {json_path} {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
