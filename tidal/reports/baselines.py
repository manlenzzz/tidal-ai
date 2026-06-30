from __future__ import annotations

from typing import Any


ENGINEERING_BASELINE: dict[str, Any] = {
    "artifact_method_key": "baseline",
    "display_label": "engineering reference",
    "role": "uncompressed serving/runtime reference",
    "not_paper_baseline": True,
}

PAPER_BASELINE_ALIGNMENT: dict[str, dict[str, Any]] = {
    "qpruner": {
        "primary": ["LLM-Pruner"],
        "recovery_baselines": ["LoRA", "LoftQ"],
        "metric_focus": ["zero-shot accuracy", "peak memory", "pruning rate"],
    },
    "cap": {
        "pruning_baselines": ["SparseGPT", "Wanda", "DSNoT", "OATS", "OWL", "AlphaPruning"],
        "joint_compression_baselines": ["SLiM", "JSQ", "L2QER", "LPAF"],
        "svd_structured_baselines": ["SVD-LLM v2", "Dobi-SVD", "Basis Sharing", "LoSparse"],
        "metric_focus": ["perplexity", "zero-shot accuracy", "throughput", "memory footprint"],
    },
    "rankadaptor": {
        "pruning_stage_baselines": ["LLM-Pruner", "Shortened LLaMA"],
        "recovery_baselines": ["LoRA", "AdaLoRA", "without recovery"],
        "metric_focus": ["zero-shot accuracy", "generation quality", "trainable adapter budget"],
    },
}


def engineering_baseline_summary() -> dict[str, Any]:
    return dict(ENGINEERING_BASELINE)


def paper_baseline_alignment() -> dict[str, dict[str, Any]]:
    return {method: dict(values) for method, values in PAPER_BASELINE_ALIGNMENT.items()}


def engineering_baseline_note() -> str:
    return "Engineering reference: `baseline` means uncompressed serving/runtime reference, not a paper baseline."


def paper_baseline_note() -> str:
    return (
        "Paper baselines: QPruner -> LLM-Pruner; CAP -> SparseGPT, Wanda; "
        "RankAdaptor -> LoRA, AdaLoRA, without recovery."
    )
