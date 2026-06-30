"""Helpers for selecting the strongest preserved multi-card demo artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MULTICARD_QWEN_COMPRESSION_GENERATE_PREFIX = "multicard_qwen_compression_generate_"
MULTICARD_QWEN_LORA_FINETUNE_PREFIX = "multicard_qwen_lora_finetune_"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, dict) else None


def _int_score(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _float_score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def multicard_qwen_compression_generate_report_name(artifact_name: str | None) -> str | None:
    if not artifact_name:
        return None
    if not artifact_name.startswith(MULTICARD_QWEN_COMPRESSION_GENERATE_PREFIX):
        return None
    if not artifact_name.endswith(".json"):
        return None
    run_label = artifact_name[len(MULTICARD_QWEN_COMPRESSION_GENERATE_PREFIX) : -len(".json")]
    return f"multicard-qwen-compression-generate-{run_label}.md"


def multicard_qwen_lora_finetune_report_name(artifact_name: str | None) -> str | None:
    if not artifact_name:
        return None
    if not artifact_name.startswith(MULTICARD_QWEN_LORA_FINETUNE_PREFIX):
        return None
    if not artifact_name.endswith(".json"):
        return None
    run_label = artifact_name[len(MULTICARD_QWEN_LORA_FINETUNE_PREFIX) : -len(".json")]
    return f"multicard-qwen-lora-finetune-{run_label}.md"


def best_multicard_qwen_compression_generate(artifacts: Path) -> dict[str, Any] | None:
    candidates: list[tuple[tuple[int, int, int, int, float, float], dict[str, Any]]] = []
    for path in artifacts.glob(f"{MULTICARD_QWEN_COMPRESSION_GENERATE_PREFIX}*.json"):
        if "_rank" in path.stem:
            continue
        payload = _read_json(path)
        if payload is None:
            continue
        aggregate = payload.get("aggregate", {}) if isinstance(payload.get("aggregate"), dict) else {}
        status_score = 1 if payload.get("status") == "PASS" else 0
        consistency_score = 1 if aggregate.get("distributed_reduce_consistent") is True else 0
        world_size = _int_score(payload.get("world_size"))
        pass_count = _int_score(aggregate.get("pass_count"))
        target_layers = _int_score(payload.get("target_layer_limit"))
        qpruner_tokens = _float_score(aggregate.get("qpruner_tokens_per_s_total"))
        selected = dict(payload)
        selected["_artifact_name"] = path.name
        selected["_report_name"] = multicard_qwen_compression_generate_report_name(path.name)
        candidates.append(
            (
                (
                    status_score,
                    consistency_score,
                    min(world_size, pass_count),
                    target_layers,
                    qpruner_tokens,
                    path.stat().st_mtime,
                ),
                selected,
            )
        )
    return max(candidates, key=lambda row: row[0])[1] if candidates else None


def best_multicard_qwen_lora_finetune(artifacts: Path) -> dict[str, Any] | None:
    candidates: list[tuple[tuple[int, int, int, int, float, float], dict[str, Any]]] = []
    for path in artifacts.glob(f"{MULTICARD_QWEN_LORA_FINETUNE_PREFIX}*.json"):
        if "_rank" in path.stem:
            continue
        payload = _read_json(path)
        if payload is None:
            continue
        aggregate = payload.get("aggregate", {}) if isinstance(payload.get("aggregate"), dict) else {}
        status_score = 1 if payload.get("status") == "PASS" else 0
        consistency_score = 1 if aggregate.get("distributed_reduce_consistent") is True else 0
        adapter_sync_score = 1 if aggregate.get("adapter_sync_consistent") is True else 0
        world_size = _int_score(payload.get("world_size"))
        pass_count = _int_score(aggregate.get("pass_count"))
        improvement = -_float_score(aggregate.get("validation_loss_delta_avg"))
        selected = dict(payload)
        selected["_artifact_name"] = path.name
        selected["_report_name"] = multicard_qwen_lora_finetune_report_name(path.name)
        candidates.append(
            (
                (
                    status_score,
                    consistency_score,
                    adapter_sync_score,
                    min(world_size, pass_count),
                    improvement,
                    path.stat().st_mtime,
                ),
                selected,
            )
        )
    return max(candidates, key=lambda row: row[0])[1] if candidates else None
