"""Selection helpers for Qwen QPruner grouped replay demo artifacts."""
from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from tidal.reports.npu_monitor_selection import (
    best_npu_utilization_monitor,
    grouped_replay_monitor_score,
    int_score,
)


MULTICARD_QWEN_QPRUNER_GROUPED_REPLAY_PREFIX = "multicard_qwen_qpruner_grouped_replay_"


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, dict) else None


def float_score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def multicard_qwen_qpruner_grouped_replay_report_name(
    replay: dict[str, Any] | None,
    fallback: str = "multicard-qwen-qpruner-grouped-replay-missing.md",
) -> str:
    run_label = (replay or {}).get("run_label")
    if not run_label:
        artifact_name = str((replay or {}).get("_artifact_name") or "")
        prefix = MULTICARD_QWEN_QPRUNER_GROUPED_REPLAY_PREFIX
        suffix = ".json"
        if artifact_name.startswith(prefix) and artifact_name.endswith(suffix):
            run_label = artifact_name[len(prefix) : -len(suffix)]
    return f"multicard-qwen-qpruner-grouped-replay-{run_label}.md" if run_label else fallback


def _command_value(command: list[Any], flag: str) -> int:
    try:
        index = command.index(flag)
    except ValueError:
        return 0
    if index + 1 >= len(command):
        return 0
    return int_score(command[index + 1])


def _replay_workload_score(payload: dict[str, Any]) -> int:
    workers = payload.get("workers")
    if not isinstance(workers, list):
        return 0
    batch_size = 0
    iters = 0
    warmup = 0
    for worker in workers:
        if not isinstance(worker, dict):
            continue
        command = worker.get("command")
        if not isinstance(command, list):
            continue
        batch_size = max(batch_size, _command_value(command, "--batch-size"))
        iters = max(iters, _command_value(command, "--iters"))
        warmup = max(warmup, _command_value(command, "--warmup"))
    return batch_size * max(iters, 1) + warmup


def _matching_monitor_score(artifacts: Path, run_label: str) -> tuple[int, int, int]:
    if not run_label:
        return (0, 0, 0)
    for path in artifacts.glob("npu_monitor_*.json"):
        payload = read_json(path)
        if payload is None:
            continue
        payload["_artifact_name"] = path.name
        if _monitor_inner_run_label(payload) != run_label:
            continue
        return (
            1 if payload.get("status") == "PASS" else 0,
            grouped_replay_monitor_score(payload),
            int_score(payload.get("samples_with_8_nonzero_aicore")),
        )
    return (0, 0, 0)


def _run_label_from_command(command: list[Any]) -> str | None:
    try:
        index = command.index("--run-label")
    except ValueError:
        return None
    if index + 1 >= len(command):
        return None
    value = command[index + 1]
    return str(value) if value is not None else None


def _monitor_inner_run_label(payload: dict[str, Any]) -> str | None:
    command = payload.get("command")
    if isinstance(command, list):
        value = _run_label_from_command(command)
        if value:
            return value
    command_text = payload.get("command_text")
    if isinstance(command_text, str) and command_text:
        try:
            value = _run_label_from_command(shlex.split(command_text))
        except ValueError:
            value = None
        if value:
            return value
    monitor_label = str(payload.get("run_label") or "")
    suffix = "_utilmon"
    if monitor_label.endswith(suffix):
        return monitor_label[: -len(suffix)]
    artifact_name = str(payload.get("_artifact_name") or "")
    prefix = "npu_monitor_"
    suffix_json = f"{suffix}.json"
    if artifact_name.startswith(prefix) and artifact_name.endswith(suffix_json):
        return artifact_name[len(prefix) : -len(suffix_json)]
    return None


def best_multicard_qwen_qpruner_grouped_replay(artifacts: Path) -> dict[str, Any] | None:
    selected_monitor = best_npu_utilization_monitor(artifacts)
    selected_monitor_inner_run_label = _monitor_inner_run_label(selected_monitor or {})
    candidates: list[tuple[tuple[int, int, int, int, int, int, int, int, float, float], dict[str, Any]]] = []
    for path in artifacts.glob(f"{MULTICARD_QWEN_QPRUNER_GROUPED_REPLAY_PREFIX}*.json"):
        if path.name.endswith(".worker.json") or path.name.endswith(".release.json"):
            continue
        payload = read_json(path)
        if payload is None:
            continue
        selected = dict(payload)
        selected["_artifact_name"] = path.name
        aggregate = selected.get("aggregate", {}) if isinstance(selected.get("aggregate"), dict) else {}
        status_score = 1 if selected.get("status") == "PASS" else 0
        world_size = int_score(selected.get("world_size"))
        memory_native_count = int_score(aggregate.get("memory_native_worker_count"))
        storage_reduction = float_score(aggregate.get("min_code_cache_storage_reduction_pct"))
        run_label = str(selected.get("run_label") or "")
        matching_monitor = _matching_monitor_score(artifacts, run_label)
        selected_monitor_match = 1 if run_label and run_label == selected_monitor_inner_run_label else 0
        candidates.append(
            (
                (
                    status_score,
                    1 if aggregate.get("all_workers_memory_native") is True else 0,
                    memory_native_count,
                    1 if storage_reduction > 0 else 0,
                    world_size,
                    selected_monitor_match,
                    matching_monitor[0] + matching_monitor[1],
                    _replay_workload_score(selected),
                    float_score(aggregate.get("best_grouped_speedup")),
                    path.stat().st_mtime,
                ),
                selected,
            )
        )
    return max(candidates, key=lambda row: row[0])[1] if candidates else None
