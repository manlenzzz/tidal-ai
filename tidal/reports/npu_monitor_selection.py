"""Selection helpers for preserved NPU utilization monitor artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def int_score(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def npu_monitor_peak_score(payload: dict[str, Any]) -> int:
    peaks = payload.get("max_aicore_by_card", {})
    if not isinstance(peaks, dict):
        return 0
    return sum(int_score(value) for value in peaks.values())


def grouped_replay_monitor_score(payload: dict[str, Any]) -> int:
    command = " ".join(str(part) for part in payload.get("command", []) if part is not None)
    command_text = str(payload.get("command_text") or "")
    run_label = str(payload.get("run_label") or "")
    artifact_name = str(payload.get("_artifact_name") or "")
    haystack = " ".join([command, command_text, run_label, artifact_name]).lower()
    if "multicard_qwen_qpruner_grouped_replay.py" in haystack:
        return 2
    if "grouped_replay" in haystack and "qpruner" in haystack:
        return 1
    return 0


def best_npu_utilization_monitor(artifacts: Path) -> dict[str, Any] | None:
    candidates: list[tuple[tuple[int, int, int, int, int, int, int, float], dict[str, Any]]] = []
    for path in artifacts.glob("npu_monitor_*.json"):
        payload = read_json(path)
        if payload is None:
            continue
        payload["_artifact_name"] = path.name
        status_score = 1 if payload.get("status") == "PASS" else 0
        max_active = int_score(payload.get("max_active_process_cards"))
        all_aicore_samples = int_score(payload.get("samples_with_8_nonzero_aicore"))
        all_active_samples = int_score(payload.get("samples_with_8_active_process_cards"))
        samples = int_score(payload.get("samples"))
        candidates.append(
            (
                (
                    status_score,
                    max_active,
                    grouped_replay_monitor_score(payload),
                    all_aicore_samples,
                    all_active_samples,
                    npu_monitor_peak_score(payload),
                    samples,
                    path.stat().st_mtime,
                ),
                payload,
            )
        )
    return max(candidates, key=lambda row: row[0])[1] if candidates else None
