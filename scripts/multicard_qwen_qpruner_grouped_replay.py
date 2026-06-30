#!/usr/bin/env python
"""Launch real Qwen/QPruner grouped replay workers together across Ascend cards."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


class WorkerSpec(NamedTuple):
    rank: int
    card: int
    command: list[str]
    log_path: Path
    payload_json: Path
    worker_json: Path
    release_json: Path
    visible_devices: str
    seed: int


START_GATE_WRAPPER = r"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

command = sys.argv[1:]
target_raw = os.environ.get("TIDAL_SYNC_START_TARGET_TS")
release_json = os.environ.get("TIDAL_SYNC_RELEASE_JSON")
target = float(target_raw) if target_raw else None
if target is not None:
    while True:
        delay = target - time.time()
        if delay <= 0:
            break
        time.sleep(min(delay, 0.05))
released_at = time.time()
release_lag = None if target is None else released_at - target
if release_json:
    path = Path(release_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "sync_start_target_ts": target,
        "released_at": released_at,
        "release_lag_s": release_lag,
        "command": command,
    }, indent=2, sort_keys=True))
raise SystemExit(subprocess.call(command))
""".strip()


def parse_cards(raw: str) -> list[int]:
    cards = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not cards:
        raise ValueError("at least one card is required")
    return cards


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def command_text(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def fmt(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def worker_run_label(*, run_label: str, rank: int, card: int) -> str:
    return f"{run_label}_rank{rank}_npu{card}"


def build_worker_specs(
    *,
    project_root: Path,
    demo_root: Path,
    cards: list[int],
    run_label: str,
    model_id: str,
    model_path: Path,
    device: str,
    dtype: str,
    batch_size: int,
    iters: int,
    warmup: int,
    max_modules: int,
    min_group_modules: int,
    max_groups: int,
    target_layer_limit: int,
    target_layer_pattern: str | None,
    qpruner_average_bits: float,
    dense_weight_bits: float,
    qpruner_release_packed_after_cache: bool,
    qpruner_prebuild_scaled_code_dtype_cache: bool,
    seed: int,
) -> list[WorkerSpec]:
    artifacts = demo_root / "artifacts"
    logs = demo_root / "logs"
    workers = []
    for rank, card in enumerate(cards):
        worker_label = worker_run_label(run_label=run_label, rank=rank, card=card)
        worker_seed = seed + rank
        payload_json = artifacts / f"qwen_qpruner_grouped_replay_{worker_label}.json"
        worker_json = artifacts / f"multicard_qwen_qpruner_grouped_replay_{run_label}_rank{rank}.worker.json"
        release_json = artifacts / f"multicard_qwen_qpruner_grouped_replay_{run_label}_rank{rank}.release.json"
        log_path = logs / f"multicard_qwen_qpruner_grouped_replay_{run_label}_rank{rank}_npu{card}.log"
        command = [
            sys.executable,
            str(project_root / "scripts" / "qwen_qpruner_grouped_replay.py"),
            "--demo-root",
            str(demo_root),
            "--model-id",
            model_id,
            "--model-path",
            str(model_path),
            "--device",
            device,
            "--dtype",
            dtype,
            "--batch-size",
            str(batch_size),
            "--iters",
            str(iters),
            "--warmup",
            str(warmup),
            "--max-modules",
            str(max_modules),
            "--min-group-modules",
            str(min_group_modules),
            "--max-groups",
            str(max_groups),
            "--target-layer-limit",
            str(target_layer_limit),
            "--qpruner-average-bits",
            str(qpruner_average_bits),
            "--dense-weight-bits",
            str(dense_weight_bits),
            "--seed",
            str(worker_seed),
            "--run-label",
            worker_label,
        ]
        if target_layer_pattern:
            command.extend(["--target-layer-pattern", target_layer_pattern])
        if qpruner_release_packed_after_cache:
            command.append("--qpruner-release-packed-after-cache")
        if qpruner_prebuild_scaled_code_dtype_cache:
            command.append("--qpruner-prebuild-scaled-code-dtype-cache")
        workers.append(
            WorkerSpec(
                rank=rank,
                card=card,
                command=command,
                log_path=log_path,
                payload_json=payload_json,
                worker_json=worker_json,
                release_json=release_json,
                visible_devices=str(card),
                seed=worker_seed,
            )
        )
    return workers


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"status": "PARSE_ERROR", "error_type": type(exc).__name__, "error": str(exc)}


def payload_status(payload: dict[str, Any] | None, returncode: int | None) -> str:
    if payload and payload.get("status"):
        return str(payload.get("status"))
    if returncode == 0:
        return "PASS"
    if returncode is None:
        return "MISSING"
    return "FAIL"


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _best_group(payload: dict[str, Any]) -> dict[str, Any]:
    best = payload.get("best_group", {})
    return best if isinstance(best, dict) else {}


def _nested_metric(row: dict[str, Any], section: str, key: str) -> Any:
    section_payload = row.get(section, {})
    if isinstance(section_payload, dict) and key in section_payload:
        return section_payload.get(key)
    return row.get(key)


def extract_replay_metrics(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    best = _best_group(payload)
    grouped_speedup = _nested_metric(best, "grouped_scaled_code_matmul", "speedup_vs_sequential_scaled_code")
    grouped_latency = _nested_metric(best, "grouped_scaled_code_matmul", "latency_ms")
    sequential_latency = _nested_metric(best, "sequential_scaled_code_matmul_group", "latency_ms")
    grouped_scaled_bytes = best.get("grouped_scaled_code_cache_bytes")
    cached_scaled_bytes = payload.get("cached_scaled_code_bytes")
    code_storage = _as_float(payload.get("code_cache_storage_reduction_pct"))
    memory_native = (
        payload.get("status") == "PASS"
        and code_storage is not None
        and code_storage > 0
        and _as_float(grouped_scaled_bytes) in (None, 0.0)
        and _as_float(cached_scaled_bytes) in (None, 0.0)
    )
    return {
        "model_id": payload.get("model_id"),
        "target_layer_limit": payload.get("target_layer_limit"),
        "targeted_layers_total": payload.get("targeted_layers_total"),
        "quantized_layers": payload.get("quantized_layers"),
        "group_count": payload.get("group_count"),
        "available_group_count": payload.get("available_group_count"),
        "best_role": best.get("role"),
        "best_module_count": best.get("module_count"),
        "best_available_module_count": best.get("available_module_count"),
        "best_grouped_speedup": _as_float(grouped_speedup),
        "best_grouped_latency_ms": _as_float(grouped_latency),
        "best_sequential_latency_ms": _as_float(sequential_latency),
        "code_cache_storage_reduction_pct": code_storage,
        "targeted_storage_reduction_pct": _as_float(payload.get("targeted_storage_reduction_pct")),
        "cached_scaled_code_bytes": _as_float(cached_scaled_bytes),
        "released_packed_code_bytes": _as_float(payload.get("released_packed_code_bytes")),
        "live_compressed_payload_storage_bytes": _as_float(payload.get("live_compressed_payload_storage_bytes")),
        "code_cache_storage_bytes": _as_float(payload.get("code_cache_storage_bytes")),
        "best_grouped_code_cache_bytes": _as_float(best.get("grouped_code_cache_bytes")),
        "best_grouped_scaled_code_cache_bytes": _as_float(grouped_scaled_bytes),
        "max_abs_diff": _as_float(best.get("max_abs_diff_vs_sequential_scaled_code")),
        "qpruner_release_packed_after_cache": bool(payload.get("qpruner_release_packed_after_cache")),
        "memory_native": memory_native,
    }


def worker_record(
    *,
    spec: WorkerSpec,
    demo_root: Path,
    status: str,
    returncode: int | None,
    started_at: float | None,
    sync_start_target_ts: float | None,
    released_at: float | None,
    release_lag_s: float | None,
    finished_at: float | None,
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    runtime = None
    if started_at is not None and finished_at is not None:
        runtime = round(finished_at - started_at, 3)
    record = {
        "status": status,
        "rank": spec.rank,
        "card": spec.card,
        "seed": spec.seed,
        "returncode": returncode,
        "payload_status": payload_status(payload, returncode),
        "started_at": started_at,
        "sync_start_target_ts": sync_start_target_ts,
        "released_at": released_at,
        "release_lag_s": round(release_lag_s, 6) if release_lag_s is not None else None,
        "finished_at": finished_at,
        "runtime_s": runtime,
        "visible_devices": spec.visible_devices,
        "command": spec.command,
        "command_text": command_text(spec.command),
        "log_path": rel(spec.log_path, demo_root),
        "payload_json": rel(spec.payload_json, demo_root),
    }
    metrics = extract_replay_metrics(payload)
    if metrics:
        record["metrics"] = metrics
    return record


def wrapped_command(command: Sequence[str]) -> list[str]:
    return [sys.executable, "-c", START_GATE_WRAPPER, *command]


def run_worker(
    spec: WorkerSpec,
    *,
    project_root: Path,
    demo_root: Path,
    dry_run: bool,
    sync_start_target_ts: float | None,
) -> dict[str, Any]:
    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    spec.payload_json.parent.mkdir(parents=True, exist_ok=True)
    spec.worker_json.parent.mkdir(parents=True, exist_ok=True)
    spec.release_json.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        record = worker_record(
            spec=spec,
            demo_root=demo_root,
            status="SKIPPED",
            returncode=None,
            started_at=None,
            sync_start_target_ts=sync_start_target_ts,
            released_at=None,
            release_lag_s=None,
            finished_at=None,
            payload=None,
        )
        spec.worker_json.write_text(json.dumps(record, indent=2, sort_keys=True))
        return record

    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = spec.visible_devices
    env["PYTHONPATH"] = f"{project_root}{os.pathsep}{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else str(project_root)
    if sync_start_target_ts is not None:
        env["TIDAL_SYNC_START_TARGET_TS"] = f"{sync_start_target_ts:.6f}"
        env["TIDAL_SYNC_RELEASE_JSON"] = str(spec.release_json)
    started = time.time()
    with spec.log_path.open("w") as log:
        log.write(f"$ {command_text(spec.command)}\n\n")
        if sync_start_target_ts is not None:
            log.write(f"sync_start_target_ts={sync_start_target_ts:.6f}\n")
            log.write(f"sync_release_json={spec.release_json}\n\n")
        log.flush()
        command = wrapped_command(spec.command) if sync_start_target_ts is not None else spec.command
        proc = subprocess.Popen(command, cwd=project_root, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    return {
        "spec": spec,
        "process": proc,
        "started_at": started,
        "sync_start_target_ts": sync_start_target_ts,
    }


def finalize_process(entry: dict[str, Any], *, demo_root: Path) -> dict[str, Any]:
    spec: WorkerSpec = entry["spec"]
    proc: subprocess.Popen[str] = entry["process"]
    returncode = proc.wait()
    finished = time.time()
    payload = read_json(spec.payload_json)
    release = read_json(spec.release_json) or {}
    released_at = release.get("released_at") if isinstance(release, dict) else None
    sync_start_target_ts = (
        release.get("sync_start_target_ts")
        if isinstance(release, dict) and release.get("sync_start_target_ts") is not None
        else entry.get("sync_start_target_ts")
    )
    release_lag_s = release.get("release_lag_s") if isinstance(release, dict) else None
    if release_lag_s is None and released_at is not None and sync_start_target_ts is not None:
        release_lag_s = float(released_at) - float(sync_start_target_ts)
    status = "PASS" if returncode == 0 and payload_status(payload, returncode) == "PASS" else "FAIL"
    record = worker_record(
        spec=spec,
        demo_root=demo_root,
        status=status,
        returncode=returncode,
        started_at=entry["started_at"],
        sync_start_target_ts=sync_start_target_ts,
        released_at=released_at,
        release_lag_s=release_lag_s,
        finished_at=finished,
        payload=payload,
    )
    spec.worker_json.write_text(json.dumps(record, indent=2, sort_keys=True))
    return record


def start_window(workers: list[dict[str, Any]]) -> float | None:
    starts = [float(worker["started_at"]) for worker in workers if worker.get("started_at") is not None]
    if len(starts) < 2:
        return 0.0 if len(starts) == 1 else None
    return round(max(starts) - min(starts), 3)


def sync_start_target(workers: list[dict[str, Any]]) -> float | None:
    targets = [float(worker["sync_start_target_ts"]) for worker in workers if worker.get("sync_start_target_ts") is not None]
    if not targets:
        return None
    return round(min(targets), 6)


def release_lag_window(workers: list[dict[str, Any]]) -> float | None:
    lags = [float(worker["release_lag_s"]) for worker in workers if worker.get("release_lag_s") is not None]
    if len(lags) < 2:
        return 0.0 if len(lags) == 1 else None
    return round(max(lags) - min(lags), 3)


def _metric_values(workers: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for worker in workers:
        metrics = worker.get("metrics", {}) if isinstance(worker.get("metrics"), dict) else {}
        value = _as_float(metrics.get(key))
        if value is not None:
            values.append(value)
    return values


def _first_metric(workers: list[dict[str, Any]], key: str) -> Any:
    for worker in workers:
        metrics = worker.get("metrics", {}) if isinstance(worker.get("metrics"), dict) else {}
        value = metrics.get(key)
        if value is not None:
            return value
    return None


def aggregate_metrics(workers: list[dict[str, Any]]) -> dict[str, Any]:
    pass_count = sum(1 for worker in workers if worker.get("status") == "PASS")
    fail_count = sum(1 for worker in workers if worker.get("status") == "FAIL")
    skipped_count = sum(1 for worker in workers if worker.get("status") == "SKIPPED")
    best_worker = None
    best_speedup = None
    for worker in workers:
        metrics = worker.get("metrics", {}) if isinstance(worker.get("metrics"), dict) else {}
        speedup = _as_float(metrics.get("best_grouped_speedup"))
        if speedup is None:
            continue
        if best_speedup is None or speedup > best_speedup:
            best_speedup = speedup
            best_worker = worker
    best_metrics = best_worker.get("metrics", {}) if isinstance(best_worker, dict) else {}
    speedups = _metric_values(workers, "best_grouped_speedup")
    code_reductions = _metric_values(workers, "code_cache_storage_reduction_pct")
    storage_reductions = _metric_values(workers, "targeted_storage_reduction_pct")
    memory_native_count = sum(
        1
        for worker in workers
        if isinstance(worker.get("metrics"), dict) and worker["metrics"].get("memory_native") is True
    )
    aggregate = {
        "pass_count": pass_count,
        "fail_count": fail_count,
        "skipped_count": skipped_count,
        "best_role": best_metrics.get("best_role"),
        "best_grouped_speedup": round(best_speedup, 3) if best_speedup is not None else None,
        "mean_grouped_speedup": round(sum(speedups) / len(speedups), 3) if speedups else None,
        "min_grouped_speedup": round(min(speedups), 3) if speedups else None,
        "max_grouped_speedup": round(max(speedups), 3) if speedups else None,
        "target_layer_limit": _first_metric(workers, "target_layer_limit"),
        "targeted_layers_total": _first_metric(workers, "targeted_layers_total"),
        "quantized_layers": _first_metric(workers, "quantized_layers"),
        "max_group_count": max(_metric_values(workers, "group_count"), default=None),
        "memory_native_worker_count": memory_native_count,
        "all_workers_memory_native": bool(pass_count) and memory_native_count == pass_count,
        "min_code_cache_storage_reduction_pct": round(min(code_reductions), 3) if code_reductions else None,
        "min_targeted_storage_reduction_pct": round(min(storage_reductions), 3) if storage_reductions else None,
        "released_packed_code_bytes_total": round(sum(_metric_values(workers, "released_packed_code_bytes")), 3),
        "live_compressed_payload_storage_bytes_total": round(
            sum(_metric_values(workers, "live_compressed_payload_storage_bytes")),
            3,
        ),
        "max_grouped_code_cache_bytes": max(_metric_values(workers, "best_grouped_code_cache_bytes"), default=None),
        "max_grouped_scaled_code_cache_bytes": max(
            _metric_values(workers, "best_grouped_scaled_code_cache_bytes"),
            default=None,
        ),
        "max_abs_diff": max(_metric_values(workers, "max_abs_diff"), default=None),
    }
    if aggregate["max_group_count"] is not None:
        aggregate["max_group_count"] = int(aggregate["max_group_count"])
    return aggregate


def markdown_summary(summary: dict[str, Any]) -> str:
    aggregate = summary.get("aggregate", {}) if isinstance(summary.get("aggregate"), dict) else {}
    lines = [
        "# Synchronized Qwen QPruner Grouped Replay",
        "",
        "Real Qwen3 QPruner grouped-replay workers are released at the same sync gate, one worker per selected Ascend card. This is module-level memory-native replay evidence, not an end-to-end serving speedup claim.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Status | {summary.get('status')} |",
        f"| Run label | {summary.get('run_label')} |",
        f"| World size | {summary.get('world_size')} |",
        f"| Cards | {','.join(str(card) for card in summary.get('cards', []))} |",
        f"| Launched synchronously | {summary.get('launched_synchronously')} |",
        f"| Sync start target | {fmt(summary.get('sync_start_target_ts'))} |",
        f"| Start window seconds | {fmt(summary.get('start_window_s'))} |",
        f"| Release lag window seconds | {fmt(summary.get('release_lag_window_s'))} |",
        f"| Passing workers | {fmt(aggregate.get('pass_count'))} |",
        f"| Failed workers | {fmt(aggregate.get('fail_count'))} |",
        f"| Best role | {fmt(aggregate.get('best_role'))} |",
        f"| Best grouped speedup | {fmt(aggregate.get('best_grouped_speedup'))}x |",
        f"| Mean grouped speedup | {fmt(aggregate.get('mean_grouped_speedup'))}x |",
        f"| Min grouped speedup | {fmt(aggregate.get('min_grouped_speedup'))}x |",
        f"| Target layers | {fmt(aggregate.get('target_layer_limit'))} / {fmt(aggregate.get('targeted_layers_total'))} |",
        f"| Quantized layers | {fmt(aggregate.get('quantized_layers'))} |",
        f"| Group count | {fmt(aggregate.get('max_group_count'))} |",
        f"| Memory-native workers | {fmt(aggregate.get('memory_native_worker_count'))} / {fmt(aggregate.get('pass_count'))} |",
        f"| Min code-cache storage reduction pct | {fmt(aggregate.get('min_code_cache_storage_reduction_pct'))} |",
        f"| Max grouped code-cache bytes | {fmt(aggregate.get('max_grouped_code_cache_bytes'))} |",
        f"| Max grouped scaled-code bytes | {fmt(aggregate.get('max_grouped_scaled_code_cache_bytes'))} |",
        f"| Max abs diff | {fmt(aggregate.get('max_abs_diff'))} |",
        "",
        "| Rank | Card | Status | Payload status | Best role | Speedup | Code-cache storage % | Runtime s |",
        "|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for worker in summary.get("workers", []):
        if not isinstance(worker, dict):
            continue
        metrics = worker.get("metrics", {}) if isinstance(worker.get("metrics"), dict) else {}
        lines.append(
            "| {rank} | {card} | {status} | {payload} | {role} | {speedup}x | {storage} | {runtime} |".format(
                rank=worker.get("rank", ""),
                card=worker.get("card", ""),
                status=worker.get("status", ""),
                payload=worker.get("payload_status", ""),
                role=fmt(metrics.get("best_role")),
                speedup=fmt(metrics.get("best_grouped_speedup")),
                storage=fmt(metrics.get("code_cache_storage_reduction_pct")),
                runtime=fmt(worker.get("runtime_s")),
            )
        )
    return "\n".join(lines) + "\n"


def write_summary(
    workers: list[dict[str, Any]],
    *,
    demo_root: Path,
    run_label: str,
    cards: list[int],
    max_start_skew_seconds: float,
) -> dict[str, Any]:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    window = start_window(workers)
    release_window = release_lag_window(workers)
    all_pass = bool(workers) and all(worker.get("status") == "PASS" for worker in workers)
    all_skipped = bool(workers) and all(worker.get("status") == "SKIPPED" for worker in workers)
    observed_window = release_window if release_window is not None else window
    launched = observed_window is not None and observed_window <= max_start_skew_seconds if not all_skipped else False
    if all_skipped:
        status = "DRY_RUN"
    elif all_pass and launched:
        status = "PASS"
    else:
        status = "FAIL"
    summary = {
        "status": status,
        "run_label": run_label,
        "platform": platform.platform(),
        "world_size": len(cards),
        "cards": cards,
        "max_start_skew_seconds": max_start_skew_seconds,
        "sync_start_target_ts": sync_start_target(workers),
        "start_window_s": window,
        "release_lag_window_s": release_window,
        "launched_synchronously": launched,
        "aggregate": aggregate_metrics(workers),
        "workers": workers,
    }
    (artifacts / f"multicard_qwen_qpruner_grouped_replay_{run_label}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    (reports / f"multicard-qwen-qpruner-grouped-replay-{run_label}.md").write_text(
        markdown_summary(summary)
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synchronized multi-card Qwen QPruner grouped replay")
    parser.add_argument("--project-root", default="/mnt/nvme/622/tidal-ai")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--cards", default="0,1")
    parser.add_argument("--run-label", default="qwen3_06b_real_grouped_replay_2card_npu")
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-path", default="/mnt/nvme/622/models/Qwen3-0.6B")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-modules", type=int, default=8)
    parser.add_argument("--min-group-modules", type=int, default=2)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--target-layer-limit", type=int, default=0)
    parser.add_argument("--target-layer-pattern")
    parser.add_argument("--qpruner-average-bits", type=float, default=8.0)
    parser.add_argument("--dense-weight-bits", type=float, default=16.0)
    parser.add_argument("--qpruner-release-packed-after-cache", action="store_true")
    parser.add_argument("--qpruner-prebuild-scaled-code-dtype-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-start-skew-seconds", type=float, default=1.0)
    parser.add_argument("--sync-start-delay-seconds", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(args.project_root)
    demo_root = Path(args.demo_root)
    cards = parse_cards(args.cards)
    for directory in (demo_root / "artifacts", demo_root / "logs", demo_root / "reports"):
        directory.mkdir(parents=True, exist_ok=True)
    for old in (demo_root / "artifacts").glob(
        f"multicard_qwen_qpruner_grouped_replay_{args.run_label}_rank*.worker.json"
    ):
        old.unlink()

    specs = build_worker_specs(
        project_root=project_root,
        demo_root=demo_root,
        cards=cards,
        run_label=args.run_label,
        model_id=args.model_id,
        model_path=Path(args.model_path),
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        iters=args.iters,
        warmup=args.warmup,
        max_modules=args.max_modules,
        min_group_modules=args.min_group_modules,
        max_groups=args.max_groups,
        target_layer_limit=args.target_layer_limit,
        target_layer_pattern=args.target_layer_pattern,
        qpruner_average_bits=args.qpruner_average_bits,
        dense_weight_bits=args.dense_weight_bits,
        qpruner_release_packed_after_cache=bool(args.qpruner_release_packed_after_cache),
        qpruner_prebuild_scaled_code_dtype_cache=bool(args.qpruner_prebuild_scaled_code_dtype_cache),
        seed=args.seed,
    )
    sync_start_target_ts = time.time() + float(args.sync_start_delay_seconds)
    launched = [
        run_worker(
            spec,
            project_root=project_root,
            demo_root=demo_root,
            dry_run=bool(args.dry_run),
            sync_start_target_ts=sync_start_target_ts,
        )
        for spec in specs
    ]
    workers = launched if args.dry_run else [finalize_process(entry, demo_root=demo_root) for entry in launched]
    summary = write_summary(
        workers,
        demo_root=demo_root,
        run_label=args.run_label,
        cards=cards,
        max_start_skew_seconds=float(args.max_start_skew_seconds),
    )
    print("MULTICARD_QWEN_QPRUNER_GROUPED_REPLAY " + json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] in {"PASS", "DRY_RUN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
