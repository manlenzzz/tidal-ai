#!/usr/bin/env python
"""Launch independent Ascend demo tasks across cards and aggregate evidence."""
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


TASK_ALIASES = {
    "cap": "workflow_cap",
    "qpruner": "workflow_qpruner",
    "rankadaptor": "rankadaptor",
    "workflow_cap": "workflow_cap",
    "workflow_qpruner": "workflow_qpruner",
    "workflow_rankadaptor": "rankadaptor",
    "torch_serving": "torch_serving_fallback",
    "torch_serving_fallback": "torch_serving_fallback",
    "vllm": "vllm_serving_benchmark",
    "vllm_serving": "vllm_serving_benchmark",
    "vllm_serving_benchmark": "vllm_serving_benchmark",
}
VLLM_METHOD_ROTATION = ("qpruner", "cap", "baseline")


class WorkerSpec(NamedTuple):
    rank: int
    card: int
    task: str
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


def parse_tasks(raw: str) -> list[str]:
    tasks = []
    for part in raw.split(","):
        name = part.strip().lower()
        if not name:
            continue
        if name not in TASK_ALIASES:
            raise ValueError(f"unknown task {name!r}; expected one of {sorted(TASK_ALIASES)}")
        tasks.append(TASK_ALIASES[name])
    if not tasks:
        raise ValueError("at least one task is required")
    return tasks


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def command_text(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def task_payload_name(*, run_label: str, rank: int, task: str, card: int) -> str:
    if task == "torch_serving_fallback":
        return f"torch_serving_fallback_{run_label}_rank{rank}_npu{card}.json"
    if task == "vllm_serving_benchmark":
        method = VLLM_METHOD_ROTATION[rank % len(VLLM_METHOD_ROTATION)]
        return f"vllm_serving_benchmark_{run_label}_rank{rank}_{method}_npu{card}.json"
    return f"multicard_parallel_suite_{run_label}_rank{rank}_{task}_npu{card}_payload.json"


def task_command(
    *,
    project_root: Path,
    demo_root: Path,
    task: str,
    payload_json: Path,
    run_label: str,
    rank: int,
    card: int,
    device: str,
    dtype: str,
    export_run_label: str,
    iters: int,
    warmup: int,
    max_new_tokens: int,
    seed: int,
) -> list[str]:
    python = sys.executable
    if task in {"workflow_cap", "workflow_qpruner", "rankadaptor"}:
        workflow = {
            "workflow_cap": "cap",
            "workflow_qpruner": "qpruner",
            "rankadaptor": "rankadaptor",
        }[task]
        return [
            python,
            str(project_root / "scripts" / "ascend_npu_workflow_smoke.py"),
            "--device",
            device,
            "--dtype",
            dtype,
            "--workflow",
            workflow,
            "--output-json",
            str(payload_json),
            "--compat-md",
            str(demo_root / "reports" / f"multicard-parallel-suite-{run_label}-rank{rank}-{task}-npu{card}.md"),
            "--seed",
            str(seed),
        ]
    if task == "torch_serving_fallback":
        return [
            python,
            str(project_root / "scripts" / "torch_serving_fallback_benchmark.py"),
            "--demo-root",
            str(demo_root),
            "--export-run-label",
            export_run_label,
            "--run-label",
            f"{run_label}_rank{rank}_npu{card}",
            "--device",
            device,
            "--dtype",
            dtype,
            "--max-new-tokens",
            str(max_new_tokens),
            "--iters",
            str(iters),
            "--warmup",
            str(warmup),
        ]
    if task == "vllm_serving_benchmark":
        method = VLLM_METHOD_ROTATION[rank % len(VLLM_METHOD_ROTATION)]
        return [
            python,
            str(project_root / "scripts" / "vllm_serving_benchmark.py"),
            "--single-method",
            method,
            "--method-path",
            str(demo_root / "serving_exports" / export_run_label / method),
            "--output-json",
            str(payload_json),
            "--dtype",
            dtype,
            "--max-new-tokens",
            str(max_new_tokens),
            "--iters",
            str(iters),
            "--warmup",
            str(warmup),
            "--gpu-memory-utilization",
            "0.05",
            "--max-model-len",
            "128",
            "--preload-vllm-ascend-patch",
            "--preload-vllm-ascend-selector-shim",
            "--preload-vllm-ascend-metadata-shim",
        ]
    raise ValueError(f"unsupported task: {task}")


def build_worker_specs(
    *,
    project_root: Path,
    demo_root: Path,
    cards: list[int],
    tasks: list[str],
    run_label: str,
    device: str,
    dtype: str,
    export_run_label: str,
    iters: int,
    warmup: int,
    max_new_tokens: int,
    seed: int,
) -> list[WorkerSpec]:
    artifacts = demo_root / "artifacts"
    logs = demo_root / "logs"
    workers = []
    for rank, card in enumerate(cards):
        task = tasks[rank % len(tasks)]
        payload_json = artifacts / task_payload_name(run_label=run_label, rank=rank, task=task, card=card)
        worker_json = artifacts / f"multicard_parallel_suite_{run_label}_rank{rank}.worker.json"
        release_json = artifacts / f"multicard_parallel_suite_{run_label}_rank{rank}.release.json"
        log_path = logs / f"multicard_parallel_suite_{run_label}_rank{rank}_{task}_npu{card}.log"
        worker_seed = seed + rank
        command = task_command(
            project_root=project_root,
            demo_root=demo_root,
            task=task,
            payload_json=payload_json,
            run_label=run_label,
            rank=rank,
            card=card,
            device=device,
            dtype=dtype,
            export_run_label=export_run_label,
            iters=iters,
            warmup=warmup,
            max_new_tokens=max_new_tokens,
            seed=worker_seed,
        )
        workers.append(
            WorkerSpec(
                rank=rank,
                card=card,
                task=task,
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


def extract_metrics(task: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    if task == "torch_serving_fallback":
        summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
        return {
            "best_method": summary.get("best_method"),
            "best_tokens_per_s": summary.get("best_tokens_per_s"),
            "cap_tokens_per_s": (
                payload.get("exports", {}).get("cap", {}).get("tokens_per_s")
                if isinstance(payload.get("exports"), dict)
                else None
            ),
            "qpruner_tokens_per_s": (
                payload.get("exports", {}).get("qpruner", {}).get("tokens_per_s")
                if isinstance(payload.get("exports"), dict)
                else None
            ),
        }
    if task == "vllm_serving_benchmark":
        metadata_shim = payload.get("metadata_shim", {}) if isinstance(payload.get("metadata_shim"), dict) else {}
        return {
            "method": payload.get("method"),
            "tokens_per_s": payload.get("tokens_per_s"),
            "latency_ms": payload.get("latency_ms"),
            "peak_mem_mb": payload.get("peak_mem_mb"),
            "metadata_shim_status": metadata_shim.get("status"),
            "metadata_backend_forward_status": metadata_shim.get("backend_forward_status"),
        }
    if task == "workflow_cap":
        cap = payload.get("cap", {}) if isinstance(payload.get("cap"), dict) else {}
        summary = cap.get("summary", {}) if isinstance(cap.get("summary"), dict) else {}
        return {"workflow_seconds": cap.get("seconds"), "packed_count": cap.get("packed_count"), "summary": summary}
    if task == "workflow_qpruner":
        qpruner = payload.get("qpruner", {}) if isinstance(payload.get("qpruner"), dict) else {}
        summary = qpruner.get("summary", {}) if isinstance(qpruner.get("summary"), dict) else {}
        return {
            "workflow_seconds": qpruner.get("seconds"),
            "quantized_count": qpruner.get("quantized_count"),
            "summary": summary,
        }
    if task == "rankadaptor":
        rankadaptor = payload.get("rankadaptor", {}) if isinstance(payload.get("rankadaptor"), dict) else {}
        summary = rankadaptor.get("summary", {}) if isinstance(rankadaptor.get("summary"), dict) else {}
        return {
            "workflow_seconds": rankadaptor.get("seconds"),
            "profile_count": rankadaptor.get("profile_count"),
            "summary": summary,
        }
    return {}


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
        "task": spec.task,
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
    metrics = extract_metrics(spec.task, payload)
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
    sync_start_target_ts: float | None = None,
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
    if spec.task == "vllm_serving_benchmark":
        env["TIDAL_VLLM_ASCEND_METADATA_SHIM"] = "1"
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
    status = "PASS" if returncode == 0 and payload_status(payload, returncode) in {"PASS", "READY"} else "FAIL"
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


def task_counts(workers: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for worker in workers:
        task = str(worker.get("task", "unknown"))
        counts[task] = counts.get(task, 0) + 1
    return dict(sorted(counts.items()))


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


def aggregate_metrics(workers: list[dict[str, Any]]) -> dict[str, Any]:
    best_tokens: float | None = None
    best_vllm_tokens: float | None = None
    for worker in workers:
        metrics = worker.get("metrics", {}) if isinstance(worker.get("metrics"), dict) else {}
        value = metrics.get("best_tokens_per_s")
        if value is None:
            value = metrics.get("tokens_per_s")
            if value is not None and worker.get("task") == "vllm_serving_benchmark":
                best_vllm_tokens = float(value) if best_vllm_tokens is None else max(best_vllm_tokens, float(value))
        if value is not None:
            best_tokens = float(value) if best_tokens is None else max(best_tokens, float(value))
    return {
        "pass_count": sum(1 for worker in workers if worker.get("status") == "PASS"),
        "fail_count": sum(1 for worker in workers if worker.get("status") == "FAIL"),
        "skipped_count": sum(1 for worker in workers if worker.get("status") == "SKIPPED"),
        "best_tokens_per_s": best_tokens,
        "best_vllm_tokens_per_s": best_vllm_tokens,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown_summary(summary: dict[str, Any]) -> str:
    aggregate = summary.get("aggregate", {}) if isinstance(summary.get("aggregate"), dict) else {}
    lines = [
        "# Synchronized Multi-Card Parallel Suite",
        "",
        "Independent demo tasks are launched together with one worker per selected card. This captures the video-friendly view that the available Ascend cards are being used concurrently.",
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
        f"| Best serving fallback tokens/s | {fmt(aggregate.get('best_tokens_per_s'))} |",
        f"| Best vLLM tokens/s | {fmt(aggregate.get('best_vllm_tokens_per_s'))} |",
        "",
        "| Task | Workers |",
        "|---|---:|",
    ]
    for task, count in (summary.get("task_counts") or {}).items():
        lines.append(f"| {task} | {count} |")
    lines.extend(
        [
            "",
            "| Rank | Card | Task | Status | Payload status | Runtime s |",
            "|---:|---:|---|---:|---:|---:|",
        ]
    )
    for worker in summary.get("workers", []):
        lines.append(
            "| {rank} | {card} | {task} | {status} | {payload_status} | {runtime} |".format(
                rank=worker.get("rank", ""),
                card=worker.get("card", ""),
                task=worker.get("task", ""),
                status=worker.get("status", ""),
                payload_status=worker.get("payload_status", ""),
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
        "task_counts": task_counts(workers),
        "aggregate": aggregate_metrics(workers),
        "workers": workers,
    }
    (artifacts / f"multicard_parallel_suite_{run_label}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    (reports / f"multicard-parallel-suite-{run_label}.md").write_text(markdown_summary(summary))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a synchronized multi-card parallel demo suite")
    parser.add_argument("--project-root", default="/mnt/nvme/622/tidal-ai")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--cards", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--tasks",
        default="workflow_cap,workflow_qpruner,rankadaptor,torch_serving_fallback",
        help=(
            "Comma-separated task rotation. Supported: workflow_cap, workflow_qpruner, "
            "rankadaptor, torch_serving_fallback, vllm_serving_benchmark."
        ),
    )
    parser.add_argument("--run-label", default="demo_parallel_suite_npu")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--export-run-label", default="tiny_qwen3_serving_export_npu")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-start-skew-seconds", type=float, default=1.0)
    parser.add_argument(
        "--sync-start-delay-seconds",
        type=float,
        default=3.0,
        help="Future start-gate delay shared by all workers before they execute their task command.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    project_root = Path(args.project_root)
    demo_root = Path(args.demo_root)
    cards = parse_cards(args.cards)
    tasks = parse_tasks(args.tasks)
    for directory in (demo_root / "artifacts", demo_root / "logs", demo_root / "reports"):
        directory.mkdir(parents=True, exist_ok=True)
    for old in (demo_root / "artifacts").glob(f"multicard_parallel_suite_{args.run_label}_rank*.worker.json"):
        old.unlink()

    specs = build_worker_specs(
        project_root=project_root,
        demo_root=demo_root,
        cards=cards,
        tasks=tasks,
        run_label=args.run_label,
        device=args.device,
        dtype=args.dtype,
        export_run_label=args.export_run_label,
        iters=args.iters,
        warmup=args.warmup,
        max_new_tokens=args.max_new_tokens,
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
    workers = (
        launched
        if args.dry_run
        else [finalize_process(entry, demo_root=demo_root) for entry in launched]
    )
    summary = write_summary(
        workers,
        demo_root=demo_root,
        run_label=args.run_label,
        cards=cards,
        max_start_skew_seconds=float(args.max_start_skew_seconds),
    )
    print("MULTICARD_PARALLEL_SUITE " + json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] in {"PASS", "DRY_RUN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
