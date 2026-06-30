#!/usr/bin/env python
"""Run a command while preserving npu-smi utilization samples."""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


AICORE_RE = re.compile(r"^\|\s*(\d+)\s+910B\d*\s+\|\s+\S+\s+\|\s+[\d.]+\s+(\d+)\s+")
PROCESS_RE = re.compile(r"^\|\s*(\d+)\s+\d+\s+\|\s+(\d+)\s+\|")


def command_text(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def parse_npu_smi_snapshot(text: str) -> dict[str, Any]:
    aicore_by_card: dict[str, int] = {}
    active_cards: set[str] = set()
    for line in text.splitlines():
        aicore_match = AICORE_RE.match(line)
        if aicore_match:
            card, aicore = aicore_match.groups()
            aicore_by_card[card] = int(aicore)
            continue
        process_match = PROCESS_RE.match(line)
        if not process_match:
            continue
        card, pid = process_match.groups()
        if int(pid) > 0:
            active_cards.add(card)
    nonzero_aicore = sum(1 for value in aicore_by_card.values() if value > 0)
    return {
        "aicore_by_card": dict(sorted(aicore_by_card.items(), key=lambda item: int(item[0]))),
        "active_process_cards": sorted(active_cards, key=int),
        "active_process_card_count": len(active_cards),
        "nonzero_aicore_card_count": nonzero_aicore,
    }


def run_npu_smi() -> tuple[int, str]:
    proc = subprocess.run(["npu-smi", "info"], check=False, text=True, capture_output=True)
    return proc.returncode, proc.stdout + proc.stderr


def summarize_samples(
    samples: list[dict[str, Any]],
    *,
    run_label: str,
    command: Sequence[str],
    returncode: int | None,
    log_path: Path,
    expected_cards: int,
) -> dict[str, Any]:
    max_aicore_by_card: dict[str, int] = {}
    max_active = 0
    all_process_samples = 0
    all_aicore_samples = 0
    for sample in samples:
        max_active = max(max_active, int(sample.get("active_process_card_count") or 0))
        if int(sample.get("active_process_card_count") or 0) >= expected_cards:
            all_process_samples += 1
        if int(sample.get("nonzero_aicore_card_count") or 0) >= expected_cards:
            all_aicore_samples += 1
        aicore_by_card = sample.get("aicore_by_card", {})
        if not isinstance(aicore_by_card, dict):
            continue
        for card, value in aicore_by_card.items():
            current = int(value or 0)
            max_aicore_by_card[str(card)] = max(current, max_aicore_by_card.get(str(card), 0))
    if returncode is None:
        status = "DRY_RUN"
    elif returncode == 0 and max_active >= expected_cards:
        status = "PASS"
    else:
        status = "FAIL"
    return {
        "status": status,
        "run_label": run_label,
        "command": list(command),
        "command_text": command_text(command),
        "returncode": returncode,
        "monitor_log": str(log_path),
        "samples": len(samples),
        "expected_cards": expected_cards,
        "max_active_process_cards": max_active,
        "samples_with_8_active_process_cards": all_process_samples if expected_cards == 8 else None,
        "samples_with_8_nonzero_aicore": all_aicore_samples if expected_cards == 8 else None,
        "samples_with_expected_active_process_cards": all_process_samples,
        "samples_with_expected_nonzero_aicore": all_aicore_samples,
        "max_aicore_by_card": dict(sorted(max_aicore_by_card.items(), key=lambda item: int(item[0]))),
    }


def write_outputs(demo_root: Path, run_label: str, payload: dict[str, Any]) -> Path:
    artifacts = demo_root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    output = artifacts / f"npu_monitor_{run_label}.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a command with npu-smi sampling")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--expected-cards", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("command after -- is required")
    return args


def monitor_command(args: argparse.Namespace) -> dict[str, Any]:
    demo_root = Path(args.demo_root)
    logs = demo_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"npu-smi-{args.run_label}.log"
    samples: list[dict[str, Any]] = []
    if args.dry_run:
        payload = summarize_samples(
            samples,
            run_label=args.run_label,
            command=args.command,
            returncode=None,
            log_path=log_path,
            expected_cards=args.expected_cards,
        )
        write_outputs(demo_root, args.run_label, payload)
        return payload

    proc = subprocess.Popen(args.command)
    with log_path.open("w") as log:
        log.write(f"$ {command_text(args.command)}\n")
        log.write(f"expected_cards={args.expected_cards}\n")
        log.write(f"interval_seconds={args.interval_seconds}\n\n")
        while proc.poll() is None:
            timestamp = time.time()
            returncode, snapshot = run_npu_smi()
            sample = parse_npu_smi_snapshot(snapshot)
            sample["timestamp"] = timestamp
            sample["npu_smi_returncode"] = returncode
            samples.append(sample)
            log.write(f"--- sample {len(samples)} ts={timestamp:.6f} rc={returncode} ---\n")
            log.write(snapshot)
            if not snapshot.endswith("\n"):
                log.write("\n")
            log.flush()
            time.sleep(max(0.1, float(args.interval_seconds)))
        returncode = proc.wait()
        timestamp = time.time()
        smi_returncode, snapshot = run_npu_smi()
        sample = parse_npu_smi_snapshot(snapshot)
        sample["timestamp"] = timestamp
        sample["npu_smi_returncode"] = smi_returncode
        samples.append(sample)
        log.write(f"--- final sample {len(samples)} ts={timestamp:.6f} rc={smi_returncode} ---\n")
        log.write(snapshot)
        if not snapshot.endswith("\n"):
            log.write("\n")
    payload = summarize_samples(
        samples,
        run_label=args.run_label,
        command=args.command,
        returncode=returncode,
        log_path=log_path,
        expected_cards=args.expected_cards,
    )
    write_outputs(demo_root, args.run_label, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = monitor_command(args)
    print("NPU_MONITOR " + json.dumps(payload, sort_keys=True))
    return 0 if payload.get("status") in {"PASS", "DRY_RUN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
