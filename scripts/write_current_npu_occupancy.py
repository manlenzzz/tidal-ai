#!/usr/bin/env python
"""Write a normalized NPU occupancy artifact from inspected runtime state."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def resolved_status(args: argparse.Namespace) -> str:
    if args.status:
        return str(args.status)
    return "FREE" if int(args.occupied_cards) == 0 else "BLOCKED"


def build_payload(args: argparse.Namespace) -> dict[str, object]:
    return {
        "status": resolved_status(args),
        "reason": args.reason,
        "occupied_cards": args.occupied_cards,
        "total_cards": args.total_cards,
        "container": {
            "id": args.container_id,
            "name": args.container_name,
            "image": args.container_image,
        },
        "service": {
            "model": args.service_model,
            "command": args.service_command,
            "tensor_parallel_size": args.tensor_parallel_size,
        },
        "policy": args.policy,
        "next_action": args.next_action,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the current NPU occupancy artifact")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--status")
    parser.add_argument("--reason", default="npu_memory_occupied_by_external_vllm_service")
    parser.add_argument("--occupied-cards", type=int, required=True)
    parser.add_argument("--total-cards", type=int, required=True)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--service-model", required=True)
    parser.add_argument("--service-command", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--policy", default="do_not_kill_unrelated_container")
    parser.add_argument(
        "--next-action",
        default="Wait for the external vLLM service to release cards, then rerun long-decode serving benchmarks.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts = Path(args.demo_root) / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    payload = build_payload(args)
    output = artifacts / "current_npu_occupancy.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print("CURRENT_NPU_OCCUPANCY " + json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
