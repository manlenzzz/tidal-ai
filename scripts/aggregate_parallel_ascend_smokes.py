#!/usr/bin/env python
"""Aggregate only the current parallel Ascend workflow smoke artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_pid_rows(pid_file: Path) -> list[tuple[str, int, Path]]:
    rows: list[tuple[str, int, Path]] = []
    if not pid_file.exists():
        return rows
    for raw in pid_file.read_text().splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        _pid, workflow, card, _log, json_path = parts[:5]
        try:
            card_id = int(card)
        except ValueError:
            card_id = -1
        rows.append((workflow, card_id, Path(json_path)))
    return rows


def read_smoke_json(workflow: str, card: int, path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "FAIL",
            "requested_workflow": workflow,
            "card": card,
            "error_type": "MissingArtifact",
            "error": f"missing smoke JSON: {path}",
        }
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        payload.setdefault("requested_workflow", workflow)
        payload.setdefault("card", card)
        return payload
    return {
        "status": "FAIL",
        "requested_workflow": workflow,
        "card": card,
        "error_type": "InvalidArtifact",
        "error": f"smoke JSON is not an object: {path}",
    }


def write_aggregate(demo_root: Path) -> dict[str, Any]:
    artifacts = demo_root / "artifacts"
    rows = read_pid_rows(artifacts / "parallel_smoke_pids.txt")
    runs = [read_smoke_json(workflow, card, path) for workflow, card, path in rows]
    aggregate = {
        "status": "PASS" if runs and all(run.get("status") == "PASS" for run in runs) else "FAIL",
        "run_artifacts": [path.name for _workflow, _card, path in rows],
        "runs": runs,
    }
    (artifacts / "parallel_ascend_smokes.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True))
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate current parallel Ascend workflow smoke JSONs")
    parser.add_argument("demo_root")
    args = parser.parse_args()
    aggregate = write_aggregate(Path(args.demo_root))
    print("PARALLEL_ASCEND_SMOKES " + json.dumps(aggregate, sort_keys=True))
    return 0 if aggregate["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
