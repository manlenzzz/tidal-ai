"""Helpers for propagating Ascend CANN Python paths into child processes."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import MutableMapping, Sequence


DEFAULT_ASCEND_ROOTS = (
    "/usr/local/Ascend/ascend-toolkit/latest",
    "/usr/local/Ascend/cann-8.5.1",
)


def unique_existing_paths(paths: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    existing: list[str] = []
    for value in paths:
        if not value or value in seen:
            continue
        seen.add(value)
        if Path(value).exists():
            existing.append(value)
    return existing


def acl_candidate_pythonpath_entries(env: MutableMapping[str, str] | None = None) -> list[str]:
    source = env if env is not None else os.environ
    roots = unique_existing_paths(
        [
            source.get("ASCEND_HOME_PATH", ""),
            source.get("ASCEND_TOOLKIT_HOME", ""),
            source.get("ASCEND_TOOLKIT_LATEST_HOME", ""),
            *DEFAULT_ASCEND_ROOTS,
        ]
    )
    candidates: list[str] = []
    for root in roots:
        candidates.extend(
            [
                str(Path(root) / "python" / "site-packages"),
                str(Path(root) / "opp" / "built-in" / "op_impl" / "ai_core" / "tbe"),
            ]
        )
    return unique_existing_paths(candidates)


def merge_pythonpath(current: str, additions: Sequence[str]) -> str:
    values = [part for part in current.split(os.pathsep) if part]
    for entry in additions:
        if entry not in values:
            values.append(entry)
    return os.pathsep.join(values)


def apply_acl_pythonpath_to_env(env: MutableMapping[str, str] | None = None) -> list[str]:
    target = env if env is not None else os.environ
    additions = acl_candidate_pythonpath_entries(target)
    if additions:
        target["PYTHONPATH"] = merge_pythonpath(target.get("PYTHONPATH", ""), additions)
    return additions


def apply_acl_pythonpath_to_sys_path(env: MutableMapping[str, str] | None = None) -> list[str]:
    additions = acl_candidate_pythonpath_entries(env)
    for entry in additions:
        if entry not in sys.path:
            sys.path.append(entry)
    return additions
