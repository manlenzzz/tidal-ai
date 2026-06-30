import os
import sys
from pathlib import Path

from tidal.integrations import ascend_runtime_path


def test_acl_candidate_pythonpath_entries_uses_existing_ascend_roots(tmp_path):
    root = tmp_path / "Ascend"
    python_site = root / "python" / "site-packages"
    tbe_path = root / "opp" / "built-in" / "op_impl" / "ai_core" / "tbe"
    python_site.mkdir(parents=True)
    tbe_path.mkdir(parents=True)

    entries = ascend_runtime_path.acl_candidate_pythonpath_entries({"ASCEND_HOME_PATH": str(root)})

    assert entries == [str(python_site), str(tbe_path)]


def test_merge_pythonpath_appends_missing_entries_without_duplicates():
    merged = ascend_runtime_path.merge_pythonpath(
        os.pathsep.join(["/existing", "/cann/python"]),
        ["/cann/python", "/cann/tbe"],
    )

    assert merged.split(os.pathsep) == ["/existing", "/cann/python", "/cann/tbe"]


def test_apply_acl_pythonpath_to_env_updates_pythonpath(monkeypatch, tmp_path):
    candidate = tmp_path / "cann" / "python" / "site-packages"
    candidate.mkdir(parents=True)
    monkeypatch.setattr(ascend_runtime_path, "acl_candidate_pythonpath_entries", lambda env=None: [str(candidate)])
    env = {"PYTHONPATH": "/existing"}

    added = ascend_runtime_path.apply_acl_pythonpath_to_env(env)

    assert added == [str(candidate)]
    assert env["PYTHONPATH"].split(os.pathsep) == ["/existing", str(candidate)]


def test_apply_acl_pythonpath_to_sys_path_adds_candidates(monkeypatch, tmp_path):
    candidate = tmp_path / "cann" / "python" / "site-packages"
    candidate.mkdir(parents=True)
    monkeypatch.setattr(ascend_runtime_path, "acl_candidate_pythonpath_entries", lambda env=None: [str(candidate)])
    monkeypatch.setattr(sys, "path", ["/existing"])

    added = ascend_runtime_path.apply_acl_pythonpath_to_sys_path({})

    assert added == [str(candidate)]
    assert sys.path == ["/existing", str(candidate)]
