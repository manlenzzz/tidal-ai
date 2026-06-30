import importlib.util
from pathlib import Path


def load_smoke_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "ascend_npu_workflow_smoke.py"
    spec = importlib.util.spec_from_file_location("ascend_npu_workflow_smoke", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_select_workflows_all_expands_in_stable_order():
    smoke = load_smoke_module()

    assert smoke.select_workflows("all") == ("cap", "qpruner", "rankadaptor")


def test_select_workflows_accepts_single_workflow():
    smoke = load_smoke_module()

    assert smoke.select_workflows("qpruner") == ("qpruner",)


def test_select_workflows_rejects_unknown_workflow():
    smoke = load_smoke_module()

    try:
        smoke.select_workflows("unknown")
    except ValueError as exc:
        assert "unknown workflow" in str(exc)
    else:
        raise AssertionError("expected ValueError")
