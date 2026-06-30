import importlib.util
import json
from pathlib import Path


def load_aggregate_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_parallel_ascend_smokes.py"
    spec = importlib.util.spec_from_file_location("aggregate_parallel_ascend_smokes", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_aggregate_uses_pid_file_jsons_not_historical_npu_artifacts(tmp_path):
    aggregate_smokes = load_aggregate_module()
    demo_root = tmp_path / "demo"
    artifacts = demo_root / "artifacts"
    artifacts.mkdir(parents=True)

    cap = artifacts / "cap_npu0.json"
    qpruner = artifacts / "qpruner_npu1.json"
    rankadaptor = artifacts / "rankadaptor_npu2.json"
    write_json(cap, {"status": "PASS", "requested_workflow": "cap", "workflows": ["cap"], "cap": {"status": "PASS"}})
    write_json(
        qpruner,
        {"status": "PASS", "requested_workflow": "qpruner", "workflows": ["qpruner"], "qpruner": {"status": "PASS"}},
    )
    write_json(
        rankadaptor,
        {
            "status": "PASS",
            "requested_workflow": "rankadaptor",
            "workflows": ["rankadaptor"],
            "rankadaptor": {"status": "PASS"},
        },
    )
    write_json(artifacts / "vllm_serving_benchmark_old_npu.json", {"status": "FAIL", "backend": "vllm"})
    (artifacts / "parallel_smoke_pids.txt").write_text(
        "\n".join(
            [
                f"111 cap 0 {demo_root / 'logs' / 'cap_npu0.log'} {cap}",
                f"222 qpruner 1 {demo_root / 'logs' / 'qpruner_npu1.log'} {qpruner}",
                f"333 rankadaptor 2 {demo_root / 'logs' / 'rankadaptor_npu2.log'} {rankadaptor}",
            ]
        )
        + "\n"
    )

    aggregate = aggregate_smokes.write_aggregate(demo_root)

    assert aggregate["status"] == "PASS"
    assert aggregate["run_artifacts"] == ["cap_npu0.json", "qpruner_npu1.json", "rankadaptor_npu2.json"]
    assert [run["requested_workflow"] for run in aggregate["runs"]] == ["cap", "qpruner", "rankadaptor"]
    assert (artifacts / "parallel_ascend_smokes.json").exists()


def test_aggregate_reports_missing_pid_json_as_failure(tmp_path):
    aggregate_smokes = load_aggregate_module()
    demo_root = tmp_path / "demo"
    artifacts = demo_root / "artifacts"
    artifacts.mkdir(parents=True)
    missing = artifacts / "rankadaptor_npu2.json"
    (artifacts / "parallel_smoke_pids.txt").write_text(
        f"333 rankadaptor 2 {demo_root / 'logs' / 'rankadaptor_npu2.log'} {missing}\n"
    )

    aggregate = aggregate_smokes.write_aggregate(demo_root)

    assert aggregate["status"] == "FAIL"
    assert aggregate["runs"] == [
        {
            "status": "FAIL",
            "requested_workflow": "rankadaptor",
            "card": 2,
            "error_type": "MissingArtifact",
            "error": f"missing smoke JSON: {missing}",
        }
    ]
