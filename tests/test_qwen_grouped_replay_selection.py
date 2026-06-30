import json
from pathlib import Path

from tidal.reports.qwen_grouped_replay_selection import best_multicard_qwen_qpruner_grouped_replay


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_matching_monitor_uses_exact_inner_run_label_not_prefix(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    common_aggregate = {
        "pass_count": 8,
        "all_workers_memory_native": True,
        "memory_native_worker_count": 8,
        "min_code_cache_storage_reduction_pct": 50.0,
    }
    write_json(
        artifacts / "multicard_qwen_qpruner_grouped_replay_qwen_run.json",
        {
            "status": "PASS",
            "run_label": "qwen_run",
            "world_size": 8,
            "aggregate": {**common_aggregate, "best_grouped_speedup": 1.4},
            "workers": [{"command": ["python", "worker.py", "--batch-size", "32", "--iters", "1000"]}],
        },
    )
    write_json(
        artifacts / "multicard_qwen_qpruner_grouped_replay_qwen_run_fast.json",
        {
            "status": "PASS",
            "run_label": "qwen_run_fast",
            "world_size": 8,
            "aggregate": {**common_aggregate, "best_grouped_speedup": 1.2},
            "workers": [{"command": ["python", "worker.py", "--batch-size", "32", "--iters", "1000"]}],
        },
    )
    write_json(
        artifacts / "npu_monitor_qwen_run_fast_utilmon.json",
        {
            "status": "PASS",
            "run_label": "qwen_run_fast_utilmon",
            "command": [
                "python",
                "scripts/multicard_qwen_qpruner_grouped_replay.py",
                "--run-label",
                "qwen_run_fast",
            ],
            "command_text": (
                "python scripts/multicard_qwen_qpruner_grouped_replay.py "
                "--run-label qwen_run_fast"
            ),
            "max_active_process_cards": 8,
            "samples_with_8_nonzero_aicore": 55,
        },
    )

    selected = best_multicard_qwen_qpruner_grouped_replay(artifacts)

    assert selected is not None
    assert selected["_artifact_name"] == "multicard_qwen_qpruner_grouped_replay_qwen_run_fast.json"
