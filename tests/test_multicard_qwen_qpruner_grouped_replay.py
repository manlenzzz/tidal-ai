import importlib.util
import json
from pathlib import Path


def load_replay_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "multicard_qwen_qpruner_grouped_replay.py"
    spec = importlib.util.spec_from_file_location("multicard_qwen_qpruner_grouped_replay", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def qwen_replay_payload(*, role: str, speedup: float, code_storage: float = 50.0) -> dict:
    return {
        "status": "PASS",
        "model_id": "Qwen/Qwen3-0.6B",
        "target_layer_limit": 196,
        "targeted_layers_total": 196,
        "quantized_layers": 196,
        "group_count": 7,
        "available_group_count": 7,
        "qpruner_release_packed_after_cache": True,
        "code_cache_storage_reduction_pct": code_storage,
        "targeted_storage_reduction_pct": 50.0,
        "cached_scaled_code_bytes": 0,
        "released_packed_code_bytes": 1024,
        "live_compressed_payload_storage_bytes": 784,
        "code_cache_storage_bytes": 220200960,
        "best_group": {
            "role": role,
            "module_count": 8,
            "available_module_count": 28,
            "grouped_code_cache_bytes": 25165824,
            "grouped_scaled_code_cache_bytes": 0,
            "max_abs_diff_vs_sequential_scaled_code": 0.00195312,
            "sequential_scaled_code_matmul_group": {"latency_ms": 1.12},
            "grouped_scaled_code_matmul": {
                "latency_ms": 0.95,
                "speedup_vs_sequential_scaled_code": speedup,
            },
        },
    }


def test_build_worker_specs_launches_real_qwen_replay_per_visible_card(tmp_path):
    replay = load_replay_module()

    workers = replay.build_worker_specs(
        project_root=tmp_path / "project",
        demo_root=tmp_path / "demo",
        cards=[0, 2],
        run_label="qwen3_sync",
        model_id="Qwen/Qwen3-0.6B",
        model_path=tmp_path / "models" / "Qwen3-0.6B",
        device="npu",
        dtype="float16",
        batch_size=8,
        iters=12,
        warmup=3,
        max_modules=8,
        min_group_modules=2,
        max_groups=0,
        target_layer_limit=0,
        target_layer_pattern=None,
        qpruner_average_bits=8.0,
        dense_weight_bits=16.0,
        qpruner_release_packed_after_cache=True,
        qpruner_prebuild_scaled_code_dtype_cache=False,
        seed=7,
    )

    assert [(worker.rank, worker.card, worker.visible_devices) for worker in workers] == [
        (0, 0, "0"),
        (1, 2, "2"),
    ]
    assert "qwen_qpruner_grouped_replay.py" in workers[0].command[1]
    assert "--qpruner-release-packed-after-cache" in workers[0].command
    assert "--target-layer-limit" in workers[0].command
    assert workers[0].command[workers[0].command.index("--target-layer-limit") + 1] == "0"
    assert workers[0].command[workers[0].command.index("--run-label") + 1] == "qwen3_sync_rank0_npu0"
    assert workers[1].payload_json.name == "qwen_qpruner_grouped_replay_qwen3_sync_rank1_npu2.json"


def test_write_summary_aggregates_memory_first_grouped_replay(tmp_path):
    replay = load_replay_module()
    demo_root = tmp_path / "demo"
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    artifacts.mkdir(parents=True)
    reports.mkdir()

    payload0 = qwen_replay_payload(role="q_proj", speedup=1.15)
    payload1 = qwen_replay_payload(role="down_proj", speedup=1.18)
    workers = []
    for rank, card, payload, released in [
        (0, 0, payload0, 1000.01),
        (1, 1, payload1, 1000.02),
    ]:
        workers.append(
            {
                "status": "PASS",
                "rank": rank,
                "card": card,
                "returncode": 0,
                "payload_status": "PASS",
                "started_at": 999.9 + rank * 0.01,
                "sync_start_target_ts": 1000.0,
                "released_at": released,
                "release_lag_s": released - 1000.0,
                "finished_at": 1005.0 + rank,
                "runtime_s": 5.0 + rank,
                "visible_devices": str(card),
                "command": ["python", "scripts/qwen_qpruner_grouped_replay.py"],
                "command_text": "python scripts/qwen_qpruner_grouped_replay.py",
                "log_path": f"logs/rank{rank}.log",
                "payload_json": f"artifacts/qwen_qpruner_grouped_replay_qwen3_sync_rank{rank}_npu{card}.json",
                "metrics": replay.extract_replay_metrics(payload),
            }
        )

    summary = replay.write_summary(
        workers,
        demo_root=demo_root,
        run_label="qwen3_sync",
        cards=[0, 1],
        max_start_skew_seconds=0.25,
    )

    assert summary["status"] == "PASS"
    assert summary["launched_synchronously"] is True
    assert summary["release_lag_window_s"] == 0.01
    assert summary["aggregate"]["pass_count"] == 2
    assert summary["aggregate"]["best_role"] == "down_proj"
    assert summary["aggregate"]["best_grouped_speedup"] == 1.18
    assert summary["aggregate"]["mean_grouped_speedup"] == 1.165
    assert summary["aggregate"]["all_workers_memory_native"] is True
    assert summary["aggregate"]["min_code_cache_storage_reduction_pct"] == 50.0
    assert summary["aggregate"]["released_packed_code_bytes_total"] == 2048
    assert (artifacts / "multicard_qwen_qpruner_grouped_replay_qwen3_sync.json").exists()
    markdown = (reports / "multicard-qwen-qpruner-grouped-replay-qwen3_sync.md").read_text()
    assert "Synchronized Qwen QPruner Grouped Replay" in markdown
    assert "| Best role | down_proj |" in markdown
    assert "| Memory-native workers | 2 / 2 |" in markdown
