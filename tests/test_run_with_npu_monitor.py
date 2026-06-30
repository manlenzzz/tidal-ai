import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def load_monitor_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_with_npu_monitor.py"
    spec = importlib.util.spec_from_file_location("run_with_npu_monitor", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_npu_smi_snapshot_counts_active_cards_and_aicore_peaks():
    monitor = load_monitor_module()
    text = """
| 0     910B1               | OK            | 80.0        8                 0    / 0             |
| 0                         | 0000:82:00.0  | 0           0    / 0          4096 / 65536         |
| 1     910B1               | OK            | 80.0        9                 0    / 0             |
| 1                         | 0000:83:00.0  | 0           0    / 0          4096 / 65536         |
| 2     910B1               | OK            | 80.0        0                 0    / 0             |
| 2                         | 0000:84:00.0  | 0           0    / 0          4096 / 65536         |
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
| 0       0                 | 1234          | python                   | 8123                    |
| 1       0                 | 1235          | python                   | 8123                    |
| 2       0                 | 1236          | python                   | 8123                    |
"""

    sample = monitor.parse_npu_smi_snapshot(text)

    assert sample["aicore_by_card"] == {"0": 8, "1": 9, "2": 0}
    assert sample["active_process_cards"] == ["0", "1", "2"]
    assert sample["active_process_card_count"] == 3
    assert sample["nonzero_aicore_card_count"] == 2


def test_summarize_samples_preserves_all_card_activity_and_log_path(tmp_path):
    monitor = load_monitor_module()
    log_path = tmp_path / "logs" / "npu-smi-demo.log"
    samples = [
        {
            "timestamp": 1.0,
            "aicore_by_card": {str(card): card for card in range(8)},
            "active_process_cards": [str(card) for card in range(8)],
            "active_process_card_count": 8,
            "nonzero_aicore_card_count": 7,
        },
        {
            "timestamp": 2.0,
            "aicore_by_card": {str(card): card + 1 for card in range(8)},
            "active_process_cards": [str(card) for card in range(8)],
            "active_process_card_count": 8,
            "nonzero_aicore_card_count": 8,
        },
    ]

    summary = monitor.summarize_samples(
        samples,
        run_label="demo",
        command=["python", "work.py"],
        returncode=0,
        log_path=log_path,
        expected_cards=8,
    )

    assert summary["status"] == "PASS"
    assert summary["samples"] == 2
    assert summary["max_active_process_cards"] == 8
    assert summary["samples_with_8_active_process_cards"] == 2
    assert summary["samples_with_8_nonzero_aicore"] == 1
    assert summary["max_aicore_by_card"] == {
        "0": 1,
        "1": 2,
        "2": 3,
        "3": 4,
        "4": 5,
        "5": 6,
        "6": 7,
        "7": 8,
    }
    assert summary["monitor_log"] == str(log_path)
    assert summary["command_text"] == "python work.py"


def test_monitor_cli_dry_run_writes_summary(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_with_npu_monitor.py"
    demo_root = tmp_path / "demo"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--demo-root",
            str(demo_root),
            "--run-label",
            "dry",
            "--dry-run",
            "--",
            sys.executable,
            "-c",
            "print('work')",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads((demo_root / "artifacts" / "npu_monitor_dry.json").read_text())
    assert payload["status"] == "DRY_RUN"
    assert payload["samples"] == 0
    assert payload["command"][-2:] == ["-c", "print('work')"]
