import json
import subprocess
import sys

import torch
from torch import nn


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def test_qpruner_compress_accepts_importances_and_saves_summary(tmp_path):
    from tidal.methods.qpruner.torch import QuantizedLinear
    from tidal.workflows import qpruner_compress

    torch.manual_seed(0)
    model = TinyBlock()

    result = qpruner_compress(
        model=model,
        importances={"fc1": 2.0, "fc2": 0.1},
        candidate_bits=(2, 4, 8),
        max_average_bits=3.5,
        seed=0,
    )
    summary_path = result.save(tmp_path / "qpruner-run")

    assert result.model is not model
    assert isinstance(result.model.fc1, QuantizedLinear)
    assert isinstance(result.model.fc2, QuantizedLinear)
    assert isinstance(model.fc1, nn.Linear)
    assert result.summary["method"] == "qpruner"
    assert result.summary["target_count"] == 2
    assert result.summary["bitwidths"] == {"fc1": 4, "fc2": 2}
    assert result.summary["memory_bits"] == 60
    assert result.summary["average_bits"] == 60 / 18
    assert result.summary["importances"] == {"fc1": 2.0, "fc2": 0.1}
    assert summary_path == tmp_path / "qpruner-run" / "summary.json"
    assert json.loads(summary_path.read_text())["method"] == "qpruner"


def test_qpruner_compress_skips_calibration_loading_when_importances_are_given():
    from tidal.workflows.compression import qpruner_compress

    model = TinyBlock()

    result = qpruner_compress(
        model=model,
        calibration_batches=[torch.randn(1, 4)],
        calibration_data=["this text should not require a tokenizer"],
        importances={"fc1": 2.0, "fc2": 0.1},
        candidate_bits=(2, 4, 8),
        max_average_bits=3.5,
    )

    assert result.summary["method"] == "qpruner"
    assert "calibration_batches" not in result.summary


def test_qpruner_compress_combines_pruner_targets_with_importances(tmp_path):
    from tidal.workflows.compression import qpruner_compress

    target_path = tmp_path / "targets.txt"
    target_path.write_text("base_model.model.fc1.weight\n")
    model = TinyBlock()

    result = qpruner_compress(
        model=model,
        pruner_targets=target_path,
        importances={"fc1": 1.0},
        candidate_bits=(2, 4),
        max_average_bits=2.5,
    )

    assert result.summary["target_count"] == 1
    assert result.summary["pruner_target_count"] == 1
    assert result.summary["bitwidths"] == {"fc1": 2}


def test_tidal_cli_compress_qpruner_help_lists_public_options():
    result = subprocess.run(
        [sys.executable, "-m", "tidal.cli.main", "compress", "qpruner", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--model-id" in result.stdout
    assert "--pruner-targets" in result.stdout
    assert "--calibration-data" in result.stdout
    assert "--candidate-bits" in result.stdout
    assert "--max-average-bits" in result.stdout
    assert "--max-memory-bits" in result.stdout
    assert "--output" in result.stdout
