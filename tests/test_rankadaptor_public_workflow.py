import json
import subprocess
import sys

from torch import nn


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def test_rankadaptor_adapt_builds_peft_model_and_saves_summary(tmp_path):
    from tidal.workflows import rankadaptor_adapt

    model = TinyBlock()

    result = rankadaptor_adapt(
        model=model,
        sensitivities={"fc1": 2.0, "fc2": 0.2},
        budget=19,
        min_rank=1,
        max_rank=2,
        rank_step=1,
        alpha_multiplier=2,
    )
    summary_path = result.save(tmp_path / "rankadaptor-run")

    assert result.model is not model
    assert not hasattr(model.fc1, "lora_A")
    assert result.model.base_model.model.fc1.lora_A["default"].weight.shape == (2, 4)
    assert result.model.base_model.model.fc2.lora_A["default"].weight.shape == (1, 3)
    assert result.summary["method"] == "rankadaptor"
    assert result.summary["rank_config"] == {"fc1": 2, "fc2": 1}
    assert result.summary["adapter_cost"] == 19
    assert result.summary["profile_count"] == 2
    assert summary_path == tmp_path / "rankadaptor-run" / "summary.json"
    assert json.loads(summary_path.read_text())["method"] == "rankadaptor"


def test_rankadaptor_adapt_canonicalizes_sensitivity_names():
    from tidal.workflows.adaptation import rankadaptor_adapt

    model = TinyBlock()

    result = rankadaptor_adapt(
        model=model,
        sensitivities={"base_model.model.fc1.weight": 2.0, "fc2": 0.2},
        budget=19,
        min_rank=1,
        max_rank=2,
        rank_step=1,
        apply_peft=False,
    )

    assert result.summary["rank_config"] == {"fc1": 2, "fc2": 1}
    assert result.summary["profiles"][0]["sensitivity"] == 2.0


def test_rankadaptor_adapt_combines_pruner_targets_with_profiles(tmp_path):
    from tidal.workflows.adaptation import rankadaptor_adapt

    target_path = tmp_path / "targets.txt"
    target_path.write_text("base_model.model.fc1.weight\n")
    model = TinyBlock()

    result = rankadaptor_adapt(
        model=model,
        pruner_targets=target_path,
        sensitivities={"fc1": 1.0},
        budget=7,
        min_rank=1,
        max_rank=2,
        rank_step=1,
        apply_peft=False,
    )

    assert result.model is model
    assert result.summary["rank_config"] == {"fc1": 1}
    assert result.summary["profile_count"] == 1
    assert result.summary["pruner_target_count"] == 1


def test_tidal_cli_adapt_rankadaptor_help_lists_public_options():
    result = subprocess.run(
        [sys.executable, "-m", "tidal.cli.main", "adapt", "rankadaptor", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--model-id" in result.stdout
    assert "--budget" in result.stdout
    assert "--sensitivities" in result.stdout
    assert "--min-rank" in result.stdout
    assert "--max-rank" in result.stdout
    assert "--pruner-targets" in result.stdout
    assert "--output" in result.stdout
