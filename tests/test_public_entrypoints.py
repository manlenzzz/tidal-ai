import json
import subprocess
import sys

import torch
from torch import nn


class FakeTokenizer:
    pad_token_id = None
    eos_token_id = 2

    def __call__(self, text, **kwargs):
        max_length = kwargs.get("max_length", 8)
        values = [(ord(char) % 17) + 3 for char in text][:max_length]
        if not values:
            values = [self.eos_token_id]
        return {"input_ids": values, "attention_mask": [1] * len(values)}


def test_public_targets_data_and_reports_reuse_common_helpers(tmp_path):
    from tidal.data import build_text_calibration_batches, causal_lm_loss, load_calibration_texts
    from tidal.reports import save_summary_json
    from tidal.targets import build_pruned_module_name_filter, load_pruner_target_names

    target_path = tmp_path / "targets.txt"
    target_path.write_text("# ignore\nbase_model.model.model.layers.0.self_attn.q_proj.weight\n")
    calibration_path = tmp_path / "calib.jsonl"
    calibration_path.write_text('{"text": "hello"}\n{"prompt": "skip"}\n')

    names = load_pruner_target_names(target_path)
    name_filter = build_pruned_module_name_filter(names, target_roles="modern")
    texts = load_calibration_texts(calibration_path)
    batches = build_text_calibration_batches(FakeTokenizer(), texts, max_length=6)

    assert names == ["base_model.model.model.layers.0.self_attn.q_proj.weight"]
    assert name_filter("model.layers.0.self_attn.q_proj")
    assert texts == ["hello"]
    assert set(batches[0]) == {"input_ids", "attention_mask", "labels"}
    assert batches[0]["input_ids"].shape == (1, 6)
    assert callable(causal_lm_loss)

    output = save_summary_json({"method": "cap", "target_count": 1}, tmp_path / "run")
    assert output.name == "summary.json"
    assert json.loads(output.read_text())["method"] == "cap"


def test_cap_compress_workflow_accepts_in_memory_model_and_saves_summary(tmp_path):
    from tidal.workflows.compression import cap_compress

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4, bias=False))

    result = cap_compress(
        model=model,
        budget=4,
        max_iter=5,
        policy_steps=2,
        samples_per_step=1,
        seed=0,
    )
    summary_path = result.save(tmp_path / "cap-run")

    assert result.model is not model
    assert result.summary["method"] == "cap"
    assert result.summary["target_count"] == 1
    assert result.summary["parameter_count"] <= 4
    assert summary_path == tmp_path / "cap-run" / "summary.json"
    assert json.loads(summary_path.read_text())["method"] == "cap"


def test_tidal_cli_compress_cap_help_and_console_script_metadata():
    import tomllib
    from pathlib import Path

    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    assert pyproject["project"]["scripts"]["tidal"] == "tidal.cli.main:main"

    result = subprocess.run(
        [sys.executable, "-m", "tidal.cli.main", "compress", "cap", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--model-id" in result.stdout
    assert "--pruner-targets" in result.stdout
    assert "--calibration-data" in result.stdout
    assert "--output" in result.stdout


def test_existing_cap_experiment_helpers_stay_importable():
    from tidal.data import build_text_calibration_batches
    from tidal.methods.global_rank_sparsity.experiments import (
        build_text_calibration_batches as compat_build_text_calibration_batches,
    )
    from tidal.targets import load_pruner_target_names
    from tidal.methods.global_rank_sparsity.experiments import load_pruner_target_names as compat_load_pruner_target_names

    assert compat_build_text_calibration_batches is build_text_calibration_batches
    assert compat_load_pruner_target_names is load_pruner_target_names
