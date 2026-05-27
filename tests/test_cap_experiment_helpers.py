import json
import subprocess
import sys

import pytest
import torch
from torch import nn

from tidal.methods.global_rank_sparsity.experiments import (
    build_text_calibration_batches,
    causal_lm_loss,
    load_calibration_texts,
    load_pruner_target_names,
    summarize_cap_run,
)
from tidal.methods.global_rank_sparsity.torch import run_cap_compression


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, **kwargs):
        max_length = kwargs.get("max_length", 8)
        values = [(ord(char) % 17) + 3 for char in text][:max_length]
        if not values:
            values = [self.eos_token_id]
        attention = [1] * len(values)
        if kwargs.get("padding") == "max_length":
            pad = max_length - len(values)
            values = values + [self.pad_token_id] * max(0, pad)
            attention = attention + [0] * max(0, pad)
        return {"input_ids": values, "attention_mask": attention}


class TinyCausalLm(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 4)
        self.proj = nn.Linear(4, 32)

    def forward(self, input_ids, attention_mask=None, labels=None):
        logits = self.proj(self.embed(input_ids))
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return type("Output", (), {"loss": loss, "logits": logits})


def test_load_pruner_target_names_from_text_json_jsonl_and_torch_state(tmp_path):
    text_path = tmp_path / "targets.txt"
    text_path.write_text("# comment\nmodel.layers.0.self_attn.q_proj.weight\n\nmodel.layers.0.mlp.down_proj.weight_mask\n")

    json_list_path = tmp_path / "targets.json"
    json_list_path.write_text(json.dumps(["a.weight", "b.weight_mask"]))

    json_map_path = tmp_path / "state.json"
    json_map_path.write_text(json.dumps({"c.weight": 1, "d.weight_mask": 0}))

    jsonl_path = tmp_path / "targets.jsonl"
    jsonl_path.write_text('{"name": "e.weight"}\n{"module": "f.weight_mask"}\n')

    torch_path = tmp_path / "state.pt"
    torch.save({"model.layers.1.self_attn.v_proj.weight": torch.ones(1)}, torch_path)

    assert load_pruner_target_names(text_path) == [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.down_proj.weight_mask",
    ]
    assert load_pruner_target_names(json_list_path) == ["a.weight", "b.weight_mask"]
    assert load_pruner_target_names(json_map_path) == ["c.weight", "d.weight_mask"]
    assert load_pruner_target_names(jsonl_path) == ["e.weight", "f.weight_mask"]
    assert load_pruner_target_names(torch_path) == ["model.layers.1.self_attn.v_proj.weight"]


def test_load_calibration_texts_and_build_batches(tmp_path):
    text_path = tmp_path / "calib.txt"
    text_path.write_text("first sample\n\nsecond sample\n")
    jsonl_path = tmp_path / "calib.jsonl"
    jsonl_path.write_text('{"text": "third"}\n{"prompt": "ignored"}\n')

    assert load_calibration_texts(text_path) == ["first sample", "second sample"]
    assert load_calibration_texts(jsonl_path, text_field="text") == ["third"]

    batches = build_text_calibration_batches(
        FakeTokenizer(),
        ["alpha", "beta"],
        max_length=6,
        batch_size=2,
        device="cpu",
    )

    assert len(batches) == 1
    batch = batches[0]
    assert set(batch) == {"input_ids", "attention_mask", "labels"}
    assert batch["input_ids"].shape == (2, 6)
    assert torch.equal(batch["input_ids"], batch["labels"])

    with pytest.raises(ValueError, match="calibration texts"):
        build_text_calibration_batches(FakeTokenizer(), [], max_length=6)


class NoPaddingTokenizer(FakeTokenizer):
    pad_token_id = None

    def __call__(self, text, **kwargs):
        if kwargs.get("padding") is not None:
            raise ValueError("tokenizer has no pad token")
        return super().__call__(text, **{key: value for key, value in kwargs.items() if key != "padding"})


def test_build_batches_pads_without_requiring_tokenizer_pad_token():
    batches = build_text_calibration_batches(
        NoPaddingTokenizer(),
        ["alpha"],
        max_length=6,
        batch_size=1,
        device="cpu",
    )

    assert batches[0]["input_ids"].shape == (1, 6)
    assert batches[0]["attention_mask"].shape == (1, 6)
    assert batches[0]["attention_mask"].sum().item() > 0


def test_causal_lm_loss_and_summary_are_json_friendly():
    torch.manual_seed(0)
    model = TinyCausalLm()
    batch = build_text_calibration_batches(FakeTokenizer(), ["alpha"], max_length=5)[0]

    loss = causal_lm_loss(model, batch)

    assert isinstance(loss, torch.Tensor)
    assert torch.isfinite(loss)

    compression_result = run_cap_compression(
        nn.Sequential(nn.Linear(4, 4, bias=False)),
        total_budget=4,
        max_iter=5,
        policy_steps=2,
        samples_per_step=1,
        seed=0,
    )
    summary = summarize_cap_run(
        compression_result,
        model_id="tiny",
        target_roles="modern",
        pruner_target_count=3,
        calibration_batches=1,
    )

    assert summary["model_id"] == "tiny"
    assert summary["target_roles"] == "modern"
    assert summary["target_count"] == len(compression_result.targets)
    assert summary["pruner_target_count"] == 3
    assert summary["calibration_batches"] == 1
    assert isinstance(summary["layers"], list)


def test_hf_cap_experiment_help_runs_without_model_download():
    script = "examples/global_rank_sparsity/hf_cap_experiment.py"
    result = subprocess.run(
        [sys.executable, script, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--pruner-targets" in result.stdout
    assert "--calibration-data" in result.stdout
    assert "--summary-json" in result.stdout
