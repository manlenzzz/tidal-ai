import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn


def load_fixture_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "create_tiny_qwen_fixture.py"
    spec = importlib.util.spec_from_file_location("create_tiny_qwen_fixture", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_accuracy_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "qwen_compression_choice_accuracy.py"
    spec = importlib.util.spec_from_file_location("qwen_compression_choice_accuracy", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TokenizerStub:
    pad_token_id = 0
    eos_token_id = 0

    def __init__(self):
        self.vocab = {"<pad>": 0, "Question": 1, "Answer": 2, "A": 3, "B": 4, "C": 5, "D": 6}

    def encode(self, text, add_special_tokens=False):
        return [self.vocab[token] for token in text.split()]


class ChoiceModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_ids, attention_mask=None):
        batch, seq_len = input_ids.shape
        logits = torch.full((batch, seq_len, 7), -8.0, device=input_ids.device)
        logits[:, :, 3] = 0.0
        logits[:, :, 4] = 5.0
        return type("Output", (), {"logits": logits})()


class CountingChoiceModel(ChoiceModel):
    def __init__(self):
        super().__init__()
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask=None):
        self.forward_calls += 1
        return super().forward(input_ids, attention_mask=attention_mask)


def test_choice_accuracy_scores_options_by_continuation_log_likelihood():
    accuracy = load_accuracy_module()
    tokenizer = TokenizerStub()
    samples = [
        {
            "id": "toy",
            "question": "Question",
            "choices": {"A": "A", "B": "B"},
            "answer": "B",
        }
    ]

    report = accuracy.evaluate_choice_accuracy(
        ChoiceModel(),
        tokenizer,
        samples,
        device=torch.device("cpu"),
        max_length=16,
    )

    assert report["accuracy"] == 1.0
    assert report["correct"] == 1
    assert report["total"] == 1
    item = report["items"][0]
    assert item["predicted"] == "B"
    assert item["is_correct"] is True
    assert item["choice_scores"]["B"] > item["choice_scores"]["A"]


def test_choice_accuracy_batches_choice_scoring_for_all_samples():
    accuracy = load_accuracy_module()
    tokenizer = TokenizerStub()
    model = CountingChoiceModel()
    samples = [
        {
            "id": "toy_1",
            "question": "Question",
            "choices": {"A": "A", "B": "B"},
            "answer": "B",
        },
        {
            "id": "toy_2",
            "question": "Question",
            "choices": {"A": "A", "B": "B"},
            "answer": "B",
        },
    ]

    report = accuracy.evaluate_choice_accuracy(
        model,
        tokenizer,
        samples,
        device=torch.device("cpu"),
        max_length=16,
    )

    assert report["accuracy"] == 1.0
    assert model.forward_calls == 1


def test_choice_accuracy_loads_jsonl_samples_with_task_groups(tmp_path):
    accuracy = load_accuracy_module()
    sample_path = tmp_path / "choice_tasks.jsonl"
    sample_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "commonsense_item",
                        "task": "commonsense",
                        "question": "Question",
                        "choices": {"A": "A", "B": "B"},
                        "answer": "B",
                    }
                ),
                json.dumps(
                    {
                        "id": "math_item",
                        "task": "math",
                        "question": "Question",
                        "choices": {"A": "A", "B": "B"},
                        "answer": "A",
                    }
                ),
            ]
        )
        + "\n"
    )

    samples = accuracy.load_samples(sample_path, sample_limit=8)

    assert [sample["id"] for sample in samples] == ["commonsense_item", "math_item"]
    assert [sample["task"] for sample in samples] == ["commonsense", "math"]
    assert samples[0]["choices"] == {"A": "A", "B": "B"}


def test_choice_accuracy_reports_task_accuracy_groups():
    accuracy = load_accuracy_module()
    tokenizer = TokenizerStub()
    samples = [
        {
            "id": "commonsense_correct",
            "task": "commonsense",
            "question": "Question",
            "choices": {"A": "A", "B": "B"},
            "answer": "B",
        },
        {
            "id": "math_wrong",
            "task": "math",
            "question": "Question",
            "choices": {"A": "A", "B": "B"},
            "answer": "A",
        },
    ]

    report = accuracy.evaluate_choice_accuracy(
        ChoiceModel(),
        tokenizer,
        samples,
        device=torch.device("cpu"),
        max_length=16,
    )
    markdown = accuracy.markdown_report(
        {
            "status": "PASS",
            "model_id": "Qwen/Qwen3-0.6B",
            "target_layer_limit": 2,
            "targeted_layers_total": 196,
            "methods": {
                "baseline": report,
                "cap": report,
                "qpruner": report,
            },
            "samples": [{"id": sample["id"], "task": sample["task"], "answer": sample["answer"]} for sample in samples],
            "task_count": 2,
        }
    )
    csv_text = accuracy.csv_report(
        {
            "methods": {
                "baseline": report,
                "cap": report,
                "qpruner": report,
            }
        }
    )

    assert report["accuracy"] == 0.5
    assert report["task_accuracy"]["commonsense"]["accuracy"] == 1.0
    assert report["task_accuracy"]["math"]["accuracy"] == 0.0
    assert report["items"][0]["task"] == "commonsense"
    assert "| baseline | commonsense | 100.000% | 1 / 1 |" in markdown
    assert "| baseline | math | 0.000% | 0 / 1 |" in markdown
    assert "method,task,status,accuracy,correct,total" in csv_text
    assert "qpruner,math,PASS,0.0,0,1" in csv_text


def test_choice_accuracy_markdown_and_csv_surface_demo_table():
    accuracy = load_accuracy_module()
    report = {
        "status": "PASS",
        "model_id": "Qwen/Qwen3-0.6B",
        "target_layer_limit": 64,
        "targeted_layers_total": 196,
        "methods": {
            "baseline": {"status": "PASS", "accuracy": 1.0, "correct": 2, "total": 2},
            "cap": {"status": "PASS", "accuracy": 0.5, "correct": 1, "total": 2},
            "qpruner": {"status": "PASS", "accuracy": 1.0, "correct": 2, "total": 2},
        },
        "samples": [{"id": "arithmetic", "answer": "B"}],
    }

    markdown = accuracy.markdown_report(report)
    csv_text = accuracy.csv_report(report)

    assert "Qwen Compression Choice Accuracy" in markdown
    assert "| baseline | PASS | 100.000% | 2 / 2 |" in markdown
    assert "| cap | PASS | 50.000% | 1 / 2 |" in markdown
    assert "Target layer limit: `64 / 196`" in markdown
    assert "method,task,status,accuracy,correct,total" in csv_text
    assert "qpruner,__overall__,PASS,1.0,2,2" in csv_text


def test_choice_accuracy_cli_runs_offline_cpu(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=96, hidden_size=32, intermediate_size=64)
    script = Path(__file__).resolve().parents[1] / "scripts" / "qwen_compression_choice_accuracy.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--demo-root",
            str(demo_root),
            "--model-id",
            "Qwen/Qwen3-0.6B",
            "--model-path",
            str(model_dir),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--target-layer-limit",
            "2",
            "--target-layer-pattern",
            "self_attn\\.(q_proj|k_proj)$",
            "--cap-budget",
            "4096",
            "--qpruner-average-bits",
            "8",
            "--max-length",
            "48",
            "--run-label",
            "cpu_test",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads((demo_root / "artifacts" / "qwen_compression_choice_accuracy_cpu_test.json").read_text())
    assert payload["status"] == "PASS"
    assert payload["target_layer_limit"] == 2
    assert payload["targeted_layers_total"] >= 2
    assert set(payload["methods"]) == {"baseline", "cap", "qpruner"}
    for method in payload["methods"].values():
        assert method["status"] == "PASS"
        assert 0.0 <= method["accuracy"] <= 1.0
        assert method["total"] == len(payload["samples"])
    assert (demo_root / "reports" / "qwen-compression-choice-accuracy-cpu_test.md").exists()
    assert (demo_root / "reports" / "qwen-compression-choice-accuracy-cpu_test.csv").exists()


def test_choice_accuracy_cli_can_run_only_selected_methods(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=96, hidden_size=32, intermediate_size=64)
    script = Path(__file__).resolve().parents[1] / "scripts" / "qwen_compression_choice_accuracy.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--demo-root",
            str(demo_root),
            "--model-id",
            "Qwen/Qwen3-0.6B",
            "--model-path",
            str(model_dir),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--target-layer-limit",
            "0",
            "--target-layer-pattern",
            "self_attn\\.(q_proj|k_proj)$",
            "--methods",
            "baseline,qpruner",
            "--qpruner-average-bits",
            "8",
            "--max-length",
            "48",
            "--run-label",
            "cpu_selected_methods",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(
        (demo_root / "artifacts" / "qwen_compression_choice_accuracy_cpu_selected_methods.json").read_text()
    )
    assert payload["status"] == "PASS"
    assert payload["target_layer_limit"] == 0
    assert set(payload["methods"]) == {"baseline", "qpruner"}
    markdown = (demo_root / "reports" / "qwen-compression-choice-accuracy-cpu_selected_methods.md").read_text()
    csv_text = (demo_root / "reports" / "qwen-compression-choice-accuracy-cpu_selected_methods.csv").read_text()
    assert "| qpruner | PASS |" in markdown
    assert "| cap |" not in markdown
    assert "cap,__overall__" not in csv_text
