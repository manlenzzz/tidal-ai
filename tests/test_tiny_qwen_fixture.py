import importlib.util
from pathlib import Path


def load_fixture():
    script = Path(__file__).resolve().parents[1] / "scripts" / "create_tiny_qwen_fixture.py"
    spec = importlib.util.spec_from_file_location("create_tiny_qwen_fixture", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_create_tiny_qwen_fixture_writes_loadable_hf_model(tmp_path):
    fixture = load_fixture()
    out = tmp_path / "TinyQwen3"

    fixture.create_fixture(out, vocab_size=128, hidden_size=32, intermediate_size=64)

    assert (out / "config.json").exists()
    assert (out / "model.safetensors").exists() or (out / "pytorch_model.bin").exists()
    assert (out / "tokenizer.json").exists()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(out), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(out), local_files_only=True)
    assert tokenizer.padding_side == "left"
    encoded = tokenizer(["hello world"], return_tensors="pt")
    generated = model.generate(**encoded, max_new_tokens=2, do_sample=False)

    assert generated.shape[-1] == encoded["input_ids"].shape[-1] + 2
