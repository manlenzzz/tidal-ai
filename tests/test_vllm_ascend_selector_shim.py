import importlib
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def reload_shim(monkeypatch, selector_module):
    attention_module = types.SimpleNamespace(selector=selector_module)
    v1_module = types.SimpleNamespace(attention=attention_module)
    vllm_module = types.SimpleNamespace(v1=v1_module)
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.v1", v1_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.attention", attention_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.attention.selector", selector_module)
    monkeypatch.setitem(
        sys.modules,
        "vllm.config.cache",
        types.SimpleNamespace(CacheDType=type(None)),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.attention.backend",
        types.SimpleNamespace(AttentionBackend=object, AttentionType=types.SimpleNamespace(DECODER="decoder")),
    )
    module_name = "tidal.integrations.vllm_ascend_selector_shim"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_selector_shim_extends_vllm_ascend_patch_signature(monkeypatch):
    calls = []

    def cached_get_attn_backend(**kwargs):
        calls.append(kwargs)
        return "backend"

    selector = types.SimpleNamespace(_cached_get_attn_backend=cached_get_attn_backend)
    shim = reload_shim(monkeypatch, selector)

    monkeypatch.setitem(
        sys.modules,
        "vllm.config",
        types.SimpleNamespace(
            get_current_vllm_config=lambda: types.SimpleNamespace(
                attention_config=types.SimpleNamespace(backend="ASCEND"),
                cache_config=types.SimpleNamespace(user_specified_block_size=True, block_size=128),
            )
        ),
    )

    result = shim.install()

    assert result["status"] == "INSTALLED"
    assert "use_compress" in selector.AttentionSelectorConfig._fields
    assert "use_per_head_quant_scales" in selector.AttentionSelectorConfig._fields

    backend = selector.get_attn_backend(
        64,
        "float16",
        None,
        use_mla=False,
        use_compress=True,
        use_per_head_quant_scales=True,
        num_heads=8,
    )

    assert backend == "backend"
    assert calls[0]["backend"] == "ASCEND"
    assert calls[0]["num_heads"] == 8
    config = calls[0]["attn_selector_config"]
    assert config.block_size == 128
    assert config.use_compress is True
    assert config.use_per_head_quant_scales is True
