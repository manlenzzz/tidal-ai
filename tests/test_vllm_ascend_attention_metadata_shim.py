import importlib.util
import sys
import types
from pathlib import Path


def load_shim_module():
    module_path = Path(__file__).resolve().parents[1] / "tidal" / "integrations" / "vllm_ascend_attention_metadata_shim.py"
    assert module_path.exists(), f"missing shim module: {module_path}"
    spec = importlib.util.spec_from_file_location("vllm_ascend_attention_metadata_shim", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def install_fake_attention_module(monkeypatch):
    attention_module = types.ModuleType("vllm.model_executor.layers.attention.attention")
    layer = types.SimpleNamespace(kv_cache=["kv0"])
    context = types.SimpleNamespace(
        attn_metadata=[
            {"model.layers.0.self_attn": "metadata-0"},
            {"model.layers.1.self_attn": "metadata-1"},
        ],
        no_compile_layers={"model.layers.1.self_attn": layer},
        virtual_engine=0,
        slot_mapping={"model.layers.1.self_attn": "slot-1"},
    )

    def get_forward_context():
        return context

    attention_module.get_forward_context = get_forward_context
    attention_module.get_attention_context = lambda layer_name: ("unpatched-list", layer, "kv0", "slot-1")
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.attention.attention", attention_module)
    return attention_module


def install_fake_ascend_attention_module(monkeypatch):
    attention_v1_module = types.ModuleType("vllm_ascend.attention.attention_v1")
    calls = []

    class AscendAttentionBackendImpl:
        def forward(self, layer, query, key, value, kv_cache, attn_metadata, **kwargs):
            calls.append((layer, attn_metadata, kwargs))
            return "forward-result"

    attention_v1_module.AscendAttentionBackendImpl = AscendAttentionBackendImpl
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention.attention_v1", attention_v1_module)
    return attention_v1_module, calls


def test_attention_metadata_shim_unwraps_list_of_layer_dicts(monkeypatch):
    shim = load_shim_module()
    attention_module = install_fake_attention_module(monkeypatch)

    result = shim.install()

    assert result["status"] == "INSTALLED"
    assert result["supports_list_metadata"] is True
    metadata, layer, kv_cache, slot_mapping = attention_module.get_attention_context("model.layers.1.self_attn")
    assert metadata == "metadata-1"
    assert kv_cache == "kv0"
    assert slot_mapping == "slot-1"
    assert layer is attention_module.get_forward_context().no_compile_layers["model.layers.1.self_attn"]


def test_attention_metadata_shim_preserves_dict_metadata(monkeypatch):
    shim = load_shim_module()
    attention_module = install_fake_attention_module(monkeypatch)
    attention_module.get_forward_context().attn_metadata = {"model.layers.1.self_attn": "metadata-dict"}

    shim.install()

    metadata, _, _, _ = attention_module.get_attention_context("model.layers.1.self_attn")
    assert metadata == "metadata-dict"


def test_attention_metadata_shim_is_idempotent(monkeypatch):
    shim = load_shim_module()
    attention_module = install_fake_attention_module(monkeypatch)

    first = shim.install()
    second = shim.install()

    assert first["status"] == "INSTALLED"
    assert second["status"] == "ALREADY_INSTALLED"
    metadata, _, _, _ = attention_module.get_attention_context("model.layers.1.self_attn")
    assert metadata == "metadata-1"


def test_attention_metadata_shim_patches_ascend_backend_forward(monkeypatch):
    shim = load_shim_module()
    install_fake_attention_module(monkeypatch)
    attention_v1_module, calls = install_fake_ascend_attention_module(monkeypatch)

    result = shim.install()

    assert result["backend_forward_status"] == "INSTALLED"
    backend = attention_v1_module.AscendAttentionBackendImpl()
    layer = types.SimpleNamespace(layer_name="model.layers.1.self_attn")
    output = backend.forward(
        layer,
        "query",
        "key",
        "value",
        ("kv",),
        [
            {"model.layers.0.self_attn": "metadata-0"},
            {"model.layers.1.self_attn": "metadata-1"},
        ],
        output="output",
    )
    assert output == "forward-result"
    assert calls == [(layer, "metadata-1", {"output": "output"})]


def test_attention_metadata_shim_unwraps_single_metadata_list_for_backend(monkeypatch):
    shim = load_shim_module()
    install_fake_attention_module(monkeypatch)
    attention_v1_module, calls = install_fake_ascend_attention_module(monkeypatch)

    shim.install()

    backend = attention_v1_module.AscendAttentionBackendImpl()
    layer = types.SimpleNamespace(layer_name="model.layers.1.self_attn")
    output = backend.forward(
        layer,
        "query",
        "key",
        "value",
        ("kv",),
        ["metadata-single"],
        output="output",
    )
    assert output == "forward-result"
    assert calls == [(layer, "metadata-single", {"output": "output"})]


def test_attention_metadata_shim_can_defer_until_vllm_import(monkeypatch, tmp_path):
    shim = load_shim_module()
    module_name = "vllm.model_executor.layers.attention.attention"
    for name in list(sys.modules):
        if name == "vllm" or name.startswith("vllm."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    package_dir = tmp_path / "vllm" / "model_executor" / "layers" / "attention"
    package_dir.mkdir(parents=True)
    for package in [
        tmp_path / "vllm",
        tmp_path / "vllm" / "model_executor",
        tmp_path / "vllm" / "model_executor" / "layers",
        package_dir,
    ]:
        (package / "__init__.py").write_text("")
    (package_dir / "attention.py").write_text(
        "\n".join(
            [
                "import types",
                "layer = types.SimpleNamespace(kv_cache=['kv0'])",
                "context = types.SimpleNamespace(",
                "    attn_metadata=[{'layer.0': 'metadata-0'}, {'layer.1': 'metadata-1'}],",
                "    no_compile_layers={'layer.1': layer},",
                "    virtual_engine=0,",
                "    slot_mapping={'layer.1': 'slot-1'},",
                ")",
                "def get_forward_context():",
                "    return context",
                "def get_attention_context(layer_name):",
                "    return ('unpatched', layer, 'kv0', 'slot-1')",
            ]
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    result = shim.install_auto()

    assert result["status"] == "DEFERRED"
    assert module_name not in sys.modules
    attention_module = importlib.import_module(module_name)
    metadata, _, kv_cache, slot_mapping = attention_module.get_attention_context("layer.1")
    assert metadata == "metadata-1"
    assert kv_cache == "kv0"
    assert slot_mapping == "slot-1"


def test_attention_metadata_shim_can_defer_backend_patch_until_backend_import(monkeypatch, tmp_path):
    shim = load_shim_module()
    module_name = "vllm_ascend.attention.attention_v1"
    for name in list(sys.modules):
        if name == "vllm_ascend" or name.startswith("vllm_ascend."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    package_dir = tmp_path / "vllm_ascend" / "attention"
    package_dir.mkdir(parents=True)
    for package in [tmp_path / "vllm_ascend", package_dir]:
        (package / "__init__.py").write_text("")
    (package_dir / "attention_v1.py").write_text(
        "\n".join(
            [
                "calls = []",
                "class AscendAttentionBackendImpl:",
                "    def forward(self, layer, query, key, value, kv_cache, attn_metadata, **kwargs):",
                "        calls.append((layer, attn_metadata, kwargs))",
                "        return 'forward-result'",
            ]
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    result = shim.install_auto()

    assert result["backend_forward_status"] == "DEFERRED"
    assert module_name not in sys.modules
    attention_v1_module = importlib.import_module(module_name)
    assert getattr(attention_v1_module.AscendAttentionBackendImpl.forward, "_tidal_list_metadata_shim", False)
    backend = attention_v1_module.AscendAttentionBackendImpl()
    layer = types.SimpleNamespace(layer_name="model.layers.1.self_attn")
    output = backend.forward(
        layer,
        "query",
        "key",
        "value",
        ("kv",),
        [
            {"model.layers.0.self_attn": "metadata-0"},
            {"model.layers.1.self_attn": "metadata-1"},
        ],
        output="output",
    )
    assert output == "forward-result"
    assert attention_v1_module.calls == [(layer, "metadata-1", {"output": "output"})]
