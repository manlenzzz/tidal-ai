import importlib.util
import sys
import types
from pathlib import Path


def load_sitecustomize_module():
    module_path = Path(__file__).resolve().parents[1] / "sitecustomize.py"
    assert module_path.exists(), f"missing sitecustomize module: {module_path}"
    spec = importlib.util.spec_from_file_location("tidal_sitecustomize_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def install_fake_metadata_shim(monkeypatch):
    calls = []
    tidal_pkg = types.ModuleType("tidal")
    tidal_pkg.__path__ = []
    integrations_pkg = types.ModuleType("tidal.integrations")
    integrations_pkg.__path__ = []
    shim_module = types.ModuleType("tidal.integrations.vllm_ascend_attention_metadata_shim")

    def install_auto():
        calls.append("install_auto")
        return {"status": "INSTALLED"}

    shim_module.install_auto = install_auto
    monkeypatch.setitem(sys.modules, "tidal", tidal_pkg)
    monkeypatch.setitem(sys.modules, "tidal.integrations", integrations_pkg)
    monkeypatch.setitem(sys.modules, "tidal.integrations.vllm_ascend_attention_metadata_shim", shim_module)
    return calls


def install_fake_ascend_runtime_path(monkeypatch):
    calls = []
    module = types.ModuleType("tidal.integrations.ascend_runtime_path")

    def apply_acl_pythonpath_to_env(env=None):
        calls.append(("env", env is not None))
        return ["/cann/python"]

    def apply_acl_pythonpath_to_sys_path(env=None):
        calls.append(("sys_path", env is not None))
        return ["/cann/python"]

    module.apply_acl_pythonpath_to_env = apply_acl_pythonpath_to_env
    module.apply_acl_pythonpath_to_sys_path = apply_acl_pythonpath_to_sys_path
    monkeypatch.setitem(sys.modules, "tidal.integrations.ascend_runtime_path", module)
    return calls


def test_sitecustomize_skips_metadata_shim_when_env_is_unset(monkeypatch):
    calls = install_fake_metadata_shim(monkeypatch)
    monkeypatch.delenv("TIDAL_VLLM_ASCEND_METADATA_SHIM", raising=False)

    load_sitecustomize_module()

    assert calls == []


def test_sitecustomize_installs_metadata_shim_when_env_is_enabled(monkeypatch):
    calls = install_fake_metadata_shim(monkeypatch)
    monkeypatch.setenv("TIDAL_VLLM_ASCEND_METADATA_SHIM", "1")

    load_sitecustomize_module()

    assert calls == ["install_auto"]


def test_sitecustomize_installs_ascend_runtime_paths_for_metadata_shim(monkeypatch):
    install_fake_metadata_shim(monkeypatch)
    calls = install_fake_ascend_runtime_path(monkeypatch)
    monkeypatch.setenv("TIDAL_VLLM_ASCEND_METADATA_SHIM", "1")

    load_sitecustomize_module()

    assert calls == [("env", True), ("sys_path", True)]
