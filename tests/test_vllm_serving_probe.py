import importlib.util
import json
import os
import subprocess
import types
from pathlib import Path
from typing import NamedTuple


def load_probe_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "probe_vllm_serving_exports.py"
    spec = importlib.util.spec_from_file_location("probe_vllm_serving_exports", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_markdown_report_lists_vllm_skip_reason():
    probe = load_probe_module()

    text = probe.markdown_report(
        {
            "status": "SKIP",
            "vllm_available": False,
            "exports": {
                "cap": {"path": "/demo/cap", "exists": True, "transformers_load": "PASS", "vllm_load": "SKIP"},
                "qpruner": {
                    "path": "/demo/qpruner",
                    "exists": True,
                    "transformers_load": "PASS",
                    "vllm_load": "SKIP",
                },
            },
            "issues": ["vllm import failed: No module named 'vllm'"],
        }
    )

    assert "vLLM Serving Export Probe" in text
    assert "| CAP | True | PASS | SKIP |" in text
    assert "vllm import failed" in text


def test_probe_exports_skips_when_vllm_missing(tmp_path, monkeypatch):
    probe = load_probe_module()
    demo_root = tmp_path / "demo"
    for method in ("cap", "qpruner"):
        export_dir = demo_root / "serving_exports" / "run" / method
        export_dir.mkdir(parents=True)
        (export_dir / "config.json").write_text("{}")

    monkeypatch.setattr(probe, "import_vllm", lambda: (None, "vllm import failed: No module named 'vllm'"))

    report = probe.probe_exports(demo_root=demo_root, run_label="run")

    assert report["status"] == "SKIP"
    assert report["vllm_available"] is False
    assert report["exports"]["cap"]["exists"] is True
    assert report["exports"]["cap"]["vllm_load"] == "SKIP"
    assert "vllm import failed" in report["issues"][0]


def test_probe_exports_records_vllm_load_error(tmp_path, monkeypatch):
    probe = load_probe_module()
    demo_root = tmp_path / "demo"
    for method in ("cap", "qpruner"):
        export_dir = demo_root / "serving_exports" / "run" / method
        export_dir.mkdir(parents=True)
        (export_dir / "config.json").write_text("{}")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout="",
            stderr="ModuleNotFoundError: No module named 'acl'\n",
        )

    monkeypatch.setattr(probe, "import_vllm", lambda: (types.ModuleType("vllm"), None))
    monkeypatch.setattr(probe.subprocess, "run", fake_run)

    report = probe.probe_exports(demo_root=demo_root, run_label="run")

    assert report["status"] == "FAIL"
    assert report["vllm_available"] is True
    assert report["exports"]["cap"]["vllm_load"] == "FAIL"
    assert any("No module named 'acl'" in issue for issue in report["issues"])


def test_probe_main_writes_artifact(tmp_path, monkeypatch):
    probe = load_probe_module()
    demo_root = tmp_path / "demo"
    monkeypatch.setattr(probe, "import_vllm", lambda: (None, "vllm import failed: No module named 'vllm'"))

    rc = probe.main(["--demo-root", str(demo_root), "--run-label", "missing"])

    assert rc == 0
    payload = json.loads((demo_root / "artifacts" / "vllm_serving_probe_missing.json").read_text())
    assert payload["status"] == "SKIP"
    assert (demo_root / "reports" / "vllm-serving-probe-missing.md").exists()


def test_probe_main_returns_zero_when_probe_records_failure(tmp_path, monkeypatch):
    probe = load_probe_module()
    demo_root = tmp_path / "demo"

    monkeypatch.setattr(
        probe,
        "probe_exports",
        lambda *,
        demo_root,
        run_label,
        export_run_label=None,
        gpu_memory_utilization=0.05,
        preload_vllm_ascend_patch=False,
        preload_vllm_ascend_selector_shim=False: {
            "status": "FAIL",
            "vllm_available": True,
            "run_label": run_label,
            "export_run_label": export_run_label or run_label,
            "gpu_memory_utilization": gpu_memory_utilization,
            "preload_vllm_ascend_patch": preload_vllm_ascend_patch,
            "preload_vllm_ascend_selector_shim": preload_vllm_ascend_selector_shim,
            "exports": {
                "cap": {"path": "/cap", "exists": True, "transformers_load": "PASS", "vllm_load": "FAIL"},
                "qpruner": {"path": "/q", "exists": True, "transformers_load": "PASS", "vllm_load": "FAIL"},
            },
            "issues": ["vLLM load failed for /cap: ModuleNotFoundError: No module named 'acl'"],
        },
    )

    rc = probe.main(["--demo-root", str(demo_root), "--run-label", "failed"])

    assert rc == 0
    payload = json.loads((demo_root / "artifacts" / "vllm_serving_probe_failed.json").read_text())
    assert payload["status"] == "FAIL"
    assert "acl" in payload["issues"][0]


def test_vllm_load_check_captures_subprocess_stderr(tmp_path, monkeypatch):
    probe = load_probe_module()
    export_dir = tmp_path / "cap"
    export_dir.mkdir()

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout="engine starting\n",
            stderr="Traceback (most recent call last):\nModuleNotFoundError: No module named 'acl'\n",
        )

    monkeypatch.setattr(probe.subprocess, "run", fake_run)

    status, issue = probe.vllm_load_check(types.ModuleType("vllm"), export_dir)

    assert status == "FAIL"
    assert issue is not None
    assert "No module named 'acl'" in issue
    assert "engine starting" in issue


def test_vllm_load_code_uses_configurable_memory_utilization(tmp_path):
    probe = load_probe_module()

    code = probe.vllm_load_code(tmp_path, gpu_memory_utilization=0.05)

    assert "gpu_memory_utilization=0.05" in code


def test_probe_exports_records_memory_pressure_issue(tmp_path, monkeypatch):
    probe = load_probe_module()
    demo_root = tmp_path / "demo"
    for method in ("cap", "qpruner"):
        export_dir = demo_root / "serving_exports" / "run" / method
        export_dir.mkdir(parents=True)
        (export_dir / "config.json").write_text("{}")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout="",
            stderr="ValueError: Free memory on device (5.06/60.96 GiB) on startup is less than desired GPU memory\n",
        )

    monkeypatch.setattr(probe, "import_vllm", lambda: (types.ModuleType("vllm"), None))
    monkeypatch.setattr(probe.subprocess, "run", fake_run)

    report = probe.probe_exports(demo_root=demo_root, run_label="run", gpu_memory_utilization=0.05)

    assert report["status"] == "FAIL"
    assert report["gpu_memory_utilization"] == 0.05
    assert any(item["id"] == "vllm_device_memory_pressure" for item in report["issue_classes"])


def test_probe_exports_records_vllm_attention_selector_api_mismatch(tmp_path, monkeypatch):
    probe = load_probe_module()
    demo_root = tmp_path / "demo"
    for method in ("cap", "qpruner"):
        export_dir = demo_root / "serving_exports" / "run" / method
        export_dir.mkdir(parents=True)
        (export_dir / "config.json").write_text("{}")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout="",
            stderr=(
                "TypeError: AttentionSelectorConfig.__new__() got an unexpected keyword argument "
                "'use_per_head_v1_attention'\n"
            ),
        )

    monkeypatch.setattr(probe, "import_vllm", lambda: (types.ModuleType("vllm"), None))
    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    monkeypatch.setattr(
        probe,
        "package_diagnostics",
        lambda: {
            "vllm": {"version": "0.18.0+empty", "file": "/vllm-workspace/vllm/vllm/__init__.py"},
            "vllm-ascend": {"version": "0.1.dev2924+ge5f7e2f43", "file": None},
        },
    )

    report = probe.probe_exports(demo_root=demo_root, run_label="run")

    assert any(item["id"] == "vllm_attention_selector_api_mismatch" for item in report["issue_classes"])
    assert report["package_diagnostics"]["vllm"]["version"] == "0.18.0+empty"
    assert report["package_diagnostics"]["vllm-ascend"]["version"] == "0.1.dev2924+ge5f7e2f43"


def test_probe_exports_records_vllm_patch_signature_mismatch(tmp_path, monkeypatch):
    probe = load_probe_module()
    demo_root = tmp_path / "demo"
    for method in ("cap", "qpruner"):
        export_dir = demo_root / "serving_exports" / "run" / method
        export_dir.mkdir(parents=True)
        (export_dir / "config.json").write_text("{}")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout="",
            stderr="TypeError: get_attn_backend() got an unexpected keyword argument 'use_per_head_quant_scales'\n",
        )

    monkeypatch.setattr(probe, "import_vllm", lambda: (types.ModuleType("vllm"), None))
    monkeypatch.setattr(probe.subprocess, "run", fake_run)

    report = probe.probe_exports(
        demo_root=demo_root,
        run_label="run",
        preload_vllm_ascend_patch=True,
    )

    assert report["preload_vllm_ascend_patch"] is True
    assert any(item["id"] == "vllm_attention_selector_api_mismatch" for item in report["issue_classes"])


def test_probe_exports_can_use_distinct_output_and_export_run_labels(tmp_path, monkeypatch):
    probe = load_probe_module()
    demo_root = tmp_path / "demo"
    for method in ("cap", "qpruner"):
        export_dir = demo_root / "serving_exports" / "exported" / method
        export_dir.mkdir(parents=True)
        (export_dir / "config.json").write_text("{}")

    monkeypatch.setattr(probe, "import_vllm", lambda: (types.ModuleType("vllm"), None))
    monkeypatch.setattr(
        probe,
        "vllm_load_check",
        lambda vllm, path, **kwargs: ("PASS", None) if "exported" in str(path) else ("FAIL", str(path)),
    )

    report = probe.probe_exports(demo_root=demo_root, run_label="probe-output", export_run_label="exported")

    assert report["status"] == "PASS"
    assert report["run_label"] == "probe-output"
    assert report["export_run_label"] == "exported"
    assert "/serving_exports/exported/cap" in report["exports"]["cap"]["path"]


def test_vllm_load_code_can_preload_vllm_ascend_patch(tmp_path):
    probe = load_probe_module()

    code = probe.vllm_load_code(
        tmp_path,
        gpu_memory_utilization=0.05,
        preload_vllm_ascend_patch=True,
    )

    assert "import vllm_ascend.patch.platform" in code
    assert code.index("import vllm_ascend.patch.platform") < code.index("vllm.LLM(")


def test_vllm_load_code_can_preload_project_selector_shim(tmp_path):
    probe = load_probe_module()

    code = probe.vllm_load_code(
        tmp_path,
        gpu_memory_utilization=0.05,
        preload_vllm_ascend_patch=True,
        preload_vllm_ascend_selector_shim=True,
    )

    assert "import vllm_ascend.patch.platform" in code
    assert "from tidal.integrations.vllm_ascend_selector_shim import install" in code
    assert "install()" in code
    assert code.index("import vllm_ascend.patch.platform") < code.index("install()") < code.index("vllm.LLM(")


def test_vllm_load_code_is_valid_python_with_optional_preloads(tmp_path):
    probe = load_probe_module()

    code = probe.vllm_load_code(
        tmp_path,
        gpu_memory_utilization=0.05,
        preload_vllm_ascend_patch=True,
        preload_vllm_ascend_selector_shim=True,
    )

    compile(code, "<vllm-load-check>", "exec")


def test_probe_exports_passes_selector_shim_flag_to_load_checks(tmp_path, monkeypatch):
    probe = load_probe_module()
    demo_root = tmp_path / "demo"
    captured = []
    for method in ("cap", "qpruner"):
        export_dir = demo_root / "serving_exports" / "run" / method
        export_dir.mkdir(parents=True)
        (export_dir / "config.json").write_text("{}")

    def fake_load_check(vllm, path, **kwargs):
        captured.append(kwargs)
        return "PASS", None

    monkeypatch.setattr(probe, "import_vllm", lambda: (types.ModuleType("vllm"), None))
    monkeypatch.setattr(probe, "vllm_load_check", fake_load_check)

    report = probe.probe_exports(
        demo_root=demo_root,
        run_label="run",
        preload_vllm_ascend_patch=True,
        preload_vllm_ascend_selector_shim=True,
    )

    assert report["status"] == "PASS"
    assert report["preload_vllm_ascend_selector_shim"] is True
    assert captured
    assert all(row["preload_vllm_ascend_selector_shim"] is True for row in captured)


def test_package_diagnostics_reports_missing_distributions(monkeypatch):
    probe = load_probe_module()

    def fake_version(name):
        if name == "vllm":
            return "0.18.0+empty"
        raise probe.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(probe.importlib.metadata, "version", fake_version)
    monkeypatch.setattr(probe.importlib.util, "find_spec", lambda name: None)

    diagnostics = probe.package_diagnostics()

    assert diagnostics["vllm"]["version"] == "0.18.0+empty"
    assert diagnostics["vllm-ascend"]["status"] == "MISSING"


def test_attention_selector_api_diagnostics_compares_vllm_and_patch_signatures(monkeypatch):
    probe = load_probe_module()

    class VllmConfig(NamedTuple):
        head_size: int
        dtype: object
        kv_cache_dtype: str | None
        block_size: int | None
        use_sparse: bool = False
        use_per_head_quant_scales: bool = False

    class PatchConfig(NamedTuple):
        head_size: int
        dtype: object
        kv_cache_dtype: str | None
        block_size: int | None
        use_sparse: bool = False
        use_compress: bool = False

    def vllm_get_attn_backend(
        head_size,
        dtype,
        kv_cache_dtype,
        use_sparse=False,
        use_per_head_quant_scales=False,
        num_heads=None,
    ):
        return None

    def patch_get_attn_backend(
        head_size,
        dtype,
        kv_cache_dtype,
        block_size=None,
        use_sparse=False,
        use_compress=False,
    ):
        return None

    modules = {
        "vllm.v1.attention.selector": types.SimpleNamespace(
            __file__="/vllm/v1/attention/selector.py",
            AttentionSelectorConfig=VllmConfig,
            get_attn_backend=vllm_get_attn_backend,
        ),
        "vllm_ascend.patch.platform.patch_selector": types.SimpleNamespace(
            __file__="/vllm_ascend/patch/platform/patch_selector.py",
            AttentionSelectorConfig=PatchConfig,
            get_attn_backend=patch_get_attn_backend,
        ),
    }

    monkeypatch.setattr(probe.importlib, "import_module", lambda name: modules[name])

    diagnostics = probe.attention_selector_api_diagnostics()

    assert diagnostics["status"] == "MISMATCH"
    assert "use_per_head_quant_scales" in diagnostics["missing_patch_config_fields"]
    assert "use_per_head_quant_scales" in diagnostics["missing_patch_get_attn_backend_parameters"]
    assert "num_heads" in diagnostics["missing_patch_get_attn_backend_parameters"]
    assert "vllm.v1.attention.selector" in diagnostics["vllm_selector"]["module"]
    assert "vllm_ascend.patch.platform.patch_selector" in diagnostics["patch_selector"]["module"]


def test_probe_exports_records_attention_selector_api_diagnostics(tmp_path, monkeypatch):
    probe = load_probe_module()
    demo_root = tmp_path / "demo"
    for method in ("cap", "qpruner"):
        export_dir = demo_root / "serving_exports" / "run" / method
        export_dir.mkdir(parents=True)
        (export_dir / "config.json").write_text("{}")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout="",
            stderr="TypeError: get_attn_backend() got an unexpected keyword argument 'use_per_head_quant_scales'\n",
        )

    monkeypatch.setattr(probe, "import_vllm", lambda: (types.ModuleType("vllm"), None))
    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    monkeypatch.setattr(
        probe,
        "attention_selector_api_diagnostics",
        lambda: {
            "status": "MISMATCH",
            "missing_patch_config_fields": ["use_per_head_quant_scales"],
            "missing_patch_get_attn_backend_parameters": ["use_per_head_quant_scales", "num_heads"],
        },
    )

    report = probe.probe_exports(demo_root=demo_root, run_label="run")

    assert any(item["id"] == "vllm_attention_selector_api_mismatch" for item in report["issue_classes"])
    assert report["api_diagnostics"]["status"] == "MISMATCH"
    assert "use_per_head_quant_scales" in report["api_diagnostics"]["missing_patch_config_fields"]


def test_acl_diagnostics_recommends_cann_pythonpath_when_acl_hidden(tmp_path, monkeypatch):
    probe = load_probe_module()
    cann_root = tmp_path / "cann-8.5.1"
    python_site = cann_root / "python" / "site-packages"
    tbe_site = cann_root / "opp" / "built-in" / "op_impl" / "ai_core" / "tbe"
    python_site.mkdir(parents=True)
    tbe_site.mkdir(parents=True)

    monkeypatch.setattr(probe.importlib.util, "find_spec", lambda name: None)

    diagnostics = probe.acl_diagnostics(
        env={
            "PYTHONPATH": "/mnt/nvme/622/tidal-ai",
            "ASCEND_HOME_PATH": str(cann_root),
        },
        sys_path=["/mnt/nvme/622/tidal-ai"],
    )

    assert diagnostics["import_acl"] == "MISSING"
    assert str(python_site) in diagnostics["candidate_pythonpath_entries"]
    assert str(tbe_site) in diagnostics["candidate_pythonpath_entries"]
    assert diagnostics["suggested_pythonpath"].startswith("/mnt/nvme/622/tidal-ai:")


def test_vllm_load_check_prepends_acl_candidates_to_subprocess_env(tmp_path, monkeypatch):
    probe = load_probe_module()
    export_dir = tmp_path / "cap"
    export_dir.mkdir()
    captured = {}

    def fake_run(*args, **kwargs):
        captured["env"] = kwargs.get("env")
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="vLLM load PASS\n", stderr="")

    monkeypatch.setattr(
        probe,
        "acl_diagnostics",
        lambda: {
            "import_acl": "MISSING",
            "candidate_pythonpath_entries": ["/usr/local/Ascend/cann/python/site-packages"],
            "suggested_pythonpath": "/project:/usr/local/Ascend/cann/python/site-packages",
        },
    )
    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    monkeypatch.setenv("PYTHONPATH", "/project")

    status, issue = probe.vllm_load_check(types.ModuleType("vllm"), export_dir)

    assert status == "PASS"
    assert issue is None
    assert captured["env"]["PYTHONPATH"].startswith("/project:")
    assert "/usr/local/Ascend/cann/python/site-packages" in captured["env"]["PYTHONPATH"]


def test_markdown_report_includes_acl_diagnostics():
    probe = load_probe_module()

    text = probe.markdown_report(
        {
            "status": "FAIL",
            "vllm_available": True,
            "exports": {
                "cap": {"path": "/demo/cap", "exists": True, "transformers_load": "PASS", "vllm_load": "FAIL"},
                "qpruner": {
                    "path": "/demo/qpruner",
                    "exists": True,
                    "transformers_load": "PASS",
                    "vllm_load": "FAIL",
                },
            },
            "issues": ["ModuleNotFoundError: No module named 'acl'"],
            "acl_diagnostics": {
                "import_acl": "MISSING",
                "pythonpath": "/mnt/nvme/622/tidal-ai",
                "candidate_pythonpath_entries": ["/usr/local/Ascend/cann-8.5.1/python/site-packages"],
            },
            "package_diagnostics": {
                "vllm": {"version": "0.18.0+empty", "file": "/vllm-workspace/vllm/vllm/__init__.py"},
                "vllm-ascend": {"version": "0.1.dev2924+ge5f7e2f43", "file": None},
            },
        }
    )

    assert "## ACL Diagnostics" in text
    assert "import acl: `MISSING`" in text
    assert "/usr/local/Ascend/cann-8.5.1/python/site-packages" in text
    assert "## Package Diagnostics" in text
    assert "0.18.0+empty" in text


def test_markdown_report_includes_attention_selector_api_diagnostics():
    probe = load_probe_module()

    text = probe.markdown_report(
        {
            "status": "FAIL",
            "vllm_available": True,
            "exports": {
                "cap": {"path": "/demo/cap", "exists": True, "transformers_load": "PASS", "vllm_load": "FAIL"},
                "qpruner": {"path": "/demo/qpruner", "exists": True, "transformers_load": "PASS", "vllm_load": "FAIL"},
            },
            "issues": ["TypeError: get_attn_backend() got an unexpected keyword argument 'use_per_head_quant_scales'"],
            "issue_classes": [{"id": "vllm_attention_selector_api_mismatch"}],
            "api_diagnostics": {
                "status": "MISMATCH",
                "missing_patch_config_fields": ["use_per_head_quant_scales"],
                "missing_patch_get_attn_backend_parameters": ["use_per_head_quant_scales", "num_heads"],
                "vllm_selector": {
                    "get_attn_backend_signature": "(head_size, dtype, kv_cache_dtype, use_per_head_quant_scales=False, num_heads=None)",
                },
                "patch_selector": {
                    "get_attn_backend_signature": "(head_size, dtype, kv_cache_dtype, block_size=None)",
                },
            },
        }
    )

    assert "## Attention Selector API Diagnostics" in text
    assert "status: `MISMATCH`" in text
    assert "use_per_head_quant_scales" in text
    assert "num_heads" in text
