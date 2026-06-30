#!/usr/bin/env python
"""Probe vLLM compatibility for exported TinyQwen serving directories."""
from __future__ import annotations

import argparse
import inspect
import importlib
import importlib.metadata
import os
import json
import platform
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Sequence


MAX_LOG_CHARS = 4000
VLLM_LOAD_TIMEOUT_S = 180
ACL_ENV_KEYS = (
    "PYTHONPATH",
    "ASCEND_HOME_PATH",
    "ASCEND_TOOLKIT_HOME",
    "ASCEND_TOOLKIT_LATEST_HOME",
)


def import_vllm() -> tuple[Any | None, str | None]:
    try:
        return importlib.import_module("vllm"), None
    except Exception as exc:
        return None, f"vllm import failed: {exc}"


def unique_existing_paths(paths: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    existing: list[str] = []
    for value in paths:
        if not value or value in seen:
            continue
        seen.add(value)
        if Path(value).exists():
            existing.append(value)
    return existing


def acl_candidate_pythonpath_entries(env: dict[str, str] | None = None) -> list[str]:
    env = env or os.environ
    roots = unique_existing_paths(
        [
            env.get("ASCEND_HOME_PATH", ""),
            env.get("ASCEND_TOOLKIT_HOME", ""),
            env.get("ASCEND_TOOLKIT_LATEST_HOME", ""),
            "/usr/local/Ascend/ascend-toolkit/latest",
            "/usr/local/Ascend/cann-8.5.1",
        ]
    )
    candidates: list[str] = []
    for root in roots:
        candidates.extend(
            [
                str(Path(root) / "python" / "site-packages"),
                str(Path(root) / "opp" / "built-in" / "op_impl" / "ai_core" / "tbe"),
            ]
        )
    return unique_existing_paths(candidates)


def merge_pythonpath(current: str, additions: Sequence[str]) -> str:
    values = [part for part in current.split(os.pathsep) if part]
    for entry in additions:
        if entry not in values:
            values.append(entry)
    return os.pathsep.join(values)


def acl_diagnostics(
    *,
    env: dict[str, str] | None = None,
    sys_path: Sequence[str] | None = None,
) -> dict[str, Any]:
    env = env or os.environ
    sys_path = list(sys.path if sys_path is None else sys_path)
    spec = importlib.util.find_spec("acl")
    candidate_entries = acl_candidate_pythonpath_entries(env)
    pythonpath = env.get("PYTHONPATH", "")
    return {
        "import_acl": "FOUND" if spec else "MISSING",
        "acl_origin": getattr(spec, "origin", None) if spec else None,
        "python_executable": sys.executable,
        "pythonpath": pythonpath,
        "sys_path_head": sys_path[:8],
        "env": {key: env.get(key, "") for key in ACL_ENV_KEYS},
        "candidate_pythonpath_entries": candidate_entries,
        "suggested_pythonpath": merge_pythonpath(pythonpath, candidate_entries),
    }


def package_diagnostics() -> dict[str, dict[str, Any]]:
    diagnostics: dict[str, dict[str, Any]] = {}
    module_for_package = {
        "vllm": "vllm",
        "vllm-ascend": "vllm_ascend",
    }
    for package_name, module_name in module_for_package.items():
        row: dict[str, Any] = {}
        try:
            row["version"] = importlib.metadata.version(package_name)
            row["status"] = "FOUND"
        except importlib.metadata.PackageNotFoundError:
            row["version"] = None
            row["status"] = "MISSING"
        spec = importlib.util.find_spec(module_name)
        row["module"] = module_name
        row["file"] = getattr(spec, "origin", None) if spec else None
        diagnostics[package_name] = row
    return diagnostics


def _signature(value: Any) -> str | None:
    try:
        return str(inspect.signature(value))
    except Exception:
        return None


def _namedtuple_fields(value: Any) -> list[str]:
    fields = getattr(value, "_fields", None)
    if fields:
        return [str(field) for field in fields]
    annotations = getattr(value, "__annotations__", None)
    if isinstance(annotations, dict):
        return [str(field) for field in annotations]
    return []


def _signature_parameters(value: Any) -> list[str]:
    try:
        return [str(name) for name in inspect.signature(value).parameters]
    except Exception:
        return []


def _attention_selector_module(module_name: str) -> dict[str, Any]:
    row: dict[str, Any] = {"module": module_name}
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        row.update({"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"})
        return row

    row.update({"status": "FOUND", "file": getattr(module, "__file__", None)})
    config = getattr(module, "AttentionSelectorConfig", None)
    get_attn_backend = getattr(module, "get_attn_backend", None)
    row["config_fields"] = _namedtuple_fields(config) if config is not None else []
    row["get_attn_backend_signature"] = _signature(get_attn_backend) if get_attn_backend is not None else None
    row["get_attn_backend_parameters"] = (
        _signature_parameters(get_attn_backend) if get_attn_backend is not None else []
    )
    return row


def attention_selector_api_diagnostics() -> dict[str, Any]:
    vllm_selector = _attention_selector_module("vllm.v1.attention.selector")
    patch_selector = _attention_selector_module("vllm_ascend.patch.platform.patch_selector")

    vllm_fields = vllm_selector.get("config_fields") or []
    patch_fields = patch_selector.get("config_fields") or []
    vllm_params = vllm_selector.get("get_attn_backend_parameters") or []
    patch_params = patch_selector.get("get_attn_backend_parameters") or []
    missing_patch_config_fields = [field for field in vllm_fields if field not in patch_fields]
    missing_patch_get_attn_backend_parameters = [name for name in vllm_params if name not in patch_params]
    status = "MISMATCH" if missing_patch_config_fields or missing_patch_get_attn_backend_parameters else "MATCH"
    if vllm_selector.get("status") != "FOUND" or patch_selector.get("status") != "FOUND":
        status = "UNAVAILABLE"

    return {
        "status": status,
        "vllm_selector": vllm_selector,
        "patch_selector": patch_selector,
        "missing_patch_config_fields": missing_patch_config_fields,
        "missing_patch_get_attn_backend_parameters": missing_patch_get_attn_backend_parameters,
    }


def transformers_load_check(path: Path) -> str:
    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained(str(path), local_files_only=True)
        return "PASS"
    except Exception:
        return "FAIL"


def tail_log(value: str | bytes | None, *, max_chars: int = MAX_LOG_CHARS) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    value = value.strip()
    if len(value) <= max_chars:
        return value
    return "...<truncated>...\n" + value[-max_chars:]


def vllm_load_code(
    path: Path,
    *,
    gpu_memory_utilization: float,
    preload_vllm_ascend_patch: bool = False,
    preload_vllm_ascend_selector_shim: bool = False,
) -> str:
    lines = ["import vllm"]
    if preload_vllm_ascend_patch:
        lines.append("import vllm_ascend.patch.platform")
    if preload_vllm_ascend_selector_shim:
        lines.extend(
            [
                "from tidal.integrations.vllm_ascend_selector_shim import install",
                "print('TIDAL vLLM-Ascend selector shim', install())",
            ]
        )
    lines.extend(
        [
            "",
            "vllm.LLM(",
            f"    model={str(path)!r},",
            "    trust_remote_code=False,",
            "    enforce_eager=True,",
            "    max_model_len=128,",
            f"    gpu_memory_utilization={gpu_memory_utilization!r},",
            ")",
            'print("vLLM load PASS")',
        ]
    )
    return "\n".join(lines) + "\n"


def format_subprocess_failure(path: Path, completed: subprocess.CompletedProcess[str]) -> str:
    details = [f"vLLM load failed for {path}: exit code {completed.returncode}"]
    stdout = tail_log(completed.stdout)
    stderr = tail_log(completed.stderr)
    if stdout:
        details.append(f"stdout:\n{stdout}")
    if stderr:
        details.append(f"stderr:\n{stderr}")
    return "\n".join(details)


def classify_issue(issue: str) -> str | None:
    if "No module named 'acl'" in issue:
        return "vllm_acl_python_module_unavailable"
    if "Free memory on device" in issue and "less than desired" in issue:
        return "vllm_device_memory_pressure"
    if "AttentionSelectorConfig.__new__()" in issue and "unexpected keyword argument" in issue:
        return "vllm_attention_selector_api_mismatch"
    if "get_attn_backend() got an unexpected keyword argument" in issue:
        return "vllm_attention_selector_api_mismatch"
    return None


def vllm_load_check(
    vllm: Any | None,
    path: Path,
    *,
    gpu_memory_utilization: float = 0.05,
    preload_vllm_ascend_patch: bool = False,
    preload_vllm_ascend_selector_shim: bool = False,
) -> tuple[str, str | None]:
    if vllm is None:
        return "SKIP", None
    diagnostics = acl_diagnostics()
    env = os.environ.copy()
    if diagnostics.get("candidate_pythonpath_entries"):
        env["PYTHONPATH"] = str(diagnostics.get("suggested_pythonpath") or env.get("PYTHONPATH", ""))
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                vllm_load_code(
                    path,
                    gpu_memory_utilization=gpu_memory_utilization,
                    preload_vllm_ascend_patch=preload_vllm_ascend_patch,
                    preload_vllm_ascend_selector_shim=preload_vllm_ascend_selector_shim,
                ),
            ],
            capture_output=True,
            text=True,
            timeout=VLLM_LOAD_TIMEOUT_S,
            env=env,
        )
        if completed.returncode == 0:
            return "PASS", None
        return "FAIL", format_subprocess_failure(path, completed)
    except subprocess.TimeoutExpired as exc:
        stdout = tail_log(exc.stdout)
        stderr = tail_log(exc.stderr)
        issue = f"vLLM load timed out for {path} after {VLLM_LOAD_TIMEOUT_S}s"
        if stdout:
            issue += f"\nstdout:\n{stdout}"
        if stderr:
            issue += f"\nstderr:\n{stderr}"
        return "FAIL", issue
    except Exception as exc:
        return "FAIL", f"vLLM load failed for {path}: {type(exc).__name__}: {exc}"


def export_path(demo_root: Path, run_label: str, method: str) -> Path:
    return demo_root / "serving_exports" / run_label / method


def issue_class_rows(issues: Sequence[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for issue in issues:
        issue_id = classify_issue(str(issue))
        if issue_id and issue_id not in seen:
            rows.append({"id": issue_id})
            seen.add(issue_id)
    return rows


def probe_exports(
    *,
    demo_root: Path,
    run_label: str,
    export_run_label: str | None = None,
    gpu_memory_utilization: float = 0.05,
    preload_vllm_ascend_patch: bool = False,
    preload_vllm_ascend_selector_shim: bool = False,
) -> dict[str, Any]:
    export_run_label = export_run_label or run_label
    vllm, issue = import_vllm()
    diagnostics = acl_diagnostics()
    packages = package_diagnostics()
    api_diagnostics = attention_selector_api_diagnostics()
    issues = []
    if issue:
        issues.append(issue)
    exports: dict[str, dict[str, Any]] = {}
    for method in ("cap", "qpruner"):
        path = export_path(demo_root, export_run_label, method)
        exists = path.exists()
        vllm_status, vllm_issue = (
            vllm_load_check(
                vllm,
                path,
                gpu_memory_utilization=gpu_memory_utilization,
                preload_vllm_ascend_patch=preload_vllm_ascend_patch,
                preload_vllm_ascend_selector_shim=preload_vllm_ascend_selector_shim,
            )
            if exists
            else ("MISSING", None)
        )
        if vllm_issue:
            issues.append(vllm_issue)
        exports[method] = {
            "path": str(path),
            "exists": exists,
            "transformers_load": transformers_load_check(path) if exists else "MISSING",
            "vllm_load": vllm_status,
        }
    if vllm is None:
        status = "SKIP"
    elif all(row["vllm_load"] == "PASS" for row in exports.values()):
        status = "PASS"
    else:
        status = "FAIL"
    return {
        "status": status,
        "vllm_available": vllm is not None,
        "run_label": run_label,
        "export_run_label": export_run_label,
        "gpu_memory_utilization": gpu_memory_utilization,
        "preload_vllm_ascend_patch": preload_vllm_ascend_patch,
        "preload_vllm_ascend_selector_shim": preload_vllm_ascend_selector_shim,
        "exports": exports,
        "issues": issues,
        "issue_classes": issue_class_rows(issues),
        "acl_diagnostics": diagnostics,
        "package_diagnostics": packages,
        "api_diagnostics": api_diagnostics,
        "platform": platform.platform(),
    }


def row(method: str, payload: dict[str, Any]) -> str:
    return "| {method} | {exists} | {hf} | {vllm} |".format(
        method=method,
        exists=payload.get("exists"),
        hf=payload.get("transformers_load"),
        vllm=payload.get("vllm_load"),
    )


def markdown_report(report: dict[str, Any]) -> str:
    exports = report.get("exports", {})
    diagnostics = report.get("acl_diagnostics") or {}
    packages = report.get("package_diagnostics") or {}
    api = report.get("api_diagnostics") or {}
    lines = [
        "# vLLM Serving Export Probe",
        "",
        f"- Status: `{report.get('status')}`",
        f"- vLLM available: `{report.get('vllm_available')}`",
        f"- Export run label: `{report.get('export_run_label')}`",
        f"- gpu_memory_utilization: `{report.get('gpu_memory_utilization')}`",
        f"- preload_vllm_ascend_patch: `{report.get('preload_vllm_ascend_patch')}`",
        f"- preload_vllm_ascend_selector_shim: `{report.get('preload_vllm_ascend_selector_shim')}`",
        "",
        "| Method | Exists | Transformers load | vLLM load |",
        "|---|---:|---:|---:|",
        row("CAP", exports.get("cap", {})),
        row("QPruner", exports.get("qpruner", {})),
        "",
        "## Issues",
        "",
    ]
    issues = report.get("issues") or []
    if issues:
        lines.extend(f"- {issue}" for issue in issues)
    else:
        lines.append("- None")
    lines.extend(["", "## Issue Classes", ""])
    classes = report.get("issue_classes") or []
    if classes:
        lines.extend(f"- `{entry.get('id')}`" for entry in classes)
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Attention Selector API Diagnostics",
            "",
            f"- status: `{api.get('status', 'missing')}`",
            "- missing patch AttentionSelectorConfig fields: "
            f"`{', '.join(api.get('missing_patch_config_fields') or []) or 'none'}`",
            "- missing patch get_attn_backend parameters: "
            f"`{', '.join(api.get('missing_patch_get_attn_backend_parameters') or []) or 'none'}`",
        ]
    )
    vllm_selector = api.get("vllm_selector") or {}
    patch_selector = api.get("patch_selector") or {}
    if vllm_selector or patch_selector:
        lines.extend(
            [
                "- vLLM selector get_attn_backend: "
                f"`{vllm_selector.get('get_attn_backend_signature', 'missing')}`",
                "- vLLM-Ascend patch get_attn_backend: "
                f"`{patch_selector.get('get_attn_backend_signature', 'missing')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## ACL Diagnostics",
            "",
            f"- import acl: `{diagnostics.get('import_acl', 'UNKNOWN')}`",
            f"- acl origin: `{diagnostics.get('acl_origin')}`",
            f"- PYTHONPATH: `{diagnostics.get('pythonpath', '')}`",
            "- Candidate PYTHONPATH entries:",
        ]
    )
    candidates = diagnostics.get("candidate_pythonpath_entries") or []
    if candidates:
        lines.extend(f"  - `{entry}`" for entry in candidates)
    else:
        lines.append("  - None")
    lines.extend(["", "## Package Diagnostics", ""])
    if packages:
        for package_name, payload in sorted(packages.items()):
            lines.append(
                "- `{name}`: status=`{status}`, version=`{version}`, file=`{file}`".format(
                    name=package_name,
                    status=payload.get("status"),
                    version=payload.get("version"),
                    file=payload.get("file"),
                )
            )
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Note",
            "",
            "A SKIP result means the exported HF directories exist, but this container cannot run a vLLM load check. Keep the artifact as a compatibility gate for a vLLM-Ascend image.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(report: dict[str, Any], demo_root: Path, run_label: str) -> None:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    (artifacts / f"vllm_serving_probe_{run_label}.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    (reports / f"vllm-serving-probe-{run_label}.md").write_text(markdown_report(report))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe vLLM compatibility for TinyQwen serving exports")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--run-label", default="tiny_qwen3_serving_export_npu")
    parser.add_argument("--export-run-label", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.05)
    parser.add_argument("--preload-vllm-ascend-patch", action="store_true")
    parser.add_argument("--preload-vllm-ascend-selector-shim", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    demo_root = Path(args.demo_root)
    report = probe_exports(
        demo_root=demo_root,
        run_label=args.run_label,
        export_run_label=args.export_run_label,
        gpu_memory_utilization=args.gpu_memory_utilization,
        preload_vllm_ascend_patch=args.preload_vllm_ascend_patch,
        preload_vllm_ascend_selector_shim=args.preload_vllm_ascend_selector_shim,
    )
    write_outputs(report, demo_root, args.run_label)
    print("VLLM_SERVING_PROBE " + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
