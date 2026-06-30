#!/usr/bin/env python
"""Probe Ascend inference readiness without downloading models."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pkgutil
import platform
import sys
from pathlib import Path
from typing import Any


DEFAULT_MODELS = (
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen2.5-0.5B-Instruct",
)


def package_info(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return {"status": "MISSING", "version": None}
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "unknown")
        return {"status": "FOUND", "version": str(version)}
    except Exception as exc:
        return {
            "status": "IMPORT_ERROR",
            "version": None,
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }


def hf_cache_dir_name(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")


def has_model_files(path: Path) -> bool:
    return (path / "config.json").exists() and (
        (path / "tokenizer.json").exists()
        or (path / "tokenizer.model").exists()
        or (path / "tokenizer_config.json").exists()
    )


def find_model_path(model_id: str, roots: list[Path]) -> tuple[Path | None, str | None]:
    direct_names = {model_id.split("/")[-1], model_id.replace("/", "--"), hf_cache_dir_name(model_id)}
    for root in roots:
        if not root.exists():
            continue
        for name in direct_names:
            direct = root / name
            if has_model_files(direct):
                return direct, "direct"
            snapshots = direct / "snapshots"
            if snapshots.exists():
                for snapshot in sorted(snapshots.iterdir()):
                    if snapshot.is_dir() and has_model_files(snapshot):
                        return snapshot, "huggingface_cache"
        cache_root = root / hf_cache_dir_name(model_id) / "snapshots"
        if cache_root.exists():
            for snapshot in sorted(cache_root.iterdir()):
                if snapshot.is_dir() and has_model_files(snapshot):
                    return snapshot, "huggingface_cache"
    return None, None


def find_local_model_candidates(model_ids: list[str], roots: list[Path]) -> list[dict[str, Any]]:
    candidates = []
    for model_id in model_ids:
        path, source = find_model_path(model_id, roots)
        candidates.append(
            {
                "model_id": model_id,
                "status": "FOUND" if path else "MISSING",
                "path": str(path) if path else None,
                "source": source,
            }
        )
    return candidates


def vllm_ascend_info() -> dict[str, Any]:
    modules = sorted(
        m.name for m in pkgutil.iter_modules() if "vllm_ascend" in m.name.lower()
    )
    return {"status": "FOUND" if modules else "MISSING", "modules": modules[:120]}


def default_roots() -> list[Path]:
    raw = [
        os.environ.get("HF_HOME"),
        os.environ.get("TRANSFORMERS_CACHE"),
        "/mnt/nvme/622/models",
        "/mnt/nvme/622/hf_cache",
        "/mnt/nvme/622",
        str(Path.home() / ".cache" / "huggingface" / "hub"),
        str(Path.home() / ".cache" / "huggingface"),
    ]
    roots: list[Path] = []
    seen: set[Path] = set()
    for item in raw:
        if not item:
            continue
        path = Path(item)
        if path not in seen:
            roots.append(path)
            seen.add(path)
    return roots


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Ascend Inference Probe",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Packages",
        "",
        "| Package | Status | Version |",
        "|---|---:|---:|",
    ]
    for name, info in report["packages"].items():
        lines.append(f"| {name} | {info.get('status')} | {info.get('version') or ''} |")
    lines.extend(
        [
            "",
            "## Model Candidates",
            "",
            "| Model | Status | Path | Source |",
            "|---|---:|---|---:|",
        ]
    )
    for model in report["models"]:
        lines.append(
            f"| {model['model_id']} | {model['status']} | {model.get('path') or ''} | {model.get('source') or ''} |"
        )
    lines.extend(
        [
            "",
            "## vLLM Ascend",
            "",
            f"Status: `{report['vllm_ascend']['status']}`",
            "",
        ]
    )
    modules = report["vllm_ascend"].get("modules") or []
    if modules:
        lines.append("Modules:")
        for module in modules:
            lines.append(f"- `{module}`")
    else:
        lines.append("No Ascend-specific vLLM module was discovered.")
    lines.extend(
        [
            "",
            "Next step: place a small modern model such as `Qwen/Qwen3-0.6B` under one of the scanned roots, then run the torch_npu/vLLM inference benchmark without changing this evidence contract.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_probe_report(report: dict[str, Any], demo_root: Path) -> None:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    (artifacts / "ascend_inference_probe.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    (reports / "ascend-inference-probe.md").write_text(markdown_report(report))


def build_report(model_ids: list[str], roots: list[Path]) -> dict[str, Any]:
    packages = {
        name: package_info(name)
        for name in ("torch", "torch_npu", "transformers", "vllm", "peft", "accelerate", "datasets")
    }
    models = find_local_model_candidates(model_ids, roots)
    has_model = any(model["status"] == "FOUND" for model in models)
    required_packages = ("torch", "torch_npu", "transformers")
    required_ready = all(packages[name]["status"] == "FOUND" for name in required_packages)
    status = "READY" if has_model and required_ready else "MODEL_MISSING" if required_ready else "BACKEND_INCOMPLETE"
    return {
        "status": status,
        "platform": platform.platform(),
        "python": sys.version,
        "roots": [str(root) for root in roots],
        "packages": packages,
        "models": models,
        "vllm_ascend": vllm_ascend_info(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Ascend inference readiness")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--model-id", action="append", dest="model_ids")
    parser.add_argument("--root", action="append", dest="roots")
    args = parser.parse_args()

    model_ids = args.model_ids or list(DEFAULT_MODELS)
    roots = [Path(root) for root in args.roots] if args.roots else default_roots()
    report = build_report(model_ids, roots)
    write_probe_report(report, Path(args.demo_root))
    print("ASCEND_INFERENCE_PROBE " + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
