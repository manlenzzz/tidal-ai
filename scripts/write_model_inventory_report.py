#!/usr/bin/env python
"""Write an offline model inventory report for Ascend demo planning."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any, Sequence


DEFAULT_TARGET_MODELS = (
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen2.5-0.5B-Instruct",
)
DEFAULT_TINY_MODELS = ("TinyQwen3-Offline",)
DEFAULT_ROOTS = (
    "/mnt/nvme/622/models",
    "/mnt/nvme/622/hf_cache",
    "/mnt/nvme/622",
    "~/.cache/huggingface/hub",
    "~/.cache/huggingface",
)


def has_model_files(path: Path) -> bool:
    return (path / "config.json").exists() and (
        (path / "tokenizer.json").exists()
        or (path / "tokenizer.model").exists()
        or (path / "tokenizer_config.json").exists()
    )


def hf_cache_dir_name(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")


def direct_names(model_id: str) -> set[str]:
    return {model_id.split("/")[-1], model_id.replace("/", "--"), hf_cache_dir_name(model_id)}


def find_model_path(model_id: str, roots: Sequence[Path]) -> tuple[Path | None, str | None]:
    for root in roots:
        expanded = root.expanduser()
        if not expanded.exists():
            continue
        for name in direct_names(model_id):
            direct = expanded / name
            if has_model_files(direct):
                return direct, "direct"
            snapshots = direct / "snapshots"
            if snapshots.exists():
                for snapshot in sorted(snapshots.iterdir()):
                    if snapshot.is_dir() and has_model_files(snapshot):
                        return snapshot, "huggingface_cache"
        cache_root = expanded / hf_cache_dir_name(model_id) / "snapshots"
        if cache_root.exists():
            for snapshot in sorted(cache_root.iterdir()):
                if snapshot.is_dir() and has_model_files(snapshot):
                    return snapshot, "huggingface_cache"
    return None, None


def parameter_bytes(path: Path | None) -> int:
    if path is None:
        return 0
    total = 0
    for pattern in ("*.safetensors", "pytorch_model*.bin", "*.bin"):
        for file in path.glob(pattern):
            if file.is_file():
                total += file.stat().st_size
    return total


def read_config(path: Path | None) -> dict[str, Any]:
    if path is None or not (path / "config.json").exists():
        return {}
    try:
        return json.loads((path / "config.json").read_text())
    except Exception:
        return {}


def model_entry(*, model_id: str, path: Path | None, source: str | None) -> dict[str, Any]:
    cfg = read_config(path)
    bytes_count = parameter_bytes(path)
    return {
        "model_id": model_id,
        "status": "FOUND" if path else "MISSING",
        "path": str(path) if path else None,
        "source": source,
        "model_type": cfg.get("model_type"),
        "architectures": cfg.get("architectures", []),
        "hidden_size": cfg.get("hidden_size"),
        "parameter_bytes": bytes_count,
        "parameter_mb": round(bytes_count / 1024**2, 3) if bytes_count else 0.0,
    }


def tiny_entry(*, name: str, roots: Sequence[Path]) -> dict[str, Any]:
    path, source = find_model_path(name, roots)
    entry = model_entry(model_id=name, path=path, source=source)
    entry["name"] = name
    return entry


def recommended_next_action(status: str) -> str:
    if status == "TARGET_MODEL_FOUND":
        return "Run torch_npu inference/compression benchmarks against the found modern local model."
    if status == "ONLY_TINY_FIXTURE":
        return "Place a modern local model such as Qwen/Qwen3.5-0.8B or Qwen/Qwen3-0.6B under /mnt/nvme/622/models, then rerun the same benchmark scripts."
    return "Create TinyQwen3-Offline for offline plumbing and add a modern Qwen snapshot for final demo metrics."


def build_report(
    *,
    roots: Sequence[Path],
    target_model_ids: Sequence[str],
    tiny_model_names: Sequence[str],
) -> dict[str, Any]:
    target_candidates = []
    for model_id in target_model_ids:
        path, source = find_model_path(model_id, roots)
        target_candidates.append(model_entry(model_id=model_id, path=path, source=source))
    tiny_fixtures = [tiny_entry(name=name, roots=roots) for name in tiny_model_names]
    has_target = any(row["status"] == "FOUND" for row in target_candidates)
    has_tiny = any(row["status"] == "FOUND" for row in tiny_fixtures)
    status = "TARGET_MODEL_FOUND" if has_target else "ONLY_TINY_FIXTURE" if has_tiny else "NO_LOCAL_MODEL"
    return {
        "status": status,
        "platform": platform.platform(),
        "roots": [str(root.expanduser()) for root in roots],
        "target_candidates": target_candidates,
        "tiny_fixtures": tiny_fixtures,
        "recommended_next_action": recommended_next_action(status),
    }


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Model Inventory",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Recommended next action: {report.get('recommended_next_action')}",
        "",
        "## Target Modern Models",
        "",
        "| Model | Status | Path | Source | Type | MB |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in report.get("target_candidates", []):
        lines.append(
            "| {model} | {status} | {path} | {source} | {model_type} | {mb} |".format(
                model=row.get("model_id", ""),
                status=row.get("status", ""),
                path=row.get("path") or "",
                source=row.get("source") or "",
                model_type=row.get("model_type") or "",
                mb=fmt(row.get("parameter_mb")),
            )
        )
    lines.extend(
        [
            "",
            "## Offline Tiny Fixtures",
            "",
            "| Model | Status | Path | Type | MB |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for row in report.get("tiny_fixtures", []):
        lines.append(
            "| {model} | {status} | {path} | {model_type} | {mb} |".format(
                model=row.get("name") or row.get("model_id", ""),
                status=row.get("status", ""),
                path=row.get("path") or "",
                model_type=row.get("model_type") or "",
                mb=fmt(row.get("parameter_mb")),
            )
        )
    lines.extend(["", "## Scanned Roots", ""])
    for root in report.get("roots", []):
        lines.append(f"- `{root}`")
    return "\n".join(lines) + "\n"


def write_model_inventory_report(
    *,
    demo_root: Path,
    roots: Sequence[Path],
    target_model_ids: Sequence[str],
    tiny_model_names: Sequence[str],
) -> Path:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    report = build_report(
        roots=roots,
        target_model_ids=target_model_ids,
        tiny_model_names=tiny_model_names,
    )
    (artifacts / "model_inventory.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    output = reports / "model-inventory.md"
    output.write_text(markdown(report))
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write offline model inventory report")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--root", action="append", dest="roots")
    parser.add_argument("--target-model-id", action="append", dest="target_model_ids")
    parser.add_argument("--tiny-model-name", action="append", dest="tiny_model_names")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    roots = [Path(root) for root in (args.roots or DEFAULT_ROOTS)]
    target_model_ids = args.target_model_ids or list(DEFAULT_TARGET_MODELS)
    tiny_model_names = args.tiny_model_names or list(DEFAULT_TINY_MODELS)
    output = write_model_inventory_report(
        demo_root=Path(args.demo_root),
        roots=roots,
        target_model_ids=target_model_ids,
        tiny_model_names=tiny_model_names,
    )
    report = json.loads((Path(args.demo_root) / "artifacts" / "model_inventory.json").read_text())
    print("MODEL_INVENTORY " + json.dumps({"status": report["status"], "report": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
