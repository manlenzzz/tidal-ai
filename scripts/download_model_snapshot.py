#!/usr/bin/env python
"""Download or reuse a model snapshot under the demo model root."""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any, Callable, Sequence


DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"
DEFAULT_MODEL_ROOT = "/mnt/nvme/622/models"
DEFAULT_ALLOW_PATTERNS = (
    "*.json",
    "*.model",
    "*.txt",
    "*.py",
    "*.safetensors",
    "tokenizer.*",
    "generation_config.json",
    "merges.txt",
    "vocab.json",
)
DEFAULT_IGNORE_PATTERNS = (
    "*.msgpack",
    "*.h5",
    "*.ot",
)
SUPPORTED_PROVIDERS = ("huggingface", "modelscope")


def has_model_files(path: Path) -> bool:
    return (path / "config.json").exists() and (
        (path / "tokenizer.json").exists()
        or (path / "tokenizer.model").exists()
        or (path / "tokenizer_config.json").exists()
    )


def local_model_dir(model_root: Path, model_id: str) -> Path:
    return model_root / model_id.split("/")[-1]


def ensure_under_root(model_root: Path, candidate: Path) -> None:
    root = model_root.resolve()
    path = candidate.resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"destination {path} is outside allowed model root {root}")


def parameter_bytes(path: Path) -> int:
    total = 0
    for pattern in ("*.safetensors", "pytorch_model*.bin", "*.bin"):
        for file in path.glob(pattern):
            if file.is_file():
                total += file.stat().st_size
    return total


def huggingface_snapshot_adapter(snapshot_download: Callable[..., str]) -> Callable[..., str]:
    def download(
        *,
        model_id: str,
        revision: str | None,
        local_dir: str,
        allow_patterns: Sequence[str],
        ignore_patterns: Sequence[str],
    ) -> str:
        return snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=list(allow_patterns),
            ignore_patterns=list(ignore_patterns),
        )

    return download


def modelscope_snapshot_adapter(snapshot_download: Callable[..., str]) -> Callable[..., str]:
    def download(
        *,
        model_id: str,
        revision: str | None,
        local_dir: str,
        allow_patterns: Sequence[str],
        ignore_patterns: Sequence[str],
    ) -> str:
        kwargs: dict[str, Any] = {"local_dir": local_dir}
        if revision:
            kwargs["revision"] = revision
        return snapshot_download(model_id, **kwargs)

    return download


def load_snapshot_downloader(provider: str) -> Callable[..., str]:
    if provider == "huggingface":
        from huggingface_hub import snapshot_download

        return huggingface_snapshot_adapter(snapshot_download)
    if provider == "modelscope":
        from modelscope.hub.snapshot_download import snapshot_download

        return modelscope_snapshot_adapter(snapshot_download)
    raise ValueError(f"unsupported provider: {provider}")


def download_or_reuse_snapshot(
    *,
    model_id: str,
    model_root: Path,
    provider: str,
    revision: str | None,
    allow_patterns: Sequence[str],
    ignore_patterns: Sequence[str],
    snapshot_download: Callable[..., str],
) -> dict[str, Any]:
    started = time.perf_counter()
    model_root.mkdir(parents=True, exist_ok=True)
    destination = local_model_dir(model_root, model_id)
    ensure_under_root(model_root, destination)
    if has_model_files(destination):
        status = "FOUND"
        attempted = False
    else:
        snapshot_download(
            model_id=model_id,
            revision=revision,
            local_dir=str(destination),
            allow_patterns=list(allow_patterns),
            ignore_patterns=list(ignore_patterns),
        )
        status = "DOWNLOADED" if has_model_files(destination) else "INCOMPLETE"
        attempted = True
    bytes_count = parameter_bytes(destination) if destination.exists() else 0
    return {
        "status": status,
        "model_id": model_id,
        "provider": provider,
        "revision": revision,
        "model_root": str(model_root),
        "model_path": str(destination),
        "download_attempted": attempted,
        "parameter_bytes": bytes_count,
        "parameter_mb": round(bytes_count / 1024**2, 3) if bytes_count else 0.0,
        "allow_patterns": list(allow_patterns),
        "ignore_patterns": list(ignore_patterns),
        "seconds": time.perf_counter() - started,
        "platform": platform.platform(),
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Model Snapshot",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Status | {report.get('status')} |",
        f"| Model | {report.get('model_id')} |",
        f"| Provider | {report.get('provider') or 'unknown'} |",
        f"| Revision | {report.get('revision') or 'default'} |",
        f"| Model path | {report.get('model_path')} |",
        f"| Download attempted | {report.get('download_attempted')} |",
        f"| Parameter MB | {report.get('parameter_mb')} |",
        f"| Seconds | {report.get('seconds')} |",
        "",
        "This model-snapshot artifact is kept for the demo recording trail.",
    ]
    if report.get("error"):
        lines.extend(["", f"Error: `{report.get('error_type')}: {report.get('error')}`"])
    return "\n".join(lines) + "\n"


def safe_label(value: str) -> str:
    return value.replace("/", "_").replace(".", "").replace("-", "_").lower()


def write_outputs(report: dict[str, Any], demo_root: Path, *, run_label: str) -> None:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    (artifacts / f"model_snapshot_{run_label}.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    (reports / f"model-snapshot-{run_label}.md").write_text(markdown_report(report))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download or reuse a demo model snapshot")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-root", default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--provider", choices=SUPPORTED_PROVIDERS, default="huggingface")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--allow-pattern", action="append", dest="allow_patterns")
    parser.add_argument("--ignore-pattern", action="append", dest="ignore_patterns")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_label = args.run_label or safe_label(args.model_id)
    try:
        snapshot_download = load_snapshot_downloader(args.provider)
        report = download_or_reuse_snapshot(
            model_id=args.model_id,
            model_root=Path(args.model_root),
            provider=args.provider,
            revision=args.revision,
            allow_patterns=args.allow_patterns or DEFAULT_ALLOW_PATTERNS,
            ignore_patterns=args.ignore_patterns or DEFAULT_IGNORE_PATTERNS,
            snapshot_download=snapshot_download,
        )
    except Exception as exc:
        model_path = local_model_dir(Path(args.model_root), args.model_id)
        report = {
            "status": "ERROR",
            "model_id": args.model_id,
            "provider": args.provider,
            "revision": args.revision,
            "model_root": args.model_root,
            "model_path": str(model_path),
            "download_attempted": True,
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
            "platform": platform.platform(),
        }
    write_outputs(report, Path(args.demo_root), run_label=run_label)
    print("MODEL_SNAPSHOT " + json.dumps(report, sort_keys=True))
    return 0 if report.get("status") in {"FOUND", "DOWNLOADED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
