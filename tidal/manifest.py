from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_METHOD_IDS = [
    "rankadaptor",
    "qpruner",
    "dynamic-operator-optimization",
    "qr-adaptor",
    "autoqra",
    "global-rank-sparsity",
]


class ManifestError(ValueError):
    """Raised when the method source manifest is incomplete or inconsistent."""


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ManifestError("manifest root must be a mapping")
    return data


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    methods = manifest.get("methods")
    if not isinstance(methods, list):
        raise ManifestError("manifest must contain a methods list")

    ids: list[str] = []
    for index, method in enumerate(methods):
        if not isinstance(method, dict):
            raise ManifestError(f"method entry {index} must be a mapping")
        method_id = method.get("id")
        if not isinstance(method_id, str) or not method_id:
            raise ManifestError(f"method entry {index} missing id")
        ids.append(method_id)
        _validate_method_entry(method_id, method)

    duplicates = sorted({method_id for method_id in ids if ids.count(method_id) > 1})
    if duplicates:
        raise ManifestError(f"duplicate method ids: {', '.join(duplicates)}")

    expected = set(REQUIRED_METHOD_IDS)
    actual = set(ids)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise ManifestError(f"missing method ids: {', '.join(missing)}")
    if extra:
        raise ManifestError(f"unexpected method ids: {', '.join(extra)}")

    return ids


def _validate_method_entry(method_id: str, method: dict[str, Any]) -> None:
    for field in ("title", "source_url", "pdf_path", "text_path", "code"):
        if field not in method:
            raise ManifestError(f"{method_id} missing {field}")

    code = method["code"]
    if not isinstance(code, dict):
        raise ManifestError(f"{method_id} code must be a mapping")

    status = code.get("status")
    if status not in {"external", "implemented"}:
        raise ManifestError(f"{method_id} code.status must be external or implemented")

    if status == "external":
        required_fields = ("repo_url", "local_path")
    else:
        required_fields = ("local_path",)

    for field in required_fields:
        if not code.get(field):
            raise ManifestError(f"{method_id} code missing {field}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Validate the TIDAL method source manifest")
    parser.add_argument("manifest", nargs="?", default="methods/sources.yaml")
    args = parser.parse_args()

    method_ids = validate_manifest(load_manifest(args.manifest))
    print(f"ok: {len(method_ids)} methods")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
