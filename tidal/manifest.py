from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_PAPER_IDS = [
    "rankadaptor",
    "qpruner",
    "dynamic-operator-optimization",
    "qr-adaptor",
    "autoqra",
    "global-rank-sparsity",
]


class ManifestError(ValueError):
    """Raised when the paper source manifest is incomplete or inconsistent."""


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ManifestError("manifest root must be a mapping")
    return data


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    papers = manifest.get("papers")
    if not isinstance(papers, list):
        raise ManifestError("manifest must contain a papers list")

    ids: list[str] = []
    for index, paper in enumerate(papers):
        if not isinstance(paper, dict):
            raise ManifestError(f"paper entry {index} must be a mapping")
        paper_id = paper.get("id")
        if not isinstance(paper_id, str) or not paper_id:
            raise ManifestError(f"paper entry {index} missing id")
        ids.append(paper_id)
        _validate_paper_entry(paper_id, paper)

    duplicates = sorted({paper_id for paper_id in ids if ids.count(paper_id) > 1})
    if duplicates:
        raise ManifestError(f"duplicate paper ids: {', '.join(duplicates)}")

    expected = set(REQUIRED_PAPER_IDS)
    actual = set(ids)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise ManifestError(f"missing paper ids: {', '.join(missing)}")
    if extra:
        raise ManifestError(f"unexpected paper ids: {', '.join(extra)}")

    return ids


def _validate_paper_entry(paper_id: str, paper: dict[str, Any]) -> None:
    for field in ("title", "source_url", "pdf_path", "text_path", "code"):
        if field not in paper:
            raise ManifestError(f"{paper_id} missing {field}")

    code = paper["code"]
    if not isinstance(code, dict):
        raise ManifestError(f"{paper_id} code must be a mapping")

    status = code.get("status")
    if status not in {"external", "paper-derived"}:
        raise ManifestError(f"{paper_id} code.status must be external or paper-derived")

    if status == "external":
        for field in ("repo_url", "local_path"):
            if not code.get(field):
                raise ManifestError(f"{paper_id} external code missing {field}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Validate the TIDAL paper source manifest")
    parser.add_argument("manifest", nargs="?", default="papers/sources.yaml")
    args = parser.parse_args()

    paper_ids = validate_manifest(load_manifest(args.manifest))
    print(f"ok: {len(paper_ids)} papers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
