from pathlib import Path

import pytest

from tidal.manifest import ManifestError, load_manifest, validate_manifest


def write_manifest(tmp_path: Path, body: str) -> Path:
    manifest = tmp_path / "sources.yaml"
    manifest.write_text(body, encoding="utf-8")
    return manifest


def test_validate_manifest_accepts_all_six_unique_papers(tmp_path: Path) -> None:
    manifest_path = write_manifest(
        tmp_path,
        """
papers:
  - id: rankadaptor
    title: "RankAdaptor: Hierarchical Rank Allocation for Efficient Fine-Tuning Pruned LLMs via Performance Model"
    source_url: https://arxiv.org/abs/2406.15734
    pdf_url: https://arxiv.org/pdf/2406.15734
    pdf_path: papers/pdf/rankadaptor.pdf
    text_path: papers/text/rankadaptor.txt
    code:
      status: implemented
      local_path: tidal/rankadaptor.py
  - id: qpruner
    title: "Qpruner: Probabilistic decision quantization for structured pruning in large language models"
    source_url: https://arxiv.org/abs/2412.11629
    pdf_url: https://arxiv.org/pdf/2412.11629
    pdf_path: papers/pdf/qpruner.pdf
    text_path: papers/text/qpruner.txt
    code:
      status: implemented
      local_path: tidal/qpruner.py
  - id: dynamic-operator-optimization
    title: Dynamic operator optimization for efficient multi-tenant LoRA model serving
    source_url: https://ojs.aaai.org/index.php/AAAI/article/view/34453
    pdf_url: https://ojs.aaai.org/index.php/AAAI/article/view/34453/36608
    pdf_path: papers/pdf/dynamic-operator-optimization.pdf
    text_path: papers/text/dynamic-operator-optimization.txt
    code:
      status: external
      repo_url: https://github.com/harrysyz99/Dop
      local_path: external/Dop
  - id: qr-adaptor
    title: "Balancing fidelity and plasticity: Aligning mixed-precision fine-tuning with linguistic hierarchies"
    source_url: https://arxiv.org/abs/2505.03802
    pdf_url: https://arxiv.org/pdf/2505.03802
    pdf_path: papers/pdf/qr-adaptor.pdf
    text_path: papers/text/qr-adaptor.txt
    code:
      status: external
      repo_url: https://github.com/harrysyz99/qr_adapter
      local_path: external/qr_adapter
  - id: autoqra
    title: "AutoQRA: Joint Optimization of Mixed-Precision Quantization and Low-rank Adapters for Efficient LLM Fine-Tuning"
    source_url: https://arxiv.org/abs/2602.22268
    pdf_url: https://arxiv.org/pdf/2602.22268
    pdf_path: papers/pdf/autoqra.pdf
    text_path: papers/text/autoqra.txt
    code:
      status: external
      repo_url: https://github.com/harrysyz99/autoqra
      local_path: external/autoqra
  - id: global-rank-sparsity
    title: Large Language Model Compression with Global Rank and Sparsity Optimization
    source_url: https://arxiv.org/abs/2505.03801
    pdf_url: https://arxiv.org/pdf/2505.03801
    pdf_path: papers/pdf/global-rank-sparsity.pdf
    text_path: papers/text/global-rank-sparsity.txt
    code:
      status: implemented
      local_path: tidal/cap.py
""",
    )

    manifest = load_manifest(manifest_path)

    assert validate_manifest(manifest) == [
        "rankadaptor",
        "qpruner",
        "dynamic-operator-optimization",
        "qr-adaptor",
        "autoqra",
        "global-rank-sparsity",
    ]


def test_validate_manifest_rejects_missing_or_duplicate_papers(tmp_path: Path) -> None:
    manifest_path = write_manifest(
        tmp_path,
        """
papers:
  - id: rankadaptor
    title: "RankAdaptor: Hierarchical Rank Allocation for Efficient Fine-Tuning Pruned LLMs via Performance Model"
    source_url: https://arxiv.org/abs/2406.15734
    pdf_path: papers/pdf/rankadaptor.pdf
    text_path: papers/text/rankadaptor.txt
    code:
      status: implemented
      local_path: tidal/rankadaptor.py
  - id: rankadaptor
    title: RankAdaptor duplicate
    source_url: https://arxiv.org/abs/2406.15734
    pdf_path: papers/pdf/rankadaptor.pdf
    text_path: papers/text/rankadaptor.txt
    code:
      status: implemented
      local_path: tidal/rankadaptor.py
""",
    )

    manifest = load_manifest(manifest_path)

    with pytest.raises(ManifestError, match="duplicate paper ids"):
        validate_manifest(manifest)
