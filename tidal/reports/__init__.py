from __future__ import annotations

import json
from pathlib import Path

from tidal.methods.global_rank_sparsity.torch import CAPRunResult

__all__ = ["save_summary_json", "summarize_cap_run"]


def save_summary_json(summary: dict[str, object], output_dir: str | Path) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path


def summarize_cap_run(
    result: CAPRunResult,
    *,
    model_id: str,
    target_roles: object | None = None,
    pruner_target_count: int | None = None,
    calibration_batches: int | None = None,
    max_layers: int = 64,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "method": "cap",
        "model_id": model_id,
        "target_roles": target_roles,
        "target_count": len(result.targets),
        "compressed_layers": len(result.layer_summaries),
        "parameter_count": result.global_result.parameter_count,
        "selected_candidates": len(result.global_result.selected_candidates),
        "best_loss": result.global_result.best_loss,
        "layers": [layer.__dict__ for layer in result.layer_summaries[: max(0, max_layers)]],
    }
    if pruner_target_count is not None:
        summary["pruner_target_count"] = int(pruner_target_count)
    if calibration_batches is not None:
        summary["calibration_batches"] = int(calibration_batches)
    return summary
