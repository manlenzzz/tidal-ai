from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

from tidal.methods.global_rank_sparsity.torch import CAPRunResult
from tidal.methods.qpruner.core import config_memory_bits
from tidal.methods.qpruner.torch import QPrunerRun

__all__ = ["save_summary_json", "summarize_cap_run", "summarize_qpruner_run"]


def _jsonable_target_roles(target_roles: object | None) -> object | None:
    if target_roles is None or isinstance(target_roles, (str, int, float, bool)):
        return target_roles
    if isinstance(target_roles, Mapping):
        return {str(key): str(value) for key, value in target_roles.items()}
    if isinstance(target_roles, Iterable):
        return [getattr(role, "value", str(role)) for role in target_roles]
    return str(target_roles)


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
        "target_roles": _jsonable_target_roles(target_roles),
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


def summarize_qpruner_run(
    result: QPrunerRun,
    *,
    model_id: str,
    target_roles: object | None = None,
    pruner_target_count: int | None = None,
    calibration_batches: int | None = None,
    max_layers: int = 64,
) -> dict[str, object]:
    names = list(result.plan.bitwidths)[: max(0, max_layers)]
    bitwidths = {name: int(result.plan.bitwidths[name]) for name in names}
    layer_sizes = {name: int(result.plan.layer_sizes[name]) for name in names}
    importances = {name: float(result.importances[name]) for name in names}
    memory_bits = config_memory_bits(result.plan.bitwidths, result.plan.layer_sizes)
    total_parameters = int(sum(result.plan.layer_sizes.values()))

    summary: dict[str, object] = {
        "method": "qpruner",
        "model_id": model_id,
        "target_roles": _jsonable_target_roles(target_roles),
        "target_count": len(result.plan.bitwidths),
        "bitwidths": bitwidths,
        "layer_sizes": layer_sizes,
        "importances": importances,
        "memory_bits": memory_bits,
        "average_bits": memory_bits / total_parameters if total_parameters else 0.0,
        "score": result.plan.score,
    }
    if pruner_target_count is not None:
        summary["pruner_target_count"] = int(pruner_target_count)
    if calibration_batches is not None:
        summary["calibration_batches"] = int(calibration_batches)
    return summary
