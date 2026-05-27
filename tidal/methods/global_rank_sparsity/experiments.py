from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import torch

from tidal.methods.global_rank_sparsity.torch import CAPRunResult


def load_pruner_target_names(path: str | Path) -> list[str]:
    target_path = Path(path)
    suffix = target_path.suffix.lower()
    if suffix in {".pt", ".pth", ".bin", ".ckpt", ".safetensors"}:
        if suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
            except ImportError as exc:  # pragma: no cover - optional dependency path
                raise ImportError("loading .safetensors pruner targets requires safetensors") from exc
            loaded = load_file(str(target_path), device="cpu")
        else:
            loaded = torch.load(target_path, map_location="cpu")
        return _names_from_loaded_object(loaded)

    if suffix == ".json":
        return _names_from_loaded_object(json.loads(target_path.read_text()))

    if suffix == ".jsonl":
        names: list[str] = []
        for line in target_path.read_text().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if isinstance(record, str):
                names.append(record)
            elif isinstance(record, Mapping):
                for key in ("name", "module", "module_name", "tensor", "key"):
                    value = record.get(key)
                    if isinstance(value, str) and value.strip():
                        names.append(value.strip())
                        break
        return names

    names = []
    for line in target_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        names.append(stripped)
    return names


def _names_from_loaded_object(value: object) -> list[str]:
    if isinstance(value, Mapping):
        if "state_dict" in value and isinstance(value["state_dict"], Mapping):
            return [str(name) for name in value["state_dict"].keys()]
        if "model" in value and isinstance(value["model"], Mapping):
            return [str(name) for name in value["model"].keys()]
        if "names" in value and isinstance(value["names"], list):
            return [str(name) for name in value["names"]]
        return [str(name) for name in value.keys()]
    if isinstance(value, list | tuple):
        names: list[str] = []
        for item in value:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, Mapping):
                for key in ("name", "module", "module_name", "tensor", "key"):
                    candidate = item.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        names.append(candidate.strip())
                        break
        return names
    raise ValueError(f"unsupported pruner target format: {type(value).__name__}")


def load_calibration_texts(path: str | Path, *, text_field: str = "text") -> list[str]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        texts: list[str] = []
        for line in source.read_text().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if isinstance(record, Mapping):
                value = record.get(text_field)
                if isinstance(value, str) and value.strip():
                    texts.append(value.strip())
            elif isinstance(record, str) and record.strip():
                texts.append(record.strip())
        return texts
    return [line.strip() for line in source.read_text().splitlines() if line.strip()]


def build_text_calibration_batches(
    tokenizer: object,
    texts: Iterable[str],
    *,
    max_length: int = 128,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
) -> list[dict[str, torch.Tensor]]:
    usable_texts = [text for text in texts if str(text).strip()]
    if not usable_texts:
        raise ValueError("calibration texts must not be empty")
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    encoded_rows: list[dict[str, list[int]]] = []
    for text in usable_texts:
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )
        input_ids = _as_int_list(encoded["input_ids"])[:max_length]
        attention_mask = _as_int_list(encoded.get("attention_mask", [1] * len(input_ids)))[: len(input_ids)]
        if len(input_ids) == 0:
            continue
        pad_length = max_length - len(input_ids)
        if pad_length > 0:
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(tokenizer, "eos_token_id", None)
            if pad_token_id is None:
                pad_token_id = 0
            input_ids = input_ids + [int(pad_token_id)] * pad_length
            attention_mask = attention_mask + [0] * pad_length
        encoded_rows.append({"input_ids": input_ids, "attention_mask": attention_mask})

    if not encoded_rows:
        raise ValueError("calibration texts produced no tokenized batches")

    batches: list[dict[str, torch.Tensor]] = []
    for start in range(0, len(encoded_rows), batch_size):
        rows = encoded_rows[start : start + batch_size]
        input_ids = torch.tensor([row["input_ids"] for row in rows], dtype=torch.long, device=device)
        attention_mask = torch.tensor([row["attention_mask"] for row in rows], dtype=torch.long, device=device)
        batches.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.clone(),
            }
        )
    return batches


def _as_int_list(value: object) -> list[int]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if value and isinstance(value, list) and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def causal_lm_loss(model: object, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    outputs = model(**dict(batch))
    loss = getattr(outputs, "loss", None)
    if loss is None:
        if isinstance(outputs, Mapping):
            loss = outputs.get("loss")
    if loss is None:
        raise ValueError("model output does not contain a loss; pass labels in calibration batches")
    return loss


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
