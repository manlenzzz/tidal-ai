import json
from pathlib import Path

from tidal.reports.multicard_selection import (
    best_multicard_qwen_lora_finetune,
    best_multicard_qwen_compression_generate,
    multicard_qwen_compression_generate_report_name,
    multicard_qwen_lora_finetune_report_name,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def generate_payload(*, target_layers: int, qpruner_tokens: float, world_size: int = 8) -> dict:
    return {
        "status": "PASS",
        "model_id": "Qwen/Qwen3-0.6B",
        "world_size": world_size,
        "backend": "hccl",
        "target_layer_limit": target_layers,
        "targeted_layers_total": 196,
        "aggregate": {
            "distributed_reduce_consistent": True,
            "baseline_tokens_per_s_total": 100.0,
            "cap_tokens_per_s_total": 200.0,
            "qpruner_tokens_per_s_total": qpruner_tokens,
            "peak_mem_mb_total": 30000.0,
            "pass_count": world_size,
        },
    }


def test_best_multicard_qwen_compression_generate_prefers_stronger_8card_pass(tmp_path):
    artifacts = tmp_path / "artifacts"
    write_json(
        artifacts / "multicard_qwen_compression_generate_qwen3_06b_compression_generate_npu.json",
        generate_payload(target_layers=2, qpruner_tokens=180.0),
    )
    write_json(
        artifacts / "multicard_qwen_compression_generate_qwen3_06b_compression_generate_8card_target8_long8_npu.json",
        generate_payload(target_layers=8, qpruner_tokens=211.526),
    )
    write_json(
        artifacts / "multicard_qwen_compression_generate_qwen3_06b_compression_generate_8card_target8_long8_npu_rank0.json",
        {"status": "PASS", "rank": 0},
    )

    selected = best_multicard_qwen_compression_generate(artifacts)

    assert selected is not None
    assert selected["_artifact_name"] == (
        "multicard_qwen_compression_generate_qwen3_06b_compression_generate_8card_target8_long8_npu.json"
    )
    assert selected["_report_name"] == "multicard-qwen-compression-generate-qwen3_06b_compression_generate_8card_target8_long8_npu.md"
    assert selected["target_layer_limit"] == 8
    assert selected["aggregate"]["qpruner_tokens_per_s_total"] == 211.526


def test_multicard_qwen_compression_generate_report_name_uses_run_label():
    assert (
        multicard_qwen_compression_generate_report_name(
            "multicard_qwen_compression_generate_qwen3_06b_compression_generate_8card_target8_long8_npu.json"
        )
        == "multicard-qwen-compression-generate-qwen3_06b_compression_generate_8card_target8_long8_npu.md"
    )


def lora_payload(*, validation_delta: float, world_size: int = 8, train_mode: str = "random") -> dict:
    return {
        "status": "PASS",
        "method": "rankadaptor_lora",
        "model_id": "Qwen/Qwen3-0.6B",
        "world_size": world_size,
        "train_mode": train_mode,
        "aggregate": {
            "distributed_reduce_consistent": True,
            "adapter_sync_consistent": True,
            "validation_loss_delta_avg": validation_delta,
            "pass_count": world_size,
        },
    }


def test_best_multicard_qwen_lora_finetune_prefers_instruction_8card_validation_improvement(tmp_path):
    artifacts = tmp_path / "artifacts"
    write_json(
        artifacts / "multicard_qwen_lora_finetune_qwen3_06b_lora_sync_npu.json",
        lora_payload(validation_delta=-0.008, train_mode="random"),
    )
    write_json(
        artifacts / "multicard_qwen_lora_finetune_qwen3_06b_lora_instruction_sync_8card_steps6_20260625.json",
        lora_payload(validation_delta=-0.132813, train_mode="instruction"),
    )
    write_json(
        artifacts / "multicard_qwen_lora_finetune_qwen3_06b_lora_instruction_sync_8card_steps6_20260625_rank0.json",
        {"status": "PASS", "rank": 0},
    )

    selected = best_multicard_qwen_lora_finetune(artifacts)

    assert selected is not None
    assert selected["_artifact_name"] == (
        "multicard_qwen_lora_finetune_qwen3_06b_lora_instruction_sync_8card_steps6_20260625.json"
    )
    assert selected["_report_name"] == (
        "multicard-qwen-lora-finetune-qwen3_06b_lora_instruction_sync_8card_steps6_20260625.md"
    )
    assert selected["train_mode"] == "instruction"
    assert selected["aggregate"]["validation_loss_delta_avg"] == -0.132813


def test_multicard_qwen_lora_finetune_report_name_uses_run_label():
    assert (
        multicard_qwen_lora_finetune_report_name(
            "multicard_qwen_lora_finetune_qwen3_06b_lora_instruction_sync_8card_steps6_20260625.json"
        )
        == "multicard-qwen-lora-finetune-qwen3_06b_lora_instruction_sync_8card_steps6_20260625.md"
    )
