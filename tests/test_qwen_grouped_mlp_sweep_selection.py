import json
from pathlib import Path

from tidal.reports.qwen_grouped_mlp_sweep import (
    best_qwen_grouped_mlp_memory_tradeoff,
    best_qwen_grouped_mlp_sweep,
    qwen_grouped_mlp_memory_tradeoff_readout,
    qwen_grouped_mlp_sweep_readout,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_best_qwen_grouped_mlp_sweep_includes_nocopy_sweep(tmp_path):
    artifacts = tmp_path / "artifacts"
    write_json(
        artifacts / "qwen3_06b_native_profile_8card_grouped_mlp_sweep_20260624.json",
        {
            "status": "PASS",
            "profile_count": 8,
            "pass_count": 8,
            "max_active_process_cards": 8,
            "samples_with_8_nonzero_aicore": 35,
            "samples_with_8_active_process_cards": 18,
            "best_speed": {"paired_latency_speedup": 0.808},
            "best_code_cache_reduction": {"code_cache_storage_reduction_pct": 50.0},
            "rows": [{"grouped_mlp_pairs": 28}],
        },
    )
    write_json(
        artifacts / "qwen3_06b_native_profile_8card_grouped_mlp_nocopy_sweep_20260624.json",
        {
            "status": "PASS",
            "profile_count": 8,
            "pass_count": 8,
            "max_active_process_cards": 8,
            "samples_with_8_nonzero_aicore": 36,
            "samples_with_8_active_process_cards": 19,
            "best_speed": {"paired_latency_speedup": 0.824},
            "best_code_cache_reduction": {"code_cache_storage_reduction_pct": 50.0},
            "rows": [{"grouped_mlp_pairs": 28, "grouped_attention_kv_pairs": 28}],
        },
    )

    sweep = best_qwen_grouped_mlp_sweep(artifacts)

    assert sweep is not None
    assert sweep["_artifact_name"] == "qwen3_06b_native_profile_8card_grouped_mlp_nocopy_sweep_20260624.json"
    readout = qwen_grouped_mlp_sweep_readout(sweep)
    assert "best paired speedup 0.824x" in readout
    assert "grouped K/V pairs 28" in readout
    assert "all-8 AICore samples 36" in readout


def test_qwen_grouped_mlp_sweep_readout_omits_kv_clause_for_legacy_sweeps():
    readout = qwen_grouped_mlp_sweep_readout(
        {
            "status": "PASS",
            "profile_count": 8,
            "pass_count": 8,
            "max_active_process_cards": 8,
            "samples_with_8_nonzero_aicore": 36,
            "best_speed": {"paired_latency_speedup": 0.824},
            "best_code_cache_reduction": {"code_cache_storage_reduction_pct": 50.0},
            "rows": [{"grouped_mlp_pairs": 28}],
        }
    )

    assert "grouped MLP pairs 28" in readout
    assert "grouped K/V pairs" not in readout


def test_best_qwen_grouped_mlp_sweep_prefers_speed_after_card_coverage(tmp_path):
    artifacts = tmp_path / "artifacts"
    write_json(
        artifacts / "qwen3_06b_native_profile_8card_grouped_mlp_nocopy_sweep_many_samples_20260624.json",
        {
            "status": "PASS",
            "profile_count": 8,
            "pass_count": 8,
            "max_active_process_cards": 8,
            "samples_with_8_nonzero_aicore": 36,
            "samples_with_8_active_process_cards": 19,
            "best_speed": {"paired_latency_speedup": 0.808},
            "best_code_cache_reduction": {"code_cache_storage_reduction_pct": 50.0},
            "rows": [{"grouped_mlp_pairs": 28}],
        },
    )
    write_json(
        artifacts / "qwen3_06b_native_profile_8card_grouped_mlp_nocopy_sweep_faster_20260624.json",
        {
            "status": "PASS",
            "profile_count": 8,
            "pass_count": 8,
            "max_active_process_cards": 8,
            "samples_with_8_nonzero_aicore": 25,
            "samples_with_8_active_process_cards": 13,
            "best_speed": {"paired_latency_speedup": 0.836},
            "best_code_cache_reduction": {"code_cache_storage_reduction_pct": 50.0},
            "rows": [{"grouped_mlp_pairs": 28}],
        },
    )

    sweep = best_qwen_grouped_mlp_sweep(artifacts)

    assert sweep is not None
    assert sweep["_artifact_name"] == "qwen3_06b_native_profile_8card_grouped_mlp_nocopy_sweep_faster_20260624.json"
    assert "best paired speedup 0.836x" in qwen_grouped_mlp_sweep_readout(sweep)


def test_best_qwen_grouped_mlp_memory_tradeoff_prefers_high_speed_with_memory_budget(tmp_path):
    artifacts = tmp_path / "artifacts"
    write_json(
        artifacts / "qwen3_06b_native_profile_8card_grouped_mlp_nocopy_sweep_20260624.json",
        {
            "status": "PASS",
            "profile_count": 8,
            "pass_count": 8,
            "max_active_process_cards": 8,
            "samples_with_8_nonzero_aicore": 36,
            "samples_with_8_active_process_cards": 19,
            "best_speed": {"label": "bits8_scaledcode_prebuilt", "paired_latency_speedup": 0.824},
            "best_code_cache_reduction": {"label": "bits4_scaledcode", "code_cache_storage_reduction_pct": 50.0},
            "rows": [
                {
                    "label": "bits8_scaledcode_prebuilt",
                    "status": "PASS",
                    "paired_latency_speedup": 0.824,
                    "code_cache_storage_reduction_pct": -50.0,
                    "grouped_mlp_pairs": 28,
                },
                {
                    "label": "bits4_scaledcode",
                    "status": "PASS",
                    "paired_latency_speedup": 0.771,
                    "code_cache_storage_reduction_pct": 50.0,
                    "grouped_mlp_pairs": 28,
                },
            ],
        },
    )
    write_json(
        artifacts / "qwen3_06b_native_profile_8card_grouped_mlp_nocopy_sweep_densebudget_20260624.json",
        {
            "status": "PASS",
            "profile_count": 8,
            "pass_count": 8,
            "max_active_process_cards": 8,
            "samples_with_8_nonzero_aicore": 35,
            "samples_with_8_active_process_cards": 18,
            "best_speed": {"label": "densebudget96m", "paired_latency_speedup": 0.810},
            "best_code_cache_reduction": {
                "label": "densebudget0m",
                "code_cache_storage_reduction_pct": 50.0,
            },
            "rows": [
                {
                    "label": "densebudget8m",
                    "status": "PASS",
                    "paired_latency_speedup": 0.808,
                    "code_cache_storage_reduction_pct": 49.524,
                    "scaled_code_dtype_cache_budget_bytes": 0,
                    "dense_cache_budget_bytes": 8 * 1024 * 1024,
                    "grouped_mlp_pairs": 28,
                },
                {
                    "label": "densebudget96m",
                    "status": "PASS",
                    "paired_latency_speedup": 0.810,
                    "code_cache_storage_reduction_pct": 44.286,
                    "dense_cache_budget_bytes": 96 * 1024 * 1024,
                    "grouped_mlp_pairs": 28,
                },
            ],
        },
    )

    tradeoff = best_qwen_grouped_mlp_memory_tradeoff(artifacts)

    assert tradeoff is not None
    assert tradeoff["_artifact_name"] == "qwen3_06b_native_profile_8card_grouped_mlp_nocopy_sweep_densebudget_20260624.json"
    assert tradeoff["_memory_tradeoff_row"]["label"] == "densebudget8m"
    readout = qwen_grouped_mlp_memory_tradeoff_readout(tradeoff)
    assert "memory-first tradeoff" in readout
    assert "densebudget8m" in readout
    assert "0.808x" in readout
    assert "49.524%" in readout
    assert "scaled-code dtype-cache budget 0 bytes" in readout
    assert "8388608 bytes" in readout
    assert "all-8 AICore samples 35" in readout


def test_best_qwen_grouped_mlp_memory_tradeoff_readout_includes_scaled_dtype_budget(tmp_path):
    artifacts = tmp_path / "artifacts"
    write_json(
        artifacts / "qwen3_06b_native_profile_8card_grouped_mlp_nocopy_sweep_scaleddtype_20260624.json",
        {
            "status": "PASS",
            "profile_count": 8,
            "pass_count": 8,
            "max_active_process_cards": 8,
            "samples_with_8_nonzero_aicore": 36,
            "samples_with_8_active_process_cards": 18,
            "best_speed": {"label": "scaleddtype8m", "paired_latency_speedup": 0.812},
            "best_code_cache_reduction": {
                "label": "scaleddtype0m",
                "code_cache_storage_reduction_pct": 50.0,
            },
            "rows": [
                {
                    "label": "scaleddtype8m",
                    "status": "PASS",
                    "paired_latency_speedup": 0.812,
                    "code_cache_storage_reduction_pct": 49.047,
                    "scaled_code_dtype_cache_budget_bytes": 8 * 1024 * 1024,
                    "dense_cache_budget_bytes": 0,
                    "grouped_mlp_pairs": 28,
                },
                {
                    "label": "scaleddtype32m",
                    "status": "PASS",
                    "paired_latency_speedup": 0.826,
                    "code_cache_storage_reduction_pct": 46.190,
                    "scaled_code_dtype_cache_budget_bytes": 32 * 1024 * 1024,
                    "dense_cache_budget_bytes": 0,
                    "grouped_mlp_pairs": 28,
                },
            ],
        },
    )

    tradeoff = best_qwen_grouped_mlp_memory_tradeoff(artifacts)

    assert tradeoff is not None
    assert tradeoff["_memory_tradeoff_row"]["label"] == "scaleddtype8m"
    readout = qwen_grouped_mlp_memory_tradeoff_readout(tradeoff)
    assert "0.812x" in readout
    assert "49.047%" in readout
    assert "scaled-code dtype-cache budget 8388608 bytes" in readout
    assert "dense-cache budget 0 bytes" in readout
