import importlib.util
from pathlib import Path

import torch

from tidal.methods.qpruner.torch import QuantizedLinear


def load_replay_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "qwen_qpruner_grouped_replay.py"
    assert script.exists(), f"missing grouped replay script: {script}"
    spec = importlib.util.spec_from_file_location("qwen_qpruner_grouped_replay", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_quantized_linear(*, in_features: int, out_features: int, seed: int) -> QuantizedLinear:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    linear = torch.nn.Linear(in_features, out_features)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(out_features, in_features, generator=generator) * 0.02)
        linear.bias.copy_(torch.randn(out_features, generator=generator) * 0.02)
    module = QuantizedLinear.from_linear(linear, bits=8).eval()
    module.enable_scaled_code_matmul(device="cpu")
    module.enable_aux_cache(dtype=torch.float32, device="cpu")
    return module


def make_released_quantized_linear(*, in_features: int, out_features: int, seed: int) -> QuantizedLinear:
    module = make_quantized_linear(in_features=in_features, out_features=out_features, seed=seed)
    module.enable_scaled_code_matmul(device="cpu", release_packed_codes=True)
    module.enable_aux_cache(dtype=torch.float32, device="cpu")
    assert module.packed_weight_codes.numel() == 0
    assert module.scaled_code_matmul_enabled
    return module


class TinyQwenLike(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList()
        for layer_index in range(3):
            layer = torch.nn.Module()
            layer.self_attn = torch.nn.Module()
            layer.self_attn.q_proj = make_quantized_linear(
                in_features=8,
                out_features=16,
                seed=100 + layer_index,
            )
            layer.self_attn.k_proj = make_quantized_linear(
                in_features=8,
                out_features=8,
                seed=200 + layer_index,
            )
            layer.mlp = torch.nn.Module()
            layer.mlp.down_proj = make_quantized_linear(
                in_features=12,
                out_features=8,
                seed=300 + layer_index,
            )
            self.model.layers.append(layer)


def test_collect_groupable_quantized_linears_groups_real_projection_roles_by_shape():
    replay = load_replay_module()
    model = TinyQwenLike()

    groups = replay.collect_groupable_quantized_linears(model)

    assert ("q_proj", 8, 16) in groups
    assert ("k_proj", 8, 8) in groups
    assert ("down_proj", 12, 8) in groups
    assert len(groups[("q_proj", 8, 16)]) == 3
    assert groups[("q_proj", 8, 16)][0].name.endswith("self_attn.q_proj")
    assert all(item.module.scaled_code_matmul_enabled for item in groups[("q_proj", 8, 16)])


def test_replay_quantized_group_reports_memory_native_grouped_speedup():
    replay = load_replay_module()
    model = TinyQwenLike()
    group = replay.collect_groupable_quantized_linears(model)[("q_proj", 8, 16)]

    row = replay.replay_quantized_group(
        group=group,
        role="q_proj",
        in_features=8,
        out_features=16,
        batch_size=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
        warmup=0,
        iters=2,
        seed=123,
        max_modules=3,
    )

    assert row["role"] == "q_proj"
    assert row["shape"] == {"batch_size": 4, "in_features": 8, "out_features": 16}
    assert row["module_count"] == 3
    assert row["sample_module_names"][0].endswith("self_attn.q_proj")
    assert row["sequential_scaled_code_matmul_group"]["runtime_strategy"] == (
        "sequential_scaled_int8_code_matmul_group"
    )
    assert row["grouped_scaled_code_matmul"]["runtime_strategy"] == "grouped_scaled_int8_code_bmm"
    assert row["grouped_scaled_code_matmul"]["speedup_vs_sequential_scaled_code"] is not None
    assert row["grouped_scaled_code_matmul"]["memory_strategy"] == "int8_code_cache_plus_aux_tensors"
    assert row["grouped_code_cache_bytes"] == sum(item.module.cached_code_bytes for item in group)
    assert row["grouped_aux_cache_bytes"] == sum(item.module.cached_aux_bytes for item in group)
    assert row["grouped_scaled_code_cache_bytes"] == sum(item.module.cached_scaled_code_bytes for item in group)
    assert row["max_abs_diff_vs_sequential_scaled_code"] <= 1e-5


def test_replay_quantized_group_reuses_cache_after_packed_codes_are_released():
    replay = load_replay_module()
    group = [
        replay.QuantizedModuleRef(
            name=f"model.layers.{index}.self_attn.o_proj",
            module=make_released_quantized_linear(in_features=8, out_features=8, seed=400 + index),
        )
        for index in range(2)
    ]

    row = replay.replay_quantized_group(
        group=group,
        role="o_proj",
        in_features=8,
        out_features=8,
        batch_size=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
        warmup=0,
        iters=1,
        seed=456,
        max_modules=2,
    )

    assert row["role"] == "o_proj"
    assert row["max_abs_diff_vs_sequential_scaled_code"] <= 1e-5
    assert all(item.module.packed_weight_codes.numel() == 0 for item in group)


def test_default_accelerator_device_request_matches_index_zero_cache():
    replay = load_replay_module()

    assert replay.device_matches_request("npu:0", "npu")
    assert replay.device_matches_request("cuda:0", "cuda")
    assert replay.device_matches_request("npu:1", "npu:1")
    assert not replay.device_matches_request("npu:1", "npu:0")
