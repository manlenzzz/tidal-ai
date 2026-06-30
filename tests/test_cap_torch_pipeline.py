import torch
from torch import nn

from tidal.model_support import build_pruned_module_name_filter
from tidal.methods.global_rank_sparsity.torch import (
    CAPPackedLinear,
    build_cap_calibration_evaluator,
    collect_cap_targets,
    run_cap_compression,
)


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)


class TinyMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = TinyAttention()
        self.mlp = TinyMlp()


class TinyCausalLmStyleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyLayer()])
        self.lm_head = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        layer = self.model.layers[0]
        hidden = layer.self_attn.o_proj(layer.self_attn.q_proj(x))
        hidden = layer.mlp.down_proj(layer.mlp.gate_proj(hidden))
        return self.lm_head(hidden)


def mse_loss(model, batch):
    inputs, targets = batch
    return torch.nn.functional.mse_loss(model(inputs), targets)


def test_collect_cap_targets_composes_modern_roles_with_pruner_filter():
    model = TinyCausalLmStyleModel()
    pruner_filter = build_pruned_module_name_filter(
        {
            "base_model.model.model.layers.0.self_attn.q_proj.weight": torch.empty(4, 4),
            "model.layers.0.mlp.down_proj.weight_mask": torch.empty(4, 8),
            "lm_head.weight": torch.empty(4, 4),
        },
        target_roles="modern",
    )

    targets = collect_cap_targets(model, target_roles="modern", name_filter=pruner_filter)

    assert [target.name for target in targets] == [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.mlp.down_proj",
    ]
    assert all(target.parameter_count > 0 for target in targets)
    assert all(target.canonical_name for target in targets)


def test_cap_calibration_evaluator_scores_and_restores_original_modules():
    torch.manual_seed(0)
    model = TinyCausalLmStyleModel()
    targets = collect_cap_targets(model, target_roles="modern")
    batches = [(torch.randn(2, 4), torch.randn(2, 4)) for _ in range(2)]
    evaluator = build_cap_calibration_evaluator(model, targets, batches, mse_loss)
    original_q_proj = model.model.layers[0].self_attn.q_proj

    compressions = {
        target.name: target.identity_compression()
        for target in targets
        if target.name in {"model.layers.0.self_attn.q_proj", "model.layers.0.mlp.down_proj"}
    }
    loss = evaluator(compressions)

    assert isinstance(loss, float)
    assert loss >= 0.0
    assert model.model.layers[0].self_attn.q_proj is original_q_proj
    assert not isinstance(model.model.layers[0].self_attn.q_proj, CAPPackedLinear)


def test_run_cap_compression_returns_compressed_model_and_metadata():
    torch.manual_seed(0)
    model = TinyCausalLmStyleModel()
    batches = [(torch.randn(2, 4), torch.randn(2, 4))]

    result = run_cap_compression(
        model,
        total_budget=24,
        target_roles="modern",
        calibration_batches=batches,
        loss_fn=mse_loss,
        max_iter=20,
        policy_steps=4,
        samples_per_step=2,
        seed=0,
    )

    assert result.compressed_model is not model
    assert result.global_result.parameter_count <= 24
    assert result.targets
    assert result.layer_summaries
    assert isinstance(result.compressed_model.model.layers[0].self_attn.q_proj, CAPPackedLinear)
    assert isinstance(model.model.layers[0].self_attn.q_proj, nn.Linear)
    assert {summary.name for summary in result.layer_summaries} <= {target.name for target in result.targets}


def test_cap_packed_linear_low_rank_zero_sparse_forward_avoids_weight_reconstruction(monkeypatch):
    left = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
    right = torch.tensor([[4.0, 5.0]], dtype=torch.float32)
    sparse = torch.zeros(3, 2, dtype=torch.float32)
    packed = CAPPackedLinear(left, right, sparse, None, parameter_count=5)
    inputs = torch.tensor([[1.0, -1.0], [2.0, 3.0]], dtype=torch.float32)

    def fail_reconstruction():
        raise AssertionError("dense reconstruction should not be used for zero-sparse low-rank forward")

    monkeypatch.setattr(packed, "reconstructed_weight", fail_reconstruction)

    out = packed(inputs)
    expected = torch.nn.functional.linear(
        torch.nn.functional.linear(inputs, right),
        left,
    )

    assert torch.allclose(out, expected)


def test_cap_packed_linear_sparse_residual_forward_avoids_weight_reconstruction(monkeypatch):
    left = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
    right = torch.tensor([[4.0, 5.0]], dtype=torch.float32)
    sparse = torch.tensor([[0.0, 0.25], [0.5, 0.0], [0.0, -0.75]], dtype=torch.float32)
    packed = CAPPackedLinear(left, right, sparse, None, parameter_count=8)
    inputs = torch.tensor([[1.0, -1.0], [2.0, 3.0]], dtype=torch.float32)
    dense_weight = left @ right + sparse

    def fail_reconstruction():
        raise AssertionError("sparse-residual forward should not reconstruct dense weight")

    monkeypatch.setattr(packed, "reconstructed_weight", fail_reconstruction)

    out = packed(inputs)
    expected = torch.nn.functional.linear(inputs, dense_weight)

    assert torch.allclose(out, expected)


def test_cap_packed_linear_stores_sparse_residual_as_coordinates():
    left = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
    right = torch.tensor([[4.0, 5.0]], dtype=torch.float32)
    sparse = torch.tensor([[0.0, 0.25], [0.5, 0.0], [0.0, -0.75]], dtype=torch.float32)
    packed = CAPPackedLinear(left, right, sparse, None, parameter_count=8)

    assert "sparse" not in dict(packed.named_buffers())
    assert packed.sparse_indices.shape == (3, 2)
    assert packed.sparse_values.shape == (3,)
    assert torch.equal(packed.sparse_indices[:, 0], torch.tensor([0, 1, 2]))
    assert torch.equal(packed.sparse_indices[:, 1], torch.tensor([1, 0, 1]))
    assert torch.allclose(packed.sparse_values, torch.tensor([0.25, 0.5, -0.75]))
    assert torch.allclose(packed.reconstructed_weight(), left @ right + sparse)


def test_cap_packed_linear_inference_cache_reuses_reconstructed_weight(monkeypatch):
    left = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
    right = torch.tensor([[4.0, 5.0]], dtype=torch.float32)
    sparse = torch.ones(3, 2, dtype=torch.float32) * 0.25
    packed = CAPPackedLinear(left, right, sparse, None, parameter_count=11)
    inputs = torch.tensor([[1.0, -1.0], [2.0, 3.0]], dtype=torch.float32)
    expected = packed(inputs)

    packed.enable_inference_cache(dtype=torch.float32, device=torch.device("cpu"))
    cached = packed(inputs)
    assert torch.allclose(cached, expected)

    def fail_reconstruction():
        raise AssertionError("cached inference should not reconstruct dense weight")

    monkeypatch.setattr(packed, "reconstructed_weight", fail_reconstruction)
    cached_again = packed(inputs)

    assert torch.allclose(cached_again, expected)
