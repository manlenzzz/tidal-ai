import torch
from torch import nn

from tidal.model_support import ModuleRole, build_modern_hf_name_filter, list_linear_modules
from tidal.methods.global_rank_sparsity.torch import CAPPackedLinear, apply_global_cap_compression
from tidal.methods.qpruner.torch import build_quantization_plan, collect_linear_layer_sizes
from tidal.methods.rankadaptor import collect_linear_profiles


class QwenAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)


class QwenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 8, bias=False)
        self.up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)


class QwenLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = QwenAttention()
        self.mlp = QwenMLP()


class QwenStyleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([QwenLayer()])
        self.lm_head = nn.Linear(4, 4, bias=False)


class FusedBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv_proj = nn.Linear(4, 12, bias=False)
        self.attn.out_proj = nn.Linear(4, 4, bias=False)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(4, 8, bias=False)
        self.mlp.fc2 = nn.Linear(8, 4, bias=False)


class FusedQKVModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([FusedBlock()])
        self.lm_head = nn.Linear(4, 4, bias=False)


class MoEExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(4, 8, bias=False)
        self.w2 = nn.Linear(8, 4, bias=False)
        self.w3 = nn.Linear(4, 8, bias=False)


class MoEBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = QwenAttention()
        self.block_sparse_moe = nn.Module()
        self.block_sparse_moe.gate = nn.Linear(4, 2, bias=False)
        self.block_sparse_moe.experts = nn.ModuleList([MoEExpert(), MoEExpert()])


class DeepSeekStyleMoEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([MoEBlock()])
        self.embed_out = nn.Linear(4, 4, bias=False)
        self.lm_head = nn.Linear(4, 4, bias=False)


def roles_by_name(model):
    return {info.name: info.role for info in list_linear_modules(model)}


def test_classifies_modern_hf_linear_module_roles():
    qwen_roles = roles_by_name(QwenStyleModel())
    fused_roles = roles_by_name(FusedQKVModel())
    moe_roles = roles_by_name(DeepSeekStyleMoEModel())

    assert qwen_roles["model.layers.0.self_attn.q_proj"] is ModuleRole.ATTENTION
    assert qwen_roles["model.layers.0.self_attn.o_proj"] is ModuleRole.ATTENTION
    assert qwen_roles["model.layers.0.mlp.gate_proj"] is ModuleRole.MLP
    assert qwen_roles["model.layers.0.mlp.down_proj"] is ModuleRole.MLP
    assert qwen_roles["lm_head"] is ModuleRole.OUTPUT_HEAD

    assert fused_roles["transformer.h.0.attn.qkv_proj"] is ModuleRole.ATTENTION
    assert fused_roles["transformer.h.0.attn.out_proj"] is ModuleRole.ATTENTION
    assert fused_roles["transformer.h.0.mlp.fc1"] is ModuleRole.MLP

    assert moe_roles["model.layers.0.block_sparse_moe.gate"] is ModuleRole.MOE_ROUTER
    assert moe_roles["model.layers.0.block_sparse_moe.experts.0.w1"] is ModuleRole.MOE_EXPERT
    assert moe_roles["model.layers.0.block_sparse_moe.experts.1.w2"] is ModuleRole.MOE_EXPERT
    assert moe_roles["embed_out"] is ModuleRole.EMBEDDING_PROJECTION
    assert moe_roles["lm_head"] is ModuleRole.OUTPUT_HEAD


def test_modern_default_selector_includes_method_targets_and_excludes_unsafe_modules():
    model = DeepSeekStyleMoEModel()
    selected = {info.name for info in list_linear_modules(model, target_roles="modern")}

    assert "model.layers.0.self_attn.q_proj" in selected
    assert "model.layers.0.block_sparse_moe.experts.0.w1" in selected
    assert "model.layers.0.block_sparse_moe.gate" not in selected
    assert "embed_out" not in selected
    assert "lm_head" not in selected

    include_router = {info.name for info in list_linear_modules(model, target_roles={ModuleRole.MOE_ROUTER})}
    assert include_router == {"model.layers.0.block_sparse_moe.gate"}

    name_filter = build_modern_hf_name_filter(target_roles="modern")
    assert name_filter("model.layers.0.self_attn.q_proj")
    assert not name_filter("model.layers.0.block_sparse_moe.gate")
    assert not name_filter("lm_head")


def test_methods_use_modern_target_selector_without_custom_name_filter():
    torch.manual_seed(0)
    model = DeepSeekStyleMoEModel()

    profiles = collect_linear_profiles(model, target_roles="modern", min_rank=1, max_rank=2)
    profiled_names = {profile.name for profile in profiles}
    assert "model.layers.0.self_attn.q_proj" in profiled_names
    assert "model.layers.0.block_sparse_moe.experts.0.w1" in profiled_names
    assert "model.layers.0.block_sparse_moe.gate" not in profiled_names
    assert "lm_head" not in profiled_names

    sizes = collect_linear_layer_sizes(model, target_roles="modern")
    plan = build_quantization_plan(
        model,
        {name: 1.0 for name in sizes},
        target_roles="modern",
        candidate_bits=(2, 4),
        max_average_bits=3.0,
    )
    assert set(plan.layer_sizes) == set(sizes)
    assert "model.layers.0.block_sparse_moe.gate" not in plan.layer_sizes

    compressed = apply_global_cap_compression(
        model,
        total_budget=sum(max(1, size // 2) for size in sizes.values()),
        target_roles="modern",
        max_iter=10,
        policy_steps=4,
        samples_per_step=2,
        seed=0,
    )
    assert isinstance(compressed.model.layers[0].self_attn.q_proj, CAPPackedLinear)
    assert isinstance(compressed.model.layers[0].block_sparse_moe.experts[0].w1, CAPPackedLinear)
    assert isinstance(compressed.model.layers[0].block_sparse_moe.gate, nn.Linear)
    assert isinstance(compressed.lm_head, nn.Linear)
