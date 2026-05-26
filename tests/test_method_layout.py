from pathlib import Path


def test_method_packages_are_the_primary_public_layout():
    import tidal.methods.rankadaptor as rankadaptor
    import tidal.methods.qpruner as qpruner
    import tidal.methods.qpruner.torch as qpruner_torch
    import tidal.methods.global_rank_sparsity as global_rank_sparsity
    import tidal.methods.global_rank_sparsity.torch as global_rank_sparsity_torch
    import tidal.methods.dynamic_operator_optimization.reference as dop_reference

    assert hasattr(rankadaptor, "online_incremental_rank_search")
    assert hasattr(qpruner, "bayesian_refine_bitwidths")
    assert hasattr(qpruner_torch, "collect_linear_mutual_information")
    assert hasattr(global_rank_sparsity, "optimize_global_rank_sparsity_for_matrices")
    assert hasattr(global_rank_sparsity_torch, "apply_global_cap_compression")
    assert hasattr(dop_reference, "lora_sgmv")


def test_legacy_imports_remain_thin_compatibility_exports():
    import tidal.rankadaptor as legacy_rankadaptor
    import tidal.qpruner_torch as legacy_qpruner_torch
    import tidal.cap_torch as legacy_cap_torch

    from tidal.methods.rankadaptor import online_incremental_rank_search
    from tidal.methods.qpruner.torch import collect_linear_mutual_information
    from tidal.methods.global_rank_sparsity.torch import apply_global_cap_compression

    assert legacy_rankadaptor.online_incremental_rank_search is online_incremental_rank_search
    assert legacy_qpruner_torch.collect_linear_mutual_information is collect_linear_mutual_information
    assert legacy_cap_torch.apply_global_cap_compression is apply_global_cap_compression


def test_examples_are_grouped_by_method():
    root = Path(__file__).resolve().parents[1]

    assert (root / "examples/rankadaptor/basic_rank_search.py").is_file()
    assert (root / "examples/rankadaptor/torch_peft_search.py").is_file()
    assert (root / "examples/qpruner/mi_bo_quantization.py").is_file()
    assert (root / "examples/qpruner/torch_quantize.py").is_file()
    assert (root / "examples/global_rank_sparsity/cap_policy_search.py").is_file()
    assert (root / "examples/global_rank_sparsity/torch_compress.py").is_file()
    assert (root / "examples/dynamic_operator_optimization/sgmv_reference.py").is_file()
