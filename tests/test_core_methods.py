import numpy as np

from tidal.cap import compress_global_rank_sparsity, optimize_global_rank_sparsity, robust_pca
from tidal.qpruner import (
    allocate_bitwidths,
    bayesian_refine_bitwidths,
    config_memory_bits,
    discrete_mutual_information,
    enumerate_bitwidth_configs,
    quantize_symmetric,
)
from tidal.rankadaptor import (
    ModuleProfile,
    allocate_ranks,
    config_cost,
    export_peft_rank_pattern,
    fit_log_performance_model,
    search_rank_allocation,
)
from tidal.sgmv import Segment, lora_sgmv, sgmv


def test_rankadaptor_allocates_more_rank_to_sensitive_modules():
    profiles = [
        ModuleProfile("low", sensitivity=0.2, min_rank=1, max_rank=5),
        ModuleProfile("high", sensitivity=2.0, min_rank=1, max_rank=5),
    ]

    config = allocate_ranks(profiles, budget=6)

    assert config_cost(config, profiles) <= 6
    assert config["high"] > config["low"]
    assert export_peft_rank_pattern(config)["high"] == {"r": config["high"], "lora_alpha": 2 * config["high"]}


def test_rankadaptor_fits_performance_model_and_returns_search_trace():
    profiles = [
        ModuleProfile("block.0.q", sensitivity=0.4, min_rank=1, max_rank=5, rank_step=1),
        ModuleProfile("block.1.q", sensitivity=1.6, min_rank=1, max_rank=5, rank_step=1),
        ModuleProfile("block.2.q", sensitivity=0.9, min_rank=1, max_rank=5, rank_step=1),
    ]
    samples = [
        ({"block.0.q": 1, "block.1.q": 1, "block.2.q": 1}, 0.35),
        ({"block.0.q": 1, "block.1.q": 3, "block.2.q": 1}, 0.72),
        ({"block.0.q": 2, "block.1.q": 4, "block.2.q": 2}, 0.91),
        ({"block.0.q": 4, "block.1.q": 4, "block.2.q": 4}, 1.02),
    ]

    model = fit_log_performance_model(samples, profiles)
    result = search_rank_allocation(profiles, budget=9, performance_model=model, max_steps=8)

    assert result.cost <= 9
    assert result.config["block.1.q"] >= result.config["block.0.q"]
    assert len(result.history) >= 2
    assert result.history[-1].score >= result.history[0].score


def test_qpruner_mi_and_bitwidth_allocation_prioritize_informative_layer():
    rng = np.random.default_rng(0)
    y = rng.normal(size=256)
    informative = y[:, None] + 0.05 * rng.normal(size=(256, 3))
    noise = rng.normal(size=(256, 3))

    mi_informative = discrete_mutual_information(informative, y, bins=8)
    mi_noise = discrete_mutual_information(noise, y, bins=8)
    config = allocate_bitwidths(
        {"informative": mi_informative, "noise": mi_noise},
        {"informative": 100, "noise": 100},
        candidate_bits=(2, 4, 8),
        max_average_bits=4,
    )

    assert mi_informative > mi_noise
    assert config["informative"] >= config["noise"]


def test_qpruner_symmetric_quantization_roundtrips_shape_and_bounds():
    tensor = np.array([-1.0, -0.2, 0.2, 1.0])
    quantized = quantize_symmetric(tensor, bits=4)

    assert quantized.codes.shape == tensor.shape
    assert quantized.codes.min() >= -7
    assert quantized.codes.max() <= 7
    assert np.allclose(quantized.dequantize(), tensor, atol=quantized.scale / 2 + 1e-12)


def test_qpruner_bayesian_refinement_evaluates_feasible_configs_and_keeps_best():
    importances = {"layers.0": 0.2, "layers.1": 1.4, "layers.2": 0.7}
    layer_sizes = {name: 64 for name in importances}
    candidates = enumerate_bitwidth_configs(
        list(importances),
        layer_sizes,
        candidate_bits=(2, 4, 8),
        max_average_bits=4,
    )

    def objective(config):
        return sum(importances[name] * np.log2(config[name]) for name in config)

    result = bayesian_refine_bitwidths(
        importances,
        layer_sizes,
        objective=objective,
        candidate_bits=(2, 4, 8),
        max_average_bits=4,
        max_trials=5,
        seed=0,
    )

    assert result.config in candidates
    assert config_memory_bits(result.config, layer_sizes) <= sum(layer_sizes.values()) * 4
    assert result.config["layers.1"] >= result.config["layers.0"]
    assert len(result.evaluations) == 5
    assert result.score == max(item.score for item in result.evaluations)


def test_cap_rpca_and_budgeted_compression_return_valid_matrix():
    low_rank = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    sparse = np.zeros_like(low_rank)
    sparse[0, 1] = 5.0
    matrix = low_rank + sparse

    result = robust_pca(matrix, max_iter=80, tol=1e-4)
    compressed = compress_global_rank_sparsity(matrix, budget=8, max_iter=80)

    assert result.low_rank.shape == matrix.shape
    assert result.sparse.shape == matrix.shape
    assert compressed.reconstructed.shape == matrix.shape
    assert compressed.parameter_count <= 8


def test_cap_policy_search_returns_feasible_global_rank_sparse_solution():
    low_rank = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]])
    sparse = np.zeros_like(low_rank)
    sparse[0, 2] = 4.0
    sparse[2, 0] = -3.0
    matrix = low_rank + sparse

    result = optimize_global_rank_sparsity(
        matrix,
        budget=9,
        max_iter=80,
        policy_steps=40,
        samples_per_step=6,
        learning_rate=0.15,
        seed=7,
    )

    assert result.compression.parameter_count <= 9
    assert result.compression.reconstructed.shape == matrix.shape
    assert len(result.history) == 40
    assert result.best_reward >= result.history[0].reward


def test_sgmv_matches_segmented_matmul_reference():
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    weights = np.array([
        [[1.0, 0.0], [0.0, 1.0]],
        [[2.0, 0.0], [0.0, 3.0]],
    ])
    segments = [Segment(0, 2, 0), Segment(2, 3, 1)]

    out = sgmv(x, weights, segments)

    expected = np.vstack([x[:2] @ weights[0], x[2:] @ weights[1]])
    assert np.allclose(out, expected)


def test_lora_sgmv_matches_two_stage_reference():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    a = np.array([[[1.0], [2.0]]])
    b = np.array([[[3.0, 4.0]]])
    segments = [Segment(0, 2, 0)]

    out = lora_sgmv(x, a, b, segments)

    assert np.allclose(out, (x @ a[0]) @ b[0])
