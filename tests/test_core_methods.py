import numpy as np

from tidal.cap import compress_global_rank_sparsity, robust_pca
from tidal.qpruner import allocate_bitwidths, discrete_mutual_information, quantize_symmetric
from tidal.rankadaptor import ModuleProfile, allocate_ranks, config_cost, export_peft_rank_pattern
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
