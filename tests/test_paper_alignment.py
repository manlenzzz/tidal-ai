import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)

from tidal.cap import optimize_global_rank_sparsity_for_matrices
from tidal.cap_torch import apply_global_cap_compression
from tidal.qpruner_torch import collect_linear_mutual_information
from tidal.rankadaptor import ModuleProfile, online_incremental_rank_search


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def test_rankadaptor_online_search_uses_task_evaluations_to_improve_rank_choice():
    profiles = [
        ModuleProfile("layer0", sensitivity=1.0, min_rank=1, max_rank=3, rank_step=1),
        ModuleProfile("layer1", sensitivity=1.0, min_rank=1, max_rank=3, rank_step=1),
    ]

    def evaluate(config):
        return -abs(config["layer0"] - 3) - 0.25 * abs(config["layer1"] - 1)

    result = online_incremental_rank_search(
        profiles,
        budget=4,
        evaluate_config=evaluate,
        initial_samples=3,
        iterations=3,
        candidate_samples=8,
        train_epochs=2,
        seed=4,
    )

    assert result.best_config == {"layer0": 3, "layer1": 1}
    assert len(result.observations) >= 4
    assert result.best_score == max(item.score for item in result.observations)


def test_qpruner_collects_mutual_information_from_pruned_model_outputs():
    torch.manual_seed(0)
    model = TinyBlock()
    batches = [torch.randn(8, 4), torch.randn(8, 4)]

    mi = collect_linear_mutual_information(model, batches, bins=4)

    assert set(mi) == {"fc1", "fc2"}
    assert all(value >= 0.0 for value in mi.values())


def test_cap_global_optimizer_allocates_one_budget_across_matrices():
    matrices = {
        "a": np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]]),
        "b": np.array([[4.0, 0.0, 0.0], [0.0, 0.2, 0.0]]),
    }

    result = optimize_global_rank_sparsity_for_matrices(
        matrices,
        total_budget=12,
        max_iter=60,
        policy_steps=8,
        samples_per_step=4,
        seed=0,
    )

    assert result.parameter_count <= 12
    assert set(result.compressions) == {"a", "b"}
    assert len(result.history) == 8


def test_cap_torch_global_compression_rewrites_linear_layers_under_one_budget():
    torch.manual_seed(0)
    model = TinyBlock()

    compressed = apply_global_cap_compression(
        model,
        total_budget=16,
        max_iter=40,
        policy_steps=6,
        samples_per_step=4,
        seed=0,
    )
    total_parameters = compressed.fc1.parameter_count + compressed.fc2.parameter_count

    assert total_parameters <= 16
    assert compressed(torch.randn(2, 4)).shape == (2, 2)
    assert isinstance(model.fc1, nn.Linear)
