import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from peft import get_peft_model

from tidal.rankadaptor import build_lora_config, collect_linear_profiles, online_incremental_rank_search


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


model = TinyBlock()
profiles = collect_linear_profiles(
    model,
    sensitivities={"fc1": 2.0, "fc2": 0.2},
    min_rank=1,
    max_rank=2,
    rank_step=1,
)

def evaluate_config(config):
    return -abs(config["fc1"] - 2) - 0.25 * abs(config["fc2"] - 1)


result = online_incremental_rank_search(
    profiles,
    budget=19,
    evaluate_config=evaluate_config,
    initial_samples=2,
    iterations=2,
    candidate_samples=4,
    train_epochs=2,
    seed=1,
)
config = build_lora_config(result.best_config, alpha_multiplier=2)
peft_model = get_peft_model(model, config)

trainable = sum(parameter.numel() for parameter in peft_model.parameters() if parameter.requires_grad)
print({"ranks": result.best_config, "adapter_parameters": trainable, "evaluations": len(result.observations)})
print(peft_model(torch.randn(2, 4)).shape)
