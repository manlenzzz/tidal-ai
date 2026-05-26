import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from tidal.cap_torch import apply_global_cap_compression, summarize_cap_layers


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


torch.manual_seed(0)
model = TinyBlock()
compressed = apply_global_cap_compression(
    model,
    total_budget=16,
    max_iter=60,
    policy_steps=12,
    samples_per_step=4,
    seed=0,
)
output = compressed(torch.randn(2, 4))

print([result.__dict__ for result in summarize_cap_layers(compressed)])
print({"output_shape": tuple(output.shape)})
