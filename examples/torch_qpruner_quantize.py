import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from tidal.qpruner_torch import apply_mixed_precision_quantization, build_quantization_plan


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
plan = build_quantization_plan(
    model,
    {"fc1": 2.0, "fc2": 0.1},
    candidate_bits=(2, 4, 8),
    max_average_bits=3.5,
)
quantized = apply_mixed_precision_quantization(model, plan)
output = quantized(torch.randn(2, 4))

print({"bitwidths": plan.bitwidths, "layer_sizes": plan.layer_sizes})
print({"backend": type(quantized.fc1).__name__, "output_shape": tuple(output.shape)})
