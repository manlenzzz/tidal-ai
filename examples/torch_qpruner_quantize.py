import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from tidal.qpruner_torch import (
    apply_mixed_precision_quantization,
    build_quantization_plan,
    collect_linear_mutual_information,
)


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
with torch.no_grad():
    model.fc1.weight.zero_()
    model.fc1.bias.zero_()
    model.fc1.weight[0, 0] = 1.0
    model.fc1.weight[1, 1] = 1.0
    model.fc1.weight[2, 2] = 1.0
    model.fc2.weight.zero_()
    model.fc2.bias[:] = torch.tensor([0.0, 1.0])
    model.fc2.weight[0, 0] = 2.0

calibration_batches = [torch.randn(8, 4), torch.randn(8, 4)]
importance = collect_linear_mutual_information(model, calibration_batches, bins=4)
plan = build_quantization_plan(
    model,
    importance,
    candidate_bits=(2, 4, 8),
    max_average_bits=3.5,
)
quantized = apply_mixed_precision_quantization(model, plan)
output = quantized(torch.randn(2, 4))

print({"importance": {name: round(value, 4) for name, value in importance.items()}})
print({"bitwidths": plan.bitwidths, "layer_sizes": plan.layer_sizes})
print({"backend": type(quantized.fc1).__name__, "output_shape": tuple(output.shape)})
