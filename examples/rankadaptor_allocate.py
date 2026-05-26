import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tidal.rankadaptor import ModuleProfile, allocate_ranks, export_peft_rank_pattern

profiles = [
    ModuleProfile("layers.0.q_proj", sensitivity=0.4, min_rank=2, max_rank=8, rank_step=2),
    ModuleProfile("layers.1.q_proj", sensitivity=1.2, min_rank=2, max_rank=8, rank_step=2),
    ModuleProfile("layers.2.q_proj", sensitivity=0.9, min_rank=2, max_rank=8, rank_step=2),
]

config = allocate_ranks(profiles, budget=16)
print(export_peft_rank_pattern(config))
