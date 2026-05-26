import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tidal.rankadaptor import ModuleProfile, export_peft_rank_pattern, fit_log_performance_model, search_rank_allocation

profiles = [
    ModuleProfile("layers.0.q_proj", sensitivity=0.4, min_rank=2, max_rank=8, rank_step=2),
    ModuleProfile("layers.1.q_proj", sensitivity=1.2, min_rank=2, max_rank=8, rank_step=2),
    ModuleProfile("layers.2.q_proj", sensitivity=0.9, min_rank=2, max_rank=8, rank_step=2),
]

samples = [
    ({"layers.0.q_proj": 2, "layers.1.q_proj": 2, "layers.2.q_proj": 2}, 0.48),
    ({"layers.0.q_proj": 2, "layers.1.q_proj": 6, "layers.2.q_proj": 2}, 0.73),
    ({"layers.0.q_proj": 4, "layers.1.q_proj": 8, "layers.2.q_proj": 4}, 0.88),
    ({"layers.0.q_proj": 8, "layers.1.q_proj": 8, "layers.2.q_proj": 8}, 0.92),
]

model = fit_log_performance_model(samples, profiles)
result = search_rank_allocation(profiles, budget=16, performance_model=model)
print(export_peft_rank_pattern(result.config))
print({"score": round(result.score, 4), "cost": result.cost, "steps": len(result.history)})
