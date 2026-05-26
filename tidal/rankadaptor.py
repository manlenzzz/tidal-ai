from __future__ import annotations

from dataclasses import dataclass
from math import log1p
from typing import Callable, Iterable

RankConfig = dict[str, int]
PerformanceModel = Callable[[RankConfig], float]


@dataclass(frozen=True)
class ModuleProfile:
    """Metadata used to allocate LoRA ranks for a pruned module."""

    name: str
    sensitivity: float
    min_rank: int = 0
    max_rank: int = 64
    rank_step: int = 1
    cost_per_rank: int = 1

    def __post_init__(self) -> None:
        if self.min_rank < 0:
            raise ValueError(f"{self.name}: min_rank must be non-negative")
        if self.max_rank < self.min_rank:
            raise ValueError(f"{self.name}: max_rank must be >= min_rank")
        if self.rank_step <= 0:
            raise ValueError(f"{self.name}: rank_step must be positive")
        if self.cost_per_rank <= 0:
            raise ValueError(f"{self.name}: cost_per_rank must be positive")
        if self.sensitivity < 0:
            raise ValueError(f"{self.name}: sensitivity must be non-negative")


def config_cost(config: RankConfig, profiles: Iterable[ModuleProfile]) -> int:
    by_name = {profile.name: profile for profile in profiles}
    return sum(config[name] * by_name[name].cost_per_rank for name in config)


def default_recovery_score(config: RankConfig, profiles: Iterable[ModuleProfile]) -> float:
    by_name = {profile.name: profile for profile in profiles}
    return sum(by_name[name].sensitivity * log1p(rank) for name, rank in config.items())


def allocate_ranks(
    profiles: Iterable[ModuleProfile],
    *,
    budget: int,
    performance_model: PerformanceModel | None = None,
) -> RankConfig:
    """Allocate hierarchical adapter ranks under a parameter/rank budget."""

    profile_list = list(profiles)
    if not profile_list:
        raise ValueError("profiles must not be empty")
    if budget < 0:
        raise ValueError("budget must be non-negative")

    config: RankConfig = {profile.name: profile.min_rank for profile in profile_list}
    spent = config_cost(config, profile_list)
    if spent > budget:
        raise ValueError("minimum ranks exceed budget")

    scorer = performance_model or (lambda cfg: default_recovery_score(cfg, profile_list))

    while True:
        current_score = scorer(dict(config))
        best_name: str | None = None
        best_rank = 0
        best_gain_per_cost = 0.0
        best_gain = 0.0
        best_added_cost = 0

        for profile in profile_list:
            current_rank = config[profile.name]
            next_rank = min(profile.max_rank, current_rank + profile.rank_step)
            if next_rank == current_rank:
                continue
            added_cost = (next_rank - current_rank) * profile.cost_per_rank
            if spent + added_cost > budget:
                continue

            candidate = dict(config)
            candidate[profile.name] = next_rank
            gain = scorer(candidate) - current_score
            gain_per_cost = gain / added_cost
            if (gain_per_cost, gain, -added_cost) > (best_gain_per_cost, best_gain, -best_added_cost):
                best_name = profile.name
                best_rank = next_rank
                best_gain_per_cost = gain_per_cost
                best_gain = gain
                best_added_cost = added_cost

        if best_name is None or best_gain <= 0:
            return config

        spent += best_added_cost
        config[best_name] = best_rank


def export_peft_rank_pattern(config: RankConfig, *, alpha_multiplier: int = 2) -> dict[str, dict[str, int]]:
    if alpha_multiplier <= 0:
        raise ValueError("alpha_multiplier must be positive")
    return {
        name: {"r": rank, "lora_alpha": rank * alpha_multiplier}
        for name, rank in sorted(config.items())
    }
