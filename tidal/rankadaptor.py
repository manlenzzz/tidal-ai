from __future__ import annotations

from dataclasses import dataclass
from math import log1p
from typing import Callable, Iterable, Sequence

import numpy as np

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


@dataclass(frozen=True)
class RankSearchStep:
    step: int
    config: RankConfig
    score: float
    cost: int
    changed_module: str | None = None


@dataclass(frozen=True)
class RankSearchResult:
    config: RankConfig
    score: float
    cost: int
    history: tuple[RankSearchStep, ...]


def config_cost(config: RankConfig, profiles: Iterable[ModuleProfile]) -> int:
    by_name = {profile.name: profile for profile in profiles}
    return sum(config[name] * by_name[name].cost_per_rank for name in config)


def default_recovery_score(config: RankConfig, profiles: Iterable[ModuleProfile]) -> float:
    by_name = {profile.name: profile for profile in profiles}
    return sum(by_name[name].sensitivity * log1p(rank) for name, rank in config.items())


def _feature_vector(config: RankConfig, profiles: Sequence[ModuleProfile]) -> np.ndarray:
    return np.array(
        [1.0] + [profile.sensitivity * log1p(float(config[profile.name])) for profile in profiles],
        dtype=np.float64,
    )


def fit_log_performance_model(
    samples: Sequence[tuple[RankConfig, float]],
    profiles: Iterable[ModuleProfile],
    *,
    ridge: float = 1e-6,
) -> PerformanceModel:
    """Fit the RankAdaptor log-rank performance surrogate from measured configs."""

    profile_list = list(profiles)
    if not profile_list:
        raise ValueError("profiles must not be empty")
    if not samples:
        raise ValueError("samples must not be empty")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")

    names = {profile.name for profile in profile_list}
    x_rows: list[np.ndarray] = []
    y_values: list[float] = []
    for config, score in samples:
        if set(config) != names:
            raise ValueError("each sample config must contain exactly the profiled modules")
        x_rows.append(_feature_vector(config, profile_list))
        y_values.append(float(score))

    x = np.vstack(x_rows)
    y = np.asarray(y_values, dtype=np.float64)
    regularizer = ridge * np.eye(x.shape[1], dtype=np.float64)
    regularizer[0, 0] = 0.0
    weights = np.linalg.solve(x.T @ x + regularizer, x.T @ y)

    def predict(config: RankConfig) -> float:
        if set(config) != names:
            raise ValueError("config must contain exactly the profiled modules")
        return float(_feature_vector(config, profile_list) @ weights)

    return predict


def search_rank_allocation(
    profiles: Iterable[ModuleProfile],
    *,
    budget: int,
    performance_model: PerformanceModel | None = None,
    initial_config: RankConfig | None = None,
    max_steps: int | None = None,
    min_gain: float = 0.0,
) -> RankSearchResult:
    """Coordinate-search rank allocation with a fitted or analytic performance model."""

    profile_list = list(profiles)
    if not profile_list:
        raise ValueError("profiles must not be empty")
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if min_gain < 0:
        raise ValueError("min_gain must be non-negative")

    names = {profile.name for profile in profile_list}
    if initial_config is None:
        config: RankConfig = {profile.name: profile.min_rank for profile in profile_list}
    else:
        if set(initial_config) != names:
            raise ValueError("initial_config must contain exactly the profiled modules")
        config = dict(initial_config)

    for profile in profile_list:
        rank = config[profile.name]
        if rank < profile.min_rank or rank > profile.max_rank:
            raise ValueError(f"{profile.name}: initial rank outside profile bounds")

    spent = config_cost(config, profile_list)
    if spent > budget:
        raise ValueError("initial ranks exceed budget")

    scorer = performance_model or (lambda cfg: default_recovery_score(cfg, profile_list))
    score = scorer(dict(config))
    history = [RankSearchStep(0, dict(config), score, spent, None)]
    limit = max_steps if max_steps is not None else sum(
        (profile.max_rank - config[profile.name]) // profile.rank_step for profile in profile_list
    )

    for step in range(1, limit + 1):
        best: tuple[float, float, str, int, int, float] | None = None
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
            candidate_score = scorer(candidate)
            gain = candidate_score - score
            key = (gain / added_cost, gain, profile.name, next_rank, added_cost, candidate_score)
            if best is None or key > best:
                best = key

        if best is None or best[1] <= min_gain:
            break

        _, gain, name, next_rank, added_cost, candidate_score = best
        config[name] = next_rank
        spent += added_cost
        score = candidate_score
        history.append(RankSearchStep(step, dict(config), score, spent, name))

    return RankSearchResult(dict(config), float(score), spent, tuple(history))


def allocate_ranks(
    profiles: Iterable[ModuleProfile],
    *,
    budget: int,
    performance_model: PerformanceModel | None = None,
) -> RankConfig:
    """Allocate hierarchical adapter ranks under a parameter/rank budget."""

    return search_rank_allocation(
        profiles,
        budget=budget,
        performance_model=performance_model,
    ).config


def export_peft_rank_pattern(config: RankConfig, *, alpha_multiplier: int = 2) -> dict[str, dict[str, int]]:
    if alpha_multiplier <= 0:
        raise ValueError("alpha_multiplier must be positive")
    return {
        name: {"r": rank, "lora_alpha": rank * alpha_multiplier}
        for name, rank in sorted(config.items())
    }
