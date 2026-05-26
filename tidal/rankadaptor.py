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


def collect_linear_profiles(
    model: object,
    *,
    sensitivities: dict[str, float] | None = None,
    min_rank: int = 0,
    max_rank: int = 64,
    rank_step: int = 1,
    name_filter: Callable[[str], bool] | None = None,
) -> list[ModuleProfile]:
    """Build RankAdaptor profiles from torch.nn.Linear modules.

    The LoRA cost per rank is in_features + out_features for each linear layer,
    matching the parameter count of A and B adapter matrices.
    """

    try:
        from torch import nn
    except ImportError as exc:
        raise ImportError("collect_linear_profiles requires PyTorch") from exc

    sensitivity_map = sensitivities or {}
    profiles: list[ModuleProfile] = []
    for name, module in model.named_modules():
        if not name or not isinstance(module, nn.Linear):
            continue
        if name_filter is not None and not name_filter(name):
            continue
        sensitivity = float(sensitivity_map.get(name, 1.0))
        profiles.append(
            ModuleProfile(
                name=name,
                sensitivity=sensitivity,
                min_rank=min_rank,
                max_rank=max_rank,
                rank_step=rank_step,
                cost_per_rank=int(module.in_features + module.out_features),
            )
        )
    if not profiles:
        raise ValueError("model does not contain matching torch.nn.Linear modules")
    return profiles


def build_lora_config(
    rank_config: RankConfig,
    *,
    alpha_multiplier: int = 2,
    target_modules: Sequence[str] | None = None,
    **kwargs: object,
) -> object:
    """Create a PEFT LoraConfig from a RankAdaptor rank allocation."""

    if alpha_multiplier <= 0:
        raise ValueError("alpha_multiplier must be positive")
    positive = {name: int(rank) for name, rank in rank_config.items() if int(rank) > 0}
    if not positive:
        raise ValueError("rank_config must contain at least one positive rank")

    try:
        from peft import LoraConfig
    except ImportError as exc:
        raise ImportError("build_lora_config requires peft") from exc

    targets = list(target_modules) if target_modules is not None else sorted(positive)
    return LoraConfig(
        r=min(positive.values()),
        lora_alpha=min(positive.values()) * alpha_multiplier,
        target_modules=targets,
        rank_pattern=dict(sorted(positive.items())),
        alpha_pattern={name: rank * alpha_multiplier for name, rank in sorted(positive.items())},
        **kwargs,
    )



@dataclass(frozen=True)
class RankObservation:
    config: RankConfig
    score: float
    source: str = "measured"


@dataclass(frozen=True)
class OnlineRankSearchResult:
    best_config: RankConfig
    best_score: float
    observations: tuple[RankObservation, ...]
    predicted_config: RankConfig
    predicted_score: float


def enumerate_rank_configs(profiles: Iterable[ModuleProfile], *, budget: int | None = None) -> list[RankConfig]:
    """Enumerate feasible hierarchical rank configurations in the solution space S."""

    from itertools import product

    profile_list = list(profiles)
    if not profile_list:
        raise ValueError("profiles must not be empty")
    values = [range(profile.min_rank, profile.max_rank + 1, profile.rank_step) for profile in profile_list]
    configs: list[RankConfig] = []
    for ranks in product(*values):
        config = {profile.name: int(rank) for profile, rank in zip(profile_list, ranks)}
        if budget is not None and config_cost(config, profile_list) > budget:
            continue
        configs.append(config)
    configs.sort(key=lambda cfg: (config_cost(cfg, profile_list), tuple(cfg[p.name] for p in profile_list)))
    return configs


def sample_rank_configs(
    profiles: Iterable[ModuleProfile],
    *,
    budget: int | None = None,
    count: int,
    seed: int | None = None,
) -> list[RankConfig]:
    """Randomly sample feasible rank configurations, using exhaustive enumeration for small spaces."""

    if count <= 0:
        raise ValueError("count must be positive")
    profile_list = list(profiles)
    all_configs = enumerate_rank_configs(profile_list, budget=budget)
    if len(all_configs) <= count:
        return all_configs
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(all_configs), size=count, replace=False)
    return [all_configs[int(index)] for index in chosen]


class TorchMLPPerformanceModel:
    """Five-layer MLP performance model used by RankAdaptor-style online search."""

    def __init__(self, profiles: Iterable[ModuleProfile], *, hidden_dims: Sequence[int] = (32, 32, 32)):
        self.profiles = list(profiles)
        if not self.profiles:
            raise ValueError("profiles must not be empty")
        if len(hidden_dims) != 3:
            raise ValueError("hidden_dims must contain three hidden dimensions")
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError("TorchMLPPerformanceModel requires PyTorch") from exc

        self._torch = torch
        layers: list[object] = []
        input_dim = len(self.profiles)
        dims = [input_dim, *[int(dim) for dim in hidden_dims], 1]
        for in_dim, out_dim in zip(dims, dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != 1:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def _tensorize(self, configs: Sequence[RankConfig]):
        rows: list[list[float]] = []
        for config in configs:
            if set(config) != {profile.name for profile in self.profiles}:
                raise ValueError("config must contain exactly the profiled modules")
            row = []
            for profile in self.profiles:
                span = max(profile.max_rank - profile.min_rank, 1)
                row.append((float(config[profile.name]) - profile.min_rank) / span)
            rows.append(row)
        return self._torch.tensor(rows, dtype=self._torch.float32)

    def fit(
        self,
        observations: Sequence[RankObservation],
        *,
        epochs: int = 200,
        learning_rate: float = 1e-2,
        seed: int | None = None,
    ) -> None:
        if not observations:
            raise ValueError("observations must not be empty")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if seed is not None:
            self._torch.manual_seed(seed)
        x = self._tensorize([item.config for item in observations])
        y = self._torch.tensor([[float(item.score)] for item in observations], dtype=self._torch.float32)
        optimizer = self._torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        loss_fn = self._torch.nn.MSELoss()
        self.network.train()
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(self.network(x), y)
            loss.backward()
            optimizer.step()

    def predict(self, config: RankConfig) -> float:
        self.network.eval()
        with self._torch.no_grad():
            return float(self.network(self._tensorize([config])).reshape(()).item())


def fit_mlp_performance_model(
    observations: Sequence[RankObservation],
    profiles: Iterable[ModuleProfile],
    *,
    hidden_dims: Sequence[int] = (32, 32, 32),
    epochs: int = 200,
    learning_rate: float = 1e-2,
    seed: int | None = None,
) -> TorchMLPPerformanceModel:
    model = TorchMLPPerformanceModel(profiles, hidden_dims=hidden_dims)
    model.fit(observations, epochs=epochs, learning_rate=learning_rate, seed=seed)
    return model


def online_incremental_rank_search(
    profiles: Iterable[ModuleProfile],
    *,
    budget: int,
    evaluate_config: Callable[[RankConfig], float],
    initial_samples: int = 8,
    iterations: int = 8,
    candidate_samples: int = 64,
    hidden_dims: Sequence[int] = (32, 32, 32),
    train_epochs: int = 200,
    learning_rate: float = 1e-2,
    seed: int | None = None,
) -> OnlineRankSearchResult:
    """RankAdaptor-style initialization, online incremental learning, and convergence search.

    ``evaluate_config`` is the expensive paper step: recover/fine-tune the pruned model
    under a rank configuration and return the downstream task score.
    """

    if budget < 0:
        raise ValueError("budget must be non-negative")
    if initial_samples <= 0 or iterations < 0 or candidate_samples <= 0:
        raise ValueError("sample counts must be positive and iterations non-negative")
    profile_list = list(profiles)
    all_configs = enumerate_rank_configs(profile_list, budget=budget)
    if not all_configs:
        raise ValueError("no feasible rank configurations")

    rng = np.random.default_rng(seed)
    initial_count = min(initial_samples, len(all_configs))
    initial_indices = rng.choice(len(all_configs), size=initial_count, replace=False)
    observed_by_key: dict[tuple[tuple[str, int], ...], RankObservation] = {}

    def key(config: RankConfig) -> tuple[tuple[str, int], ...]:
        return tuple(sorted((name, int(rank)) for name, rank in config.items()))

    def observe(config: RankConfig, source: str) -> RankObservation:
        item = RankObservation(dict(config), float(evaluate_config(dict(config))), source)
        observed_by_key[key(config)] = item
        return item

    for index in initial_indices:
        observe(all_configs[int(index)], "initial")

    predicted_config = dict(next(iter(observed_by_key.values())).config)
    predicted_score = float("-inf")
    for step in range(iterations):
        observations = list(observed_by_key.values())
        model = fit_mlp_performance_model(
            observations,
            profile_list,
            hidden_dims=hidden_dims,
            epochs=train_epochs,
            learning_rate=learning_rate,
            seed=None if seed is None else seed + step,
        )
        remaining = [config for config in all_configs if key(config) not in observed_by_key]
        if not remaining:
            break
        if len(remaining) > candidate_samples:
            indices = rng.choice(len(remaining), size=candidate_samples, replace=False)
            candidates = [remaining[int(index)] for index in indices]
        else:
            candidates = remaining
        scored = [(model.predict(config), config) for config in candidates]
        predicted_score, predicted_config = max(scored, key=lambda item: item[0])
        observe(predicted_config, "online")

    final_observations = tuple(observed_by_key.values())
    best = max(final_observations, key=lambda item: item.score)
    final_model = fit_mlp_performance_model(
        final_observations,
        profile_list,
        hidden_dims=hidden_dims,
        epochs=train_epochs,
        learning_rate=learning_rate,
        seed=seed,
    )
    final_candidates = sample_rank_configs(profile_list, budget=budget, count=candidate_samples, seed=seed)
    predicted_score, predicted_config = max(
        ((final_model.predict(config), config) for config in final_candidates),
        key=lambda item: item[0],
    )
    return OnlineRankSearchResult(dict(best.config), float(best.score), final_observations, dict(predicted_config), float(predicted_score))

def export_peft_rank_pattern(config: RankConfig, *, alpha_multiplier: int = 2) -> dict[str, dict[str, int]]:
    if alpha_multiplier <= 0:
        raise ValueError("alpha_multiplier must be positive")
    return {
        name: {"r": rank, "lora_alpha": rank * alpha_multiplier}
        for name, rank in sorted(config.items())
    }
