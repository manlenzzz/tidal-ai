## ADDED Requirements

### Requirement: Canonicalize Pruned Model Paths
The system SHALL canonicalize linear module identifiers produced by Hugging Face wrappers, PEFT wrappers, torch compile wrappers, LLM-Pruner outputs, WANDA masks, and checkpoint parameter keys into comparable module paths.

#### Scenario: Wrapper and tensor suffix normalization
- **WHEN** the input name is `base_model.model.model.layers.0.self_attn.q_proj.weight`
- **THEN** the canonical module name is `model.layers.0.self_attn.q_proj`

#### Scenario: Torch compile wrapper normalization
- **WHEN** the input name is `_orig_mod.model.layers.0.mlp.down_proj.weight_mask`
- **THEN** the canonical module name is `model.layers.0.mlp.down_proj`

### Requirement: Build Pruning Output Filters
The system SHALL build a linear-module predicate from pruning output names while preserving TIDAL role filtering semantics.

#### Scenario: WANDA state-dict keys select only default TIDAL roles
- **WHEN** the input mapping contains attention, MLP, router, and output-head tensor keys
- **THEN** the generated predicate matches only attention, MLP, and expert linear modules by default

#### Scenario: Original module names remain usable
- **WHEN** the target model exposes prefixed wrapper names
- **THEN** the generated predicate can match the original exposed module name by comparing canonical paths
