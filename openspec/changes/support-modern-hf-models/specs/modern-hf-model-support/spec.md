## ADDED Requirements

### Requirement: Linear module classification
The system SHALL discover `torch.nn.Linear` modules in Hugging Face-style language models and classify each module into a stable role: `attention`, `mlp`, `moe_expert`, `moe_router`, `output_head`, `embedding_projection`, or `other`.

#### Scenario: Qwen-style decoder block
- **WHEN** a model contains projection names such as `model.layers.0.self_attn.q_proj`, `model.layers.0.self_attn.k_proj`, `model.layers.0.self_attn.v_proj`, `model.layers.0.self_attn.o_proj`, `model.layers.0.mlp.gate_proj`, `model.layers.0.mlp.up_proj`, and `model.layers.0.mlp.down_proj`
- **THEN** the system classifies attention projections as `attention` and MLP projections as `mlp`

#### Scenario: Fused attention projection
- **WHEN** a model contains fused projection names such as `attn.qkv_proj`, `attn.Wqkv`, `self_attn.c_attn`, or `attention.query_key_value`
- **THEN** the system classifies those modules as `attention`

#### Scenario: MoE decoder block
- **WHEN** a model contains expert projection names and router/gate projection names in a MoE block
- **THEN** the system classifies expert projections as `moe_expert` and router/gate projections as `moe_router`

### Requirement: Safe default target selection
The system SHALL provide a default target selector for TIDAL methods that includes compressible/adaptable projection roles and excludes output heads, routers, and embedding projections unless explicitly requested.

#### Scenario: Default selector excludes unsafe modules
- **WHEN** a model contains attention, MLP, MoE expert, MoE router, embedding projection, and `lm_head` linear modules
- **THEN** the default selector includes attention, MLP, and MoE expert modules and excludes MoE router, embedding projection, and output head modules

#### Scenario: Caller overrides selected roles
- **WHEN** a caller requests additional roles such as `moe_router` or `output_head`
- **THEN** the selector includes matching modules according to the requested role set

### Requirement: Method integration
The system SHALL allow RankAdaptor, QPruner, and Global Rank-Sparsity Torch integrations to use the shared target selector without requiring callers to provide custom `name_filter` functions.

#### Scenario: RankAdaptor profiles selected modern modules
- **WHEN** `collect_linear_profiles` is called with the modern-model default selector on a Hugging Face-style model
- **THEN** profiles are produced only for selected target modules and their LoRA costs match each module shape

#### Scenario: QPruner plans selected modern modules
- **WHEN** `build_quantization_plan` is called with the modern-model default selector on a Hugging Face-style model
- **THEN** its required importances and layer sizes refer only to selected target modules

#### Scenario: CAP compresses selected modern modules
- **WHEN** `apply_global_cap_compression` is called with the modern-model default selector on a Hugging Face-style model
- **THEN** only selected linear modules are replaced by CAP packed modules

### Requirement: Backward compatibility
The system SHALL preserve existing `name_filter` behavior and public imports while adding the modern-model selector as an additive API.

#### Scenario: Existing name_filter continues to work
- **WHEN** callers pass an existing `name_filter` lambda to RankAdaptor, QPruner, or Global Rank-Sparsity integrations
- **THEN** the method behavior remains determined by that filter
