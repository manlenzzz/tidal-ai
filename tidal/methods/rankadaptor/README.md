# RankAdaptor

Paper: **RankAdaptor: Hierarchical Rank Allocation for Efficient Fine-Tuning Pruned LLMs via Performance Model**

This package contains the local TIDAL implementation of the RankAdaptor workflow:

1. represent the hierarchical LoRA rank configuration as a per-layer vector;
2. train a five-layer MLP performance model on measured rank configurations;
3. run online incremental search by evaluating selected candidate ranks on the target task;
4. export the selected ranks as a PEFT `LoraConfig`.

Primary APIs:

- `ModuleProfile`
- `online_incremental_rank_search`
- `TorchMLPPerformanceModel`
- `collect_linear_profiles`
- `build_lora_config`

Modern Hugging Face-style models can be targeted without hand-written layer filters:

```python
profiles = collect_linear_profiles(model, target_roles="modern", min_rank=1, max_rank=64)
```

The `modern` selector includes attention, MLP, and MoE expert linear modules while excluding `lm_head`, embedding projections, and MoE routers by default. Existing `name_filter` callables remain supported and are composed with the role selector when both are provided.

Examples:

- `examples/rankadaptor/basic_rank_search.py`
- `examples/rankadaptor/torch_peft_search.py`

