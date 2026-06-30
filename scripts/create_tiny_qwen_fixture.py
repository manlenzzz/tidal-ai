#!/usr/bin/env python
"""Create a tiny local Qwen-style HF model for offline Ascend benchmark plumbing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_tokenizer(output_dir: Path, vocab_size: int) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    base_tokens = ["<pad>", "<unk>", "<bos>", "<eos>", "hello", "world", "Explain", "Summarize"]
    vocab = {token: index for index, token in enumerate(base_tokens)}
    for index in range(len(vocab), vocab_size):
        vocab[f"tok_{index}"] = index
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    fast.padding_side = "left"
    fast.model_input_names = ["input_ids", "attention_mask"]
    fast.save_pretrained(output_dir)
    tokenizer_config_path = output_dir / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_config_path.read_text())
    tokenizer_config["padding_side"] = "left"
    tokenizer_config["model_input_names"] = ["input_ids", "attention_mask"]
    tokenizer_config_path.write_text(json.dumps(tokenizer_config, indent=2, sort_keys=True))


def qwen_config(*, vocab_size: int, hidden_size: int, intermediate_size: int):
    try:
        from transformers import Qwen3Config

        return Qwen3Config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=256,
            bos_token_id=2,
            eos_token_id=3,
            pad_token_id=0,
        )
    except ImportError:
        from transformers import Qwen2Config

        return Qwen2Config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=256,
            bos_token_id=2,
            eos_token_id=3,
            pad_token_id=0,
        )


def create_fixture(
    output_dir: Path,
    *,
    vocab_size: int = 256,
    hidden_size: int = 64,
    intermediate_size: int = 128,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    build_tokenizer(output_dir, vocab_size)
    cfg = qwen_config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    try:
        from transformers import Qwen3ForCausalLM

        model = Qwen3ForCausalLM(cfg)
    except ImportError:
        from transformers import Qwen2ForCausalLM

        model = Qwen2ForCausalLM(cfg)
    model.eval()
    model.save_pretrained(output_dir, safe_serialization=True)
    (output_dir / "fixture_manifest.json").write_text(
        json.dumps(
            {
                "purpose": "offline tiny Qwen-style model for Ascend benchmark plumbing",
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "model_type": cfg.model_type,
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Create tiny Qwen-style HF fixture")
    parser.add_argument("--output-dir", default="/mnt/nvme/622/models/TinyQwen3-Offline")
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--intermediate-size", type=int, default=128)
    args = parser.parse_args()

    create_fixture(
        Path(args.output_dir),
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
    )
    print(f"TINY_QWEN_FIXTURE {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
