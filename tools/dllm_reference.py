#!/usr/bin/env python3
"""PyTorch reference candidates for a qwen3-dllm block-masked denoise step.

This is the *reference* path for the Rust kernel: it loads the pinned HF
checkpoint in BF16 with Transformers' stock Qwen3 model and applies exactly the
mask FriedFlaxia uses (prefix causal, active block bidirectional, tail
ignored). It writes the top-k candidates per requested position, which the
ignored `real_dllm_matches_pytorch_reference` Rust test consumes.

Environment (one-off, outside the Rust build):

    pip install torch transformers safetensors numpy gguf

Usage:

    python3 tools/convert_dllm.py <hf-checkpoint-dir> /tmp/dllm-q8.gguf --quant q8_0
    python3 tools/dllm_reference.py \
      --model-dir <hf-checkpoint-dir> \
      --tokens 151644,872,151645,151669,100,200,300,400 \
      --block-start 0 --block-end 8 --positions 4,5,6 \
      --out crates/friedflaxia-core/tests/fixtures/dllm-pytorch-top8.json
    FRIEDFLAXIA_DLLM_Q8_GGUF=/tmp/dllm-q8.gguf \
      cargo test --release --test dllm -- --ignored --nocapture
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--tokens", required=True, help="comma-separated token ids")
    parser.add_argument("--block-start", type=int, required=True)
    parser.add_argument("--block-end", type=int, required=True)
    parser.add_argument("--positions", required=True, help="comma-separated positions")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--out", required=True)
    parser.add_argument("--logits-out", help="optional raw f32 logits for the positions")
    args = parser.parse_args()

    tokens = [int(value) for value in args.tokens.split(",")]
    positions = [int(value) for value in args.positions.split(",")]
    n = len(tokens)
    if not 0 <= args.block_start < args.block_end <= n:
        raise SystemExit("block must be a non-empty range inside tokens")
    if not positions or any(p >= args.block_end for p in positions):
        raise SystemExit("positions must be inside the block end")

    config = Qwen3Config(**json.loads((Path(args.model_dir) / "config.json").read_text()))
    model = Qwen3ForCausalLM.from_pretrained(
        args.model_dir, config=config, dtype=torch.bfloat16
    )
    model.eval()

    allowed = torch.zeros(n, n, dtype=torch.bool)
    for query in range(n):
        key_end = (
            args.block_end
            if args.block_start <= query < args.block_end
            else query + 1
        )
        allowed[query, :key_end] = True
    mask = torch.zeros(1, 1, n, n, dtype=torch.bfloat16)
    mask[0, 0][~allowed] = float("-inf")

    with torch.no_grad():
        logits = model(
            input_ids=torch.tensor([tokens], dtype=torch.long),
            attention_mask=mask,
            use_cache=False,
        ).logits[0]
    rows = logits[positions].float()
    values, indices = torch.topk(rows, args.top_k, dim=-1)

    result = {
        "tokens": tokens,
        "block": [args.block_start, args.block_end],
        "positions": positions,
        "vocab_size": logits.shape[-1],
        "top_k": args.top_k,
        "token_ids": indices.tolist(),
        "top_logits": values.tolist(),
    }
    Path(args.out).write_text(json.dumps(result))
    if args.logits_out:
        rows.numpy().astype("<f4").tofile(args.logits_out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
