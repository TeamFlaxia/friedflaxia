#!/usr/bin/env python3
"""Convert an A2D Qwen3 dLLM checkpoint to a FriedFlaxia `qwen3-dllm` GGUF.

The upstream dLLM release ships BF16 safetensors with a `a2d-qwen3` config.
FriedFlaxia reads the Qwen3 layer layout from GGUF, so this script maps the
tensor names, transposes to GGML `[cols, rows]` order, quantizes 2-D weights
to Q8_0 (or Q4_0) and keeps the 1-D norms in F32.

Usage:
  python3 tools/convert_dllm.py <model_dir> <out.gguf> [--quant q8_0|q4_0]
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

QK = 32
Q8_0_BLOCK_BYTES = 2 + QK
Q4_0_BLOCK_BYTES = 2 + QK // 2

TOKEN_NORMAL = 1
TOKEN_CONTROL = 3
TOKEN_UNUSED = 5


def read_safetensors(path: Path) -> dict[str, np.ndarray]:
    raw = path.read_bytes()
    (header_len,) = struct.unpack_from("<Q", raw, 0)
    header = json.loads(raw[8 : 8 + header_len])
    data = memoryview(raw)[8 + header_len :]
    tensors: dict[str, np.ndarray] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        dtype = info["dtype"]
        shape = tuple(info["shape"])
        chunk = data[start:end]
        if dtype == "BF16":
            bits = np.frombuffer(chunk, dtype="<u2").astype(np.uint32)
            values = (bits << 16).view(np.float32).astype(np.float32)
        elif dtype == "F16":
            values = np.frombuffer(chunk, dtype="<f2").astype(np.float32)
        elif dtype == "F32":
            values = np.frombuffer(chunk, dtype="<f4").astype(np.float32)
        else:
            raise SystemExit(f"unsupported safetensors dtype {dtype} for {name}")
        tensors[name] = values.reshape(shape)
    return tensors


def f16_bytes(values: np.ndarray) -> bytes:
    return values.astype("<f2").tobytes()


def quantize_q8_0(rows: np.ndarray) -> bytes:
    """`rows` is [rows, cols]; cols must be a multiple of 32."""
    rows = np.ascontiguousarray(rows, dtype=np.float32)
    if rows.shape[1] % QK != 0:
        raise SystemExit(f"Q8_0 needs cols%32==0, got {rows.shape}")
    blocks = rows.reshape(rows.shape[0], -1, QK)
    amax = np.max(np.abs(blocks), axis=2)
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    quant = np.rint(blocks / scale[:, :, None])
    quant = np.clip(quant, -127, 127).astype(np.int8)
    out = np.empty((blocks.shape[0], blocks.shape[1], Q8_0_BLOCK_BYTES), dtype=np.uint8)
    out[:, :, 0:2] = np.frombuffer(f16_bytes(scale), dtype=np.uint8).reshape(scale.shape + (2,))
    out[:, :, 2:] = quant.view(np.uint8)
    return out.tobytes()


def quantize_q4_0(rows: np.ndarray) -> bytes:
    rows = np.ascontiguousarray(rows, dtype=np.float32)
    if rows.shape[1] % QK != 0:
        raise SystemExit(f"Q4_0 needs cols%32==0, got {rows.shape}")
    blocks = rows.reshape(rows.shape[0], -1, QK)
    amax = np.max(np.abs(blocks), axis=2)
    scale = np.where(amax > 0, amax / 8.0, 1.0).astype(np.float32)
    quant = np.rint(blocks / scale[:, :, None]) + 8.0
    quant = np.clip(quant, 0, 15).astype(np.uint8)
    low = quant[:, :, : QK // 2]
    high = quant[:, :, QK // 2 :]
    packed = (low | (high << 4)).astype(np.uint8)
    out = np.empty((blocks.shape[0], blocks.shape[1], Q4_0_BLOCK_BYTES), dtype=np.uint8)
    out[:, :, 0:2] = np.frombuffer(f16_bytes(scale), dtype=np.uint8).reshape(scale.shape + (2,))
    out[:, :, 2:] = packed
    return out.tobytes()


def build_tokens(model_dir: Path, vocab_size: int) -> tuple[list[str], list[int]]:
    tokenizer = json.loads((model_dir / "tokenizer.json").read_text())
    tokens: dict[int, str] = {}
    types: dict[int, int] = {}
    for token, index in tokenizer["model"]["vocab"].items():
        tokens[index] = token
        types[index] = TOKEN_NORMAL
    for added in tokenizer.get("added_tokens", []):
        index = added["id"]
        tokens[index] = added["content"]
        types[index] = TOKEN_CONTROL if added.get("special") else TOKEN_NORMAL
    # The embedding matrix is larger than the serialized tokenizer (Qwen3
    # reserves ids for future specials). Fill the gaps so ids stay positional.
    for index in range(vocab_size):
        if index not in tokens:
            tokens[index] = f"<|unused_{index}|>"
            types[index] = TOKEN_UNUSED
    return [tokens[i] for i in range(vocab_size)], [types[i] for i in range(vocab_size)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--quant", choices=["q8_0", "q4_0"], default="q8_0")
    parser.add_argument("--name", default="Qwen3-0.6B-diffusion-bd3lm-v0.1")
    args = parser.parse_args()

    import gguf

    model_dir: Path = args.model_dir
    config = json.loads((model_dir / "config.json").read_text())
    tensors = read_safetensors(model_dir / "model.safetensors")

    n_layer = config["num_hidden_layers"]
    n_embd = config["hidden_size"]
    n_ff = config["intermediate_size"]
    n_head = config["num_attention_heads"]
    n_head_kv = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    vocab_size = config["vocab_size"]
    tokenizer = json.loads((model_dir / "tokenizer.json").read_text())
    merges = [" ".join(pair) for pair in tokenizer["model"]["merges"]]
    tokens, token_types = build_tokens(model_dir, vocab_size)

    writer = gguf.GGUFWriter(args.out, "qwen3-dllm")
    writer.add_name(args.name)
    writer.add_block_count(n_layer)
    writer.add_embedding_length(n_embd)
    writer.add_feed_forward_length(n_ff)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_head_kv)
    writer.add_key_length(head_dim)
    writer.add_layer_norm_rms_eps(float(config["rms_norm_eps"]))
    writer.add_rope_freq_base(float(config["rope_theta"]))
    writer.add_tokenizer_model("qwen3")
    writer.add_token_list(tokens)
    writer.add_token_scores([0.0] * vocab_size)
    writer.add_token_types(token_types)
    writer.add_token_merges(merges)
    writer.add_bos_token_id(int(config["bos_token_id"]))
    writer.add_eos_token_id(int(config["eos_token_id"]))
    writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)

    quantize = quantize_q8_0 if args.quant == "q8_0" else quantize_q4_0
    raw_type = gguf.GGMLQuantizationType.Q8_0 if args.quant == "q8_0" else gguf.GGMLQuantizationType.Q4_0

    def add_matrix(name: str, value: np.ndarray) -> None:
        # PyTorch Linear is [out, in] and already row-major over `in`, which is
        # exactly GGML's [in, out] contiguous layout. gguf-py reverses the
        # shape on write, so pass [out, in] unchanged.
        matrix = np.ascontiguousarray(value, dtype=np.float32)
        data = quantize(matrix)
        raw = np.frombuffer(data, dtype=np.uint8).reshape(matrix.shape[0], -1)
        writer.add_tensor(name, raw, raw_dtype=raw_type)

    def add_vector(name: str, value: np.ndarray) -> None:
        writer.add_tensor(name, np.ascontiguousarray(value, dtype=np.float32))

    embed = tensors["model.embed_tokens.weight"]
    if embed.shape != (vocab_size, n_embd):
        raise SystemExit(f"unexpected embedding shape {embed.shape}")
    add_matrix("token_embd.weight", embed)
    add_vector("output_norm.weight", tensors["model.norm.weight"])

    # Tied embeddings: the loader falls back to token_embd when output.weight
    # is absent. Verify the release really ties before dropping lm_head.
    if not config.get("tie_word_embeddings", False):
        add_matrix("output.weight", tensors["lm_head.weight"])

    for layer in range(n_layer):
        prefix = f"model.layers.{layer}"
        out = f"blk.{layer}"
        add_vector(f"{out}.attn_norm.weight", tensors[f"{prefix}.input_layernorm.weight"])
        add_matrix(f"{out}.attn_q.weight", tensors[f"{prefix}.self_attn.q_proj.weight"])
        add_matrix(f"{out}.attn_k.weight", tensors[f"{prefix}.self_attn.k_proj.weight"])
        add_matrix(f"{out}.attn_v.weight", tensors[f"{prefix}.self_attn.v_proj.weight"])
        add_matrix(f"{out}.attn_output.weight", tensors[f"{prefix}.self_attn.o_proj.weight"])
        add_vector(f"{out}.attn_q_norm.weight", tensors[f"{prefix}.self_attn.q_norm.weight"])
        add_vector(f"{out}.attn_k_norm.weight", tensors[f"{prefix}.self_attn.k_norm.weight"])
        add_vector(f"{out}.ffn_norm.weight", tensors[f"{prefix}.post_attention_layernorm.weight"])
        add_matrix(f"{out}.ffn_gate.weight", tensors[f"{prefix}.mlp.gate_proj.weight"])
        add_matrix(f"{out}.ffn_up.weight", tensors[f"{prefix}.mlp.up_proj.weight"])
        add_matrix(f"{out}.ffn_down.weight", tensors[f"{prefix}.mlp.down_proj.weight"])

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes, {args.quant})")


if __name__ == "__main__":
    main()
