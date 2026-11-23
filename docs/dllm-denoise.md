# Theia dLLM denoise API

Theia runs small masked-diffusion language models (dLLMs) across Flaxia Crowd
nodes and merges their per-position token distributions. This note records the
implemented FriedFlaxia extension and what is still missing.

> Status: the qwen3-dllm forward pass, block attention mask, full-logit and
> top-k APIs, and the wasm bindings are implemented. LoRA adapters, a block/KV
> cache and real-model parity are next.

## Implemented

`crates/friedflaxia-core/src/models/dllm.rs`:

- **`qwen3-dllm` architecture** with the Qwen3 layer layout: pre-attention
  RMSNorm, GQA q/k/v, per-head q/k RMSNorm, NeoX RoPE at the absolute canvas
  position, SwiGLU FFN, and a tied or explicit output projection.
- **Block-diffusion attention mask.** Queries inside `[start, end)` attend to
  the clean prefix `0..start` and bidirectionally to every token in the block.
  Queries outside the block fall back to causal attention; their outputs are not
  scored. Tokens at or after `end` are never read, so the backbone only runs over
  `tokens[..end]`.
- **Full logits or sparse candidates.** `denoise_step` returns raw row-major
  `[positions][vocab]` logits and the max-softmax confidence. `denoise_topk`
  returns the top-k token ids with **softmaxed** probabilities (one full-vocab
  pass per position, no duplicate exponentiation) and the effective `topK` after
  clamping to the vocabulary.
- **Validation.** Token ids, block bounds, unique positions, and `top_k` are
  checked before any tensor is read. Positions may be anywhere before `end`
  (including the clean prefix), because the tail is the only part that is not
  computed.

The wasm `Model` exposes both entry points:

```ts
model.denoise_step(tokens, positions, blockStart, blockEnd)
// { logits: Float32Array, positions: Uint32Array,
//   confidence: Float32Array, vocabSize: number }

model.denoise_topk(tokens, positions, blockStart, blockEnd, topK)
// { positions: Uint32Array, tokenIds: Uint32Array,
//   probabilities: Float32Array, confidence: Float32Array,
//   vocabSize: number, topK: number }
```

`logits` is row-major `[positions.len()][vocabSize]` and is **raw** (apply
softmax on the host if you need probabilities). `tokenIds` and `probabilities`
are row-major `[positions.len()][topK]`; `probabilities` are already softmaxed,
so `confidence` equals the first probability of each row.

## Still missing

1. **Block/KV cache.** The step already truncates the backbone at `end` and
   only scores the requested positions, but the clean prefix is still recomputed
   every round. A prefix cache and a block cache are the next large performance
   win.
2. **A session handle.** Config, tensor lookups and norm tensors are re-read on
   every call. A `DllmSession` is also the natural home for the KV cache.
3. **LoRA adapters.** Theia's v1 experts are LoRA adapters on one base model;
   the engine does not apply adapters yet.
4. **SDAR self-conditioning.** QwenDLM's residual adapter is not implemented;
   the base Qwen3 forward is.
5. **Real-model parity.** The synthetic tiny GGUFs cover the mask and API; a
   pinned QwenDLM/Qwen3-0.6B-bd3lm comparison is still required.
6. **Sampling and ensemble.** The host owns sampling, Self-Repulsion and delta
   merging. Rust only returns scores.

## Tests

`crates/friedflaxia-core/tests/dllm.rs` builds tiny Qwen3-style GGUFs and
covers:

- logits/confidence shape and determinism,
- the block mask (future tokens are not read, the clean prefix is read, and the
  block is bidirectional),
- tied output embeddings,
- top-k vs the full distribution and effective `top_k` clamping,
- a hardened two-layer / 4-query-head / 2-KV-head fixture with non-uniform norms
  and golden logits,
- invalid token ids, blocks, duplicate positions and an out-of-block tail.

A real-model smoke test is available behind
`FRIEDFLAXIA_DLLM_GGUF=/path/to/model.gguf`:

```sh
cargo test --release --test dllm -- --ignored --nocapture
```

## Performance

A denoise step is a full forward over the active block; a response costs
`K paths × T steps` of them. Before committing to a 4B model:

1. Measure one step on the target hardware (native first).
2. Add the block/KV cache so the clean prefix is not recomputed.
3. Use the existing batched quantized matmul, then wasm SIMD / native threads,
   and finally `wgpu` if the step is still short.

## Model candidates

License safety comes first (avoid PolyForm Noncommercial):

| Model | Params | License | Role |
|---|---|---|---|
| `yifanyu/QwenDLM-4B` | 4B | Apache-2.0 | primary |
| `nvidia/Nemotron-Labs-Diffusion-3B` | 3B | NVIDIA Open Model License | 3B alternative |
| `dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1` | 0.8B | Apache-2.0 | algorithm validation |

**Quantization:** the kernels cover F32, BF16, Q4_0, Q4_1, Q8_0 and Q6_K. A
typical `Q4_K_M` conversion is not supported yet, so pin the converted dLLM GGUF
to Q8_0 or Q4_0 (or add a Q4_K kernel before shipping it).

## Related

- `TeamFlaxia/theia` — chat UI, portable `/api`, merge functions, Crowd adapter.
- `TeamFlaxia/flaxia-crowd` `docs/theia-integration.md` — `theia-inference`
  workload contract.
