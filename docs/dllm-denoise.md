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
  scored. This is the mask a masked-diffusion sampler needs.
- **Full logits or sparse candidates.** `denoise_step` returns row-major
  `[positions][vocab]` logits and the max-softmax confidence. `denoise_topk`
  returns the top-k token ids/probabilities so a 4B vocabulary does not have to
  cross the wasm boundary on every step.
- **Validation.** Token ids, block bounds, positions, and `top_k` are checked
  before any tensor is read.

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

`logits` is row-major `[positions.len()][vocabSize]`; `tokenIds` and
`probabilities` are row-major `[positions.len()][topK]`.

## Still missing

1. **Block/KV cache.** The current step recomputes the whole sequence (with the
   block mask) and only scores the requested positions. A prefix cache and a
   block cache are the next large performance win.
2. **LoRA adapters.** Theia's v1 experts are LoRA adapters on one base model;
   the engine does not apply adapters yet.
3. **SDAR self-conditioning.** QwenDLM's residual adapter is not implemented;
   the base Qwen3 forward is.
4. **Real-model parity.** The synthetic tiny GGUF covers the mask and API; a
   pinned QwenDLM/Qwen3-0.6B-bd3lm comparison is still required.
5. **Sampling and ensemble.** The host owns sampling, Self-Repulsion and delta
   merging. Rust only returns scores.

## Tests

`crates/friedflaxia-core/tests/dllm.rs` builds a tiny Qwen3-style GGUF and
covers:

- logits/confidence shape and determinism,
- the block mask (future tokens do not affect the block; the clean prefix does),
- tied output embeddings,
- top-k vs the full distribution,
- invalid token ids, blocks and positions.

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

## Related

- `TeamFlaxia/theia` — chat UI, portable `/api`, merge functions, Crowd adapter.
- `TeamFlaxia/flaxia-crowd` `docs/theia-integration.md` — `theia-inference`
  workload contract.
