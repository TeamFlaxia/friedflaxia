# Theia dLLM denoise API

Theia runs small masked-diffusion language models (dLLMs) across Flaxia Crowd
nodes and merges their per-position token distributions. This note records the
implemented FriedFlaxia extension and what is still missing.

> Status: the qwen3-dllm forward pass, block attention mask, full-logit and
> top-k APIs, LoRA adapters and the wasm bindings are implemented. A block/KV
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

Model.withLora(baseBytes, loraBytes)   // base dLLM + adapter
model.loraRank()                       // number | undefined
model.loraExpert()                     // string | undefined
```

`logits` is row-major `[positions.len()][vocabSize]` and is **raw** (apply
softmax on the host if you need probabilities). `tokenIds` and `probabilities`
are row-major `[positions.len()][topK]`; `probabilities` are already softmaxed,
so `confidence` equals the first probability of each row.

## LoRA adapters

Theia's v1 experts are LoRA adapters over one base dLLM. An adapter is a
separate `qwen3-dllm-lora` GGUF:

```text
general.architecture      = "qwen3-dllm-lora"
qwen3-dllm-lora.rank      = 16
qwen3-dllm-lora.alpha     = 32.0     (optional, defaults to rank)
qwen3-dllm-lora.expert    = "code"   (optional)

blk.{i}.attn_q.lora_a.weight      [in, rank]
blk.{i}.attn_q.lora_b.weight      [rank, out]
... attn_k, attn_v, attn_output, ffn_gate, ffn_up, ffn_down
```

Each projection becomes `W x + (alpha / rank) * B (A x)`. Modules without
tensors are left untouched; a module with only one of `lora_a`/`lora_b` is
rejected. Unknown module suffixes, non-canonical or malformed tensor names,
layers beyond the base model, rank/shape mismatches, out-of-range tensor offsets,
and an adapter that matches no modules are rejected at load time. A zero
`lora_b` is a verified no-op. `Model.withLora` loads the adapter next to the base
model, and `loraRank`/`loraExpert` expose its identity for node reporting and
delta attribution.

## Still missing

1. **Block/KV cache.** The step already truncates the backbone at `end` and
   only scores the requested positions, but the clean prefix is still recomputed
   every round. A prefix cache and a block cache are the next large performance
   win.
2. **A session handle.** Config, tensor lookups and norm tensors are re-read on
   every call. A `DllmSession` is also the natural home for the KV cache.
3. **SDAR self-conditioning.** QwenDLM's residual adapter is not implemented;
   the base Qwen3 forward is.
4. **Real-model parity.** The synthetic tiny GGUFs cover the mask and API, and
   the converted BD3LM validation model now runs end to end. A logit parity
   check against the PyTorch reference is still required.
5. **Sampling and ensemble.** The host owns sampling, Self-Repulsion and delta
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

### Converted validation model

The 0.6B BD3LM validation checkpoint is converted and pinned in
`model-manifest.json` as `dllm`:

```text
RemydreScarlet/qwen3-0.6b-diffusion-bd3lm-GGUF  (Q4_0, 342 MB)
sha256 0f13278bde4d700118c8cf13d293af897a8974e74293de887aa383c567352a8e
```

`tools/convert_dllm.py` maps the upstream `a2d-qwen3` safetensors to the
`qwen3-dllm` GGUF layout (Q8_0 or Q4_0), including the Qwen3 tokenizer. The
`qwen3` BPE mode splits every digit (`\p{N}`, not GPT-2's `\p{N}{1,3}`) and
matches control tokens such as `<|im_start|>`/`<|mask|>` verbatim before BPE,
which Qwen prompts require. Input is assumed to be NFC-normalized. Chat prompts are built from
`<|im_start|>`/`<|im_end|>` ids plus `<|mask|>` (151669) for the response
block. This is an algorithm-validation model, not a production chat model.

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
| `dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1` | 0.6B | Apache-2.0 | algorithm validation |

**Quantization:** the kernels cover F32, BF16, Q4_0, Q4_1, Q8_0 and Q6_K. A
typical `Q4_K_M` conversion is not supported yet, so pin the converted dLLM GGUF
to Q8_0 or Q4_0 (or add a Q4_K kernel before shipping it). The pinned validation
artifact is Q4_0 (342 MB); the Q8_0 conversion (640 MB) is available from the
same converter for quality comparisons.

## Related

- `TeamFlaxia/theia` — chat UI, portable `/api`, merge functions, Crowd adapter.
- `TeamFlaxia/flaxia-crowd` `docs/theia-integration.md` — `theia-inference`
  workload contract.
