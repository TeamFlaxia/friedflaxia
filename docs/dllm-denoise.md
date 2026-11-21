# Theia dLLM denoise API (design)

Theia runs small masked-diffusion language models (dLLMs) across Flaxia Crowd
nodes and merges their per-position token distributions. This note records the
proposed FriedFlaxia extension: what already fits, what is missing, and the API
shape the Crowd workload layer can target.

> Status: design only. No runtime change yet.

## What already fits

FriedFlaxia is a good substrate for a denoiser even though it cannot run one
today:

- **Bidirectional attention exists.** `models/d1omni.rs` scores every query
  against all keys (no causal mask), and `models/embeddinggemma.rs` uses
  non-causal attention with an optional centered window.
- **All-position hidden states exist.** `trunk_forward` (d1-omni) and
  `forward_hidden_until` (PLaMo) return a vector per token; the PLaMo helper is
  currently private.
- **Per-position logits are a small step.** `decision_logits(marker_positions)`
  already returns one score per requested position, and the host applies
  softmax. Generalizing this from 3 decision classes to vocabulary-sized logits
  is the natural path.
- **GGUF is a generic container.** Quantized loading (`Q4_0`, `Q4_1`, `Q6_K`,
  `Q8_0`) and the tensor index do not care about the architecture name.
- **The wasm API already returns raw values for host-side post-processing**, so
  Self-Repulsion and distribution merging can stay in the Crowd JS workload
  layer while Rust focuses on the forward pass.

## What is missing

1. **Architecture whitelist.** `ModelKind::from_architecture` accepts only
   `gemma-embedding2`, `lfm2`, `plamo3`, `yolov8-nudenet`.
2. **Generation is AR greedy only.** `generate()` is prefill → one-token decode
   → `argmax`; there is no way to score a full masked canvas.
3. **No attention-mask machinery.** Attention ranges are hard-coded causal or
   full-range. Block diffusion needs a per-token mask (bidirectional inside the
   active block, causal across the clean prefix).
4. **RoPE takes a scalar sequential position.** dLLMs need per-token positions
   and a stable canvas index.
5. **KV cache is append-only and monotonic.** Iterative unmasking revises
   earlier positions, so either the cache is rebuilt per step or a block cache
   is introduced.
6. **No sampling and no ensemble.** `argmax` is the only selector; there is no
   multi-path pool or distribution merge.
7. **Scalar f32 kernels.** The current loops are fine for small encoders but
   will not carry a 2B dLLM across K paths × T steps at interactive latency.

## Proposed API

Keep the engine responsible for the forward pass and return **logits**, not
sampled tokens. The host owns Self-Repulsion and merging.

```rust
pub struct DenoiseStepInput<'a> {
    /// Canvas: prompt tokens followed by `mask_token_id` for unresolved slots.
    pub tokens: &'a [u32],
    /// Positions to score (the still-masked slots of the active block).
    pub positions: &'a [u32],
    /// Active block range `[start, end)`.
    pub block: (usize, usize),
}

pub struct DenoiseStepOutput {
    /// Row-major `positions.len() * vocab_size` logits.
    pub logits: Vec<f32>,
    pub vocab_size: usize,
    /// Per-position max-softmax confidence, for the host's commit rule.
    pub confidence: Vec<f32>,
}

impl Model {
    /// One masked-diffusion denoising step over the active block.
    pub fn denoise_step(
        &self,
        input: DenoiseStepInput<'_>,
    ) -> Result<DenoiseStepOutput, ModelError>;
}
```

A lower-level primitive makes the step easy to test and reuse:

```rust
impl Model {
    /// Hidden state for every token in `tokens`, with block-aware attention.
    pub fn forward_positions(&self, tokens: &[u32], block: (usize, usize))
        -> Result<Vec<Vec<f32>>, ModelError>;

    /// Vocabulary logits for the requested positions.
    pub fn position_logits(&self, tokens: &[u32], positions: &[u32], block: (usize, usize))
        -> Result<DenoiseStepOutput, ModelError>;
}
```

`softmax` stays out of Rust: Crowd's decision path already does it in JS, and
the merge needs the raw logits anyway.

## Implementation sketch

1. Add `models/qwen3_dlm.rs` (or the chosen dLLM backbone) with:
   - block-bidirectional attention mask,
   - per-token RoPE positions,
   - mask-token embedding lookup,
   - a block cache so the clean prefix is not recomputed every step.
2. Add `ModelKind` and the architecture string. Because we own the engine, the
   GGUF metadata can use a project-specific architecture name; llama.cpp
   compatibility is not required.
3. Expose `denoise_step` through `friedflaxia-wasm` and re-vendor the artifact
   into Crowd (`packages/node/src/vendor/friedflaxia`).
4. Pin the converted GGUF in Crowd's model registry with a SHA-256, as the
   existing models do.

## Performance

A denoise step is a full forward over the active block; a response costs
`K paths × T steps` of them. Even at block size 32 this is far more compute than
AR decoding. Before committing to a 2B model:

1. Measure one `denoise_step` on the target hardware.
2. If scalar f32 is too slow, add wasm SIMD for the browser and native threads
   for the desktop node.
3. If that is still short, a `wgpu` backend is the larger but durable option.

Start on the native node: it is easier to profile and does not need the wasm
rebuild loop. Port to wasm once the step shape and merge behaviour are settled.

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
