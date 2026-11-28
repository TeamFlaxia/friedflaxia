# Theia dLLM denoise API

Theia runs small masked-diffusion language models (dLLMs) across Flaxia Crowd
nodes and merges their per-position token distributions. This note records the
implemented FriedFlaxia extension and what is still missing.

> Status: the qwen3-dllm forward pass, block attention mask, full-logit and
> top-k APIs, LoRA adapters, the wasm bindings, the native `DllmSession`
> block/KV prefix cache, the offline Rust training CLI, and the LoRA→GGUF
> adapter factory (RQ3) are implemented. Top-candidate parity against the
> pinned PyTorch checkpoint is validated on the converted Q8_0 model;
> bit-exact BF16 logit parity and a *trained* expert adapter are still open.
> Internal-test seed adapter pins are published on Hugging Face (see RQ3 below).

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
- **Native block/KV prefix cache.** `DllmSession` keeps the committed causal
  prefix's final hidden rows and per-layer post-RoPE K/V rows. Each step runs
  the backbone only over `[previous_prefix, end)`; the free functions remain the
  uncached differential oracle. See [Block/KV cache](#blockkv-cache) below. The
  session is native-only; no wasm binding was added.

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

## RQ3: LoRA → GGUF factory (solved)

`crates/friedflaxia-core/src/train/lora_factory.rs` builds a complete adapter
with the Rust `Writer`; `crates/friedflaxia-core/examples/make_lora.rs` is its
CLI. Shapes come from the **actual pinned base GGUF** by default (verified:
`qwen3-dllm.block_count=28`, `embedding_length=1024`,
`feed_forward_length=3072`, `head_count=16`, `head_count_kv=8`,
`key_length=128`, vocab 151936 → `q_dim=2048`, `kv_dim=1024`). A documented
`--pinned-config` fallback uses these verified constants without a local base.

```sh
# Obtain the pinned base from crates/friedflaxia-core/model-manifest.json.
curl -L -o /tmp/base-qwen3-bd3lm-q4_0.gguf \
  'https://huggingface.co/RemydreScarlet/qwen3-0.6b-diffusion-bd3lm-GGUF/resolve/fd3fb1978befe31cff16ab6cfcd4d5184cca0e9b/qwen3-bd3lm-q4_0.gguf'

# Generate rank=32 / alpha=64 adapters, 7 modules × 28 layers, F32 tensors.
cargo run --release -p friedflaxia-core --example make_lora -- \
  --base /tmp/base-qwen3-bd3lm-q4_0.gguf --expert code \
  --out "$PWD/artifacts/theia-code-f32.gguf" \
  --round-out "$PWD/artifacts/theia-code-round.json"
cargo run --release -p friedflaxia-core --example make_lora -- \
  --base /tmp/base-qwen3-bd3lm-q4_0.gguf --expert math \
  --out "$PWD/artifacts/theia-math-f32.gguf"
cargo run --release -p friedflaxia-core --example make_lora -- \
  --base /tmp/base-qwen3-bd3lm-q4_0.gguf --expert chat \
  --out "$PWD/artifacts/theia-chat-f32.gguf"

# Apply one offline SFT update (mechanical demo, see below).
cargo run --release -p friedflaxia-train -- sft \
  artifacts/theia-code-f32.gguf artifacts/theia-code-round.json \
  artifacts/theia-code-sft-update-f32.gguf
```

**Warning: these weights are deterministic SplitMix64 internal-test seeds, not
trained experts.** They prove the GGUF format, `LoraModel::from_gguf`, expert
identity and the offline update path; they carry no quality claim. A real
trained adapter must replace them before any evaluation or release. Filenames
say `f32` because the adapter tensors are F32 (the *base* is Q4_0).

`--round-out` writes a deterministic SFT round for `blk.0.attn_q`: `x` is a
seeded normal vector, `logits` are the generated adapter's own LoRA delta
(`alpha/rank · B(Ax)`), and `target` is the argmax of those synthetic logits.
No base forward, reward model or evaluation produced them; the resulting
`*-sft-update-*` file only demonstrates that the offline CLI performs a real
AdamW step and re-serializes a loadable adapter.

Local artifacts (absolute paths under
`/home/remydre/flaxian/friedflaxia/artifacts/`, generated 2026-11-28). The three
expert files are published at an immutable HF revision in
`RemydreScarlet/theia-lora-adapters` and mirrored in Theia
`model-dev/adapters.json`:

| File | Bytes | SHA-256 |
|---|---:|---|
| `theia-code-f32.gguf` | 80767200 | `a051b00f8fc70c2506eded1cd91df8a54a4fb0592befd28a394d6dd9bd20e755` |
| `theia-math-f32.gguf` | 80767200 | `e4b6e02df51dc06f9cd9179a0d7bad75ae9092ac840a73e921a4e3595aae4d3b` |
| `theia-chat-f32.gguf` | 80767200 | `e58285ca4482343f8fa0f33a3e38667a158da74a2c69a394d945b15f114e9427` |
| `theia-code-sft-update-f32.gguf` | 80767200 | `7795f2ecbacc1fc5042c5f6badd0a47c55766835f13c78c9e46a58fe0e6f3747` |
| `theia-code-round.json` | 42140 | `124275db081f3f460d600f80f6489033e5743c57ae1769fa0607e162636ffd75` |

`crates/friedflaxia-core/tests/lora_factory.rs` covers the production path:
writer → GGUF parser → `LoraModel::from_gguf` (rank/alpha/expert/pair count),
canonical names and shapes, determinism, the train-writer round trip, and
`denoise_step_with_lora` (the core equivalent of `Model.withLora`). An ignored
test checks the real pinned base and real artifacts via
`FRIEDFLAXIA_DLLM_GGUF` / `FRIEDFLAXIA_LORA_GGUF` /
`FRIEDFLAXIA_LORA_UPDATED_GGUF`.

## Offline training and the off-time policy

**The production training path is Rust**: `friedflaxia-core` for GGUF IO and the
adapter `Writer`, and `friedflaxia-train` for offline LoRA updates. The Python
`tools/convert_dllm.py` converter and the upstream dLLM Python trainer
(PEFT/SFT recipes) are **reference implementations only**, kept during the
migration for validation and comparison. They are never called from a Crowd
node or from Theia's inference path. See [tools/README.md](../tools/README.md)
for the replacement status and compatibility constraints.

Inference is forward-only. A node denoises with a pinned base dLLM plus a
pinned expert adapter, reports scores and commit outcomes, and **never trains
or mutates weights while serving**. Commit logs and cached activations are the
training *input*, collected during inference and consumed later by a separate
off-time job. The output is a new immutable adapter pin (`url` / `sha256` /
`maxBytes` / `revision`), rolled out via canary with the previous pin kept for
immediate rollback. This is the T5 (expert pin / capability) policy described
in `theia/model-dev/research/06-offline-adapter-factory.md` and
`flaxia-crowd/docs/theia-integration.md` in the sibling repositories.

The native CLI (not part of the wasm build) is:

```sh
cargo run -p friedflaxia-train -- es-lora adapter.gguf round.json out.gguf
cargo run -p friedflaxia-train -- sft     adapter.gguf round.json out.gguf
cargo run -p friedflaxia-train -- grpo    adapter.gguf round.json out.gguf
```

- `es-lora`: evolution strategies over the LoRA A/B vector, driven by
  `{"alpha": 0.01, "samples": [{"seed": 1, "reward": 0.7}]}`; no backward pass.
- `sft`: cross-entropy on cached base-model hidden activations, driven by
  `{"lr": 0.001, "examples": [{"x": [...], "logits": [...], "target": 0}]}`.
- `grpo`: the same round format with `sampled` and `advantage` instead of
  `target`; a zero advantage leaves the adapter byte-identical.

`x` is a frozen hidden vector from a base-model forward (there is no autograd
tape in the runtime), and `logits` are the cached output scores. The CLI only
updates LoRA A/B tensors and re-serializes a loadable F32
`qwen3-dllm-lora` GGUF through `crates/friedflaxia-core/src/gguf/writer.rs`; it
does not run a base forward and never changes inference-time behavior. Run it
as its own process on its own schedule, outside inference traffic.

### Expert pin / capability status (T5 wiring)

| Item | Status |
|---|---|
| `Model.withLora`, `loraRank()`, `loraExpert()` | implemented in Rust/wasm |
| `THEIA_LORA_ADAPTERS` pins and `theia-expert-*` capability routing | implemented in Crowd |
| LoRA adapter GGUF production (`qwen3-dllm-lora`) | **solved (RQ3)**: Rust `Writer` + `lora_factory` + `make_lora` CLI + train CLI |
| Real LoRA GGUF artifact and actual pin values | internal-test seed pins published on HF (`RemydreScarlet/theia-lora-adapters`, mirrored in Theia `model-dev/adapters.json`); **trained expert still open** |
| Block/KV cache (`DllmSession`, native) | implemented |
| PyTorch top-candidate parity (Q8_0) | validated; bit-exact BF16 parity not claimed |

## Block/KV cache

`DllmSession` parses the GGUF once and owns a committed prefix:

- `tokens[..P]` — the token ids the cached rows were computed from,
- `hidden[..P]` — final `n_embd`-wide rows (fed to `output_norm` and the output
  projection when a caller scores a prefix position),
- per-layer `k[..P]` / `v[..P]` — post q/k-RMSNorm, post-RoPE GQA rows,
- the identity of the adapter used for those rows.

Because a query `q < block.0` is causal (`key_end = q + 1`), its row never sees
a block token. Such rows are committed only after a fully successful step; a
step with `block.0 > P` first computes the causal gap `[P..block.0)` and commits
it, while a step with `block.0 < P` drops cached rows down to `block.0`.

Invalidation rules:

- if a committed token at index `i < P` changed, the prefix is truncated to `i`;
- if `block.0 < P`, the prefix is truncated to `block.0`;
- if `block.0 > P`, rows `[P..block.0)` are computed causally and committed on
  success;
- changing the LoRA instance, or switching between `None` and `Some`, clears the
  entire prefix (identity is compared by pointer address);
- `positions`, `top_k` and `block.1` changes do not invalidate anything, and
  tokens at or after `block.1` stay ignored;
- any `ModelError` leaves the cache exactly as it was — nothing is committed on
  a failed step.

`denoise_step` / `denoise_step_with_lora` and `denoise_topk` /
`denoise_topk_with_lora` mirror the free functions. Only fresh rows are
projected and attended; cached keys/values are read for earlier positions. The
same `scores` vector, `softmax_in_place`, `crate::simd::dot`, LoRA order and
residual order are used by both paths, so the synthetic F32 fixtures are
bit-identical to the uncached oracle (SIMD on and off). When only one fresh row
remains, the cached projection batch is padded with an unused row so the Q4_0
`matvec`/`matmul` scalar accumulation orders cannot diverge.
## Still missing

1. **SDAR self-conditioning.** QwenDLM's residual adapter is not implemented;
   the base Qwen3 forward is.
2. **Bit-exact BF16 parity.** The synthetic tiny GGUFs cover the mask, API and
   cache equivalence; the converted BD3LM validation model runs end to end and
   its top candidates match the PyTorch reference (see below). Q4_0 tail
   candidates can still swap when logits are near-tied, so bit-exact logit
   parity with the BF16 checkpoint is not claimed.
3. **Sampling and ensemble.** The host owns sampling, Self-Repulsion and delta
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
- invalid token ids, blocks, duplicate positions and an out-of-block tail,
- `DllmSession` cache equivalence: empty/warm/gap steps, prefix mutation,
  backwards blocks, prefix scoring, LoRA switches, tail changes, hardened golden
  logits, error/clear handling, and a randomized multi-step differential with
  SIMD enabled and disabled.

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
block. This is an algorithm-validation model, not a production chat model, and
the converter is a **reference** tool: production artifacts follow the Rust path
described above, with the Rust GGUF loader as the compatibility authority.

## PyTorch reference parity

`tools/dllm_reference.py` loads the pinned HF checkpoint in BF16 with
Transformers' stock Qwen3 model and applies the same block mask as the Rust
kernel. It writes top-k candidates (and optionally raw f32 logits) for a fixed
token sequence, which the ignored
`real_dllm_matches_pytorch_reference` test compares against a converted Q8_0
GGUF:

```sh
python3 tools/convert_dllm.py <hf-checkpoint-dir> /tmp/dllm-q8.gguf --quant q8_0
python3 tools/dllm_reference.py --model-dir <hf-checkpoint-dir> \
  --tokens 151644,872,151645,151669,100,200,300,400 \
  --block-start 0 --block-end 8 --positions 4,5,6 \
  --out crates/friedflaxia-core/tests/fixtures/dllm-pytorch-top8.json
FRIEDFLAXIA_DLLM_Q8_GGUF=/tmp/dllm-q8.gguf \
  cargo test --release --test dllm -- --ignored --nocapture
```

On the pinned 0.6B checkpoint, the Q8_0 Rust logits have cosine similarity
0.9999 with the BF16 reference and agree on top-1 for all three positions and
on 7 of 8 candidates each. Q4_0 quantization is coarser and swaps near-tied
tail candidates, which is why the parity fixture uses Q8_0. The CI `dllm-parity`
job separately verifies the pinned Q4_0 model end to end and the cached-vs-
uncached equivalence.

## Performance

A denoise step is a full forward over the active block; a response costs
`K paths × T steps` of them. Before committing to a 4B model:

1. Measure one step on the target hardware (native first).
2. Use `DllmSession` so the clean prefix is not recomputed.
3. Use the existing batched quantized matmul, then wasm SIMD / native threads,
   and finally `wgpu` if the step is still short.

### Native measurement

`crates/friedflaxia-core/examples/dllm_bench.rs` walks an eight-position block
across a 64-token canvas, runs the free function and `DllmSession` on the same
inputs, asserts identical top-8 candidates on every step, and prints per-step
timings:

```sh
FRIEDFLAXIA_DLLM_GGUF=/path/to/qwen3-bd3lm-q4_0.gguf \
  cargo run --release --example dllm_bench
```

On this host (release profile, pinned Q4_0 0.6B model, eight scored positions
per step) the eight steps took 27.3 s uncached vs 12.6 s cached, **2.17x**,
with the gap widening from step 2 onward as the committed prefix grows. Both
paths re-read the GGUF layer tensors on every call, so the remaining floor is
weight streaming; caching removes the prefix projections/norms/FFN, not the
attention over cached prefix keys. These are absolute timings for this machine
only — re-run on the target hardware before sizing a 4B model.

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
