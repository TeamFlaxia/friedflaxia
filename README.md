# FriedFlaxia

**GGUF-first, memory-bounded inference engine for Flaxia Crowd.**

FriedFlaxia is a Rust inference runtime that replaces ONNX Runtime in Flaxia Crowd's
browser and native nodes. It reads standard GGUF files, keeps weights quantized, and
loads only the tensors a task needs.

## Scope

The initial target is the set of models Flaxia Crowd actually runs:

| Task | Model | Format |
|---|---|---|
| Translation (`text-generation`) | PLaMo 3 Translate 0.8B | GGUF (Q4_0) |
| AI tags (`decision`) | LiquidAI d1-omni-600M | GGUF (Q8_0 + mmproj) |
| Post/CC embeddings | EmbeddingGemma 2 | GGUF (Q8_0 + mmproj) |
| NSFW detection | NudeNet 320n + NMS | GGUF (converted from ONNX) |

Everything else in `ai-inference` is out of scope.

## Principles

- **GGUF is the weight container.** No custom weight format.
- **Quantization stays in the weights.** Q4_0/Q8_0 blocks are dequantized on the fly;
  embedding gathers work directly on quantized rows.
- **Memory is bounded before loading.** Header parsing and tensor indexes are separate
  from tensor payload loading.
- **Browser and native share the core.** `friedflaxia-core` is dependency-free Rust and
  builds for `wasm32-unknown-unknown` as well as native targets.
- **TDD.** Behavior is defined by tests first, including golden values for quant blocks.

## Status

Implemented and tested:

- GGUF v2/v3 metadata and tensor-index parser, with tokenizer metadata skipping
- Q4_0, Q4_1, Q6_K, Q8_0 and BF16 dequantization, plus F16 scale decoding
- Quantized matrix-vector kernels for Q4_0/Q4_1/Q6_K/Q8_0/F32/BF16
- GGUF tokenizer metadata and tokenizers:
  - BPE for `gpt2` (d1-omni) and `gemma4` (EmbeddingGemma 2)
  - Unigram for `plamo3` (PLaMO 3 Translate)
- Architecture-scoped configuration reader
- Tensor views with quantized row access
- RMSNorm, LayerNorm, tanh/erf GELU, SiLU, softmax, NeoX/YaRN RoPE and pooling
- **EmbeddingGemma 2 text forward pass**
- **d1-omni decision forward pass**
- **PLaMO 3 Translate forward pass** (YaRN/full + plain/SWA RoPE, fused QKV,
  GQA, SwiGLU, post norms, greedy next-token)

Real-model parity:

| Model | Check | Result |
|---|---|---|
| EmbeddingGemma 2 Q8_0 | ONNX `sentence_embedding` | cosine `0.99994`, max abs `0.00136` |
| EmbeddingGemma 2 Q8_0 | HF tokenizer IDs / decode | exact |
| d1-omni Q8_0 | HF tokenizer IDs / decode | exact |
| d1-omni Q8_0 | ONNX decision logits (3 qtypes) | max abs `0.022` |
| PLaMO 3 Translate Q4_0 | llama.cpp top-10 logits | exact top-10, max logit delta `<0.5` |
| PLaMO 3 Translate Q4_0 | HF tokenizer IDs / decode | exact |

```sh
FRIEDFLAXIA_EMBED_GEMMA_GGUF=/path/embeddinggemma-2-Q8_0.gguf \
  cargo test --release --test embeddinggemma -- --ignored --nocapture
FRIEDFLAXIA_D1_GGUF=/path/d1-omni-600M-Q8_0.gguf \
  cargo test --release --test d1omni -- --ignored --nocapture
FRIEDFLAXIA_PLAMO_GGUF=/path/plamo-3-translate-0.8b-Q4_0.gguf \
  cargo test --release --test plamo3 -- --ignored --nocapture
```

CI downloads each pinned GGUF, verifies its SHA-256, then runs the tokenizer and
forward parity jobs.

Additionally:

- PLaMO KV cache (`PlamoSession`) with prefill + single-token decode
- wasm-bindgen package (`friedflaxia-wasm`) with `Model`/`Engine`
- Node smoke test against the real Q8_0 model
- Crowd opt-in backends (draft PR): embed-gemma, text-only vector-embed,
  d1 decision and PLaMO translation

Next slices:

1. Tensor range loader for streaming GGUF payloads
2. Multimodal EmbeddingGemma (vision/audio encoders)
3. NudeNet YOLOv8 + NMS (last ONNX dependency)
4. Default flip in Crowd and ONNX dependency removal
5. Further SIMD kernels and wasm size/load tuning

## wasm

```sh
./scripts/build-wasm.sh web          # browser ESM package
./scripts/build-wasm.sh nodejs /tmp/friedflaxia-nodejs
node scripts/wasm-smoke.cjs /tmp/friedflaxia-nodejs /path/embeddinggemma-2-Q8_0.gguf
```

## Development

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Inspect a GGUF header:

```sh
cargo run --example inspect -- /path/to/model.gguf
```

The parser only needs the header and the alignment padding, so a bounded HTTP Range
request is enough for inspection.

## License

MIT
