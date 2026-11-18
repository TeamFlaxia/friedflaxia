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

The first slice is complete:

- GGUF v2/v3 metadata and tensor-index parser
- Q4_0, Q8_0 and BF16 dequantization
- F16 scale decoding
- Real-header verification against PLaMo, d1-omni and EmbeddingGemma GGUF files

Next slices:

1. Q4_1 and Q6_K (PLaMo's output tensor is Q6_K)
2. GGUF tokenizer metadata reader (BPE and Unigram)
3. Quantized matrix-vector kernels
4. EmbeddingGemma text embedder
5. d1-omni decision forward
6. PLaMo generation
7. NudeNet YOLOv8 + NMS

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
