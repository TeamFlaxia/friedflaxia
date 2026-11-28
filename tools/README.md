# `tools/` — reference tooling only

**These Python scripts are reference implementations, not the production path.**
The production path is Rust: `friedflaxia-core` (GGUF/tokenizer/model IO and
forward pass) and `friedflaxia-train` (offline LoRA updates). The scripts are
kept during the migration so the checkpoint mapping and the training algorithm
remain inspectable and validation artifacts can be reproduced. Do not call
them from Crowd nodes, Theia, or any serving path. The upstream dLLM Python
trainer (PEFT/SFT recipes) has the same status: reference only.

## `convert_dllm.py`

Maps an upstream `a2d-qwen3` BF16 safetensors checkpoint to the
`qwen3-dllm` GGUF layout read by `friedflaxia-core`:

```sh
python3 tools/convert_dllm.py <model_dir> <out.gguf> [--quant q8_0|q4_0] [--name ...]
```

Requires `numpy` and `gguf` (gguf-py). It quantizes 2-D weights to Q8_0 or
Q4_0, keeps 1-D norms in F32, and carries the Qwen3 tokenizer (`tokenizer_model
= qwen3`, digit-wise `\p{N}` splitting and verbatim control tokens such as
`<|im_start|>` / `<|mask|>`; inputs are assumed NFC-normalized).

The pinned validation artifact
(`RemydreScarlet/qwen3-0.6b-diffusion-bd3lm-GGUF`, Q4_0) was made with this
script. It is an algorithm-validation model, not a production chat model. Use
this converter for understanding or reproducing that artifact; new artifacts
must be checked against the Rust GGUF reader, which is the compatibility
authority.

`tools/convert_nudenet.py` has the same status: a reference ONNX→GGUF
converter for the pinned NudeNet 320n F32 artifact.

## Replacement status: Rust `Writer`

| Area | Status |
|---|---|
| LoRA adapter GGUF (`qwen3-dllm-lora`) generation / updates | **Replaced (RQ3 solved).** `crates/friedflaxia-core/src/gguf/writer.rs` writes GGUF v3 with `rank` / `alpha` / `expert` metadata and F32 `lora_a` / `lora_b` tensors; `crates/friedflaxia-core/src/train/lora_factory.rs` + `examples/make_lora.rs` generate complete adapters from a base config; `lora_weights.rs` round-trips them; `friedflaxia-train` emits updated adapters. |
| Adapter-based training | **Replaced.** `friedflaxia-train` implements `es-lora`, `sft` and `grpo` offline (see below). No autograd tape in the runtime; cached hidden activations and logits are the input. |
| Full base dLLM conversion (safetensors → `qwen3-dllm`) | **Not replaced.** `convert_dllm.py` is currently the only converter for this layout and stays as reference while the migration is in progress. Do not delete it yet. |

## Compatibility notes

- **Two architectures.** Base weights are `qwen3-dllm`; adapters are a
  separate `qwen3-dllm-lora` GGUF. `Model.withLora(baseBytes, loraBytes)` adds
  the adapter at load time; `loraRank()` / `loraExpert()` report its identity.
- **Adapter metadata.** `qwen3-dllm-lora.rank` (U32, non-zero),
  `qwen3-dllm-lora.alpha` (F32, finite and positive; defaults to `rank`),
  `qwen3-dllm-lora.expert` (string, informational but used for capability
  routing and delta attribution).
- **Tensor names and shapes.** Canonical
  `blk.{i}.{module}.lora_a.weight` / `lora_b.weight` for the seven modules
  `attn_q`, `attn_k`, `attn_v`, `attn_output`, `ffn_gate`, `ffn_up`,
  `ffn_down`. GGML dims are `[in, rank]` for A and `[rank, out]` for B.
  Tensors must be F32; malformed names, unknown suffixes, duplicates, shape /
  rank mismatches and layers beyond the base model are rejected.
- **Bytes and alignment.** The Rust writer emits GGUF v3 little-endian with
  32-byte alignment before the tensor payload, so its output loads directly
  without another conversion step. Python gguf-py writes and the Rust writer
  are not guaranteed to be byte-identical; compare loaded tensors, not files.
- **Quantization.** Base conversion supports Q8_0 / Q4_0 only. The runtime
  kernels cover F32, BF16, Q4_0, Q4_1, Q6_K and Q8_0; Q4_K_M is not supported.
  Keep pinned dLLM GGUFs at Q8_0 or Q4_0. Adapter tensors stay F32.
- **Pinning.** Published adapters are immutable: `THEIA_LORA_ADAPTERS` records
  `url` / `sha256` / `maxBytes` / `revision` and nodes advertise
  `theia-expert-<id>` capabilities. Local `code` / `math` / `chat` artifacts
  exist for internal testing (below), but no trained expert is published and no
  real pin is configured yet.

## Adapter factory (production path, RQ3)

```sh
# Prefer reading shapes from the real pinned base GGUF (see model-manifest.json).
cargo run --release -p friedflaxia-core --example make_lora -- \
  --base /path/to/qwen3-bd3lm-q4_0.gguf --expert code \
  --out "$PWD/artifacts/theia-code-f32.gguf" \
  --round-out "$PWD/artifacts/theia-code-round.json"

# Verified fallback config, no base file needed: --pinned-config
# Defaults: rank 32, alpha 64, all 7 modules × 28 layers, F32 tensors.
```

**The generated weights are deterministic SplitMix64 internal-test seeds, NOT
trained experts.** They only prove loadability and expert identity; a real
trained adapter must replace them before any quality claim. Filenames say
`f32` because the adapter tensors are F32, not the Q4_0 base.

`--round-out` produces a deterministic SFT round derived from the adapter
itself: `x` is seeded, `logits` are the LoRA delta `B(Ax) · alpha/rank`, and
`target` is their argmax. This is a mechanical training-input example, not an
evaluation result. See [docs/dllm-denoise.md](../docs/dllm-denoise.md#rq3-lora--gguf-factory-solved)
for exact commands, artifact hashes and tests.

## Offline training CLI (production path)

```sh
cargo run -p friedflaxia-train -- es-lora adapter.gguf round.json out.gguf
cargo run -p friedflaxia-train -- sft     adapter.gguf round.json out.gguf
cargo run -p friedflaxia-train -- grpo    adapter.gguf round.json out.gguf
```

Round files (the stable shared-`x` form):

```json
{"alpha": 0.01, "samples": [{"seed": 1, "reward": 0.7}]}
{"lr": 0.001, "examples": [{"x": [0.1, 0.2], "logits": [0.5, -0.2], "target": 0}]}
{"lr": 0.01, "examples": [{"x": [0.1, 0.2], "logits": [0.5, -0.2], "sampled": 0, "advantage": 0.7}]}
```

`x` is a frozen hidden activation from a base-model forward and `logits` are
cached output scores; the CLI updates only LoRA A/B and writes a new loadable
adapter GGUF. Training never runs during inference (forward-only node, off-time
job, canary + rollback). See
[docs/dllm-denoise.md](../docs/dllm-denoise.md#offline-training-and-the-off-time-policy).
