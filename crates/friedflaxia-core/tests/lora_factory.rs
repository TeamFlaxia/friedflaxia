//! Tests for the RQ3 adapter factory: generated seed GGUFs must take the same
//! load path as production adapters (parser -> `LoraModel` -> denoise), and
//! must be accepted by the offline train writer.

use friedflaxia_core::gguf::{Gguf, ParseOptions, Value};
use friedflaxia_core::models::dllm::{DllmConfig, denoise_step, denoise_step_with_lora};
use friedflaxia_core::models::lora::{LoraModel, LoraModule};
use friedflaxia_core::train::lora_factory::{build_seed_adapter, expert_seed};
use friedflaxia_core::train::lora_weights::LoraWeights;
use friedflaxia_core::train::rng::SplitMix64;

const RANK: u32 = 4;
const ALPHA: f32 = 8.0;
const SEED: u64 = 17;

fn config() -> DllmConfig {
    DllmConfig {
        n_layer: 1,
        n_embd: 8,
        n_ff: 16,
        n_head: 2,
        n_head_kv: 1,
        head_dim: 4,
        eps: 1e-6,
        rope_freq_base: 10_000.0,
        vocab_size: 5,
    }
}

fn put_string(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

fn put_kv_u32(out: &mut Vec<u8>, key: &str, value: u32) {
    put_string(out, key);
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_kv_f32(out: &mut Vec<u8>, key: &str, value: f32) {
    put_string(out, key);
    out.extend_from_slice(&6u32.to_le_bytes());
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_kv_string(out: &mut Vec<u8>, key: &str, value: &str) {
    put_string(out, key);
    out.extend_from_slice(&8u32.to_le_bytes());
    put_string(out, value);
}

struct Spec {
    name: String,
    dims: Vec<u64>,
    values: Vec<f32>,
}

fn values(count: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    (0..count).map(|_| rng.next_f32() * 0.2 - 0.1).collect()
}

fn mat(name: &str, cols: usize, rows: usize, seed: u64) -> Spec {
    Spec {
        name: name.to_owned(),
        dims: vec![cols as u64, rows as u64],
        values: values(cols * rows, seed),
    }
}

fn vec1(name: &str, len: usize, uniform: bool, seed: u64) -> Spec {
    Spec {
        name: name.to_owned(),
        dims: vec![len as u64],
        values: if uniform {
            vec![1.0; len]
        } else {
            values(len, seed)
        },
    }
}

/// Minimal `qwen3-dllm` base with the same dimensions as `config()`.
fn base_bytes() -> Vec<u8> {
    let c = config();
    let q = c.q_dim();
    let kv = c.kv_dim();
    let mut specs = vec![mat("token_embd.weight", c.n_embd, c.vocab_size, 1)];
    for layer in 0..c.n_layer {
        let p = format!("blk.{layer}");
        let seed = 100 * (layer as u64 + 1);
        specs.push(vec1(&format!("{p}.attn_norm.weight"), c.n_embd, true, 0));
        specs.push(mat(&format!("{p}.attn_q.weight"), c.n_embd, q, seed + 1));
        specs.push(mat(&format!("{p}.attn_k.weight"), c.n_embd, kv, seed + 2));
        specs.push(mat(&format!("{p}.attn_v.weight"), c.n_embd, kv, seed + 3));
        specs.push(mat(
            &format!("{p}.attn_output.weight"),
            q,
            c.n_embd,
            seed + 4,
        ));
        specs.push(vec1(
            &format!("{p}.attn_q_norm.weight"),
            c.head_dim,
            true,
            0,
        ));
        specs.push(vec1(
            &format!("{p}.attn_k_norm.weight"),
            c.head_dim,
            true,
            0,
        ));
        specs.push(vec1(&format!("{p}.ffn_norm.weight"), c.n_embd, true, 0));
        specs.push(mat(
            &format!("{p}.ffn_gate.weight"),
            c.n_embd,
            c.n_ff,
            seed + 5,
        ));
        specs.push(mat(
            &format!("{p}.ffn_up.weight"),
            c.n_embd,
            c.n_ff,
            seed + 6,
        ));
        specs.push(mat(
            &format!("{p}.ffn_down.weight"),
            c.n_ff,
            c.n_embd,
            seed + 7,
        ));
    }
    specs.push(vec1("output_norm.weight", c.n_embd, true, 0));

    let mut metadata = Vec::new();
    put_kv_u32(&mut metadata, "general.alignment", 32);
    put_kv_string(&mut metadata, "general.architecture", "qwen3-dllm");
    put_kv_u32(&mut metadata, "qwen3-dllm.block_count", c.n_layer as u32);
    put_kv_u32(
        &mut metadata,
        "qwen3-dllm.embedding_length",
        c.n_embd as u32,
    );
    put_kv_u32(
        &mut metadata,
        "qwen3-dllm.feed_forward_length",
        c.n_ff as u32,
    );
    put_kv_u32(
        &mut metadata,
        "qwen3-dllm.attention.head_count",
        c.n_head as u32,
    );
    put_kv_u32(
        &mut metadata,
        "qwen3-dllm.attention.head_count_kv",
        c.n_head_kv as u32,
    );
    put_kv_u32(
        &mut metadata,
        "qwen3-dllm.attention.key_length",
        c.head_dim as u32,
    );
    put_kv_f32(
        &mut metadata,
        "qwen3-dllm.attention.layer_norm_rms_epsilon",
        c.eps,
    );
    put_kv_f32(&mut metadata, "qwen3-dllm.rope.freq_base", c.rope_freq_base);

    let mut offsets = Vec::with_capacity(specs.len());
    let mut cursor = 0usize;
    for spec in &specs {
        offsets.push(cursor);
        cursor += spec.values.len() * 4;
        cursor = cursor.div_ceil(32) * 32;
    }
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(specs.len() as u64).to_le_bytes());
    out.extend_from_slice(&10u64.to_le_bytes());
    out.extend_from_slice(&metadata);
    for (spec, offset) in specs.iter().zip(&offsets) {
        put_string(&mut out, &spec.name);
        out.extend_from_slice(&(spec.dims.len() as u32).to_le_bytes());
        for dim in &spec.dims {
            out.extend_from_slice(&dim.to_le_bytes());
        }
        out.extend_from_slice(&0u32.to_le_bytes()); // F32
        out.extend_from_slice(&(*offset as u64).to_le_bytes());
    }
    let data_start = out.len().div_ceil(32) * 32;
    out.resize(data_start, 0);
    for (spec, offset) in specs.iter().zip(&offsets) {
        out.resize(data_start + offset, 0);
        for value in &spec.values {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
    out
}

fn f32_data(bytes: &[u8], gguf: &Gguf, name: &str) -> Vec<f32> {
    gguf.tensor_data(bytes, name)
        .unwrap_or_else(|error| panic!("tensor payload {name}: {error}"))
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

#[test]
fn seed_adapter_loads_with_the_production_parser_and_lora_model() {
    let c = config();
    let bytes = build_seed_adapter(&c, RANK, ALPHA, "code", SEED).expect("adapter bytes");
    let gguf = Gguf::parse(&bytes).expect("writer output parses");
    assert_eq!(
        gguf.metadata("general.architecture"),
        Some(&Value::String("qwen3-dllm-lora".into()))
    );
    assert_eq!(
        gguf.metadata("qwen3-dllm-lora.rank"),
        Some(&Value::U32(RANK))
    );
    assert_eq!(
        gguf.metadata("qwen3-dllm-lora.alpha"),
        Some(&Value::F32(ALPHA))
    );
    assert_eq!(
        gguf.metadata("qwen3-dllm-lora.expert"),
        Some(&Value::String("code".into()))
    );
    assert_eq!(gguf.tensor_count(), c.n_layer * 7 * 2);

    for (layer, module, side, dims) in [
        (
            0,
            LoraModule::AttnQ,
            "lora_a",
            vec![c.n_embd as u64, RANK as u64],
        ),
        (
            0,
            LoraModule::AttnQ,
            "lora_b",
            vec![RANK as u64, c.q_dim() as u64],
        ),
        (
            0,
            LoraModule::AttnK,
            "lora_a",
            vec![c.n_embd as u64, RANK as u64],
        ),
        (
            0,
            LoraModule::FfnDown,
            "lora_a",
            vec![c.n_ff as u64, RANK as u64],
        ),
        (
            0,
            LoraModule::FfnDown,
            "lora_b",
            vec![RANK as u64, c.n_embd as u64],
        ),
    ] {
        let name = format!("blk.{layer}.{}.{side}.weight", module.suffix());
        let info = gguf.tensor(&name).unwrap_or_else(|| panic!("{name}"));
        assert_eq!(info.dims, dims, "{name}");
        let data = f32_data(&bytes, &gguf, &name);
        assert_eq!(data.len(), dims[0] as usize * dims[1] as usize);
        assert!(data.iter().all(|value| value.is_finite() && *value != 0.0));
    }

    let lora = LoraModel::from_gguf(gguf, bytes, &c).expect("LoraModel::from_gguf");
    assert_eq!(lora.rank(), RANK as usize);
    assert!((lora.scale() - ALPHA / RANK as f32).abs() < 1e-6);
    assert_eq!(lora.expert(), Some("code"));
    assert_eq!(lora.pair_count(), c.n_layer * 7);
}

#[test]
fn seed_adapter_is_deterministic_and_expert_separated() {
    let c = config();
    let a = build_seed_adapter(&c, RANK, ALPHA, "code", SEED).unwrap();
    let b = build_seed_adapter(&c, RANK, ALPHA, "code", SEED).unwrap();
    assert_eq!(a, b, "same seed and expert must produce identical bytes");
    let other = build_seed_adapter(&c, RANK, ALPHA, "math", SEED).unwrap();
    assert_ne!(a, other, "expert id must separate the weight stream");
    assert_ne!(expert_seed(SEED, "code"), expert_seed(SEED, "math"));
}

#[test]
fn seed_adapter_round_trips_through_the_train_writer() {
    let c = config();
    let bytes = build_seed_adapter(&c, RANK, ALPHA, "chat", SEED).unwrap();
    let gguf = Gguf::parse(&bytes).unwrap();
    let weights = LoraWeights::from_gguf(&gguf, &bytes).expect("train CLI can parse the adapter");
    assert_eq!(weights.rank(), RANK as usize);
    assert_eq!(weights.alpha(), ALPHA);
    assert_eq!(weights.expert(), "chat");
    let rewritten = weights.write_gguf().expect("train writer output");
    let rewritten_gguf = Gguf::parse(&rewritten).expect("train output parses");
    assert_eq!(rewritten_gguf.tensor_count(), c.n_layer * 7 * 2);
    let reloaded = LoraModel::from_gguf(rewritten_gguf, rewritten, &c).expect("train output loads");
    assert_eq!(reloaded.expert(), Some("chat"));
    assert_eq!(reloaded.pair_count(), c.n_layer * 7);
}

#[test]
fn seed_adapter_runs_through_the_with_lora_equivalent_path() {
    let base = base_bytes();
    let gguf = Gguf::parse(&base).expect("tiny base parses");
    let config = DllmConfig::from_gguf(&gguf).expect("base config");
    let adapter = build_seed_adapter(&config, RANK, ALPHA, "code", SEED).unwrap();
    let adapter_gguf = Gguf::parse(&adapter).expect("adapter parses");
    let lora = LoraModel::from_gguf(adapter_gguf, adapter, &config).expect("adapter loads");

    let tokens = [1, 2, 3, 4];
    let positions = [2, 3];
    let block = (2, 4);
    let plain = denoise_step(&gguf, &base, &tokens, &positions, block).expect("base step");
    let adapted = denoise_step_with_lora(&gguf, &base, &tokens, &positions, block, Some(&lora))
        .expect("adapter step");
    assert_eq!(plain.logits.len(), adapted.logits.len());
    assert_ne!(plain.logits, adapted.logits, "seed LoRA must affect logits");
    assert!(adapted.logits.iter().all(|value| value.is_finite()));
}

#[test]
#[ignore = "requires FRIEDFLAXIA_DLLM_GGUF with the pinned qwen3-dllm base"]
fn pinned_base_shapes_match_the_generator_config() {
    let path = std::env::var("FRIEDFLAXIA_DLLM_GGUF").expect("set FRIEDFLAXIA_DLLM_GGUF");
    let bytes = std::fs::read(path).expect("read base GGUF");
    let gguf = Gguf::parse_with_options(
        &bytes,
        ParseOptions {
            skip_tokenizer: true,
        },
    )
    .expect("parse base GGUF");
    let c = DllmConfig::from_gguf(&gguf).expect("qwen3-dllm config");
    assert_eq!(c.n_layer, 28);
    assert_eq!(c.n_embd, 1024);
    assert_eq!(c.n_ff, 3072);
    assert_eq!(c.n_head, 16);
    assert_eq!(c.n_head_kv, 8);
    assert_eq!(c.head_dim, 128);
    assert_eq!(c.q_dim(), 2048);
    assert_eq!(c.kv_dim(), 1024);
    assert_eq!(c.vocab_size, 151_936);

    // When an artifact generated by `make_lora` is available, verify the real
    // file (not a tiny fixture) through the same loader `Model.withLora` uses.
    if let Ok(adapter_path) = std::env::var("FRIEDFLAXIA_LORA_GGUF") {
        let adapter = std::fs::read(adapter_path).expect("read adapter GGUF");
        let adapter_gguf = Gguf::parse(&adapter).expect("parse adapter GGUF");
        let lora = LoraModel::from_gguf(adapter_gguf, adapter.clone(), &c).expect("load adapter");
        assert_eq!(lora.rank(), 32);
        assert!((lora.scale() - 2.0).abs() < 1e-6);
        assert_eq!(lora.pair_count(), c.n_layer * 7);
        assert!(["code", "math", "chat"].contains(&lora.expert().unwrap()));

        if let Ok(updated_path) = std::env::var("FRIEDFLAXIA_LORA_UPDATED_GGUF") {
            let updated = std::fs::read(updated_path).expect("read trained-update adapter");
            let updated_gguf = Gguf::parse(&updated).expect("parse trained-update adapter");
            let updated_lora = LoraModel::from_gguf(updated_gguf, updated.clone(), &c)
                .expect("trained-update loads");
            assert_eq!(updated_lora.expert(), lora.expert());
            assert_eq!(updated_lora.pair_count(), lora.pair_count());
            // The offline SFT step must have changed at least one tensor.
            let name = "blk.0.attn_q.lora_a.weight";
            assert_ne!(
                f32_data(&adapter, &Gguf::parse(&adapter).unwrap(), name),
                f32_data(&updated, &Gguf::parse(&updated).unwrap(), name),
            );
        }
    }
}
