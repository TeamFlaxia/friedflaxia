use friedflaxia_core::gguf::{Gguf, TensorType};
use friedflaxia_core::models::dllm::{denoise_step, denoise_topk};

const ALIGNMENT: usize = 32;
const GGUF_VERSION: u32 = 3;
const F32_TYPE: u32 = 0;

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

fn align_up(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}

fn pseudo_weights(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 33) as f32) / ((1u64 << 31) as f32);
            (unit - 0.5) * 0.4
        })
        .collect()
}

struct TensorSpec {
    name: &'static str,
    dims: Vec<u64>,
    values: Vec<f32>,
}

fn matrix(name: &'static str, cols: usize, rows: usize, seed: u64) -> TensorSpec {
    TensorSpec {
        name,
        dims: vec![cols as u64, rows as u64],
        values: pseudo_weights(cols * rows, seed),
    }
}

fn vector(name: &'static str, len: usize) -> TensorSpec {
    TensorSpec {
        name,
        dims: vec![len as u64],
        values: vec![1.0; len],
    }
}

/// Tiny Qwen3-style dLLM: one layer, 8-d hidden, 2 query heads, 1 KV head.
fn model_specs(tie_output: bool) -> Vec<TensorSpec> {
    let mut specs = vec![
        matrix("token_embd.weight", 8, 5, 1),
        vector("blk.0.attn_norm.weight", 8),
        matrix("blk.0.attn_q.weight", 8, 8, 2),
        matrix("blk.0.attn_k.weight", 8, 4, 3),
        matrix("blk.0.attn_v.weight", 8, 4, 4),
        vector("blk.0.attn_q_norm.weight", 4),
        vector("blk.0.attn_k_norm.weight", 4),
        matrix("blk.0.attn_output.weight", 8, 8, 5),
        vector("blk.0.ffn_norm.weight", 8),
        matrix("blk.0.ffn_gate.weight", 8, 16, 6),
        matrix("blk.0.ffn_up.weight", 8, 16, 7),
        matrix("blk.0.ffn_down.weight", 16, 8, 8),
        vector("output_norm.weight", 8),
    ];
    if !tie_output {
        specs.push(matrix("output.weight", 8, 5, 9));
    }
    specs
}

fn build_model(tie_output: bool) -> Vec<u8> {
    let specs = model_specs(tie_output);
    let mut metadata = Vec::new();
    put_kv_u32(&mut metadata, "general.alignment", ALIGNMENT as u32);
    put_kv_string(&mut metadata, "general.architecture", "qwen3-dllm");
    put_kv_u32(&mut metadata, "qwen3-dllm.block_count", 1);
    put_kv_u32(&mut metadata, "qwen3-dllm.embedding_length", 8);
    put_kv_u32(&mut metadata, "qwen3-dllm.feed_forward_length", 16);
    put_kv_u32(&mut metadata, "qwen3-dllm.attention.head_count", 2);
    put_kv_u32(&mut metadata, "qwen3-dllm.attention.head_count_kv", 1);
    put_kv_u32(&mut metadata, "qwen3-dllm.attention.key_length", 4);
    put_kv_f32(
        &mut metadata,
        "qwen3-dllm.attention.layer_norm_rms_epsilon",
        1e-6,
    );
    put_kv_f32(&mut metadata, "qwen3-dllm.rope.freq_base", 10_000.0);
    let metadata_count = 10u64;

    let mut offsets = Vec::with_capacity(specs.len());
    let mut cursor = 0usize;
    for spec in &specs {
        offsets.push(cursor);
        cursor = align_up(cursor + spec.values.len() * 4, ALIGNMENT);
    }

    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    out.extend_from_slice(&(specs.len() as u64).to_le_bytes());
    out.extend_from_slice(&metadata_count.to_le_bytes());
    out.extend_from_slice(&metadata);

    for (spec, offset) in specs.iter().zip(&offsets) {
        put_string(&mut out, spec.name);
        out.extend_from_slice(&(spec.dims.len() as u32).to_le_bytes());
        for dim in &spec.dims {
            out.extend_from_slice(&dim.to_le_bytes());
        }
        out.extend_from_slice(&F32_TYPE.to_le_bytes());
        out.extend_from_slice(&(*offset as u64).to_le_bytes());
    }

    let data_start = align_up(out.len(), ALIGNMENT);
    out.resize(data_start, 0);
    for (spec, offset) in specs.iter().zip(&offsets) {
        out.resize(data_start + offset, 0);
        for value in &spec.values {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
    out
}

fn assert_close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (index, (left, right)) in a.iter().zip(b).enumerate() {
        assert!(
            (left - right).abs() < 1e-5,
            "value {index} differs: {left} != {right}"
        );
    }
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0usize;
    for index in 1..row.len() {
        if row[index] > row[best] {
            best = index;
        }
    }
    best
}

#[test]
fn denoise_step_returns_logits_and_confidence() {
    let bytes = build_model(false);
    let gguf = Gguf::parse(&bytes).expect("parse tiny dLLM");
    let out = denoise_step(&gguf, &bytes, &[1, 2, 3, 4], &[2, 3], (2, 4)).expect("denoise");

    assert_eq!(out.vocab_size, 5);
    assert_eq!(out.positions, vec![2, 3]);
    assert_eq!(out.logits.len(), 2 * 5);
    assert_eq!(out.confidence.len(), 2);
    for confidence in &out.confidence {
        assert!(*confidence > 0.0 && *confidence <= 1.0);
    }
    for row in out.logits.chunks_exact(5) {
        assert!(row.iter().any(|value| value.abs() > 1e-9));
    }
}

#[test]
fn denoise_step_is_deterministic() {
    let bytes = build_model(false);
    let gguf = Gguf::parse(&bytes).expect("parse tiny dLLM");
    let first = denoise_step(&gguf, &bytes, &[1, 2, 3, 4], &[2, 3], (2, 4)).expect("first");
    let second = denoise_step(&gguf, &bytes, &[1, 2, 3, 4], &[2, 3], (2, 4)).expect("second");
    assert_close(&first.logits, &second.logits);
}

#[test]
fn block_mask_hides_future_tokens() {
    let bytes = build_model(false);
    let gguf = Gguf::parse(&bytes).expect("parse tiny dLLM");
    let base = denoise_step(&gguf, &bytes, &[1, 2, 3, 4], &[0, 1], (0, 2)).expect("base");
    // Position 3 is outside the active block, so it must not influence the
    // block's queries.
    let changed = denoise_step(&gguf, &bytes, &[1, 2, 3, 0], &[0, 1], (0, 2)).expect("changed");
    assert_close(&base.logits, &changed.logits);
}

#[test]
fn block_mask_reads_the_clean_prefix() {
    let bytes = build_model(false);
    let gguf = Gguf::parse(&bytes).expect("parse tiny dLLM");
    let base = denoise_step(&gguf, &bytes, &[1, 2, 3, 4], &[2, 3], (2, 4)).expect("base");
    let changed = denoise_step(&gguf, &bytes, &[0, 2, 3, 4], &[2, 3], (2, 4)).expect("changed");
    assert_ne!(base.logits, changed.logits);
}

#[test]
fn tied_output_projection_is_supported() {
    let bytes = build_model(true);
    let gguf = Gguf::parse(&bytes).expect("parse tiny dLLM");
    let out = denoise_step(&gguf, &bytes, &[1, 2, 3, 4], &[2], (2, 4)).expect("tied output");
    assert_eq!(out.logits.len(), 5);
}

#[test]
fn top_k_matches_the_full_distribution() {
    let bytes = build_model(false);
    let gguf = Gguf::parse(&bytes).expect("parse tiny dLLM");
    let tokens = [1, 2, 3, 4];
    let positions = [2, 3];
    let full = denoise_step(&gguf, &bytes, &tokens, &positions, (2, 4)).expect("full");
    let top = denoise_topk(&gguf, &bytes, &tokens, &positions, (2, 4), 2).expect("top-k");

    assert_eq!(top.vocab_size, 5);
    assert_eq!(top.positions, vec![2, 3]);
    assert_eq!(top.token_ids.len(), 2 * 2);
    assert_eq!(top.probabilities.len(), 2 * 2);

    for (row_index, row) in full.logits.chunks_exact(5).enumerate() {
        let expected = argmax(row);
        let ids = &top.token_ids[row_index * 2..row_index * 2 + 2];
        assert_eq!(ids[0] as usize, expected);
        let probabilities = &top.probabilities[row_index * 2..row_index * 2 + 2];
        assert!(probabilities[0] >= probabilities[1]);
        assert!((probabilities[0] - full.confidence[row_index]).abs() < 1e-5);
    }
}

#[test]
fn rejects_invalid_inputs() {
    let bytes = build_model(false);
    let gguf = Gguf::parse(&bytes).expect("parse tiny dLLM");

    assert!(denoise_step(&gguf, &bytes, &[], &[0], (0, 1)).is_err());
    assert!(denoise_step(&gguf, &bytes, &[1, 2], &[], (0, 2)).is_err());
    assert!(denoise_step(&gguf, &bytes, &[1, 2], &[0], (2, 4)).is_err());
    assert!(denoise_step(&gguf, &bytes, &[1, 2], &[3], (0, 2)).is_err());
    assert!(denoise_step(&gguf, &bytes, &[1, 9], &[0], (0, 2)).is_err());
    assert!(denoise_topk(&gguf, &bytes, &[1, 2], &[0], (0, 2), 0).is_err());
    assert!(denoise_topk(&gguf, &bytes, &[1, 2], &[0], (0, 2), 99).is_ok());
}

#[test]
fn fixture_tensors_are_f32() {
    let bytes = build_model(false);
    let gguf = Gguf::parse(&bytes).expect("parse tiny dLLM");
    assert_eq!(gguf.tensor_count(), 14);
    assert_eq!(
        gguf.tensor("blk.0.attn_q.weight").unwrap().tensor_type,
        TensorType::F32
    );
}

#[test]
#[ignore = "requires FRIEDFLAXIA_DLLM_GGUF pointing at a qwen3-dllm GGUF"]
fn real_dllm_denoise_smoke() {
    let Ok(path) = std::env::var("FRIEDFLAXIA_DLLM_GGUF") else {
        eprintln!("skipping: FRIEDFLAXIA_DLLM_GGUF is not set");
        return;
    };
    let bytes = std::fs::read(path).expect("read dLLM GGUF");
    let gguf = Gguf::parse(&bytes).expect("parse dLLM GGUF");
    let config = friedflaxia_core::models::dllm::DllmConfig::from_gguf(&gguf).expect("config");
    let tokens: Vec<u32> = vec![0; 8];
    let positions: Vec<usize> = (0..8).collect();
    let started = std::time::Instant::now();
    let out = denoise_step(&gguf, &bytes, &tokens, &positions, (0, 8)).expect("denoise");
    println!(
        "dllm smoke vocab={} positions={} ms={}",
        config.vocab_size,
        out.positions.len(),
        started.elapsed().as_millis()
    );
}
