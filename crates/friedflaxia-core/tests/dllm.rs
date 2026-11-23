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
    name: String,
    dims: Vec<u64>,
    values: Vec<f32>,
}

fn matrix(name: &str, cols: usize, rows: usize, seed: u64) -> TensorSpec {
    TensorSpec {
        name: name.to_string(),
        dims: vec![cols as u64, rows as u64],
        values: pseudo_weights(cols * rows, seed),
    }
}

fn vector(name: &str, len: usize, uniform: bool) -> TensorSpec {
    TensorSpec {
        name: name.to_string(),
        dims: vec![len as u64],
        values: if uniform {
            vec![1.0; len]
        } else {
            (0..len).map(|index| 0.55 + 0.07 * index as f32).collect()
        },
    }
}

#[derive(Clone, Copy)]
struct ModelDims {
    n_layer: usize,
    n_embd: usize,
    n_ff: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
}

impl ModelDims {
    /// One layer, 8-d hidden, 2 query heads, 1 KV head.
    const SIMPLE: Self = Self {
        n_layer: 1,
        n_embd: 8,
        n_ff: 16,
        n_head: 2,
        n_head_kv: 1,
        head_dim: 4,
    };
    /// Two layers, 16-d hidden, 4 query heads / 2 KV heads, non-uniform norms.
    /// This shape catches GQA mapping, multi-layer residuals and norm/RoPE order.
    const HARDENED: Self = Self {
        n_layer: 2,
        n_embd: 16,
        n_ff: 32,
        n_head: 4,
        n_head_kv: 2,
        head_dim: 4,
    };

    fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }

    fn kv_dim(&self) -> usize {
        self.n_head_kv * self.head_dim
    }
}

fn model_specs(dims: ModelDims, tie_output: bool, uniform_norms: bool) -> Vec<TensorSpec> {
    let mut specs = vec![matrix("token_embd.weight", dims.n_embd, 5, 1)];
    for layer in 0..dims.n_layer {
        let p = format!("blk.{layer}");
        let seed = 100 * (layer as u64 + 1);
        specs.push(vector(
            &format!("{p}.attn_norm.weight"),
            dims.n_embd,
            uniform_norms,
        ));
        specs.push(matrix(
            &format!("{p}.attn_q.weight"),
            dims.n_embd,
            dims.q_dim(),
            seed + 1,
        ));
        specs.push(matrix(
            &format!("{p}.attn_k.weight"),
            dims.n_embd,
            dims.kv_dim(),
            seed + 2,
        ));
        specs.push(matrix(
            &format!("{p}.attn_v.weight"),
            dims.n_embd,
            dims.kv_dim(),
            seed + 3,
        ));
        specs.push(matrix(
            &format!("{p}.attn_output.weight"),
            dims.q_dim(),
            dims.n_embd,
            seed + 4,
        ));
        specs.push(vector(
            &format!("{p}.attn_q_norm.weight"),
            dims.head_dim,
            uniform_norms,
        ));
        specs.push(vector(
            &format!("{p}.attn_k_norm.weight"),
            dims.head_dim,
            uniform_norms,
        ));
        specs.push(vector(
            &format!("{p}.ffn_norm.weight"),
            dims.n_embd,
            uniform_norms,
        ));
        specs.push(matrix(
            &format!("{p}.ffn_gate.weight"),
            dims.n_embd,
            dims.n_ff,
            seed + 5,
        ));
        specs.push(matrix(
            &format!("{p}.ffn_up.weight"),
            dims.n_embd,
            dims.n_ff,
            seed + 6,
        ));
        specs.push(matrix(
            &format!("{p}.ffn_down.weight"),
            dims.n_ff,
            dims.n_embd,
            seed + 7,
        ));
    }
    specs.push(vector("output_norm.weight", dims.n_embd, uniform_norms));
    if !tie_output {
        specs.push(matrix("output.weight", dims.n_embd, 5, 900));
    }
    specs
}

fn build_gguf(specs: Vec<TensorSpec>, dims: ModelDims) -> Vec<u8> {
    let mut metadata = Vec::new();
    put_kv_u32(&mut metadata, "general.alignment", ALIGNMENT as u32);
    put_kv_string(&mut metadata, "general.architecture", "qwen3-dllm");
    put_kv_u32(&mut metadata, "qwen3-dllm.block_count", dims.n_layer as u32);
    put_kv_u32(
        &mut metadata,
        "qwen3-dllm.embedding_length",
        dims.n_embd as u32,
    );
    put_kv_u32(
        &mut metadata,
        "qwen3-dllm.feed_forward_length",
        dims.n_ff as u32,
    );
    put_kv_u32(
        &mut metadata,
        "qwen3-dllm.attention.head_count",
        dims.n_head as u32,
    );
    put_kv_u32(
        &mut metadata,
        "qwen3-dllm.attention.head_count_kv",
        dims.n_head_kv as u32,
    );
    put_kv_u32(
        &mut metadata,
        "qwen3-dllm.attention.key_length",
        dims.head_dim as u32,
    );
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
        put_string(&mut out, &spec.name);
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

fn build_model(tie_output: bool) -> Vec<u8> {
    build_gguf(
        model_specs(ModelDims::SIMPLE, tie_output, true),
        ModelDims::SIMPLE,
    )
}

fn build_hardened_model() -> Vec<u8> {
    build_gguf(
        model_specs(ModelDims::HARDENED, false, false),
        ModelDims::HARDENED,
    )
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
fn block_mask_is_bidirectional_inside_the_block() {
    let bytes = build_model(false);
    let gguf = Gguf::parse(&bytes).expect("parse tiny dLLM");
    let base = denoise_step(&gguf, &bytes, &[1, 2, 3, 4], &[0], (0, 2)).expect("base");
    // Query 0 is inside the block, so token 1 (a later block position) must be
    // visible. A causal-only mask would leave these logits unchanged.
    let changed = denoise_step(&gguf, &bytes, &[1, 0, 3, 4], &[0], (0, 2)).expect("changed");
    assert_ne!(base.logits, changed.logits);
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
fn hardened_model_matches_golden_logits() {
    let bytes = build_hardened_model();
    let gguf = Gguf::parse(&bytes).expect("parse hardened dLLM");
    let tokens = [1, 2, 3, 4, 0, 1, 2, 3];
    let out = denoise_step(&gguf, &bytes, &tokens, &[0, 4, 7], (0, 8)).expect("hardened");
    assert_eq!(out.positions, vec![0, 4, 7]);
    assert_eq!(out.logits.len(), 3 * 5);
    // Regression lock for the block mask, q/k norm -> RoPE order, residual
    // addition, GQA mapping (4 query / 2 KV heads) and two-layer stacking.
    const GOLDEN: [f32; 15] = [
        -0.5107393,
        -0.2641858,
        -1.0002869,
        0.42415598,
        1.2973803,
        -0.74567235,
        -0.18163148,
        -0.8792292,
        0.6298983,
        1.3569688,
        -0.052192926,
        -0.10673007,
        -0.9995232,
        0.49606305,
        1.153955,
    ];
    for (index, (left, right)) in out.logits.iter().zip(&GOLDEN).enumerate() {
        assert!(
            (left - right).abs() < 1e-4,
            "golden value {index} differs: {left} != {right}"
        );
    }
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
    assert_eq!(top.top_k, 2);
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
fn future_tokens_are_not_read() {
    let bytes = build_model(false);
    let gguf = Gguf::parse(&bytes).expect("parse tiny dLLM");
    let short = denoise_step(&gguf, &bytes, &[1, 2], &[0, 1], (0, 2)).expect("short");
    let long = denoise_step(&gguf, &bytes, &[1, 2, 3, 4], &[0, 1], (0, 2)).expect("long");
    assert_close(&short.logits, &long.logits);
}

#[test]
fn scoring_a_prefix_position_is_allowed() {
    let bytes = build_model(false);
    let gguf = Gguf::parse(&bytes).expect("parse tiny dLLM");
    let out = denoise_step(&gguf, &bytes, &[1, 2, 3, 4], &[0], (2, 4)).expect("prefix score");
    assert_eq!(out.logits.len(), 5);
}

#[test]
fn rejects_invalid_inputs() {
    let bytes = build_model(false);
    let gguf = Gguf::parse(&bytes).expect("parse tiny dLLM");

    assert!(denoise_step(&gguf, &bytes, &[], &[0], (0, 1)).is_err());
    assert!(denoise_step(&gguf, &bytes, &[1, 2], &[], (0, 2)).is_err());
    assert!(denoise_step(&gguf, &bytes, &[1, 2], &[0], (2, 4)).is_err());
    assert!(denoise_step(&gguf, &bytes, &[1, 2], &[3], (0, 2)).is_err());
    assert!(denoise_step(&gguf, &bytes, &[1, 2], &[0, 0], (0, 2)).is_err());
    assert!(denoise_step(&gguf, &bytes, &[1, 9], &[0], (0, 2)).is_err());
    assert!(denoise_topk(&gguf, &bytes, &[1, 2], &[0], (0, 2), 0).is_err());
    let clamped = denoise_topk(&gguf, &bytes, &[1, 2], &[0], (0, 2), 99).expect("clamped");
    assert_eq!(clamped.top_k, 5);
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
    assert_eq!(out.vocab_size, config.vocab_size);
    assert_eq!(out.positions.len(), positions.len());
    assert_eq!(out.logits.len(), positions.len() * config.vocab_size);
    assert!(out.logits.iter().all(|value| value.is_finite()));
    assert!(
        out.confidence
            .iter()
            .all(|value| *value >= 0.0 && *value <= 1.0)
    );
    println!(
        "dllm smoke vocab={} positions={} ms={}",
        config.vocab_size,
        out.positions.len(),
        started.elapsed().as_millis()
    );
}
