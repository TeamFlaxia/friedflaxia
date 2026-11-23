use friedflaxia_core::gguf::{Gguf, ParseOptions};
use friedflaxia_core::models::embeddinggemma::embed_text;
use std::env;
use std::fs;

const TOKEN_IDS: [u32; 9] = [2, 3689, 563, 506, 5279, 529, 7001, 236881, 1];
const GOLDEN: &[u8] = include_bytes!("fixtures/embedgemma_golden.bin");

#[test]
#[ignore = "requires FRIEDFLAXIA_EMBED_GEMMA_GGUF pointing at embeddinggemma-2-Q8_0.gguf"]
fn embeddinggemma_matches_onnx_golden() {
    let path = env::var("FRIEDFLAXIA_EMBED_GEMMA_GGUF").expect("set FRIEDFLAXIA_EMBED_GEMMA_GGUF");
    let file = fs::read(path).expect("read EmbeddingGemma GGUF");
    let gguf = Gguf::parse_with_options(
        &file,
        ParseOptions {
            skip_tokenizer: true,
        },
    )
    .expect("parse EmbeddingGemma GGUF");

    let embedding = embed_text(&gguf, &file, &TOKEN_IDS).expect("EmbeddingGemma forward");
    assert_eq!(embedding.len(), 768);
    assert_eq!(GOLDEN.len(), 768 * 4);

    let golden: Vec<f32> = GOLDEN
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect();

    let norm = embedding
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    assert!((norm - 1.0).abs() < 1e-4, "embedding norm is {norm}");

    let dot: f32 = embedding.iter().zip(&golden).map(|(a, b)| a * b).sum();
    let golden_norm = golden.iter().map(|value| value * value).sum::<f32>().sqrt();
    let cosine = dot / (norm * golden_norm);
    let max_abs = embedding
        .iter()
        .zip(&golden)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("cosine={cosine} max_abs={max_abs} norm={norm}");
    assert!(
        cosine > 0.99,
        "cosine similarity {cosine} is too low (max abs diff {max_abs})"
    );
    assert!(
        max_abs < 0.05,
        "max absolute difference {max_abs} is too high (cosine {cosine})"
    );
}

/// Scaling check for the batched forward pass. The old per-token path streamed
/// the whole Q8_0 matrix once per token, so 256 tokens cost ~256 passes; with
/// the batched kernels the weight matrix is streamed once per layer tile.
#[test]
#[ignore = "requires FRIEDFLAXIA_EMBED_GEMMA_GGUF; prints timings"]
fn embeddinggemma_batched_long_sequence_timing() {
    let path = env::var("FRIEDFLAXIA_EMBED_GEMMA_GGUF").expect("set FRIEDFLAXIA_EMBED_GEMMA_GGUF");
    let file = fs::read(path).expect("read EmbeddingGemma GGUF");
    let gguf = Gguf::parse_with_options(
        &file,
        ParseOptions {
            skip_tokenizer: true,
        },
    )
    .expect("parse EmbeddingGemma GGUF");

    let short = TOKEN_IDS;
    let started = std::time::Instant::now();
    embed_text(&gguf, &file, &short).expect("short embed");
    println!(
        "tokens={} time_ms={}",
        short.len(),
        started.elapsed().as_millis()
    );

    for tokens in [64usize, 256] {
        let ids: Vec<u32> = (0..tokens)
            .map(|index| TOKEN_IDS[1 + index % (TOKEN_IDS.len() - 2)])
            .collect();
        let started = std::time::Instant::now();
        embed_text(&gguf, &file, &ids).expect("long embed");
        println!(
            "tokens={} time_ms={} ms_per_token={:.2}",
            tokens,
            started.elapsed().as_millis(),
            started.elapsed().as_secs_f64() * 1000.0 / tokens as f64
        );
    }
}
