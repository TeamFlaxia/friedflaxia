use friedflaxia_core::gguf::{Gguf, ParseOptions};
use friedflaxia_core::models::d1_audio::encode_audio;
use std::env;
use std::fs;

const SAMPLES: &[u8] = include_bytes!("fixtures/d1a_samples.bin");
const EXPECTED: &[u8] = include_bytes!("fixtures/d1a_out.bin");

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect()
}

#[test]
#[ignore = "requires FRIEDFLAXIA_D1_MMPROJ pointing at mmproj-d1-omni-600M-Q8_0.gguf"]
fn d1_audio_matches_onnx() {
    let Ok(path) = env::var("FRIEDFLAXIA_D1_MMPROJ") else {
        eprintln!("skipping: FRIEDFLAXIA_D1_MMPROJ is not set");
        return;
    };
    let file = fs::read(path).expect("read mmproj");
    let gguf = Gguf::parse_with_options(
        &file,
        ParseOptions {
            skip_tokenizer: true,
        },
    )
    .expect("parse mmproj");
    let actual = encode_audio(&gguf, &file, &floats(SAMPLES)).expect("encode audio");
    let expected = floats(EXPECTED);
    assert_eq!(actual.len(), expected.len());
    let dot: f32 = actual.iter().zip(&expected).map(|(a, b)| a * b).sum();
    let norm_a: f32 = actual.iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_b: f32 = expected.iter().map(|v| v * v).sum::<f32>().sqrt();
    let cosine = dot / (norm_a * norm_b);
    let mean_abs: f64 = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs() as f64)
        .sum::<f64>()
        / expected.len() as f64;
    println!("d1 audio cosine={cosine} mean_abs={mean_abs}");
    assert!(cosine > 0.99, "cosine too low: {cosine}");
    assert!(mean_abs < 0.2, "audio differs from ONNX (mean {mean_abs})");
}
