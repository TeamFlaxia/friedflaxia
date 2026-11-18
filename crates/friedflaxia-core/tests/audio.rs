use friedflaxia_core::gguf::{Gguf, ParseOptions};
use friedflaxia_core::models::gemma_audio::encode_audio;
use std::env;
use std::fs;

const MEL: &[u8] = include_bytes!("fixtures/audio_mel.bin");
const EXPECTED: &[u8] = include_bytes!("fixtures/audio_out.bin");

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect()
}

#[test]
#[ignore = "requires FRIEDFLAXIA_MMPROJ pointing at mmproj-embeddinggemma-2-Q8_0.gguf"]
fn audio_encoder_matches_onnx() {
    let Ok(path) = env::var("FRIEDFLAXIA_MMPROJ") else {
        eprintln!("skipping: FRIEDFLAXIA_MMPROJ is not set");
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
    let mel = floats(MEL);
    let actual = encode_audio(&gguf, &file, &mel, 280).expect("encode audio");
    let expected = floats(EXPECTED);
    assert_eq!(actual.len(), expected.len());
    let mut max_abs = 0.0f32;
    let mut total = 0.0f64;
    for (actual, expected) in actual.iter().zip(&expected) {
        let difference = (actual - expected).abs();
        max_abs = max_abs.max(difference);
        total += difference as f64;
    }
    println!(
        "audio max_abs={max_abs} mean_abs={}",
        total / expected.len() as f64
    );
    let mean_abs = total / expected.len() as f64;
    assert!(
        max_abs < 0.75 && mean_abs < 0.1,
        "audio differs from ONNX (max {max_abs}, mean {mean_abs})"
    );
}
