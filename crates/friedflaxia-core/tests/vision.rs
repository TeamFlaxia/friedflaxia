use friedflaxia_core::gguf::{Gguf, ParseOptions};
use friedflaxia_core::models::gemma_vision::encode_image;
use std::env;
use std::fs;

const PATCHES: &[u8] = include_bytes!("fixtures/vision_patches.bin");
const EXPECTED: &[u8] = include_bytes!("fixtures/vision_out.bin");

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect()
}

#[test]
#[ignore = "requires FRIEDFLAXIA_MMPROJ pointing at mmproj-embeddinggemma-2-Q8_0.gguf"]
fn vision_encoder_matches_onnx() {
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
    let patches = floats(PATCHES);
    let actual = encode_image(&gguf, &file, &patches, 6, 9).expect("encode image");
    let expected = floats(EXPECTED);
    assert_eq!(actual.len(), expected.len());
    let mut max_abs = 0.0f32;
    for (actual, expected) in actual.iter().zip(&expected) {
        max_abs = max_abs.max((actual - expected).abs());
    }
    println!("vision max_abs={max_abs}");
    assert!(max_abs < 0.2, "vision differs from ONNX (max {max_abs})");
}
