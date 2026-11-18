use friedflaxia_core::gguf::{Gguf, ParseOptions};
use friedflaxia_core::models::d1_audio::debug_p_transposed;
use std::env;
use std::fs;
fn main() {
    let path = env::args().nth(1).unwrap();
    let file = fs::read(path).unwrap();
    let gguf = Gguf::parse_with_options(
        &file,
        ParseOptions {
            skip_tokenizer: true,
        },
    )
    .unwrap();
    let samples: Vec<f32> = include_bytes!("../tests/fixtures/d1a_samples.bin")
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let out = debug_p_transposed(&gguf, &file, &samples).unwrap();
    let mut bytes = Vec::new();
    for v in &out {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write("/tmp/opencode/ours-d1a-pt.bin", bytes).unwrap();
}
