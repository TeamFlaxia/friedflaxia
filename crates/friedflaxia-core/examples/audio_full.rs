use friedflaxia_core::gguf::{Gguf, ParseOptions};
use friedflaxia_core::models::gemma_audio::encode_audio;
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
    let mel: Vec<f32> = include_bytes!(
        "/home/waipu/flaxian/friedflaxia/crates/friedflaxia-core/tests/fixtures/audio_mel.bin"
    )
    .chunks_exact(4)
    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    .collect();
    let out = encode_audio(&gguf, &file, &mel, 280).unwrap();
    let mut bytes = Vec::new();
    for v in &out {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write("/tmp/opencode/ours-audio-full.bin", bytes).unwrap();
}
