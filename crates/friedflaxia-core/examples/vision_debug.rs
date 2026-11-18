use friedflaxia_core::gguf::{Gguf, ParseOptions};
use friedflaxia_core::models::gemma_vision::debug_image_layers;
use std::env;
use std::fs;

fn main() {
    let path = env::args().nth(1).expect("usage: vision_debug <mmproj>");
    let file = fs::read(path).expect("read");
    let gguf = Gguf::parse_with_options(
        &file,
        ParseOptions {
            skip_tokenizer: true,
        },
    )
    .expect("parse");
    let patches: Vec<f32> = include_bytes!("../tests/fixtures/vision_patches.bin")
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    for layers in [1usize, 2, 16] {
        let out = debug_image_layers(&gguf, &file, &patches, 6, 9, layers).expect("layers");
        println!("layers{layers} first8 {:?}", &out[..8]);
        let mut bytes = Vec::new();
        for value in &out {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(format!("/tmp/opencode/ours-vision-l{layers}.bin"), bytes).expect("write");
    }
}
