use friedflaxia_core::gguf::{Gguf, ParseOptions};
use friedflaxia_core::models::d1_vision::debug_image_layers;
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
    let patches: Vec<f32> = include_bytes!("../tests/fixtures/d1v_patches.bin")
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    for layers in [1usize, 2, 12] {
        let out = debug_image_layers(&gguf, &file, &patches, 32, 32, layers).unwrap();
        println!("layers{layers} first8 {:?}", &out[..8]);
        let mut bytes = Vec::new();
        for v in &out {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        fs::write(format!("/tmp/opencode/ours-d1v-l{layers}.bin"), bytes).unwrap();
    }
}
