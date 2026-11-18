use friedflaxia_core::gguf::{Gguf, ParseOptions};
use friedflaxia_core::models::d1_vision::{debug_image_embedding, debug_patch_embedding};
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
    let patch = debug_patch_embedding(&gguf, &file, &patches, 32, 32).unwrap();
    println!("patch first8 {:?}", &patch[..8]);
    let mut bytes = Vec::new();
    for v in &patch {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write("/tmp/opencode/ours-d1v-patch.bin", bytes).unwrap();
    let out = debug_image_embedding(&gguf, &file, &patches, 32, 32).unwrap();
    println!("first8 {:?}", &out[..8]);
    let mut bytes = Vec::new();
    for v in &out {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write("/tmp/opencode/ours-d1v-embedding.bin", bytes).unwrap();
}
