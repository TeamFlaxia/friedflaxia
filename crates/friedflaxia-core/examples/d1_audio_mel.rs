use friedflaxia_core::models::d1_audio::debug_mel;
use std::fs;
fn main() {
    let samples: Vec<f32> = include_bytes!("../tests/fixtures/d1a_samples.bin")
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let out = debug_mel(&samples).unwrap();
    println!("first12 {:?}", &out[..12]);
    let mut bytes = Vec::new();
    for v in &out {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write("/tmp/opencode/ours-d1a-mel.bin", bytes).unwrap();
}
