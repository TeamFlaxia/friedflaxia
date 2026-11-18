use friedflaxia_core::models::d1_audio::debug_positions;
use std::fs;
fn main() {
    let out = debug_positions();
    let mut bytes = Vec::new();
    for v in &out {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write("/tmp/opencode/ours-d1a-pos.bin", bytes).unwrap();
}
