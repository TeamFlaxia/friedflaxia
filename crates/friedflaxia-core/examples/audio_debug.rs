use friedflaxia_core::gguf::{Gguf, ParseOptions};
use friedflaxia_core::models::gemma_audio::{
    debug_audio_attention, debug_audio_attn_out, debug_audio_ffn1, debug_audio_layer0,
    debug_audio_qkv, debug_audio_raw_attention, debug_audio_relative_keys, debug_audio_subsample,
};
use std::env;
use std::fs;

fn main() {
    let path = env::args().nth(1).expect("usage: audio_debug <mmproj>");
    let file = fs::read(path).expect("read");
    let gguf = Gguf::parse_with_options(
        &file,
        ParseOptions {
            skip_tokenizer: true,
        },
    )
    .expect("parse");
    let mel: Vec<f32> = include_bytes!("../tests/fixtures/audio_mel.bin")
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    for (name, out) in [
        (
            "subsample",
            debug_audio_subsample(&gguf, &file, &mel, 280).expect("subsample"),
        ),
        (
            "ffn1",
            debug_audio_ffn1(&gguf, &file, &mel, 280).expect("ffn1"),
        ),
        (
            "attention",
            debug_audio_attention(&gguf, &file, &mel, 280).expect("attention"),
        ),
        (
            "q",
            debug_audio_qkv(&gguf, &file, &mel, 280).expect("qkv").0,
        ),
        (
            "k",
            debug_audio_qkv(&gguf, &file, &mel, 280).expect("qkv").1,
        ),
        (
            "v",
            debug_audio_qkv(&gguf, &file, &mel, 280).expect("qkv").2,
        ),
        (
            "relative",
            debug_audio_relative_keys(&gguf, &file).expect("relative"),
        ),
        (
            "attn_out",
            debug_audio_attn_out(&gguf, &file, &mel, 280).expect("attn_out"),
        ),
        (
            "raw_attention",
            debug_audio_raw_attention(&gguf, &file, &mel, 280).expect("raw_attention"),
        ),
        (
            "light_conv",
            debug_audio_layer0(&gguf, &file, &mel, 280, 4).expect("light_conv"),
        ),
        (
            "ffn2",
            debug_audio_layer0(&gguf, &file, &mel, 280, 5).expect("ffn2"),
        ),
        (
            "layer0",
            debug_audio_layer0(&gguf, &file, &mel, 280, 0).expect("layer0"),
        ),
    ] {
        println!("{name} first8 {:?}", &out[..8]);
        let mut bytes = Vec::new();
        for value in &out {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(format!("/tmp/opencode/ours-audio-{name}.bin"), bytes).expect("write");
    }
}
