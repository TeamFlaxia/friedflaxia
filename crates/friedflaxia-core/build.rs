//! Generate shared preprocess constants from `model-manifest.json`.
//!
//! The same manifest is copied into flaxia-crowd by its sync script, so the
//! Rust model code and the browser preprocessing code cannot drift on patch
//! sizes, mel bins or media token ids.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=model-manifest.json");
    let manifest: serde_json::Value = serde_json::from_str(
        &fs::read_to_string("model-manifest.json").expect("model-manifest.json"),
    )
    .expect("valid model-manifest.json");
    let preprocess = &manifest["preprocess"];
    let special = &manifest["specialTokens"];

    let mut out = String::from("// Generated from model-manifest.json. Do not edit.\n");
    for key in [
        "patchSize",
        "pooling",
        "melBins",
        "frameLength",
        "hopLength",
        "fftLength",
        "sampleRate",
        "tileSize",
        "d1ImageTokens",
        "embeddingWidth",
    ] {
        let value = preprocess[key]
            .as_u64()
            .expect("numeric preprocess constant");
        out.push_str(&format!(
            "pub const {}: usize = {};\n",
            to_snake_case(key),
            value
        ));
    }
    for key in ["image", "audio", "video", "boi", "eoi", "boa", "eoa"] {
        let value = special[key].as_u64().expect("numeric special token");
        out.push_str(&format!(
            "pub const TOKEN_{}: u32 = {};\n",
            key.to_uppercase(),
            value
        ));
    }

    let out_path = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR")).join("model_manifest.rs");
    fs::write(out_path, out).expect("write generated constants");
}

fn to_snake_case(value: &str) -> String {
    let mut out = String::new();
    for (index, character) in value.chars().enumerate() {
        if character.is_ascii_uppercase() {
            if index > 0 {
                out.push('_');
            }
            out.push(character.to_ascii_lowercase());
        } else {
            out.push(character);
        }
    }
    out.to_uppercase()
}
