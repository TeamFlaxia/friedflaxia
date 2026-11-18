use friedflaxia_core::gguf::{Gguf, ParseOptions};
use friedflaxia_core::models::embeddinggemma::embed_text_with_features;
use std::env;
use std::fs;

const IDS: &[u8] = include_bytes!("fixtures/mm_ids.bin");
const IMAGE: &[u8] = include_bytes!("fixtures/mm_image.bin");
const AUDIO: &[u8] = include_bytes!("fixtures/mm_audio.bin");
const EXPECTED: &[u8] = include_bytes!("fixtures/mm_emb.bin");

fn u32s(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect()
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect()
}

#[test]
#[ignore = "requires FRIEDFLAXIA_EMBED_GGUF pointing at embeddinggemma-2-Q8_0.gguf"]
fn multimodal_embedding_matches_onnx() {
    let Ok(path) = env::var("FRIEDFLAXIA_EMBED_GGUF") else {
        eprintln!("skipping: FRIEDFLAXIA_EMBED_GGUF is not set");
        return;
    };
    let file = fs::read(path).expect("read embedding GGUF");
    let gguf = Gguf::parse_with_options(
        &file,
        ParseOptions {
            skip_tokenizer: true,
        },
    )
    .expect("parse embedding GGUF");
    let ids = u32s(IDS);
    let image = floats(IMAGE);
    let audio = floats(AUDIO);
    let actual = embed_text_with_features(&gguf, &file, &ids, &image, &[], &audio)
        .expect("embed multimodal");
    let expected = floats(EXPECTED);
    assert_eq!(actual.len(), expected.len());
    let dot: f32 = actual.iter().zip(&expected).map(|(a, b)| a * b).sum();
    let norm_a: f32 = actual.iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_b: f32 = expected.iter().map(|v| v * v).sum::<f32>().sqrt();
    let cosine = dot / (norm_a * norm_b);
    let max_abs = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("multimodal cosine={cosine} max_abs={max_abs}");
    assert!(cosine > 0.99, "cosine too low: {cosine}");
    assert!(
        max_abs < 0.05,
        "embedding differs from ONNX (max {max_abs})"
    );
}

const PATCHES: &[u8] = include_bytes!("fixtures/vision_patches.bin");
const MEL: &[u8] = include_bytes!("fixtures/audio_mel.bin");

#[test]
#[ignore = "requires FRIEDFLAXIA_EMBED_GGUF and FRIEDFLAXIA_MMPROJ"]
fn multimodal_end_to_end_matches_onnx() {
    let (Ok(embed_path), Ok(mmproj_path)) = (
        env::var("FRIEDFLAXIA_EMBED_GGUF"),
        env::var("FRIEDFLAXIA_MMPROJ"),
    ) else {
        eprintln!("skipping: FRIEDFLAXIA_EMBED_GGUF or FRIEDFLAXIA_MMPROJ is not set");
        return;
    };
    let embed_file = fs::read(embed_path).expect("read embedding GGUF");
    let embed_gguf = Gguf::parse_with_options(
        &embed_file,
        ParseOptions {
            skip_tokenizer: true,
        },
    )
    .expect("parse embedding GGUF");
    let mmproj_file = fs::read(mmproj_path).expect("read mmproj");
    let mmproj = Gguf::parse_with_options(
        &mmproj_file,
        ParseOptions {
            skip_tokenizer: true,
        },
    )
    .expect("parse mmproj");

    let image = friedflaxia_core::models::gemma_vision::encode_image(
        &mmproj,
        &mmproj_file,
        &floats(PATCHES),
        6,
        9,
    )
    .expect("encode image");
    let audio = friedflaxia_core::models::gemma_audio::encode_audio(
        &mmproj,
        &mmproj_file,
        &floats(MEL),
        280,
    )
    .expect("encode audio");
    let image = &image[..2 * 512];
    let audio = &audio[..2 * 512];
    let actual = embed_text_with_features(&embed_gguf, &embed_file, &u32s(IDS), image, &[], audio)
        .expect("embed");
    let expected = floats(EXPECTED);
    let dot: f32 = actual.iter().zip(&expected).map(|(a, b)| a * b).sum();
    let norm_a: f32 = actual.iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_b: f32 = expected.iter().map(|v| v * v).sum::<f32>().sqrt();
    let cosine = dot / (norm_a * norm_b);
    println!("end-to-end cosine={cosine}");
    assert!(cosine > 0.99, "end-to-end cosine too low: {cosine}");
}
