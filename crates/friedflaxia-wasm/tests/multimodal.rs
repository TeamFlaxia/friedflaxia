use friedflaxia_wasm::MultimodalModel;
use std::env;
use std::fs;

const PATCHES: &[u8] = include_bytes!("../../friedflaxia-core/tests/fixtures/vision_patches.bin");
const MEL: &[u8] = include_bytes!("../../friedflaxia-core/tests/fixtures/audio_mel.bin");
const IDS: &[u8] = include_bytes!("../../friedflaxia-core/tests/fixtures/mm_ids.bin");
const EXPECTED: &[u8] = include_bytes!("../../friedflaxia-core/tests/fixtures/mm_emb.bin");

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect()
}

fn u32s(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect()
}

#[test]
#[ignore = "requires FRIEDFLAXIA_EMBED_GGUF and FRIEDFLAXIA_MMPROJ"]
fn multimodal_model_end_to_end() {
    let (Ok(embed_path), Ok(mmproj_path)) = (
        env::var("FRIEDFLAXIA_EMBED_GGUF"),
        env::var("FRIEDFLAXIA_MMPROJ"),
    ) else {
        eprintln!("skipping: FRIEDFLAXIA_EMBED_GGUF or FRIEDFLAXIA_MMPROJ is not set");
        return;
    };
    let model = MultimodalModel::new(
        fs::read(embed_path).expect("read text GGUF"),
        fs::read(mmproj_path).expect("read mmproj"),
    )
    .expect("load multimodal model");

    let image = model.encode_image(&floats(PATCHES), 6, 9).expect("image");
    let audio = model.encode_audio(&floats(MEL), 280).expect("audio");
    let actual = model
        .embed(&u32s(IDS), &image[..1024], &[], &audio[..1024])
        .expect("embed");
    let expected = floats(EXPECTED);
    let dot: f32 = actual.iter().zip(&expected).map(|(a, b)| a * b).sum();
    let norm_a: f32 = actual.iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_b: f32 = expected.iter().map(|v| v * v).sum::<f32>().sqrt();
    let cosine = dot / (norm_a * norm_b);
    println!("wasm multimodal cosine={cosine}");
    assert!(cosine > 0.99);
}
