use friedflaxia_core::gguf::{Gguf, ParseOptions};
use friedflaxia_core::models::nudenet::{LABELS, NUDENET_SIZE, NudeNet};
use std::env;
use std::fs;

const RAW: &[u8] = include_bytes!("fixtures/nudenet_raw.bin");
const NMS_RAW: &[u8] = include_bytes!("fixtures/nudenet_nms_raw.bin");
const NMS_SELECTED: &[u8] = include_bytes!("fixtures/nudenet_nms_selected.bin");

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect()
}

#[test]
fn nms_matches_onnx_non_max_suppression() {
    let raw = floats(NMS_RAW);
    let selected = floats(NMS_SELECTED);
    let detections = NudeNet::debug_nms(&raw, 100, 0.45, 0.25);
    println!("ours={} onnx={}", detections.len(), selected.len() / 22);
    for (index, detection) in detections.iter().take(5).enumerate() {
        println!(
            "ours {index} {:?} {:?}",
            detection.label, detection.box_xyxy
        );
    }
    for index in 0..3 {
        let row = &selected[index * 22..(index + 1) * 22];
        println!("onnx {index} box {:?}", &row[..4]);
    }
    assert_eq!(detections.len(), selected.len() / 22);

    for (index, detection) in detections.iter().enumerate() {
        let row = &selected[index * 22..(index + 1) * 22];
        let mut label = 0usize;
        let mut score = 0.0f32;
        for class in 0..18 {
            if row[4 + class] > score {
                score = row[4 + class];
                label = class;
            }
        }
        assert_eq!(detection.label, LABELS[label]);
        assert!((detection.score - score).abs() < 1e-5);
        let cx = row[0];
        let cy = row[1];
        let w = row[2];
        let h = row[3];
        let expected = [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5];
        for (actual, expected) in detection.box_xyxy.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-4);
        }
    }
}

fn input() -> Vec<f32> {
    (0..3 * NUDENET_SIZE * NUDENET_SIZE)
        .map(|index| ((index * 37) % 256) as f32 / 255.0)
        .collect()
}

#[test]
#[ignore = "requires FRIEDFLAXIA_NUDENET_GGUF pointing at nudenet-320n-f32.gguf"]
fn nudenet_raw_output_matches_onnx() {
    let Ok(path) = env::var("FRIEDFLAXIA_NUDENET_GGUF") else {
        eprintln!("skipping: FRIEDFLAXIA_NUDENET_GGUF is not set");
        return;
    };
    let file = fs::read(path).expect("read NudeNet GGUF");
    let gguf = Gguf::parse_with_options(
        &file,
        ParseOptions {
            skip_tokenizer: true,
        },
    )
    .expect("parse NudeNet GGUF");
    let net = NudeNet::load(&gguf, &file).expect("load NudeNet");
    let raw = net.raw_output(&input()).expect("raw output");

    let golden = floats(RAW);
    assert_eq!(golden.len(), 22 * 2100);
    assert_eq!(raw.len(), 22 * 2100);

    // ONNX emits [22, anchors]; our raw helper emits [anchor, 22].
    let mut max_abs = 0.0f32;
    let mut total = 0.0f64;
    for channel in 0..22 {
        for anchor in 0..2100 {
            let ours = raw[anchor * 22 + channel];
            let expected = golden[channel * 2100 + anchor];
            let difference = (ours - expected).abs();
            max_abs = max_abs.max(difference);
            total += difference as f64;
        }
    }
    let mean_abs = total / (22.0 * 2100.0);
    println!("NudeNet max_abs={max_abs} mean_abs={mean_abs}");
    assert!(
        max_abs < 0.05,
        "NudeNet raw output differs from ONNX (max {max_abs}, mean {mean_abs})"
    );
}
