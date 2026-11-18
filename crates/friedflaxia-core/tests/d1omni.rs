use friedflaxia_core::gguf::{Gguf, ParseOptions};
use friedflaxia_core::models::d1omni::decision_logits;
use std::env;
use std::fs;

const IDS: [u32; 3] = [1, 36309, 2031];
const MARKERS: [usize; 3] = [0, 1, 2];
const GOLDEN: [&[f32]; 3] = [
    &[1.148196, 0.198898, -0.225154],
    &[1.220866, 0.268705, -0.219265],
    &[1.007995, 0.208699, -0.399619],
];

#[test]
#[ignore = "requires FRIEDFLAXIA_D1_GGUF pointing at d1-omni-600M-Q8_0.gguf"]
fn d1_decision_logits_match_onnx() {
    let Ok(path) = env::var("FRIEDFLAXIA_D1_GGUF") else {
        eprintln!("skipping: FRIEDFLAXIA_D1_GGUF is not set");
        return;
    };
    let file = fs::read(path).expect("read d1 GGUF");
    let gguf = Gguf::parse_with_options(
        &file,
        ParseOptions {
            skip_tokenizer: true,
        },
    )
    .expect("parse d1 GGUF");

    for (qtype, expected) in GOLDEN.iter().enumerate() {
        let logits =
            decision_logits(&gguf, &file, &IDS, &MARKERS, qtype).expect("decision forward");
        let max_abs = logits
            .iter()
            .zip(*expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        println!("qtype={qtype} logits={logits:?} max_abs={max_abs}");
        assert!(
            max_abs < 0.05,
            "qtype {qtype} max abs difference {max_abs} (got {logits:?})"
        );
    }
}
