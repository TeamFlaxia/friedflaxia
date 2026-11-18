use friedflaxia_core::gguf::{Gguf, ParseOptions};
use friedflaxia_core::models::plamo3::{PlamoSession, last_token_logits};
use std::env;
use std::fs;

const PROMPT_IDS: [u32; 21] = [
    1, 16, 48472, 19, 22125, 16, 2225, 17, 1943, 18, 1109, 19, 7272, 1675, 16, 4160, 17, 1943, 18,
    1385, 19,
];
const GOLDEN_TOP10: [usize; 10] = [
    74637, 7272, 50358, 53380, 54190, 54847, 51871, 21479, 54704, 58035,
];

#[test]
#[ignore = "requires FRIEDFLAXIA_PLAMO_GGUF pointing at plamo-3-translate-0.8b-Q4_0.gguf"]
fn plamo_last_token_logits_match_llama_cpp() {
    let Ok(path) = env::var("FRIEDFLAXIA_PLAMO_GGUF") else {
        eprintln!("skipping: FRIEDFLAXIA_PLAMO_GGUF is not set");
        return;
    };
    let file = fs::read(path).expect("read PLaMO GGUF");
    let gguf = Gguf::parse_with_options(
        &file,
        ParseOptions {
            skip_tokenizer: true,
        },
    )
    .expect("parse PLaMO GGUF");
    let logits = last_token_logits(&gguf, &file, &PROMPT_IDS).expect("PLaMo forward");

    let mut top: Vec<usize> = (0..logits.len()).collect();
    top.sort_by(|a, b| logits[*b].partial_cmp(&logits[*a]).unwrap());
    println!("top10={:?}", &top[..10]);
    assert_eq!(
        &top[..10],
        &GOLDEN_TOP10,
        "top-10 tokens differ from llama.cpp Q4_0"
    );
    // llama.cpp Q4_0 top values: 20.875, 19.442, 16.781, 14.991, 14.737,
    // 13.758, 13.388, 12.939, 12.531, 12.156.
    let golden_values = [
        20.875, 19.442, 16.781, 14.991, 14.737, 13.758, 13.388, 12.939, 12.531, 12.156,
    ];
    for (index, id) in GOLDEN_TOP10.iter().enumerate() {
        let difference = (logits[*id] - golden_values[index]).abs();
        assert!(
            difference < 0.5,
            "token {id} logit difference {difference} is too large"
        );
    }

    let mut session = PlamoSession::new(&gguf, &file).expect("session");
    let cached = session.prefill(&PROMPT_IDS).expect("cached prefill");
    let mut cached_top: Vec<usize> = (0..cached.len()).collect();
    cached_top.sort_by(|a, b| cached[*b].partial_cmp(&cached[*a]).unwrap());
    assert_eq!(&cached_top[..10], &GOLDEN_TOP10, "cached prefill top-10");

    let next = GOLDEN_TOP10[0] as u32;
    let cached_next = session.decode(next).expect("cached decode");
    let mut with_next = PROMPT_IDS.to_vec();
    with_next.push(next);
    let full_next = last_token_logits(&gguf, &file, &with_next).expect("full next logits");
    let max_difference = cached_next
        .iter()
        .zip(&full_next)
        .map(|(cached, full)| (cached - full).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_difference < 1e-3,
        "cached decode differs from full forward by {max_difference}"
    );
}
