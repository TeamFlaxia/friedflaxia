use friedflaxia_core::gguf::Gguf;
use friedflaxia_core::gguf::tokenizer::TokenizerSpec;
use friedflaxia_core::tokenizer::UnigramTokenizer;
use std::env;
use std::fs;

const HELLO: &str = "Hello world";
const HELLO_IDS: [u32; 3] = [1, 7272, 1675];
const PROMPT: &str = "<|plamo:tag|>dataset<|plamo:msg|>translation<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>Hello world<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>";
const PROMPT_IDS: [u32; 21] = [
    1, 16, 48472, 19, 22125, 16, 2225, 17, 1943, 18, 1109, 19, 7272, 1675, 16, 4160, 17, 1943, 18,
    1385, 19,
];

#[test]
#[ignore = "requires FRIEDFLAXIA_PLAMO_GGUF pointing at a PLaMO 3 GGUF (header is enough)"]
fn plamo_tokenizer_matches_huggingface() {
    let Ok(path) = env::var("FRIEDFLAXIA_PLAMO_GGUF") else {
        eprintln!("skipping: FRIEDFLAXIA_PLAMO_GGUF is not set");
        return;
    };
    let file = fs::read(path).expect("read PLaMO GGUF");
    let gguf = Gguf::parse(&file).expect("parse PLaMO GGUF");
    let spec = TokenizerSpec::from_gguf(&gguf).expect("tokenizer metadata");
    let tokenizer = UnigramTokenizer::from_spec(&spec).expect("Unigram tokenizer");

    assert_eq!(tokenizer.encode(HELLO, true).unwrap(), HELLO_IDS);
    assert_eq!(tokenizer.encode(PROMPT, true).unwrap(), PROMPT_IDS);
    assert_eq!(tokenizer.decode(&HELLO_IDS).unwrap(), HELLO);
}
