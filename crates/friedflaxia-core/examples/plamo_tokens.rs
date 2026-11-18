use friedflaxia_core::gguf::Gguf;
use friedflaxia_core::gguf::tokenizer::TokenizerSpec;
use friedflaxia_core::tokenizer::UnigramTokenizer;
use std::env;
use std::fs;

fn main() {
    let path = env::args()
        .nth(1)
        .expect("usage: plamo_tokens <gguf> <id>...");
    let ids: Vec<u32> = env::args()
        .skip(2)
        .map(|value| value.parse().expect("id"))
        .collect();
    let file = fs::read(path).expect("read GGUF");
    let gguf = Gguf::parse(&file).expect("parse GGUF");
    let spec = TokenizerSpec::from_gguf(&gguf).expect("tokenizer");
    let tokenizer = UnigramTokenizer::from_spec(&spec).expect("Unigram");
    for id in ids {
        let text = tokenizer.decode(&[id]).unwrap_or_else(|_| "<error>".into());
        println!("{id} -> {text:?}");
    }
}
