use friedflaxia_core::gguf::tokenizer::{TokenType, TokenizerSpec};
use friedflaxia_core::gguf::{Gguf, ParseOptions};
use friedflaxia_core::models::plamo3::next_token;
use friedflaxia_core::tokenizer::UnigramTokenizer;
use std::env;
use std::fs;

fn main() {
    let path = env::args()
        .nth(1)
        .expect("usage: plamo_generate <gguf> [tokens]");
    let max_tokens: usize = env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(8);
    let prompt = "<|plamo:tag|>dataset<|plamo:msg|>translation<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>Hello world<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>";
    let file = fs::read(&path).expect("read GGUF");
    let tokenizer_gguf = Gguf::parse(&file).expect("parse tokenizer GGUF");
    let spec = TokenizerSpec::from_gguf(&tokenizer_gguf).expect("tokenizer metadata");
    let tokenizer = UnigramTokenizer::from_spec(&spec).expect("Unigram tokenizer");
    let mut ids = tokenizer.encode(prompt, true).expect("encode prompt");
    let gguf = Gguf::parse_with_options(
        &file,
        ParseOptions {
            skip_tokenizer: true,
        },
    )
    .expect("parse model GGUF");
    println!("prompt ids {}", ids.len());
    for _ in 0..max_tokens {
        let next = next_token(&gguf, &file, &ids).expect("next token");
        let control = spec
            .token_types
            .get(next as usize)
            .is_some_and(|token_type| *token_type == TokenType::Control);
        if next == spec.eos_token_id || control {
            break;
        }
        ids.push(next);
    }
    println!("{}", tokenizer.decode(&ids).unwrap());
    println!("all ids {ids:?}");
}
