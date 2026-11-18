use friedflaxia_core::gguf::Gguf;
use friedflaxia_core::gguf::tokenizer::TokenizerSpec;
use friedflaxia_core::tokenizer::BpeTokenizer;
use std::env;
use std::fs;

const TEXT: &str = "What is the capital of France?";
const GOLDEN_IDS: [u32; 9] = [2, 3689, 563, 506, 5279, 529, 7001, 236881, 1];

#[test]
#[ignore = "requires FRIEDFLAXIA_EMBED_GEMMA_GGUF pointing at embeddinggemma-2-Q8_0.gguf"]
fn embeddinggemma_tokenizer_matches_huggingface() {
    let Ok(path) = env::var("FRIEDFLAXIA_EMBED_GEMMA_GGUF") else {
        eprintln!("skipping: FRIEDFLAXIA_EMBED_GEMMA_GGUF is not set");
        return;
    };
    let file = fs::read(path).expect("read EmbeddingGemma GGUF");
    let gguf = Gguf::parse(&file).expect("parse EmbeddingGemma GGUF");
    let spec = TokenizerSpec::from_gguf(&gguf).expect("tokenizer metadata");
    let tokenizer = BpeTokenizer::from_spec(&spec).expect("BPE tokenizer");

    let ids = tokenizer.encode(TEXT, true).expect("encode");
    assert_eq!(ids, GOLDEN_IDS);
    assert_eq!(tokenizer.decode(&ids).expect("decode"), TEXT);
}

#[test]
#[ignore = "requires FRIEDFLAXIA_D1_GGUF pointing at d1-omni-600M-Q8_0.gguf"]
fn d1_tokenizer_matches_huggingface() {
    let Ok(path) = env::var("FRIEDFLAXIA_D1_GGUF") else {
        eprintln!("skipping: FRIEDFLAXIA_D1_GGUF is not set");
        return;
    };
    let file = fs::read(path).expect("read d1 GGUF");
    let gguf = Gguf::parse(&file).expect("parse d1 GGUF");
    let spec = TokenizerSpec::from_gguf(&gguf).expect("tokenizer metadata");
    let tokenizer = BpeTokenizer::from_spec(&spec).expect("BPE tokenizer");

    assert_eq!(
        tokenizer.encode("Hello world", true).unwrap(),
        vec![1, 36309, 2031]
    );
    assert_eq!(tokenizer.decode(&[1, 36309, 2031]).unwrap(), "Hello world");
}
